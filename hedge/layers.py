import torch
from torch import nn
from torch_geometric.nn.inits import glorot, zeros

class ChannelAttention(nn.Module):

    def __init__(self, hid_dim, num_relations, heads=4, attn_drop=0.1):
        super().__init__()
        self.R = num_relations
        self.h = heads
        self.q = nn.Parameter(torch.Tensor(heads, hid_dim))
        self.W = nn.Linear(hid_dim, hid_dim, bias=False)
        self.proj = nn.Linear(hid_dim * heads, hid_dim, bias=False)
        self.drop = nn.Dropout(attn_drop)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, per_rel_dict):
        R = self.R
        keys = list(range(R))
        mats = []
        N = None
        H = None
        device = None
        for r in keys:
            h = per_rel_dict.get(r, None)
            if h is None:
                if N is None:
                    for v in per_rel_dict.values():
                        if v is not None:
                            N, H = v.shape
                            device = v.device
                            break
                mats.append(torch.zeros(N, H, device=device))
            else:
                mats.append(h)
        X = torch.stack(mats, dim=0)
        R, N, H = X.shape
        Xw = self.W(X)
        attn_list = []
        head_out = []
        for i in range(self.h):
            qi = self.q[i].view(1, 1, H)
            score = torch.tanh(Xw * qi).sum(-1)
            alpha = torch.softmax(score, dim=0)
            attn_list.append(alpha)
            zi = (alpha.unsqueeze(-1) * X).sum(0)
            head_out.append(zi)
        Z = torch.cat(head_out, dim=-1)
        Z = self.proj(Z)
        A = torch.stack(attn_list, dim=0)
        A = self.drop(A)
        return (Z, A)

class ConvergenceGate(nn.Module):

    def __init__(self, hid_dim, counts_dim=7):
        super().__init__()
        self.G = nn.Parameter(torch.eye(3))
        self.mlp = nn.Sequential(nn.Linear(counts_dim, hid_dim // 2), nn.ReLU(), nn.Linear(hid_dim // 2, 3), nn.Sigmoid())
        self.evidence_scale = nn.Parameter(torch.ones(3))

    def forward(self, h_r, h_h, h_c, counts_gene):
        H = torch.stack([h_r, h_h, h_c], dim=1)
        A_static = torch.softmax(self.G, dim=1)
        Hs = torch.einsum('ij,njh->nih', A_static, H)
        gates = self.mlp(counts_gene)
        gates = gates * self.evidence_scale.view(1, -1)
        gates = torch.clamp(gates, 0.0, 1.5)
        fused = (Hs * gates.unsqueeze(-1)).sum(dim=1)
        aux = {'gates': gates, 'A_static': A_static}
        return (fused, aux)

def _get_act(name: str, dim: int=None):
    name = (name or 'relu').lower()
    if name == 'silu':
        return torch.nn.SiLU()
    if name == 'gelu':
        return torch.nn.GELU()
    if name == 'prelu':
        num_param = int(dim) if dim is not None and int(dim) > 0 else 1
        return torch.nn.PReLU(num_parameters=num_param)
    return torch.nn.ReLU()

class BasisRGCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_relations, num_bases=8, bias=True, dropout=0.0, act: str='relu', norm: str='layer', pre_ln: bool=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.R = num_relations
        self.B = max(1, min(num_bases, self.R))
        self.basis = nn.Parameter(torch.Tensor(self.B, in_dim, out_dim))
        self.coeff = nn.Parameter(torch.Tensor(self.R, self.B))
        self.root = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_dim)) if bias else None
        self.drop = nn.Dropout(dropout)
        norm = (norm or 'layer').lower()
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(out_dim)
            self._use_bn = True
        elif norm == 'none':
            self.norm = nn.Identity()
            self._use_bn = False
        else:
            self.norm = nn.LayerNorm(out_dim)
            self._use_bn = False
        self.pre_ln = bool(pre_ln)
        self.act = _get_act(act, dim=self.out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.basis)
        glorot(self.coeff)
        glorot(self.root.weight)
        if self.bias is not None:
            zeros(self.bias)
        if hasattr(self.norm, 'reset_parameters'):
            try:
                self.norm.reset_parameters()
            except Exception:
                pass

    def _aggregate_per_relation(self, x, edge_index, edge_type, edge_weight):
        N = x.size(0)
        Cin, Cout = (self.in_dim, self.out_dim)
        W = torch.einsum('rb,bio->rio', self.coeff, self.basis)
        per_rel = {}
        out = x.new_zeros(N, Cout)
        for r in range(self.R):
            mask = edge_type == r
            if not torch.any(mask):
                continue
            e_idx = edge_index[:, mask]
            src, dst = (e_idx[0], e_idx[1])
            msg = x[src] @ W[r]
            if edge_weight is not None:
                msg = msg * edge_weight[mask].unsqueeze(-1)
            deg = torch.bincount(dst, minlength=N).clamp(min=1).float()
            msg_agg = torch.zeros(N, Cout, device=x.device)
            msg_agg.index_add_(0, dst, msg)
            msg_agg = msg_agg / deg.unsqueeze(-1)
            per_rel[r] = msg_agg
            out += msg_agg
        return (out, per_rel)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        h_in = x
        if self.pre_ln:
            x = self.norm(x)
        out, per_rel = self._aggregate_per_relation(x, edge_index, edge_type, edge_weight)
        out = out + self.root(x)
        if self.bias is not None:
            out = out + self.bias
        if not self.pre_ln:
            out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        if h_in.shape == out.shape:
            out = out + h_in
        return (out, per_rel)
