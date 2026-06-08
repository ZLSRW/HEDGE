
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros

class BasisRGCNLayer(nn.Module):
    """
    R-GCN with basis decomposition + relation-degree normalization.
    Returns:
        h_out: (N, hidden)
        per_rel: dict[rid] -> (N, hidden) aggregated messages before sum (for interpretability)
    """
    def __init__(self, in_dim, out_dim, num_relations, num_bases=8, bias=True, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.R = num_relations
        self.B = max(1, min(num_bases, self.R))

        self.basis = nn.Parameter(torch.Tensor(self.B, in_dim, out_dim))
        self.coeff = nn.Parameter(torch.Tensor(self.R, self.B))

        self.root = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_dim)) if bias else None
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.basis); glorot(self.coeff)
        glorot(self.root.weight)
        if self.bias is not None: zeros(self.bias)
        self.ln.reset_parameters()

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        """
        x: (N, Cin)
        edge_index: (2, E)
        edge_type: (E, )
        edge_weight: (E, ) optional
        """
        N = x.size(0)
        Cin, Cout = self.in_dim, self.out_dim

        # materialize per-relation weight from basis
        W = torch.einsum("rb,bio->rio", self.coeff, self.basis)  # (R, Cin, Cout)

        # split edges per relation
        per_rel = {}
        out = x.new_zeros(N, Cout)

        for r in range(self.R):
            mask = (edge_type == r)
            if not torch.any(mask):
                continue
            e_idx = edge_index[:, mask]
            src, dst = e_idx[0], e_idx[1]
            w = W[r]  # (Cin, Cout)
            msg = x[src] @ w  # (Er, Cout)

            if edge_weight is not None:
                wr = edge_weight[mask].unsqueeze(-1)
                msg = msg * wr

            # degree normalize per relation on dst
            deg = torch.bincount(dst, minlength=N).clamp(min=1).float()
            msg_agg = torch.zeros(N, Cout, device=x.device)
            msg_agg.index_add_(0, dst, msg)
            msg_agg = msg_agg / deg.unsqueeze(-1)

            per_rel[r] = msg_agg
            out += msg_agg

        out = out + self.root(x)
        if self.bias is not None: out = out + self.bias
        out = self.ln(out)
        out = F.relu(out)
        out = self.drop(out)
        return out, per_rel

class ChannelAttention(nn.Module):
    """
    Relation-channel attention over per-relation aggregated messages.
    Multi-head attention across R channels for each node.
    Returns mixing weights per relation (for interpretability) and the mixed vector.
    """
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
        """
        per_rel_dict: dict[rid] -> (N, H)
        """
        R = self.R
        # stack (R, N, H), missing relations become zeros
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
        X = torch.stack(mats, dim=0)  # (R, N, H)
        R, N, H = X.shape
        Xw = self.W(X)                # (R, N, H)

        # heads: compute attention per head
        attn_list = []
        head_out = []
        for i in range(self.h):
            qi = self.q[i].view(1, 1, H)         # (1,1,H)
            score = torch.tanh(Xw * qi).sum(-1)  # (R, N)
            alpha = torch.softmax(score, dim=0)  # across relations
            attn_list.append(alpha)              # (R, N)
            zi = (alpha.unsqueeze(-1) * X).sum(0)  # (N, H)
            head_out.append(zi)
        Z = torch.cat(head_out, dim=-1)          # (N, H*h)
        Z = self.proj(Z)                          # (N, H)
        A = torch.stack(attn_list, dim=0)         # (h, R, N)
        A = self.drop(A)
        return Z, A  # A for interpretability: heads x R x N

class CrosstalkGate(nn.Module):
    """
    CrosstalkGate++:
    - Static 3x3 attention (learnable)
    - Dynamic gate from 7-d counts
    - Per-relation evidence scalers
    Returns fused vector and a dict of components for logging.
    """
    def __init__(self, hid_dim, counts_dim=7):
        super().__init__()
        self.G = nn.Parameter(torch.eye(3))  # static 3x3
        self.mlp = nn.Sequential(
            nn.Linear(counts_dim, hid_dim//2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, 3),
            nn.Sigmoid()
        )
        self.evidence_scale = nn.Parameter(torch.ones(3))

    def forward(self, h_me, h_h, h_c, counts_gene):
        # static attention
        H = torch.stack([h_me, h_h, h_c], dim=1)           # (Ng, 3, H)
        A_static = torch.softmax(self.G, dim=1)            # (3,3)
        Hs = torch.einsum('ij,njh->nih', A_static, H)      # (Ng, 3, H)  mix along channel dim

        # dynamic gates
        gates = self.mlp(counts_gene)                      # (Ng, 3)
        gates = gates * self.evidence_scale.view(1, -1)    # learnable scaler per channel
        gates = torch.clamp(gates, 0.0, 1.5)               # keep in a reasonable range

        # fuse (weighted sum across channels)
        fused = (Hs * gates.unsqueeze(-1)).sum(dim=1)      # (Ng, H)

        aux = {
            "gates": gates,                # (Ng,3)
            "A_static": A_static,          # (3,3)
        }
        return fused, aux

# ==== Appended: Activation factory and BasisRGCNLayerV2 (backwards compatible) ====
# 原: def _get_act(name: str):
def _get_act(name: str, dim: int = None):
    name = (name or "relu").lower()
    if name == "silu":
        return torch.nn.SiLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "prelu":
        # 通道级 PReLU：当提供 dim (通道数) 时用逐通道参数，否则退化为单参数
        num_param = int(dim) if (dim is not None and int(dim) > 0) else 1
        return torch.nn.PReLU(num_parameters=num_param)
    return torch.nn.ReLU()


class BasisRGCNLayerV2(nn.Module):
    # Enhanced R-GCN layer with basis decomposition, optional norm (Layer/Batch/None),
    # Pre-LN or Post-LN, configurable activation (relu/silu/gelu), and residual.
    # Forward signature matches original: (x, edge_index, edge_type, edge_weight=None)
    def __init__(self, in_dim, out_dim, num_relations, num_bases=8, bias=True, dropout=0.0,
                 act: str = "relu", norm: str = "layer", pre_ln: bool = True):
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

        norm = (norm or "layer").lower()
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
            self._use_bn = True
        elif norm == "none":
            self.norm = nn.Identity()
            self._use_bn = False
        else:
            self.norm = nn.LayerNorm(out_dim)
            self._use_bn = False

        self.pre_ln = bool(pre_ln)
        self.act = _get_act(act, dim=self.out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.basis); glorot(self.coeff); glorot(self.root.weight)
        if self.bias is not None:
            zeros(self.bias)
        if hasattr(self.norm, "reset_parameters"):
            try:
                self.norm.reset_parameters()
            except Exception:
                pass

    def _aggregate_per_relation(self, x, edge_index, edge_type, edge_weight):
        N = x.size(0)
        Cin, Cout = self.in_dim, self.out_dim
        W = torch.einsum("rb,bio->rio", self.coeff, self.basis)  # (R,Cin,Cout)
        per_rel = {}
        out = x.new_zeros(N, Cout)
        for r in range(self.R):
            mask = (edge_type == r)
            if not torch.any(mask):
                continue
            e_idx = edge_index[:, mask]
            src, dst = e_idx[0], e_idx[1]
            msg = x[src] @ W[r]
            if edge_weight is not None:
                msg = msg * edge_weight[mask].unsqueeze(-1)
            deg = torch.bincount(dst, minlength=N).clamp(min=1).float()
            msg_agg = torch.zeros(N, Cout, device=x.device)
            msg_agg.index_add_(0, dst, msg)
            msg_agg = msg_agg / deg.unsqueeze(-1)
            per_rel[r] = msg_agg
            out += msg_agg
        return out, per_rel

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
        return out, per_rel
