from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_directed_edges(edge_index: torch.Tensor, node_type: torch.Tensor | None, n_nodes: int) -> torch.Tensor:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f'edge_index must be (2,E), got {tuple(edge_index.shape)}')
    src = edge_index[0].long()
    dst = edge_index[1].long()
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    if node_type is not None:
        nt = node_type.long()
        if nt.numel() != n_nodes:
            raise ValueError(f'node_type length {nt.numel()} != n_nodes {n_nodes}')
        src_t = nt[src]
        dst_t = nt[dst]
        snp_to_gene = (src_t == 1) & (dst_t == 0)
        gene_to_snp = (src_t == 0) & (dst_t == 1)
        same_type = src_t == dst_t
        same_src = torch.minimum(src, dst)
        same_dst = torch.maximum(src, dst)
        out_src = torch.cat([src[snp_to_gene], dst[gene_to_snp], same_src[same_type]])
        out_dst = torch.cat([dst[snp_to_gene], src[gene_to_snp], same_dst[same_type]])
    else:
        out_src = torch.minimum(src, dst)
        out_dst = torch.maximum(src, dst)
    key = out_src * n_nodes + out_dst
    uniq = torch.unique(key, sorted=False)
    out_src = (uniq // n_nodes).long()
    out_dst = (uniq % n_nodes).long()
    return torch.stack([out_src, out_dst], dim=0)

def _build_S_matrices(edge_index_dir: torch.Tensor, n_nodes: int, device: torch.device):
    src = edge_index_dir[0].to(device)
    dst = edge_index_dir[1].to(device)
    outdeg0 = torch.bincount(src, minlength=n_nodes).float()
    outdeg0 = torch.clamp(outdeg0, min=1.0)
    vals0 = 1.0 / outdeg0[src]
    idx0 = torch.stack([dst, src], dim=0)
    S0 = torch.sparse_coo_tensor(idx0, vals0, (n_nodes, n_nodes), device=device).coalesce()
    outdeg1 = outdeg0 + 1.0
    vals1_e = 1.0 / outdeg1[src]
    idx1_e = idx0
    self_nodes = torch.arange(n_nodes, device=device)
    idx1_s = torch.stack([self_nodes, self_nodes], dim=0)
    vals1_s = 1.0 / outdeg1[self_nodes]
    idx1 = torch.cat([idx1_e, idx1_s], dim=1)
    vals1 = torch.cat([vals1_e, vals1_s], dim=0)
    S1 = torch.sparse_coo_tensor(idx1, vals1, (n_nodes, n_nodes), device=device).coalesce()
    return (S0, S1)

class GrIDNet(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_networks: int=1, dropout: float=0.3, grid_num_layers: int=10, n_nodes: int | None=None, node_type: torch.Tensor | None=None):
        super().__init__()
        if n_nodes is None:
            raise ValueError('n_nodes must be provided for GrID-Net (per-node W,b parameters).')
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.grid_num_layers = int(grid_num_layers)
        self.n_nodes = int(n_nodes)
        if node_type is not None:
            self.register_buffer('node_type', node_type.long())
        else:
            self.node_type = None
        self.in_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.W_list = nn.ParameterList()
        self.b_list = nn.ParameterList()
        for _ in range(self.grid_num_layers):
            W = nn.Parameter(torch.ones((self.n_nodes, 1)), requires_grad=True)
            b = nn.Parameter(torch.ones((self.n_nodes, 1)), requires_grad=True)
            nn.init.xavier_normal_(W.data)
            nn.init.xavier_normal_(b.data)
            self.W_list.append(W)
            self.b_list.append(b)
        self.classifier = nn.Linear(self.hidden_dim, 1)
        self.act = nn.Tanh()
        self._cached = {'built': False}

    def _ensure_S(self, edge_index: torch.Tensor, device: torch.device):
        if self._cached.get('built', False) and self._cached.get('device', None) == device:
            return
        edge_dir = _build_directed_edges(edge_index, self.node_type if hasattr(self, 'node_type') else None, self.n_nodes)
        S0, S1 = _build_S_matrices(edge_dir, self.n_nodes, device)
        self._cached = {'built': True, 'device': device, 'S0': S0, 'S1': S1}

    def forward(self, x: torch.Tensor, edge_indices_list, edge_weights_list=None):
        if not isinstance(edge_indices_list, (list, tuple)) or len(edge_indices_list) == 0:
            raise ValueError('edge_indices_list must be a non-empty list/tuple')
        edge_index = edge_indices_list[0]
        device = x.device
        self._ensure_S(edge_index, device)
        S0 = self._cached['S0']
        S1 = self._cached['S1']
        h = self.act(self.in_proj(x))
        out_list = []
        for l in range(self.grid_num_layers):
            S = S0 if l == 0 else S1
            agg = torch.sparse.mm(S, h)
            h = self.act(agg * self.W_list[l] + self.b_list[l])
            h = F.dropout(h, p=self.dropout, training=self.training)
            out_list.append(h)
        h_shared = torch.stack(out_list, dim=0).mean(0)
        logits = self.classifier(h_shared).squeeze(-1)
        aux = {'h_shared': h_shared, 'grid_h_layers_mean': h_shared}
        return (logits, aux)
