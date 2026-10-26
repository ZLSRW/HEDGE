import torch
from torch import nn
from torch_geometric.nn import APPNP
from .config import RELATION_LIST, HyperParams
from .layers import BasisRGCNLayer, ChannelAttention, ConvergenceGate, _get_act

def _find_rel(name):
    for i, r in enumerate(RELATION_LIST):
        if name.lower() in r.lower():
            return i
    return None
RID_R = _find_rel('rqtl') or 0
RID_H = _find_rel('hqtl') or 1
RID_CIRC = _find_rel('circ') or 2
RID_PPI = _find_rel('ppi') or len(RELATION_LIST) - 1

class HEDGE(nn.Module):

    def __init__(self, hparams=None):
        super().__init__()
        hparams = hparams or HyperParams()
        in_dim = 64
        hidden_dim = hparams.dim
        num_relations = len(RELATION_LIST)
        num_bases = hparams.basis
        dropout = hparams.dropout
        use_convergence_gate = hparams.enable_convergence_gate
        attn_heads = hparams.attn_heads
        appnp_K = hparams.appnp_K
        appnp_alpha = hparams.appnp_alpha
        edge_drop_rate = hparams.edge_drop_rate
        smooth_lambda = hparams.smooth_lambda
        if hidden_dim != in_dim:
            raise ValueError('hidden dimension must match the 64-dimensional graph input')
        self.use_convergence_gate = use_convergence_gate
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.edge_drop_rate = float(edge_drop_rate)
        self.smooth_lambda = float(smooth_lambda)
        self.gene_proj = nn.Linear(in_dim + 7, hidden_dim)
        act = getattr(hparams, 'activation', 'silu')
        norm = getattr(hparams, 'norm', 'layer')
        pre_ln = bool(getattr(hparams, 'pre_ln', True))
        L = int(getattr(hparams, 'num_layers', 2))
        self.rgcn_layers = nn.ModuleList([BasisRGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases=num_bases, dropout=dropout, act=act, norm=norm, pre_ln=pre_ln) for _ in range(L)])
        self.rel_attn = ChannelAttention(hidden_dim, num_relations, heads=attn_heads, attn_drop=dropout)
        self.cgate = ConvergenceGate(hidden_dim, counts_dim=7) if use_convergence_gate else None
        self.appnp = APPNP(K=appnp_K, alpha=appnp_alpha, dropout=dropout)
        self.cls_rel = nn.Linear(hidden_dim, 1)
        self.cls_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), _get_act(act, dim=hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.cls_prior = nn.Linear(hidden_dim, 1)
        self.mix_logits = nn.Parameter(torch.zeros(3))

    def _drop_edges(self, edge_index, edge_type, edge_weight):
        if self.training and self.edge_drop_rate > 0.0:
            E = edge_index.size(1)
            keep = torch.rand(E, device=edge_index.device) > self.edge_drop_rate
            if keep.sum() < 10:
                return (edge_index, edge_type, edge_weight)
            return (edge_index[:, keep], edge_type[keep], edge_weight[keep])
        return (edge_index, edge_type, edge_weight)

    def forward(self, x, edge_index, edge_type, edge_weight, gene_mask, convergence_counts, ablate=None, return_aux=True):
        ablate = ablate or {}
        if ablate.get('rgcn', False):
            edge_weight = torch.zeros_like(edge_weight)
        N = x.size(0)
        device = x.device
        H0 = x.clone()
        counts_pad = torch.zeros_like(x[:, :7])
        counts_pad[gene_mask] = convergence_counts[gene_mask]
        gene_cat = torch.cat([H0[gene_mask], counts_pad[gene_mask]], dim=1)
        H0g = self.gene_proj(gene_cat)
        H = H0.clone()
        H[gene_mask] = H0g
        ei1, et1, ew1 = self._drop_edges(edge_index, edge_type, edge_weight)
        h_cur, per_rel1 = self.rgcn_layers[0](H, ei1, et1, ew1)
        z_rel, A_rel = self.rel_attn(per_rel1)
        z_rel_g = z_rel[gene_mask]

        def pick_rel(rid):
            v = per_rel1.get(rid, None)
            if v is None:
                return torch.zeros_like(h_cur)
            return v
        h_r = pick_rel(RID_R)[gene_mask]
        h_h = pick_rel(RID_H)[gene_mask]
        h_c = pick_rel(RID_CIRC)[gene_mask]
        if ablate.get('r', False):
            h_r = torch.zeros_like(h_r)
        if ablate.get('h', False):
            h_h = torch.zeros_like(h_h)
        if ablate.get('circ', False):
            h_c = torch.zeros_like(h_c)
        if self.use_convergence_gate:
            fused, aux_gate = self.cgate(h_r, h_h, h_c, convergence_counts[gene_mask])
        else:
            fused = (h_r + h_h + h_c) / 3.0
            aux_gate = {'gates': torch.ones(h_r.size(0), 3, device=device), 'A_static': torch.eye(3, device=device)}
        for li in range(1, len(self.rgcn_layers)):
            ei2, et2, ew2 = self._drop_edges(edge_index, edge_type, edge_weight)
            h_cur, _ = self.rgcn_layers[li](h_cur, ei2, et2, ew2)
        et = edge_type
        ppi_mask = et == RID_PPI
        ei_ppi = edge_index[:, ppi_mask]
        is_gene_src = gene_mask[ei_ppi[0]]
        is_gene_dst = gene_mask[ei_ppi[1]]
        keep = is_gene_src & is_gene_dst
        ei_ppi = ei_ppi[:, keep]
        Hg = h_cur[gene_mask]
        Ng = Hg.size(0)
        gene_ids = gene_mask.nonzero(as_tuple=False).view(-1)
        gid2loc = torch.full((N,), -1, dtype=torch.long, device=device)
        gid2loc[gene_ids] = torch.arange(Ng, device=device)
        ei_ppi_local = gid2loc[ei_ppi]
        Hg_diff = Hg if ablate.get('appnp_off', False) else self.appnp(Hg, ei_ppi_local)
        Zg = torch.cat([fused, Hg_diff], dim=1)
        logit_rel = self.cls_rel(z_rel_g).squeeze(-1)
        logit_gate = self.cls_gate(Zg).squeeze(-1)
        logit_prior = self.cls_prior(H0g).squeeze(-1)
        mix = torch.softmax(self.mix_logits, dim=0)
        logit_mixed = mix[0] * logit_rel + mix[1] * logit_gate + mix[2] * logit_prior
        logits = torch.full((N,), -1000000000.0, device=device)
        logits[gene_mask] = logit_mixed
        if not return_aux:
            return logits
        if ei_ppi_local.numel() > 0:
            u, v = ei_ppi_local
            diff = Zg[u] - Zg[v]
            smooth = (diff * diff).sum(dim=1).mean()
        else:
            smooth = torch.tensor(0.0, device=device)
        aux = {'rel_attn': A_rel.detach().cpu(), 'gates': aux_gate['gates'].detach().cpu(), 'A_static': aux_gate['A_static'].detach().cpu(), 'Ng': int(gene_mask.sum().item()), 'gene_ids': gene_ids.detach().cpu(), 'heads': {'rel': logit_rel.detach(), 'gate': logit_gate.detach(), 'prior': logit_prior.detach()}, 'z_rel_g': z_rel_g, 'mix': mix.detach().cpu(), 'smooth': self.smooth_lambda * smooth}
        return (logits, aux)
