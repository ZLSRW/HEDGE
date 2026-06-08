
import torch
import torch.nn as nn
import torch.nn.functional as F

# Robust APPNP import with fallbacks
try:
    from torch_geometric.nn import APPNP  # most versions expose it here
except Exception:
    try:
        from torch_geometric.nn.models import APPNP  # older API
    except Exception:
        APPNP = None

from config1 import RELATION_LIST
from model_layers import BasisRGCNLayerV2 as BasisRGCNLayer, ChannelAttention, CrosstalkGate, _get_act


def _find_rel(name):
    for i, r in enumerate(RELATION_LIST):
        if name.lower() in r.lower():
            return i
    return None

RID_ME   = _find_rel("meqtl")   or 0
RID_H    = _find_rel("hqtl")    or 1
RID_CIRC = _find_rel("circ")    or 2
RID_PPI  = _find_rel("ppi")     or (len(RELATION_LIST)-1)

class IdentityAPPNP(nn.Module):
    # \"\"\"Fallback if APPNP is unavailable: no diffusion (identity mapping).\"\"\"
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, edge_index, edge_weight=None):
        return x

class XTalkRGCN(nn.Module):
    # \"\"\"
    # Upgraded XTalk-RGCN++ with:
    #   (I) Relation-channel attention (ChannelAttention)
    #   (II) CrosstalkGate++ (static 3x3 + dynamic gate)
    #   (III) APPNP diffusion head on PPI subgraph
    #   (IV) Tri-head logits (rel/gate+diff/prior) + learnable mixture
    #   (V) Laplacian smoothing regularizer on gene embeddings
    #   (VI) Training-time DropEdge
    # \"\"\"
    def __init__(self, hparams=None, gene_extra_dim=0,
                 in_dim=64, hidden_dim=128, num_relations=None, num_bases=8,
                 dropout=0.1, use_xtalk_gate=True, use_gene_counts_proj=True,
                 attn_heads=4, appnp_K=10, appnp_alpha=0.1, aux_lambda=0.1,
                 edge_drop_rate=0.1, smooth_lambda=1e-3):
        super().__init__()
        if hparams is not None:
            in_dim = 64
            hidden_dim = hparams.dim
            num_relations = len(RELATION_LIST)
            num_bases = hparams.basis
            dropout = hparams.dropout
            use_xtalk_gate = hparams.enable_xtalk_gate
            attn_heads = getattr(hparams, "attn_heads", 4)
            appnp_K = getattr(hparams, "appnp_K", 10)
            appnp_alpha = getattr(hparams, "appnp_alpha", 0.1)
            aux_lambda = getattr(hparams, "aux_lambda", 0.1)
            edge_drop_rate = getattr(hparams, "edge_drop_rate", 0.1)
            smooth_lambda = getattr(hparams, "smooth_lambda", 1e-3)

        self.use_xtalk_gate = use_xtalk_gate
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.edge_drop_rate = float(edge_drop_rate)
        self.smooth_lambda = float(smooth_lambda)

        # (0) gene projection: [64 (+ extra) + 7] -> hidden
        self.gene_proj = nn.Linear(in_dim + gene_extra_dim + 7, hidden_dim)

        # (1) stacked R-GCN layers (configurable)
        act = getattr(hparams, "activation", "silu")
        norm = getattr(hparams, "norm", "layer")  # "layer" | "batch" | "none"
        pre_ln = bool(getattr(hparams, "pre_ln", True))
        L = int(getattr(hparams, "num_layers", 2))
        self.rgcn_layers = nn.ModuleList([
            BasisRGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases=num_bases,
                            dropout=dropout, act=act, norm=norm, pre_ln=pre_ln)
            for _ in range(L)
        ])

        # (I) relation-channel attention
        self.rel_attn = ChannelAttention(hidden_dim, num_relations, heads=attn_heads, attn_drop=dropout)

        # (II) crosstalk gate++ for me/h/circ
        self.cgate = CrosstalkGate(hidden_dim, counts_dim=7) if use_xtalk_gate else None

        # (III) APPNP diffusion head on PPI subgraph, gene-only
        self.appnp = (APPNP(K=appnp_K, alpha=appnp_alpha, dropout=dropout) if APPNP is not None else IdentityAPPNP())

        # (IV) tri-head classifiers
        self.cls_rel   = nn.Linear(hidden_dim, 1)        # head-A: relation attention vector (gene)
        # 原: nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), _get_act(act), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.cls_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            _get_act(act, dim=hidden_dim),  # 这里传 hidden_dim 以便 prelu 做逐通道参数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.cls_prior = nn.Linear(hidden_dim, 1)        # head-C: prior gene projection H0g

        # learnable mixture weights (softmax to [0,1] and sum to 1)
        self.mix_logits = nn.Parameter(torch.zeros(3))

    def _drop_edges(self, edge_index, edge_type, edge_weight):
        if self.training and self.edge_drop_rate > 0.0:
            E = edge_index.size(1)
            keep = torch.rand(E, device=edge_index.device) > self.edge_drop_rate
            if keep.sum() < 10:  # guard
                return edge_index, edge_type, edge_weight
            return edge_index[:, keep], edge_type[keep], edge_weight[keep]
        return edge_index, edge_type, edge_weight

    def forward(self, x, edge_index, edge_type, edge_weight, gene_mask, xtalk_counts, gene_extra=None,
                ablate=None, return_aux=True):
        N = x.size(0)
        device = x.device

        # build enhanced gene input
        H0 = x.clone()
        counts_pad = torch.zeros_like(x[:, :7]); counts_pad[gene_mask] = xtalk_counts[gene_mask]
        if gene_extra is None or gene_extra.size(1) == 0:
            gene_cat = torch.cat([H0[gene_mask], counts_pad[gene_mask]], dim=1)
        else:
            gene_cat = torch.cat([H0[gene_mask], gene_extra[gene_mask], counts_pad[gene_mask]], dim=1)
        H0g = self.gene_proj(gene_cat)                     # (Ng, H)
        H = H0.clone()
        H[gene_mask] = H0g

        # DropEdge & first R-GCN layer
        ei1, et1, ew1 = self._drop_edges(edge_index, edge_type, edge_weight)
        h_cur, per_rel1 = self.rgcn_layers[0](H, ei1, et1, ew1)
        # relation-channel attention on first-layer per-relation outputs
        z_rel, A_rel = self.rel_attn(per_rel1)            # (N,H), (heads,R,N)
        z_rel_g = z_rel[gene_mask]                        # (Ng, H)

        # crosstalk paths from per_rel1 (me/h/circ)
        def pick_rel(rid):
            v = per_rel1.get(rid, None)
            if v is None: return torch.zeros_like(h_cur)
            return v

        h_me = pick_rel(RID_ME)[gene_mask]                # (Ng, H)
        h_h  = pick_rel(RID_H)[gene_mask]
        h_c  = pick_rel(RID_CIRC)[gene_mask]

        if ablate is None: ablate = {}
        if ablate.get('me', False):   h_me = torch.zeros_like(h_me)
        if ablate.get('h', False):    h_h  = torch.zeros_like(h_h)
        if ablate.get('circ', False): h_c  = torch.zeros_like(h_c)

        if self.use_xtalk_gate:
            fused, aux_gate = self.cgate(h_me, h_h, h_c, xtalk_counts[gene_mask])
        else:
            fused = (h_me + h_h + h_c) / 3.0
            aux_gate = {"gates": torch.ones(h_me.size(0), 3, device=device),
                        "A_static": torch.eye(3, device=device)}

        # remaining R-GCN layers (if any)
        for li in range(1, len(self.rgcn_layers)):
            ei2, et2, ew2 = self._drop_edges(edge_index, edge_type, edge_weight)
            h_cur, _ = self.rgcn_layers[li](h_cur, ei2, et2, ew2)

        # APPNP diffusion on PPI subgraph, gene-only
        et = edge_type
        ppi_mask = (et == RID_PPI)
        ei_ppi = edge_index[:, ppi_mask]
        is_gene_src = gene_mask[ei_ppi[0]]
        is_gene_dst = gene_mask[ei_ppi[1]]
        keep = is_gene_src & is_gene_dst
        ei_ppi = ei_ppi[:, keep]

        Hg = h_cur[gene_mask]
        # re-index PPI edges to gene-only local indices [0..Ng-1] to avoid out-of-bounds on APPNP
        Ng = Hg.size(0)
        gene_ids = gene_mask.nonzero(as_tuple=False).view(-1)          # (Ng,)
        gid2loc = torch.full((N,), -1, dtype=torch.long, device=device)
        gid2loc[gene_ids] = torch.arange(Ng, device=device)
        ei_ppi_local = gid2loc[ei_ppi]                                 # (2, E_ppi_gene)
        Hg_diff = self.appnp(Hg, ei_ppi_local)                          # (Ng, H)

        # final gene representation (head-B backbone)
        Zg = torch.cat([fused, Hg_diff], dim=1)           # (Ng, 2H)

        # --- tri-head logits ---
        logit_rel   = self.cls_rel(z_rel_g).squeeze(-1)   # (Ng,)
        logit_gate  = self.cls_gate(Zg).squeeze(-1)       # (Ng,)
        logit_prior = self.cls_prior(H0g).squeeze(-1)     # (Ng,)

        mix = torch.softmax(self.mix_logits, dim=0)       # (3,)
        logit_mixed = mix[0]*logit_rel + mix[1]*logit_gate + mix[2]*logit_prior

        logits = torch.full((N,), -1e9, device=device)
        logits[gene_mask] = logit_mixed

        if not return_aux:
            return logits

        # Laplacian smoothing loss on Zg (gene-only PPI)
        if ei_ppi_local.numel() > 0:
            u, v = ei_ppi_local
            diff = Zg[u] - Zg[v]
            smooth = (diff*diff).sum(dim=1).mean()
        else:
            smooth = torch.tensor(0.0, device=device)

        aux = {
            "rel_attn": A_rel.detach().cpu(),            # (heads,R,N)
            "gates": aux_gate["gates"].detach().cpu(),   # (Ng,3)
            "A_static": aux_gate["A_static"].detach().cpu(),  # (3,3)
            "Ng": int(gene_mask.sum().item()),
            "gene_ids": gene_ids.detach().cpu(),         # global ids for mapping
            "heads": {
                "rel": logit_rel.detach(),
                "gate": logit_gate.detach(),
                "prior": logit_prior.detach()
            },
            "z_rel_g": z_rel_g,
            "mix": mix.detach().cpu(),
            "smooth": self.smooth_lambda * smooth
        }
        return logits, aux
