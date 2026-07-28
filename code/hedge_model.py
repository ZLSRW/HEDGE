import torch
import torch.nn as nn
from torch_geometric.nn import APPNP

from config import (
    REL2ID,
    RELATION_LIST,
    REL_CIRCQTL,
    REL_HQTL,
    REL_PPI,
    REL_RQTL,
)
from model_layers import BasisRGCNLayerV2, ChannelAttention, ConvergenceGate, _get_act


RID_R = REL2ID[REL_RQTL]
RID_H = REL2ID[REL_HQTL]
RID_CIRC = REL2ID[REL_CIRCQTL]
RID_PPI = REL2ID[REL_PPI]


class HEDGE(nn.Module):
    """Heterogeneous epigenetic diffusion-and-gating encoder."""

    def __init__(
        self,
        hparams=None,
        gene_extra_dim=0,
        in_dim=64,
        hidden_dim=128,
        num_relations=None,
        num_bases=8,
        dropout=0.1,
        use_convergence_gate=True,
        attn_heads=4,
        appnp_K=10,
        appnp_alpha=0.1,
        edge_drop_rate=0.1,
        smooth_lambda=1e-3,
    ):
        super().__init__()
        if hparams is not None:
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
            activation = hparams.activation
            normalization = hparams.norm
            pre_layer_norm = hparams.pre_ln
            num_layers = hparams.num_layers
        else:
            activation = "silu"
            normalization = "layer"
            pre_layer_norm = True
            num_layers = 2

        if num_relations is None:
            num_relations = len(RELATION_LIST)

        self.use_convergence_gate = use_convergence_gate
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.edge_drop_rate = float(edge_drop_rate)
        self.smooth_lambda = float(smooth_lambda)

        self.gene_proj = nn.Linear(in_dim + gene_extra_dim + 7, hidden_dim)
        self.rgcn_layers = nn.ModuleList(
            [
                BasisRGCNLayerV2(
                    hidden_dim,
                    hidden_dim,
                    num_relations,
                    num_bases=num_bases,
                    dropout=dropout,
                    act=activation,
                    norm=normalization,
                    pre_ln=pre_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.rel_attn = ChannelAttention(
            hidden_dim, num_relations, heads=attn_heads, attn_drop=dropout
        )
        self.convergence_gate = (
            ConvergenceGate(hidden_dim, counts_dim=7)
            if use_convergence_gate
            else None
        )
        self.appnp = APPNP(K=appnp_K, alpha=appnp_alpha, dropout=dropout)

        self.cls_rel = nn.Linear(hidden_dim, 1)
        self.cls_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            _get_act(activation, dim=hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.cls_prior = nn.Linear(hidden_dim, 1)
        self.mixture_layer = nn.Linear(3, 3)

    def _drop_edges(self, edge_index, edge_type, edge_weight):
        if self.training and self.edge_drop_rate > 0.0:
            edge_count = edge_index.size(1)
            keep = torch.rand(edge_count, device=edge_index.device) > self.edge_drop_rate
            if keep.sum() >= 10:
                return edge_index[:, keep], edge_type[keep], edge_weight[keep]
        return edge_index, edge_type, edge_weight

    def forward(
        self,
        x,
        edge_index,
        edge_type,
        edge_weight,
        gene_mask,
        convergence_counts,
        gene_extra=None,
        ablate=None,
        return_aux=True,
    ):
        node_count = x.size(0)
        device = x.device

        initial = x.clone()
        count_features = torch.zeros_like(x[:, :7])
        count_features[gene_mask] = convergence_counts[gene_mask]
        if gene_extra is None or gene_extra.size(1) == 0:
            gene_input = torch.cat(
                [initial[gene_mask], count_features[gene_mask]], dim=1
            )
        else:
            gene_input = torch.cat(
                [initial[gene_mask], gene_extra[gene_mask], count_features[gene_mask]],
                dim=1,
            )
        prior_gene_embedding = self.gene_proj(gene_input)
        hidden = initial.clone()
        hidden[gene_mask] = prior_gene_embedding

        first_edge_index, first_edge_type, first_edge_weight = self._drop_edges(
            edge_index, edge_type, edge_weight
        )
        hidden, first_relation_messages = self.rgcn_layers[0](
            hidden, first_edge_index, first_edge_type, first_edge_weight
        )
        relation_embedding, relation_attention = self.rel_attn(
            first_relation_messages
        )
        gene_relation_embedding = relation_embedding[gene_mask]

        def relation_message(relation_id):
            message = first_relation_messages.get(relation_id)
            if message is None:
                return torch.zeros_like(hidden)
            return message

        z_r = relation_message(RID_R)[gene_mask]
        z_h = relation_message(RID_H)[gene_mask]
        z_circ = relation_message(RID_CIRC)[gene_mask]

        ablate = {} if ablate is None else ablate
        if ablate.get("rqtl", False):
            z_r = torch.zeros_like(z_r)
        if ablate.get("hqtl", False):
            z_h = torch.zeros_like(z_h)
        if ablate.get("circqtl", False):
            z_circ = torch.zeros_like(z_circ)

        if self.use_convergence_gate:
            fused, gate_output = self.convergence_gate(
                z_r, z_h, z_circ, convergence_counts[gene_mask]
            )
        else:
            fused = (z_r + z_h + z_circ) / 3.0
            gate_output = {
                "gates": torch.ones(z_r.size(0), 3, device=device),
                "A_static": torch.eye(3, device=device),
            }

        for layer in self.rgcn_layers[1:]:
            next_edge_index, next_edge_type, next_edge_weight = self._drop_edges(
                edge_index, edge_type, edge_weight
            )
            hidden, _ = layer(
                hidden, next_edge_index, next_edge_type, next_edge_weight
            )

        ppi_mask = edge_type == RID_PPI
        ppi_edge_index = edge_index[:, ppi_mask]
        gene_edge_mask = gene_mask[ppi_edge_index[0]] & gene_mask[ppi_edge_index[1]]
        ppi_edge_index = ppi_edge_index[:, gene_edge_mask]

        gene_hidden = hidden[gene_mask]
        gene_ids = gene_mask.nonzero(as_tuple=False).view(-1)
        global_to_local = torch.full(
            (node_count,), -1, dtype=torch.long, device=device
        )
        global_to_local[gene_ids] = torch.arange(gene_hidden.size(0), device=device)
        local_ppi_edge_index = global_to_local[ppi_edge_index]
        diffused_gene_embedding = self.appnp(gene_hidden, local_ppi_edge_index)

        gate_diffusion_embedding = torch.cat(
            [fused, diffused_gene_embedding], dim=1
        )
        relation_logit = self.cls_rel(gene_relation_embedding).squeeze(-1)
        gate_logit = self.cls_gate(gate_diffusion_embedding).squeeze(-1)
        prior_logit = self.cls_prior(prior_gene_embedding).squeeze(-1)

        stream_logits = torch.stack(
            [relation_logit, gate_logit, prior_logit], dim=1
        )
        mixture = torch.softmax(self.mixture_layer(stream_logits), dim=1)
        mixed_logit = (mixture * stream_logits).sum(dim=1)
        logits = torch.full((node_count,), -1e9, device=device)
        logits[gene_mask] = mixed_logit

        if not return_aux:
            return logits

        if local_ppi_edge_index.numel() > 0:
            source, target = local_ppi_edge_index
            difference = (
                gate_diffusion_embedding[source]
                - gate_diffusion_embedding[target]
            )
            smooth_loss = difference.square().sum(dim=1).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=device)

        auxiliary = {
            "rel_attn": relation_attention.detach().cpu(),
            "gates": gate_output["gates"].detach().cpu(),
            "gates_train": gate_output["gates"],
            "A_static": gate_output["A_static"].detach().cpu(),
            "Ng": int(gene_mask.sum().item()),
            "gene_ids": gene_ids.detach().cpu(),
            "heads": {
                "rel": relation_logit,
                "gate": gate_logit,
                "prior": prior_logit,
            },
            "z_r": z_r,
            "z_h": z_h,
            "z_circ": z_circ,
            "z_rel_g": gene_relation_embedding,
            "z_fused": fused,
            "z_prior": prior_gene_embedding,
            "mix": mixture.detach().cpu(),
            "smooth": self.smooth_lambda * smooth_loss,
        }
        return logits, auxiliary
