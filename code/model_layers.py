import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot, zeros


def _get_act(name: str, dim: int = None):
    activation = name.lower()
    if activation == "silu":
        return nn.SiLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "prelu":
        parameter_count = int(dim) if dim is not None else 1
        return nn.PReLU(num_parameters=parameter_count)
    if activation == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class BasisRGCNLayerV2(nn.Module):
    """Basis-decomposed relation-specific graph convolution layer."""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_relations,
        num_bases=8,
        bias=True,
        dropout=0.0,
        act="relu",
        norm="layer",
        pre_ln=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = max(1, min(num_bases, num_relations))

        self.basis = nn.Parameter(
            torch.empty(self.num_bases, in_dim, out_dim)
        )
        self.coeff = nn.Parameter(
            torch.empty(self.num_relations, self.num_bases)
        )
        self.root = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None
        self.dropout = nn.Dropout(dropout)

        normalization = norm.lower()
        if normalization == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        elif normalization == "none":
            self.norm = nn.Identity()
        elif normalization == "layer":
            self.norm = nn.LayerNorm(out_dim)
        else:
            raise ValueError(f"Unsupported normalization: {norm}")

        self.pre_ln = bool(pre_ln)
        self.activation = _get_act(act, dim=out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.basis)
        glorot(self.coeff)
        glorot(self.root.weight)
        if self.bias is not None:
            zeros(self.bias)
        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()

    def _aggregate_per_relation(
        self, x, edge_index, edge_type, edge_weight
    ):
        node_count = x.size(0)
        relation_weights = torch.einsum(
            "rb,bio->rio", self.coeff, self.basis
        )
        per_relation = {}
        output = x.new_zeros(node_count, self.out_dim)

        for relation_id in range(self.num_relations):
            relation_mask = edge_type == relation_id
            if not torch.any(relation_mask):
                continue
            relation_edges = edge_index[:, relation_mask]
            source, target = relation_edges
            messages = x[source] @ relation_weights[relation_id]
            if edge_weight is not None:
                messages = messages * edge_weight[relation_mask].unsqueeze(-1)

            source_degree = torch.bincount(
                source, minlength=node_count
            ).clamp(min=1).float()
            target_degree = torch.bincount(
                target, minlength=node_count
            ).clamp(min=1).float()
            normalization = torch.rsqrt(
                source_degree[source] * target_degree[target]
            ).unsqueeze(-1)
            messages = messages * normalization

            aggregated = x.new_zeros(node_count, self.out_dim)
            aggregated.index_add_(0, target, messages)
            per_relation[relation_id] = aggregated
            output = output + aggregated

        return output, per_relation

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        residual = x
        normalized_input = self.norm(x) if self.pre_ln else x
        output, per_relation = self._aggregate_per_relation(
            normalized_input, edge_index, edge_type, edge_weight
        )
        output = output + self.root(normalized_input)
        if self.bias is not None:
            output = output + self.bias
        if not self.pre_ln:
            output = self.norm(output)
        output = self.dropout(self.activation(output))
        if residual.shape == output.shape:
            output = output + residual
        return output, per_relation


class ChannelAttention(nn.Module):
    """Multi-head attention over relation-specific node messages."""

    def __init__(self, hidden_dim, num_relations, heads=4, attn_drop=0.1):
        super().__init__()
        self.num_relations = num_relations
        self.heads = heads
        self.query = nn.Parameter(torch.empty(heads, hidden_dim))
        self.relation_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(
            hidden_dim * heads, hidden_dim, bias=False
        )
        self.dropout = nn.Dropout(attn_drop)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.relation_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(self, per_relation):
        first_message = next(iter(per_relation.values()))
        node_count, hidden_dim = first_message.shape
        relation_stack = torch.stack(
            [
                per_relation.get(
                    relation_id, first_message.new_zeros(node_count, hidden_dim)
                )
                for relation_id in range(self.num_relations)
            ],
            dim=0,
        )
        projected = self.relation_projection(relation_stack)

        attention_by_head = []
        output_by_head = []
        for head_id in range(self.heads):
            query = self.query[head_id].view(1, 1, hidden_dim)
            score = torch.tanh(projected * query).sum(-1)
            attention = torch.softmax(score, dim=0)
            attention_by_head.append(attention)
            output_by_head.append(
                (attention.unsqueeze(-1) * relation_stack).sum(0)
            )

        output = self.output_projection(torch.cat(output_by_head, dim=-1))
        attention = torch.stack(attention_by_head, dim=0)
        return output, self.dropout(attention)


class ConvergenceGate(nn.Module):
    """Static inter-channel mixing with gene-specific convergence gates."""

    def __init__(self, hidden_dim, counts_dim=7):
        super().__init__()
        self.static_matrix = nn.Parameter(torch.eye(3))
        self.dynamic_gate = nn.Sequential(
            nn.Linear(counts_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, z_r, z_h, z_circ, convergence_counts):
        channel_embeddings = torch.stack([z_r, z_h, z_circ], dim=1)
        static_attention = torch.softmax(self.static_matrix, dim=1)
        mixed_channels = torch.einsum(
            "ij,njh->nih", static_attention, channel_embeddings
        )
        gates = self.dynamic_gate(convergence_counts)
        fused = (mixed_channels * gates.unsqueeze(-1)).sum(dim=1)
        return fused, {"gates": gates, "A_static": static_attention}
