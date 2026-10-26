import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DeepND(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_networks: int=1, dropout: float=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_networks = num_networks
        self.dropout = dropout
        self.shared_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout))
        self.gcns = nn.ModuleList()
        for _ in range(num_networks):
            conv1 = GCNConv(hidden_dim, hidden_dim)
            conv2 = GCNConv(hidden_dim, hidden_dim)
            self.gcns.append(nn.ModuleList([conv1, conv2]))
        gate_hidden = max(16, hidden_dim // 2)
        self.gate_mlp = nn.Sequential(nn.Linear(hidden_dim, gate_hidden), nn.LeakyReLU(0.1), nn.Dropout(dropout), nn.Linear(gate_hidden, num_networks))
        self.cls = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_indices_list, edge_weights_list=None):
        R = self.num_networks
        assert len(edge_indices_list) == R, f'num_networks={R}, but got {len(edge_indices_list)} edge_index tensors'
        if edge_weights_list is None:
            edge_weights_list = [None] * R
        h_shared = self.shared_mlp(x)
        h_experts = []
        for r in range(R):
            edge_index = edge_indices_list[r]
            edge_weight = edge_weights_list[r]
            h = F.relu(self.gcns[r][0](h_shared, edge_index, edge_weight))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcns[r][1](h, edge_index, edge_weight)
            h_experts.append(h)
        h_stack = torch.stack(h_experts, dim=0).transpose(0, 1)
        gate_logits = self.gate_mlp(h_shared)
        alpha = torch.softmax(gate_logits, dim=-1)
        z = (alpha.unsqueeze(-1) * h_stack).sum(dim=1)
        logits = self.cls(z).squeeze(-1)
        aux = {'h_shared': h_shared, 'h_experts': h_stack, 'gate_alpha': alpha}
        return (logits, aux)
