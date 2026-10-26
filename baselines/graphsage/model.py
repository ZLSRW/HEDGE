import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_networks: int=1, dropout: float=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_networks = num_networks
        self.dropout = dropout
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_indices_list, edge_weights_list=None):
        assert len(edge_indices_list) >= 1, '需要至少一张图的 edge_index'
        edge_index = edge_indices_list[0]
        h = self.sage1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.sage2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.cls(h).squeeze(-1)
        N, H = h.size()
        h_experts = h.unsqueeze(1)
        gate_alpha = torch.ones(N, 1, device=h.device)
        aux = {'h_shared': h, 'h_experts': h_experts, 'gate_alpha': gate_alpha}
        return (logits, aux)
