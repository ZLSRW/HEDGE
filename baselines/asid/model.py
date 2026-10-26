import torch
import torch.nn as nn

class ASiD(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int=64, num_networks: int=1, dropout: float=0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_networks = int(num_networks)
        self.dropout = float(dropout)
        self.cls = nn.Linear(self.in_dim, 1, bias=True)

    def forward(self, x, edge_indices_list=None, edge_weights_list=None):
        logits = self.cls(x).squeeze(-1)
        aux = {'h_shared': x}
        return (logits, aux)
