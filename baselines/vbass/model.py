from typing import List
import torch
from torch import nn
from .layers import Classifier, Encoder, SharedLayers

class VBASSBackbone(nn.Module):

    def __init__(self, in_dim: int, shared_hidden: List[int], latent_dim=32, num_classes=2, dropout=0.1):
        super().__init__()
        self.shared = SharedLayers(dims=[in_dim, shared_hidden], use_batch_norm=True, use_layer_norm=True, dropout_rate=dropout)
        width = shared_hidden[-1] if shared_hidden else in_dim
        self.encoder = Encoder(dims=[width, [], latent_dim], use_batch_norm=True, use_layer_norm=True, dropout_rate=dropout)
        self.classifier = Classifier(dims=[width, [], num_classes], use_batch_norm=True, use_layer_norm=True, dropout_rate=dropout)

    def forward(self, x):
        h = self.shared(x)
        z, mu, sigma = self.encoder(h)
        log_probability = self.classifier(h)
        logits = log_probability.squeeze(1) if log_probability.size(1) == 1 else log_probability[:, 1] - log_probability[:, 0]
        return (logits, {'h_shared': h, 'z_mu': mu, 'z_sigma': sigma, 'z': z})

class VBASS(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_networks=1, dropout=0.1, vbass_latent_dim=32):
        super().__init__()
        hidden = [hidden_dim, hidden_dim] if isinstance(hidden_dim, int) else list(hidden_dim)
        self.backbone = VBASSBackbone(in_dim=in_dim, shared_hidden=hidden, latent_dim=vbass_latent_dim, num_classes=2, dropout=dropout)
        self.num_networks = num_networks

    def forward(self, x, edge_indices_list=None, edge_weights_list=None):
        return self.backbone(x)
