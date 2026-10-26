import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from itertools import repeat
from torch.distributions import Normal

class GaussianSampler(nn.Module):

    def __init__(self, in_features, out_features):
        super(GaussianSampler, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.sigma = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        return (Normal(mu, sigma).rsample(), mu, sigma)

class SharedLayers(nn.Module):

    def __init__(self, dims, use_bias: bool=True, use_batch_norm: bool=False, use_layer_norm: bool=True, use_activation: bool=True, activation_fn: nn.Module=nn.ELU, dropout_rate: float=0.1):
        super(SharedLayers, self).__init__()
        [x_dim, h_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = []
        for i, (n_in, n_out) in enumerate(zip(neurons[:-1], neurons[1:])):
            linear_layers += [nn.Linear(in_features=n_in, out_features=n_out, bias=use_bias), nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None, nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None, activation_fn() if use_activation else None, nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None]
        linear_layers = [x for x in linear_layers if x]
        self.hidden = nn.Sequential(*linear_layers)

    def forward(self, x):
        return self.hidden(x)

class Encoder(nn.Module):

    def __init__(self, dims, sample_layer=GaussianSampler, use_bias: bool=True, use_batch_norm: bool=False, use_layer_norm: bool=True, use_activation: bool=True, activation_fn: nn.Module=nn.ELU, dropout_rate: float=0.1):
        super(Encoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        if len(h_dim) != 0:
            neurons = [x_dim, *h_dim]
            linear_layers = []
            for i, (n_in, n_out) in enumerate(zip(neurons[:-1], neurons[1:])):
                linear_layers += [nn.Linear(in_features=n_in, out_features=n_out, bias=use_bias), nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None, nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None, activation_fn() if use_activation else None, nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None]
            linear_layers = [x for x in linear_layers if x]
            self.hidden = nn.Sequential(*linear_layers)
            self.sample = sample_layer(h_dim[-1], z_dim)
        else:
            self.hidden = nn.Sequential(*[])
            self.sample = sample_layer(x_dim, z_dim)

    def forward(self, x):
        return self.sample(self.hidden(x))

class Classifier(nn.Module):

    def __init__(self, dims, use_bias: bool=True, use_batch_norm: bool=False, use_layer_norm: bool=True, use_activation: bool=True, activation_fn: nn.Module=nn.ELU, dropout_rate: float=0.1):
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        if len(h_dim) != 0:
            neurons = [x_dim, *h_dim]
            linear_layers = []
            for i, (n_in, n_out) in enumerate(zip(neurons[:-1], neurons[1:])):
                linear_layers += [nn.Linear(in_features=n_in, out_features=n_out, bias=use_bias), nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None, nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None, activation_fn() if use_activation else None, nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None]
            linear_layers = [x for x in linear_layers if x]
            self.hidden = nn.Sequential(*linear_layers)
            self.logits = nn.Linear(h_dim[-1], y_dim)
        else:
            self.hidden = nn.Sequential(*[])
            self.logits = nn.Linear(x_dim, y_dim)

    def forward(self, x):
        pi = F.log_softmax(self.logits(self.hidden(x)), dim=-1)
        return pi
