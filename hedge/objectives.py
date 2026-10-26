import torch
from torch import nn
from torch.nn import functional as F

class FixedNoiseMapping(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.time_proj = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.net = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, z, log_sigma):
        return self.net(z + self.time_proj(log_sigma))

def supervised(logits, labels, mask):
    if not bool(mask.any()):
        return logits.sum() * 0.0
    y = labels[mask]
    n_pos = y.sum()
    n_neg = y.numel() - n_pos
    weight = torch.clamp(n_neg / n_pos.clamp_min(1.0), min=1.0)
    return F.binary_cross_entropy_with_logits(logits[mask], y, pos_weight=weight)

def consistency(first, second, mask):
    if not bool(mask.any()):
        return first.sum() * 0.0
    p = torch.sigmoid(first[mask]).clamp(1e-06, 1 - 1e-06)
    q = torch.sigmoid(second[mask]).clamp(1e-06, 1 - 1e-06)
    return 0.5 * (p * torch.log(p / q) + q * torch.log(q / p)).mean()

def noise_reconstruction(z, selected, mapping, sigma_min=0.1, sigma_max=0.5):
    if int(selected.sum()) < 2:
        return z.sum() * 0.0
    z = z[selected]
    sigma = torch.empty((z.shape[0], 1), device=z.device).uniform_(sigma_min, sigma_max)
    eps = torch.randn_like(z)
    return ((mapping(z + sigma * eps, torch.log(sigma).expand_as(z)) - eps) ** 2).mean()

def objective(model, data, labels, gene_mask, counts, train_mask, hps, mapping, ablate=None, protocol='main'):
    ablate = ablate or {}
    sel = gene_mask & train_mask & (labels >= 0)
    args = (data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, counts)
    first, aux = model(*args, ablate=ablate)
    second, _ = model(*args, ablate=ablate)
    parts = {'supervised': supervised(first, labels, sel), 'smooth': aux['smooth']}
    if protocol == 'component':
        extra = F.binary_cross_entropy_with_logits(first[sel], labels[sel]) if bool(sel.any()) else first.sum() * 0.0
        parts['supervised'] = parts['supervised'] + extra
        parts['smooth'] = first.sum() * 0.0
    parts['consistency'] = hps.rdrop_lambda * consistency(first, second, sel)
    inv = first.sum() * 0.0
    if bool(sel.any()) and hps.irm_lambda:
        for name in ['h', 'circ']:
            env = dict(ablate)
            env[name] = True
            logits, _ = model(*args, ablate=env)
            scale = torch.ones((), device=first.device, requires_grad=True)
            risk = F.binary_cross_entropy_with_logits(scale * logits[sel], labels[sel])
            inv = inv + torch.autograd.grad(risk, scale, create_graph=True)[0].square()
    parts['invariance'] = hps.irm_lambda * inv
    parts['noise'] = hps.diffusion_lambda * noise_reconstruction(aux['z_rel_g'], sel[gene_mask], mapping, hps.diffusion_sigma_min, hps.diffusion_sigma_max)
    return (sum(parts.values()), parts)
