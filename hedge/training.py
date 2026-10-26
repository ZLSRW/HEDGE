from pathlib import Path
from dataclasses import asdict
import copy, json, random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score
from .model import HEDGE
from .objectives import FixedNoiseMapping, objective
CHANNEL_CONFIGS = {'all_removed': (False, False, False), 'only_r': (True, False, False), 'only_h': (False, True, False), 'only_circ': (False, False, True), 'remove_r': (False, True, True), 'remove_h': (True, False, True), 'remove_circ': (True, True, False), 'all_kept': (True, True, True)}
COMPONENTS = ['full', 'minus_gate', 'minus_rgcn', 'minus_appnp', 'minus_dropout', 'minus_losses']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def metrics(y, p):
    return {'AUC': float(roc_auc_score(y, p)), 'AUPR': float(average_precision_score(y, p)), 'ACC': float(accuracy_score(y, p >= 0.5)), 'Precision': float(precision_score(y, p >= 0.5, zero_division=0))}

def graph_intervention(data, counts, keep):
    data = data.clone()
    counts = counts.clone()
    mask = torch.ones_like(data.edge_type, dtype=torch.bool)
    for rid, yes in enumerate(keep):
        if not yes:
            mask &= data.edge_type != rid
    data.edge_index = data.edge_index[:, mask]
    data.edge_type = data.edge_type[mask]
    data.edge_weight = data.edge_weight[mask]
    columns = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    for col, channels in enumerate(columns):
        if not all((keep[c] for c in channels)):
            counts[:, col] = 0
    return (data, counts, {name: not keep[i] for i, name in enumerate(['r', 'h', 'circ'])})

def setup_component(hps, name):
    hps = copy.deepcopy(hps)
    ablate = {}
    if name == 'minus_gate':
        hps.enable_convergence_gate = False
    elif name == 'minus_rgcn':
        ablate = {'rgcn': True, 'appnp_off': True}
        hps.appnp_K = 0
        hps.appnp_alpha = 0.0
    elif name == 'minus_appnp':
        ablate = {'appnp_off': True}
        hps.appnp_K = 0
        hps.appnp_alpha = 0.0
    elif name == 'minus_dropout':
        hps.dropout = 0.0
        hps.edge_drop_rate = 0.0
    elif name == 'minus_losses':
        hps.rdrop_lambda = 0.0
        hps.irm_lambda = 0.0
        hps.diffusion_lambda = 0.0
        hps.smooth_lambda = 0.0
    elif name != 'full':
        raise ValueError(name)
    return (hps, ablate)

def mask_from_ids(labels, ids):
    m = torch.zeros_like(labels, dtype=torch.bool)
    m[torch.as_tensor(ids, device=labels.device)] = True
    return m

def interpretation(model, data, counts, gm, labels, unify2orig, out, fold):
    model.eval()
    with torch.no_grad():
        logits, aux = model(data.x, data.edge_index, data.edge_type, data.edge_weight, gm, counts)
    gids = torch.where(gm)[0].cpu().numpy()
    att = aux['rel_attn'].mean(0).T.numpy()[gids]
    d = pd.DataFrame(att, columns=['relatt_r%d' % i for i in range(9)])
    d.insert(0, 'gene_unified_id', gids)
    d.to_csv(out / f'fold{fold}_all_gene_relatt_mean.csv', index=False)
    gates = aux['gates'].numpy()
    coupling = np.zeros((len(gids), 3))
    loc = {int(v): i for i, v in enumerate(gids)}
    ei = data.edge_index.cpu().numpy()
    et = data.edge_type.cpu().numpy()
    ew = data.edge_weight.cpu().numpy()
    for rid in range(3):
        mask = (et == rid) & np.isin(ei[1], gids)
        np.add.at(coupling[:, rid], np.array([loc[int(x)] for x in ei[1, mask]], dtype=int), ew[mask])
    tab = pd.DataFrame({'gene_unified_id': gids, 'gene_index': [unify2orig[int(i)] for i in gids]})
    for rid, name in enumerate(['r', 'h', 'circ']):
        tab['edgesum_' + name] = coupling[:, rid]
        tab['gate_' + name] = gates[:, rid]
        tab['affinity_' + name] = coupling[:, rid] * gates[:, rid]
    tab.to_csv(out / f'fold{fold}_all_gene_channel_affinity.csv', index=False)
    pd.DataFrame(aux['A_static'].numpy(), index=['r', 'h', 'circ'], columns=['r', 'h', 'circ']).to_csv(out / f'fold{fold}_channel_mixing.csv')
    np.save(out / f'fold{fold}_relation_gene_embedding.npy', aux['z_rel_g'].detach().cpu().numpy())

def run(loaded, hps, out, device='cpu', folds=5, train_fraction=None, repeats=5, channel='all_kept', component='full', feature_mode='standardized', protocol='main', patience=25):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(hps.seed)
    data = loaded.data.clone().to(device)
    labels = loaded.label.to(device)
    gm = loaded.gene_mask.to(device)
    counts = torch.zeros((data.num_nodes, 7), device=device)
    counts[gm] = torch.from_numpy(loaded.convergence_counts).to(device)
    data, counts, ablate = graph_intervention(data, counts, CHANNEL_CONFIGS[channel])
    hps, comp = setup_component(hps, component)
    ablate.update(comp)
    ids = torch.where(gm & (labels >= 0))[0].cpu().numpy()
    y = labels[ids].cpu().numpy().astype(int)
    if train_fraction is None:
        splits = list(StratifiedKFold(folds, shuffle=True, random_state=hps.seed).split(ids, y))
    elif train_fraction == 0:
        splits = [(np.array([], dtype=int), np.arange(len(ids))) for _ in range(repeats)]
    else:
        if not 0 < train_fraction < 1:
            raise ValueError('train fraction must be in [0,1)')
        splits = list(StratifiedShuffleSplit(repeats, train_size=train_fraction, random_state=hps.seed).split(ids, y))
    unify2orig = {int(a): int(b) for a, b in zip(loaded.node_ids['gene_unified_ids'], loaded.node_ids['gene_index'])}
    settings = asdict(hps)
    settings.update(feature_mode=feature_mode, protocol=protocol, channel=channel, component=component, train_fraction=train_fraction, folds=folds, patience=patience)
    (out / 'run_settings.json').write_text(json.dumps(settings, indent=2))
    rows = []
    for fold, (train, val) in enumerate(splits, 1):
        model = HEDGE(hparams=hps).to(device)
        mapping = FixedNoiseMapping(hps.dim).to(device)
        if protocol == 'component':
            opt = torch.optim.Adam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=8)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hps.epochs, eta_min=hps.lr / 50.0)
        tr = mask_from_ids(labels, ids[train])
        va = mask_from_ids(labels, ids[val])
        best = -float('inf')
        state = None
        bad = 0
        history = []
        for epoch in range(1, hps.epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss, parts = objective(model, data, labels, gm, counts, tr, hps, mapping, ablate, protocol)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0 if protocol == 'component' else 1.0)
            opt.step()
            model.eval()
            with torch.no_grad():
                logits, _ = model(data.x, data.edge_index, data.edge_type, data.edge_weight, gm, counts, ablate=ablate)
            values = metrics(labels[va].cpu().numpy(), logits[va].sigmoid().cpu().numpy())
            history.append(dict(epoch=epoch, loss=float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()}, val_AUPR=values['AUPR']))
            if protocol == 'component':
                scheduler.step(values['AUPR'])
            else:
                scheduler.step()
            if values['AUPR'] > best:
                best = values['AUPR']
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                bad = 0
            else:
                bad += 1
            if patience > 0 and bad >= patience:
                break
        model.load_state_dict(state)
        model.eval()
        torch.save({'model': state, 'fixed_noise_mapping': mapping.state_dict(), 'hparams': asdict(hps), 'feature_mode': feature_mode, 'best_epoch': best_epoch, 'ablate': ablate, 'channel': channel}, out / f'fold{fold}_best.pt')
        pd.DataFrame(history).to_csv(out / f'fold{fold}_history.csv', index=False)
        with torch.no_grad():
            logits, _ = model(data.x, data.edge_index, data.edge_type, data.edge_weight, gm, counts, ablate=ablate)
        for split, mask in [('train', tr), ('val', va)]:
            i = torch.where(mask)[0].cpu().numpy()
            pr = logits[mask].sigmoid().cpu().numpy()
            yy = labels[mask].cpu().numpy()
            pd.DataFrame({'gene_unified_id': i, 'gene_index': [unify2orig[int(j)] for j in i], 'label': yy.astype(int), 'prob': pr}).to_csv(out / f'fold{fold}_{split}_prob.csv', index=False)
            if len(np.unique(yy)) == 2:
                rows.append(dict(fold=fold, split=split, best_epoch=best_epoch, **metrics(yy, pr)))
        interpretation_model = model
        if ablate:

            class WithIntervention(torch.nn.Module):

                def __init__(self, base):
                    super().__init__()
                    self.base = base

                def forward(self, *args, **kwargs):
                    kwargs['ablate'] = ablate
                    return self.base(*args, **kwargs)
            interpretation_model = WithIntervention(model)
        interpretation(interpretation_model, data, counts, gm, labels, unify2orig, out, fold)
        print('fold', fold, 'best epoch', best_epoch, 'validation AUPR', round(best, 6), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'metrics_per_fold.csv', index=False)
    summary = result.groupby('split')[['AUC', 'AUPR', 'ACC', 'Precision']].agg(['mean', 'std'])
    summary.to_csv(out / 'metrics_summary.csv')
    return result
