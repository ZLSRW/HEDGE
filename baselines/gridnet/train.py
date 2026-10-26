import argparse
from pathlib import Path
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
from .config import TrainConfig
from .data import load_asd_data, default_data_root_from_here
from .model import GrIDNet

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_prob, thr=0.5):
    out = {}
    try:
        out['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out['roc_auc'] = float('nan')
    try:
        out['pr_auc'] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out['pr_auc'] = float('nan')
    y_pred = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    out['precision'] = float(p)
    out['recall'] = float(r)
    out['f1'] = float(f1)
    out['accuracy'] = float((y_pred == y_true).mean())
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if tn + fp > 0 else float('nan')
    except Exception:
        spec = float('nan')
    out['specificity'] = float(spec)
    return out

def bce_loss_with_pos_weight(logits, labels, mask):
    m = (labels >= 0) & mask
    if m.sum() == 0:
        return (torch.tensor(0.0, device=logits.device, requires_grad=True), torch.tensor(1.0, device=logits.device))
    y = labels[m]
    pos = (y > 0.5).sum().item()
    neg = (y <= 0.5).sum().item()
    if pos == 0:
        pos_weight = torch.tensor(1.0, device=logits.device)
    else:
        pos_weight = torch.tensor(neg / pos, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits[m], y, pos_weight=pos_weight)
    return (loss, pos_weight)

def focal_bce(logits, labels, mask, gamma: float, pos_weight=None):
    m = (labels >= 0) & mask
    if m.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    y = labels[m]
    x = logits[m]
    bce = F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight, reduction='none')
    if gamma <= 0:
        return bce.mean()
    p = torch.sigmoid(x)
    pt = torch.where(y > 0.5, p, 1 - p)
    loss = (1 - pt) ** gamma * bce
    return loss.mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default=None, help='Processed input directory; default: data/baselines')
    ap.add_argument('--out_dir', type=str, default=None, help='结果输出目录; 默认: ./GrIDNet_cv5')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.data_root is None:
        data_root = default_data_root_from_here()
    else:
        data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f'Input directory does not exist: {data_root}')
    script_dir = Path(__file__).resolve().parent
    if args.out_dir is None:
        out_dir = script_dir / 'GrIDNet_cv5'
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[info] data_root = {data_root}')
    print(f'[info] out_dir   = {out_dir}')
    data_dict = load_asd_data(data_root)
    x = data_dict['x'].to(device)
    y = torch.from_numpy(data_dict['y']).float().to(device)
    edge_indices_list = [e.to(device) for e in data_dict['edge_indices_list']]
    edge_weights_list = [w.to(device) for w in data_dict['edge_weights_list']]
    gene_index = data_dict['gene_index']
    gene_symbol = data_dict['gene_symbol']
    node_type = data_dict.get('node_type', None)
    N, F_dim = x.size()
    print(f'[info] N(nodes)= {N}, F={F_dim}, edges per net={[e.size(1) for e in edge_indices_list]}')
    y_np = y.cpu().numpy()
    labeled_mask = y_np >= 0
    labeled_ids = np.where(labeled_mask)[0]
    y_lab = y_np[labeled_mask].astype(int)
    if len(np.unique(y_lab)) < 2:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(labeled_ids)
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(labeled_ids, y_lab)
    cfg = TrainConfig()
    hps = cfg.hparams
    if args.epochs is not None:
        hps.epochs = args.epochs
    with open(out_dir / 'hparams_used.json', 'w') as f:
        json.dump(hps.__dict__, f, indent=2)
    print(f'[hparams] lr={hps.lr}, weight_decay={hps.weight_decay}, epochs={hps.epochs}, hidden_dim={hps.hidden_dim}, dropout={hps.dropout}, gamma={hps.focal_gamma}')
    rows = []
    fold_id = 0
    for tr_idx, va_idx in splitter:
        fold_id += 1
        print(f'\n[Fold {fold_id}/{args.folds}] ---------------------')
        tr_ids = labeled_ids[tr_idx]
        va_ids = labeled_ids[va_idx]
        train_mask = torch.zeros(N, dtype=torch.bool, device=device)
        val_mask = torch.zeros(N, dtype=torch.bool, device=device)
        train_mask[tr_ids] = True
        val_mask[va_ids] = True
        model = GrIDNet(in_dim=F_dim, hidden_dim=hps.hidden_dim, num_networks=hps.num_networks, dropout=hps.dropout, grid_num_layers=hps.grid_num_layers, n_nodes=N, node_type=torch.from_numpy(node_type).long().to(device) if node_type is not None else None).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay)
        best = {'pr': -1.0, 'state': None}
        bad = 0
        for ep in range(1, hps.epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x, edge_indices_list, edge_weights_list)
            base_loss, posw = bce_loss_with_pos_weight(logits, y, train_mask)
            loss = focal_bce(logits, y, train_mask, gamma=hps.focal_gamma, pos_weight=posw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            model.eval()
            with torch.no_grad():
                logits_val, _ = model(x, edge_indices_list, edge_weights_list)
                prob_val = torch.sigmoid(logits_val)
                mv = (y >= 0) & val_mask
                if mv.sum() > 0:
                    y_true = y[mv].cpu().numpy().astype(int)
                    y_prob = prob_val[mv].cpu().numpy()
                    try:
                        pr_auc = average_precision_score(y_true, y_prob)
                    except Exception:
                        pr_auc = float('nan')
                else:
                    pr_auc = float('nan')
            cur = pr_auc if not np.isnan(pr_auc) else 0.0
            print(f"[Fold {fold_id}] epoch={ep:03d} loss={loss.item():.4f} val_AUPR={cur:.4f} (best={best['pr']:.4f})")
            if cur > best['pr']:
                best['pr'] = cur
                best['state'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= hps.early_stop_patience:
                print(f"[EarlyStop] fold={fold_id}, epoch={ep}, best_val_AUPR={best['pr']:.4f}")
                break
        if best['state'] is not None:
            model.load_state_dict(best['state'])
        model.eval()
        with torch.no_grad():
            logits_all, aux = model(x, edge_indices_list, edge_weights_list)
            prob_all = torch.sigmoid(logits_all).cpu().numpy()
            h_experts = aux.get('h_experts', None)
            gate_alpha = aux.get('gate_alpha', None)
            if h_experts is not None and gate_alpha is not None:
                z_all = (gate_alpha.unsqueeze(-1) * h_experts).sum(dim=1)
            else:
                z_all = aux['h_shared']
            z_all_np = z_all.detach().cpu().numpy()
            H_dim = z_all_np.shape[1]
        feat_cols = {f'feat_{i}': z_all_np[:, i] for i in range(H_dim)}
        df_all_dict = {'node_id': np.arange(N), 'gene_index': gene_index, 'gene_symbol': gene_symbol}
        if node_type is not None:
            df_all_dict['node_type'] = node_type
        df_all = pd.DataFrame(df_all_dict)
        for k, v in feat_cols.items():
            df_all[k] = v
        m_train = ((y >= 0) & train_mask).cpu().numpy()
        m_val = ((y >= 0) & val_mask).cpu().numpy()
        df_train_repr = df_all[m_train]
        df_val_repr = df_all[m_val]
        df_train_repr.to_csv(out_dir / f'fold{fold_id}_train_node_repr.csv', index=False)
        df_val_repr.to_csv(out_dir / f'fold{fold_id}_val_node_repr.csv', index=False)

        def dump_split(split_name, mask_bool, fold_id):
            m = ((y >= 0) & mask_bool).cpu().numpy()
            idx = np.where(m)[0]
            y_true = y.cpu().numpy()[m].astype(int)
            y_prob = prob_all[m]
            y_pred = (y_prob >= 0.5).astype(int)
            df_prob = pd.DataFrame({'gene_index': gene_index[idx], 'gene_symbol': gene_symbol[idx], 'label': y_true, 'prob': y_prob})
            df_pred = pd.DataFrame({'gene_index': gene_index[idx], 'gene_symbol': gene_symbol[idx], 'label': y_true, 'pred': y_pred})
            df_prob.to_csv(out_dir / f'fold{fold_id}_{split_name}_prob.csv', index=False)
            df_pred.to_csv(out_dir / f'fold{fold_id}_{split_name}_pred.csv', index=False)
            metrics = compute_metrics(y_true, y_prob)
            return metrics
        metrics_train = dump_split('train', train_mask, fold_id)
        metrics_val = dump_split('val', val_mask, fold_id)
        metrics_train['fold'] = fold_id
        metrics_train['split'] = 'train'
        metrics_val['fold'] = fold_id
        metrics_val['split'] = 'val'
        rows.append(metrics_train)
        rows.append(metrics_val)
    df = pd.DataFrame(rows)
    cols_order = ['fold', 'split', 'roc_auc', 'pr_auc', 'accuracy', 'f1', 'precision', 'recall', 'specificity']
    df = df[cols_order]
    df.to_csv(out_dir / 'cv5_metrics_per_fold.csv', index=False)

    def agg(group):
        g = group.drop(columns=['fold', 'split'])
        res = {}
        for c in g.columns:
            res[f'mean_{c}'] = g[c].mean()
            res[f'std_{c}'] = g[c].std()
        return pd.Series(res)
    summary = df.groupby('split').apply(agg).reset_index()
    summary.to_csv(out_dir / 'cv5_metrics_summary.csv', index=False)
    print('\n[Done] 所有折完成, 结果已保存到:', out_dir)
if __name__ == '__main__':
    main()
