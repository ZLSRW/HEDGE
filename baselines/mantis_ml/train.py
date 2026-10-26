import argparse
from pathlib import Path
import random
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
from .config import TrainConfig
from .data import load_asd_data, default_data_root_from_here
from .model import MantisML

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default=None, help='Processed input directory; default: data/baselines')
    ap.add_argument('--out_dir', type=str, default=None, help='结果输出目录; 默认: ./MantisML_cv5')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    if args.data_root is None:
        data_root = default_data_root_from_here()
    else:
        data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f'Input directory does not exist: {data_root}')
    script_dir = Path(__file__).resolve().parent
    if args.out_dir is None:
        out_root = script_dir / 'MantisML_cv5'
    else:
        out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'[info] data_root = {data_root}')
    print(f'[info] out_root   = {out_root}')
    data_dict = load_asd_data(data_root)
    x = data_dict['x'].detach().cpu().numpy()
    y = np.asarray(data_dict['y'], dtype=float)
    gene_index = data_dict['gene_index']
    gene_symbol = data_dict['gene_symbol']
    node_type = data_dict.get('node_type', None)
    N, F_dim = x.shape
    print(f'[info] N(nodes)= {N}, F={F_dim}')
    labeled_mask = y >= 0
    labeled_ids = np.where(labeled_mask)[0]
    y_lab = y[labeled_mask].astype(int)
    if labeled_ids.size == 0:
        raise ValueError('没有任何带标签(y>=0)的节点，无法训练/验证。')
    cfg = TrainConfig()
    hps = cfg.hparams
    print(f'[hparams][ensemble] mode={hps.ensemble_mode}, superv_models={hps.superv_models}, nthreads={hps.nthreads}, class_weight={hps.class_weight}')
    with open(out_root / 'run_hparams.json', 'w', encoding='utf-8') as f:
        json.dump(dict(seed=args.seed, folds=args.folds, ensemble_mode=hps.ensemble_mode, superv_models=hps.superv_models, nthreads=hps.nthreads, class_weight=hps.class_weight), f, ensure_ascii=False, indent=2)
    if len(np.unique(y_lab)) >= 2:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(labeled_ids, y_lab)
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(labeled_ids)
    rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        tr_ids = labeled_ids[tr_idx]
        va_ids = labeled_ids[va_idx]
        print(f'\n[Fold {fold_id}/{args.folds}] n_train={len(tr_ids)} n_val={len(va_ids)}')
        X_tr = x[tr_ids]
        y_tr = y[tr_ids].astype(int)
        model = MantisML(superv_models=hps.superv_models, mode=hps.ensemble_mode, nthreads=hps.nthreads, random_state=args.seed + fold_id, class_weight=hps.class_weight)
        model.fit(X_tr, y_tr)
        prob_all = model.predict_proba(x)[:, 1]
        z_all, z_names = model.compute_repr(x)
        df_all_dict = {'node_id': np.arange(N), 'gene_index': gene_index, 'gene_symbol': gene_symbol}
        if node_type is not None:
            df_all_dict['node_type'] = node_type
        df_all = pd.DataFrame(df_all_dict)
        H_dim = z_all.shape[1]
        if H_dim == 0:
            df_all['feat_0'] = prob_all
        else:
            for j in range(H_dim):
                col = f'feat_{j}'
                df_all[col] = z_all[:, j]
        df_train_repr = df_all.iloc[tr_ids]
        df_val_repr = df_all.iloc[va_ids]
        df_train_repr.to_csv(out_root / f'fold{fold_id}_train_node_repr.csv', index=False)
        df_val_repr.to_csv(out_root / f'fold{fold_id}_val_node_repr.csv', index=False)

        def dump_split(split_name, ids, fold_id_inner):
            y_true_split = y[ids].astype(int)
            y_prob_split = prob_all[ids]
            y_pred_split = (y_prob_split >= 0.5).astype(int)
            df_prob = pd.DataFrame({'gene_index': gene_index[ids], 'gene_symbol': gene_symbol[ids], 'label': y_true_split, 'prob': y_prob_split})
            df_pred = pd.DataFrame({'gene_index': gene_index[ids], 'gene_symbol': gene_symbol[ids], 'label': y_true_split, 'pred': y_pred_split})
            df_prob.to_csv(out_root / f'fold{fold_id_inner}_{split_name}_prob.csv', index=False)
            df_pred.to_csv(out_root / f'fold{fold_id_inner}_{split_name}_pred.csv', index=False)
            metrics = compute_metrics(y_true_split, y_prob_split)
            return metrics
        metrics_train = dump_split('train', tr_ids, fold_id)
        metrics_val = dump_split('val', va_ids, fold_id)
        metrics_train['fold'] = fold_id
        metrics_train['split'] = 'train'
        metrics_val['fold'] = fold_id
        metrics_val['split'] = 'val'
        rows.append(metrics_train)
        rows.append(metrics_val)
    df = pd.DataFrame(rows)
    cols_order = ['fold', 'split', 'roc_auc', 'pr_auc', 'accuracy', 'f1', 'precision', 'recall', 'specificity']
    df = df[cols_order]
    df.to_csv(out_root / 'cv5_metrics_per_fold.csv', index=False)

    def agg(group):
        g = group.drop(columns=['fold', 'split'])
        res = {}
        for c in g.columns:
            res[f'mean_{c}'] = g[c].mean()
            res[f'std_{c}'] = g[c].std()
        return pd.Series(res)
    summary = df.groupby('split').apply(agg).reset_index()
    summary.to_csv(out_root / 'cv5_metrics_summary.csv', index=False)
    print('\n[Done] 5-fold completed. Output:', out_root)
if __name__ == '__main__':
    main()
