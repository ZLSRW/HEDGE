# -*- coding: utf-8 -*-
"""
grid_search.py  (3 components: Diffusion / RGCN / Gate)
- 仅搜索 3 个组件参数：
    appnp_K:        [2, 4, 6, 8]         # 扩散（非零档）
    num_layers:     [2, 3, 4, 6]         # RGCN 深度
    enable_xtalk_gate: [0, 1]            # gate 开/关（当前模型仅二值）
- 目录名：纯数字序列（p1..p3，'-' 分隔），避免 File name too long
- 输出：param_legend.json / 每个 run 的 hparams_readable.json、per_fold_metrics.csv、summary_by_split.csv

运行示例：
  python grid_search.py --data_dir . --out_root Results_Search --strategy grid-mod
  python grid_search.py --data_dir . --out_root Results_Search --strategy random --trials 24
"""
import argparse, itertools, json, math, os, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from config import TrainConfig, RELATION_LIST
from data_io import load_csv_graph
from model_xtalk_rgcn import XTalkRGCN
from cv_5fold import (
    set_seed, compute_metrics, bce_pos_weight, focal_bce_with_logits,
    rdrop_kl, relation_contrastive_loss, diffusion_loss, irm_invariance_loss,
    ns_logic_loss, subgraph_consistency_loss,
)

# —— 3 个要搜索的参数顺序（决定 p1..p3）——
PARAM_ORDER = [
    "appnp_K",         # p1: 扩散步数（APPNP）
    "num_layers",      # p2: R-GCN 层数
    "enable_xtalk_gate"# p3: gate 开关 (0/1)
]

# 固定的默认值（不参与搜索的部分）
FIXED_DEFAULTS = {
    "attn_heads":        2,      # 关系通道注意力头数（稳定且高效）  # ChannelAttention: heads 见源码
    "basis":             4,      # R-GCN basis 分解数（紧凑）       # BasisRGCNLayer: num_bases
    "appnp_alpha":       0.3,    # 扩散残差比例
    "edge_drop_rate":    0.1,    # 丢边率（温和抗噪）
    "smooth_lambda":     0.0,    # 关闭附加平滑正则（已在扩散里体现）
    "contrast_lambda":   0.0,    # 关闭对比正则（避免因素干扰）
    "diffusion_lambda":  0.05,   # 轻量扩散一致性损失
    "subgraph_lambda":   0.2,    # 适度子图一致性约束
}

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _clone_hps(hps):
    class _H: ...
    new = _H()
    for k in vars(hps):
        setattr(new, k, getattr(hps, k))
    return new

def _build_model(hps, gene_extra_dim):
    dev = _device()
    model = XTalkRGCN(hparams=hps, gene_extra_dim=gene_extra_dim).to(dev)
    return model

def _forward(model, x, ei, et, ew, gm, xtalk, gextra):
    out = model(x, ei, et, ew, gm, xtalk, gextra, ablate=None, return_aux=True)
    return out

def _fmt_val(v):
    if isinstance(v, int) or (isinstance(v, float) and float(v).is_integer()):
        return str(int(round(float(v))))
    if isinstance(v, float):
        return f"{v:.4g}".replace(".", "p").replace("+", "")
    return str(v)

def numeric_tag_from_params(params: dict, keys_order: list, max_len: int = 120):
    tokens = []
    for k in keys_order:
        if k in params:
            tokens.append(_fmt_val(params[k]))
    base = "-".join(tokens) if tokens else "0"
    if len(base) <= max_len:
        return base
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    short = "-".join(tokens[:12])
    return f"{short}-{digest}"

def param_space(strategy: str):
    if strategy == "grid-mod":
        # 全组合 = 4 × 4 × 2 = 32
        return {
            "appnp_K":         [1,2, 4, 6, 8],
            "num_layers":      [2, 3, 4, 6],
            "enable_xtalk_gate":[0, 1],
        }
    # random：同一范围，按 --trials 采样
    return {
        "appnp_K":         [1, 2, 4, 6, 8],
        "num_layers":      [2, 3, 4, 6],
        "enable_xtalk_gate":[0, 1],
    }

def generate_trials(space: dict, strategy: str, n_trials: int):
    if strategy == "grid-mod":
        keys = list(space.keys())
        vals = [space[k] for k in keys]
        for combo in itertools.product(*vals):
            yield {k: v for k, v in zip(keys, combo)}
    else:
        keys = list(space.keys())
        for _ in range(n_trials):
            yield {k: random.choice(space[k]) for k in keys}

def run_cv_once(cfg, loaded, folds=5, seed=42):
    device = _device()
    data = loaded.data
    data.x          = data.x.to(device).float()
    data.edge_index = data.edge_index.to(device).long()
    data.edge_type  = data.edge_type.to(device).long()
    data.edge_weight= data.edge_weight.to(device).float()

    gene_mask = loaded.gene_mask.bool().to(device)
    labels    = loaded.label.float().to(device)

    xtalk = torch.zeros(data.num_nodes, 7, device=device)
    xtalk[gene_mask] = torch.from_numpy(loaded.xtalk_counts).to(device)

    gextra_np = getattr(loaded, "gene_extra", None)
    if gextra_np is None or (hasattr(gextra_np, "shape") and gextra_np.shape[1]==0):
        gextra = torch.zeros(data.num_nodes, 0, device=device); gene_extra_dim = 0
    else:
        gene_extra_dim = int(gextra_np.shape[1])
        gextra = torch.zeros(data.num_nodes, gene_extra_dim, device=device)
        gextra[gene_mask] = torch.from_numpy(gextra_np).to(device)

    gene_ids = torch.where(gene_mask)[0].detach().cpu().numpy()
    y_gene   = labels[gene_mask].detach().cpu().numpy()
    labeled  = y_gene >= 0
    labeled_gene_ids = gene_ids[labeled]
    y_lab    = y_gene[labeled].astype(int)

    from sklearn.model_selection import StratifiedKFold, KFold
    if len(np.unique(y_lab)) < 2:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed).split(labeled_gene_ids)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(labeled_gene_ids, y_lab)

    rows = []
    fold_id = 0
    hps = cfg.hparams

    for tr_idx, va_idx in splitter:
        fold_id += 1
        tr_ids = labeled_gene_ids[tr_idx]; va_ids = labeled_gene_ids[va_idx]
        m_tr = torch.zeros_like(gene_mask); m_tr[tr_ids] = True
        m_va = torch.zeros_like(gene_mask); m_va[va_ids] = True

        model = _build_model(hps, gene_extra_dim=gene_extra_dim)
        opt = torch.optim.AdamW(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hps.epochs, eta_min=max(hps.lr/50.0, 1e-6))

        best = {"pr": -1.0, "state": None}; bad = 0; patience = 10

        for ep in range(1, hps.epochs+1):
            model.train(); opt.zero_grad(set_to_none=True)

            out1 = _forward(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra)
            out2 = _forward(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra)
            if isinstance(out1, tuple): logits1, aux1 = out1
            else: logits1, aux1 = out1, {}
            if isinstance(out2, tuple): logits2, aux2 = out2
            else: logits2, aux2 = out2, {}

            base_loss, posw = bce_pos_weight(logits1, labels, m_tr)
            sup_loss = focal_bce_with_logits(logits1, labels, m_tr,
                                             gamma=float(getattr(hps, "focal_gamma", 0.0) or 0.0),
                                             pos_weight=posw)
            loss = sup_loss

            lam_c = float(getattr(hps, "contrast_lambda", 0.0) or 0.0)
            tau_c = float(getattr(hps, "contrast_tau", 0.2) or 0.2)
            if lam_c > 0 and isinstance(aux1, dict):
                loss = loss + lam_c * relation_contrastive_loss(aux1, labels, m_tr, gene_mask, tau=tau_c)

            loss = loss + diffusion_loss(aux1, labels, gene_mask, m_tr, hps, device, model)
            loss = loss + irm_invariance_loss(model, data, labels, m_tr, gene_mask, xtalk, gextra, hps, device)
            loss = loss + ns_logic_loss(aux1, logits1, labels, gene_mask, m_tr, hps, device, model)
            loss = loss + subgraph_consistency_loss(aux1, logits1, model, data, labels, m_tr, gene_mask, xtalk, gextra, hps, device)

            lam_r = float(getattr(hps, "rdrop_lambda", 0.0) or 0.0)
            if lam_r > 0:
                loss = loss + lam_r * rdrop_kl(logits1, logits2, (labels>=0) & m_tr & gene_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sch.step()

            model.eval()
            with torch.no_grad():
                outv = _forward(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra)
                logits_v = outv[0] if isinstance(outv, tuple) else outv
                pv = torch.sigmoid(logits_v)
                mv = (labels>=0) & m_va
                if mv.sum()>0:
                    from sklearn.metrics import average_precision_score
                    pr = float(average_precision_score(labels[mv].detach().cpu().numpy().astype(int),
                                                       pv[mv].detach().cpu().numpy()))
                else:
                    pr = float("nan")
            cur = pr if not math.isnan(pr) else -1.0
            if cur > best["pr"]:
                best = {"pr": cur, "state": {k:v.detach().cpu() for k,v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                print(f"[EarlyStop] fold={fold_id}, ep={ep}, best_val_aupr={best['pr']:.4f}")
                break

        if best["state"] is not None:
            model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
        model.eval()
        with torch.no_grad():
            out = _forward(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits)

        mtr = ((labels>=0) & m_tr & gene_mask)
        mva = ((labels>=0) & m_va & gene_mask)
        ytr = labels[mtr].detach().cpu().numpy().astype(int); ptr = probs[mtr].detach().cpu().numpy()
        yva = labels[mva].detach().cpu().numpy().astype(int); pva = probs[mva].detach().cpu().numpy()

        met_tr = compute_metrics(ytr, ptr)
        met_va = compute_metrics(yva, pva)
        rows.append({"fold": fold_id, "split": "train", **met_tr})
        rows.append({"fold": fold_id, "split": "val",   **met_va})

    df = pd.DataFrame(rows)

    def agg(group):
        g = group.drop(columns=["fold","split"], errors="ignore")
        return pd.Series({**{f"mean_{c}": g[c].mean() for c in g.columns},
                          **{f"std_{c}":  g[c].std()  for c in g.columns}})
    summary = df.groupby("split").apply(agg).reset_index()
    return df, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".", help="数据目录")
    ap.add_argument("--out_root", type=str, default="Results_Search", help="输出根目录")
    ap.add_argument("--strategy", type=str, choices=["grid-mod", "random"], default="grid-mod")
    ap.add_argument("--trials", type=int, default=24, help="random 采样次数")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.data_dir)

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "search_summary.csv"
    all_folds_path = out_root / "all_folds_metrics.csv"
    legend_path = out_root / "param_legend.json"

    legend = {f"p{i+1}": name for i, name in enumerate(PARAM_ORDER)}
    legend_path.write_text(json.dumps(legend, indent=2), encoding="utf-8")

    cfg0 = TrainConfig()
    loaded = load_csv_graph(cfg0.paths)

    space = param_space(args.strategy)
    trials = generate_trials(space, args.strategy, args.trials)

    all_rows, all_fold_rows = [], []

    for i, params in enumerate(trials, 1):
        # 先确定 tag 和对应目录，用于判断是否已跑过
        tag = numeric_tag_from_params(params, PARAM_ORDER, max_len=120)
        run_dir = out_root / tag

        # 如果该组合对应的目录已经存在，则跳过该组合
        # 例如存在 Results_Search/2-3-1/ 就不再重跑
        if run_dir.exists():
            print(f"[Skip] Found existing folder for params={params} -> {run_dir}, skip.")
            continue

        # 真正需要跑的新组合才继续
        run_dir.mkdir(parents=True, exist_ok=True)

        hps = _clone_hps(cfg0.hparams)
        # 覆盖 3 个搜索参数
        for k, v in params.items():
            if hasattr(hps, k):
                setattr(hps, k, v)
        # gate: 0/1 -> bool
        if hasattr(hps, "enable_xtalk_gate"):
            setattr(hps, "enable_xtalk_gate",
                    bool(int(getattr(hps, "enable_xtalk_gate"))))
        # 固定其余默认值
        for k, v in FIXED_DEFAULTS.items():
            setattr(hps, k, v)

        with open(run_dir / "hparams_readable.json", "w") as f:
            json.dump({k: getattr(hps, k) for k in vars(hps)}, f, indent=2)

        print("\n" + "=" * 100)
        print(f"[Search] Run {i}: {tag}  (p1=appnp_K, p2=num_layers, p3=enable_xtalk_gate)")
        print("=" * 100)

        cfg = TrainConfig(paths=cfg0.paths, hparams=hps,
                          val_ratio=cfg0.val_ratio, test_ratio=cfg0.test_ratio,
                          stratify_by_label=cfg0.stratify_by_label, topk=cfg0.topk)

        df_folds, df_summary = run_cv_once(cfg, loaded, folds=args.folds, seed=args.seed)

        # 打印
        print("\nPer-fold metrics:")
        print(df_folds.to_string(index=False))
        print("\nAggregated (mean±std) by split:")
        for _, row in df_summary.iterrows():
            split = row["split"]
            print(f"[{split}] "
                  f"ROC-AUC {row['mean_roc_auc']:.4f}±{row['std_roc_auc']:.4f} | "
                  f"PR-AUC {row['mean_pr_auc']:.4f}±{row['std_pr_auc']:.4f} | "
                  f"F1 {row['mean_f1']:.4f}±{row['std_f1']:.4f} | "
                  f"ACC {row['mean_accuracy']:.4f}±{row['std_accuracy']:.4f} | "
                  f"PREC {row['mean_precision']:.4f}±{row['std_precision']:.4f} | "
                  f"RECALL {row['mean_recall']:.4f}±{row['std_recall']:.4f} | "
                  f"SPEC {row['mean_specificity']:.4f}±{row['std_specificity']:.4f}")

        df_folds.to_csv(run_dir / "per_fold_metrics.csv", index=False)
        df_summary.to_csv(run_dir / "summary_by_split.csv", index=False)

        row_rec = {"tag": tag}
        for idx, name in enumerate(PARAM_ORDER, 1):
            if name in params:
                row_rec[f"p{idx}"] = params[name]
        def _get(d, k):
            try: return float(d.loc[d["split"]=="val", f"mean_{k}"].values[0])
            except Exception: return float("nan")
        row_rec.update({
            "val_pr_auc": _get(df_summary, "pr_auc"),
            "val_roc_auc": _get(df_summary, "roc_auc"),
            "val_f1": _get(df_summary, "f1"),
            "val_acc": _get(df_summary, "accuracy"),
            "val_precision": _get(df_summary, "precision"),
            "val_recall": _get(df_summary, "recall"),
            "val_specificity": _get(df_summary, "specificity"),
        })
        all_rows.append(row_rec)

        df_folds_run = df_folds.copy()
        for idx, name in enumerate(PARAM_ORDER, 1):
            if name in params:
                df_folds_run[f"p{idx}"] = params[name]
        df_folds_run["tag"] = tag
        all_fold_rows.append(df_folds_run)

    if all_fold_rows:
        pd.concat(all_fold_rows, ignore_index=True).to_csv(all_folds_path, index=False)
        print(f"\n[Saved] ALL folds -> {all_folds_path}")

    # 仅当本轮有新跑的组合时，才更新 search_summary
    if all_rows:
        new_df = pd.DataFrame(all_rows)
        if summary_path.exists():
            try:
                old_df = pd.read_csv(summary_path)
                merged = pd.concat([old_df, new_df], ignore_index=True)
            except Exception:
                # 如果旧文件损坏或列不兼容，就只保留本次结果
                merged = new_df
        else:
            merged = new_df

        merged.to_csv(summary_path, index=False)
        print(f"[Saved] Appended summary -> {summary_path}")
    else:
        print("[Info] No new runs executed; search_summary.csv left unchanged.")

    print(f"[Legend] 参数位置说明 -> {legend_path}")
    print("Legend:", json.dumps({f'p{i + 1}': n for i, n in enumerate(PARAM_ORDER)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
