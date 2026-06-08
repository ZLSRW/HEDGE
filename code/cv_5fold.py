import argparse
from pathlib import Path
import random, json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

from config import TrainConfig, RELATION_LIST
from data_io import load_csv_graph
from model_xtalk_rgcn import XTalkRGCN

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def set_seed(seed: int = 42, deterministic: bool = True):
#     import os
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     # CUBLAS 可复现（CUDA >= 10.2），注意必须在第一次触发 CUDA 之前设置
#     os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 或 ":16:8"
#
#     import random, numpy as np, torch
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#     # CuDNN：关闭基准搜索 + 开启确定性
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
#     # 要求 PyTorch 选用确定性算子（如有不支持的算子会报错，便于排查）
#     if deterministic:
#         try:
#             torch.use_deterministic_algorithms(True)
#         except Exception:
#             pass   # 旧版 PyTorch 可忽略


def fwd_full(model, x, ei, et, ew, gm, xtalk, gextra, ablate=None):
    try:
        out = model(x, ei, et, ew, gm, xtalk, gextra, ablate=ablate, return_aux=True)
    except TypeError:
        out = model(x, ei, et, ew, gm, xtalk, ablate=ablate, return_aux=True)
    return out

# ---------------- basic metrics / utils ----------------
def compute_metrics(y_true, y_prob, thr=0.5):
    out = {}
    try: out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception: out["roc_auc"] = float("nan")
    try: out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception: out["pr_auc"] = float("nan")
    y_pred = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out.update(dict(precision=float(p), recall=float(r), f1=float(f1), accuracy=float((y_pred==y_true).mean())))
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        spec = tn/(tn+fp) if (tn+fp)>0 else float("nan")
    except Exception: spec=float("nan")
    out["specificity"]=float(spec); return out

def bce_pos_weight(logits, labels, mask):
    m = (labels>=0) & mask
    if m.sum()==0: return torch.tensor(0.0, device=logits.device, requires_grad=True), torch.tensor(1.0, device=logits.device)
    y = labels[m]; pos=(y>0.5).sum().item(); neg=(y<=0.5).sum().item()
    posw = torch.tensor(max(1.0, neg/max(1,pos)), device=logits.device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[m], y, pos_weight=posw)
    return loss, posw

def eval_and_dump(split, logits, labels, mask, fold_id, out_dir, ids_map):
    with torch.no_grad():
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        m = ((labels>=0) & mask).detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy()[m].astype(int)
        y_prob = probs[m]; y_pred=(y_prob>=0.5).astype(int)
        idx = np.where(m)[0].astype(int)
        gene_unified_id = idx
        gene_index = [int(ids_map.get(i, -1)) for i in gene_unified_id]

    df_prob = pd.DataFrame({"gene_unified_id":gene_unified_id,"gene_index":gene_index,"label":y_true,"prob":y_prob})
    df_pred = pd.DataFrame({"gene_unified_id":gene_unified_id,"gene_index":gene_index,"label":y_true,"pred":y_pred})
    df_prob.to_csv(out_dir / f"fold{fold_id}_{split}_prob.csv", index=False)
    df_pred.to_csv(out_dir / f"fold{fold_id}_{split}_pred.csv", index=False)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        print(f"[Fold {fold_id} | {split}] CM: TN={tn} FP={fp} | FN={fn} TP={tp}")
    except Exception:
        pass

    return compute_metrics(y_true, y_prob)

def save_interpret(fold_id, split, aux, mask, labels, out_dir, ids_map):
    heads, R, N = aux["rel_attn"].shape
    Ng = aux["Ng"]
    m = ((labels>=0) & mask).detach().cpu().numpy()
    idx = np.where(m)[0].astype(int)

    gene_unified_id = idx
    gene_index = [int(ids_map.get(int(i), -1)) for i in gene_unified_id]

    rel_attn = aux["rel_attn"][:,:,idx]  # (h,R,n_sel)
    rel_attn_mean = rel_attn.mean(0).transpose(0,1).numpy()  # (n_sel,R)

    gene_ids_all = sorted([k for k in ids_map.keys()])
    id2rank = {u:i for i,u in enumerate(gene_ids_all)}
    ranks = [id2rank.get(u, -1) for u in gene_unified_id]
    gates = aux["gates"].numpy()  # (Ng,3)
    gate_rows = np.array([gates[r] if 0<=r<Ng else [np.nan,np.nan,np.nan] for r in ranks])  # (n_sel,3)

    A_static = aux["A_static"].numpy()  # (3,3)

    df = pd.DataFrame({
        "gene_unified_id": gene_unified_id,
        "gene_index": gene_index
    })
    df["gate_me"] = gate_rows[:,0]; df["gate_h"] = gate_rows[:,1]; df["gate_circ"] = gate_rows[:,2]
    for r in range(R):
        df[f"relatt_r{r}"] = rel_attn_mean[:, r]

    df.to_csv(out_dir / f"fold{fold_id}_{split}_interpret.csv", index=False)
    with open(out_dir / f"fold{fold_id}_Astatic.json","w") as f:
        import json; json.dump(A_static.tolist(), f, indent=2)

def aux_heads_loss(aux, labels, train_mask, gene_mask, head_aux_lambda=0.2):
    if "heads" not in aux:
        return torch.tensor(0.0, device=labels.device)
    heads = aux["heads"]
    gene_ids = aux.get("gene_ids", None)
    if gene_ids is None:
        return torch.tensor(0.0, device=labels.device)
    device = labels.device
    sel = ((labels>=0) & train_mask & gene_mask)
    idx_global = torch.where(sel)[0].detach().cpu().numpy().tolist()
    # build global->local map
    gene_ids_np = gene_ids.numpy().tolist()
    local_map = {int(g): i for i, g in enumerate(gene_ids_np)}
    idx_local = [local_map.get(int(g), -1) for g in idx_global]
    idx_local = torch.tensor([i for i in idx_local if i>=0], dtype=torch.long, device=device)
    if idx_local.numel() == 0:
        return torch.tensor(0.0, device=device)

    y = labels[sel].float()
    # because we filtered idx_local to valid ones, slice y accordingly
    y = y[:idx_local.numel()]

    loss = 0.0
    for k in ["rel","gate","prior"]:
        logit = heads[k].to(device)[idx_local]
        loss += torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
    return head_aux_lambda * loss / 3.0

# ---------------- Training enhancements: focal / rdrop / relation-contrast ----------------
def focal_bce_with_logits(logits, labels, mask, gamma=0.0, pos_weight=None):
    m = (labels>=0) & mask
    if m.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    y = labels[m]
    x = logits[m]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight, reduction='none')
    if gamma <= 0:
        return bce.mean()
    p = torch.sigmoid(x)
    pt = torch.where(y>0.5, p, 1-p)
    loss = ((1-pt)**gamma) * bce
    return loss.mean()

def rdrop_kl(logits1, logits2, mask):
    m = mask
    if m.sum() == 0:
        return torch.tensor(0.0, device=logits1.device, requires_grad=True)
    p1 = torch.sigmoid(logits1[m]).clamp(1e-6, 1-1e-6)
    p2 = torch.sigmoid(logits2[m]).clamp(1e-6, 1-1e-6)
    kl1 = torch.nn.functional.kl_div(p1.log(), p2, reduction='batchmean')
    kl2 = torch.nn.functional.kl_div(p2.log(), p1, reduction='batchmean')
    return 0.5 * (kl1 + kl2)

def relation_contrastive_loss(aux, labels, train_mask, gene_mask, tau=0.2):
    if not isinstance(aux, dict) or not all(k in aux for k in ["z_me", "z_h", "z_c"]):
        return torch.tensor(0.0, device=labels.device, requires_grad=True)
    sel = ((labels>=0) & train_mask & gene_mask)
    if sel.sum() < 2:
        return torch.tensor(0.0, device=labels.device, requires_grad=True)
    gene_sel = sel[gene_mask]
    z_me = torch.nn.functional.normalize(aux["z_me"][gene_sel], dim=-1)
    z_h  = torch.nn.functional.normalize(aux["z_h"][gene_sel],  dim=-1)
    z_c  = torch.nn.functional.normalize(aux["z_c"][gene_sel],  dim=-1)
    def _nce(a, b):
        logits = (a @ b.t()) / tau
        targets = torch.arange(a.size(0), device=a.device)
        return torch.nn.functional.cross_entropy(logits, targets) + torch.nn.functional.cross_entropy(logits.t(), targets)
    return (_nce(z_me, z_h) + _nce(z_me, z_c) + _nce(z_h, z_c)) / 3.0

# ---------------- Structural modules: diffusion / IRM / neuro-symbolic / subgraph ----------------
class TinyTimeMLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class DiffusionDenoiser(torch.nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.time_proj = TinyTimeMLP(hid)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hid, hid),
            torch.nn.SiLU(),
            torch.nn.Linear(hid, hid)
        )
    def forward(self, z_noisy, log_sigma):
        t_emb = self.time_proj(log_sigma)
        return self.net(z_noisy + t_emb)

def diffusion_loss(aux, labels, gene_mask, train_mask, hps, device, model):
    lam = float(getattr(hps, "diffusion_lambda", 0.0) or 0.0)
    if lam <= 0 or not isinstance(aux, dict) or "z_rel_g" not in aux:
        return torch.tensor(0.0, device=device, requires_grad=True)
    sel = ((labels>=0) & train_mask & gene_mask)
    if sel.sum() < 2: return torch.tensor(0.0, device=device, requires_grad=True)
    Z = aux["z_rel_g"][sel[gene_mask]]
    H = Z.size(-1)
    if not hasattr(model, "_diff_denoiser"):
        model._diff_denoiser = DiffusionDenoiser(H).to(device)
    denoiser = model._diff_denoiser
    sigma_min = float(getattr(hps, "diffusion_sigma_min", 0.1))
    sigma_max = float(getattr(hps, "diffusion_sigma_max", 0.5))
    sigma = torch.exp(torch.empty(Z.size(0), 1, device=device).uniform_(sigma_min, sigma_max).log())
    eps = torch.randn_like(Z)
    Z_noisy = Z + sigma * eps
    eps_hat = denoiser(Z_noisy, torch.log(sigma).expand_as(Z))
    return lam * torch.mean((eps_hat - eps)**2)

def irmv1_penalty(loss, scale):
    g = torch.autograd.grad(loss, scale, create_graph=True)[0]
    return g.pow(2).sum()

def irm_invariance_loss(model, data, labels, m_tr, gene_mask, xtalk, gextra, hps, device):
    lam = float(getattr(hps, "irm_lambda", 0.0) or 0.0)
    if lam <= 0: return torch.tensor(0.0, device=device, requires_grad=True)
    envs = [ {"h": True}, {"circ": True} ]  # two synthetic environments
    penalty_total = 0.0
    scale = torch.tensor(1.0, device=device).requires_grad_()
    for ab in envs:
        out = fwd_full(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra, ablate=ab)
        logits = out[0] if isinstance(out, tuple) else out
        m = (labels>=0) & m_tr & gene_mask
        if m.sum()==0: continue
        loss_e = torch.nn.functional.binary_cross_entropy_with_logits(scale*logits[m], labels[m])
        penalty_total = penalty_total + irmv1_penalty(loss_e, scale)
    return lam * penalty_total

class NSPredicates(torch.nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.me = torch.nn.Linear(hid, 1)
        self.h  = torch.nn.Linear(hid, 1)
        self.c  = torch.nn.Linear(hid, 1)
    def forward(self, z_me, z_h, z_c):
        p_me = torch.sigmoid(self.me(z_me)).squeeze(-1)
        p_h  = torch.sigmoid(self.h (z_h )).squeeze(-1)
        p_c  = torch.sigmoid(self.c (z_c )).squeeze(-1)
        return p_me, p_h, p_c

def ns_logic_loss(aux, logits, labels, gene_mask, train_mask, hps, device, model):
    lam = float(getattr(hps, "ns_lambda", 0.0) or 0.0)
    if lam <= 0 or not isinstance(aux, dict):
        return torch.tensor(0.0, device=device, requires_grad=True)
    required = all(k in aux for k in ["z_me","z_h","z_c"])
    if not required: return torch.tensor(0.0, device=device, requires_grad=True)
    sel = ((labels>=0) & train_mask & gene_mask)
    if sel.sum() < 2: return torch.tensor(0.0, device=device, requires_grad=True)
    gene_sel = sel[gene_mask]
    z_me = aux["z_me"][gene_sel]; z_h = aux["z_h"][gene_sel]; z_c = aux["z_c"][gene_sel]
    H = z_me.size(-1)
    if not hasattr(model, "_ns_pred"):
        model._ns_pred = NSPredicates(H).to(device)
    p_me, p_h, p_c = model._ns_pred(z_me, z_h, z_c)
    y = torch.sigmoid(logits[sel])
    def AND(a,b): return torch.relu(a + b - 1.0)
    def IMPL(a,b): return torch.clamp(1.0 - a + b, 0.0, 1.0)
    r1 = IMPL(AND(p_me, p_h), y)        # ME ∧ H ⇒ y
    r2 = IMPL(p_c, 1.0 - y)             # C ⇒ ¬y  (if not desired, set ns_lambda=0)
    loss = (1.0 - r1).mean() + 0.5 * (1.0 - r2).mean()
    return lam * loss

# Subgraph (relation-channel) strategy via Gumbel-TopK
def _gumbel_noise_like(x):
    u = torch.rand_like(x)
    return -torch.log(-torch.log(u.clamp_min(1e-9)).clamp_min(1e-9))

def _straight_through_topk(scores, k, tau=0.5):
    C = scores.numel()
    g = _gumbel_noise_like(scores)
    logits = scores + g
    soft = torch.softmax(logits / tau, dim=-1)
    topk = torch.topk(logits, k=min(k, C), dim=-1).indices
    hard = torch.zeros_like(soft)
    hard.scatter_(0, topk, 1.0)
    mask = hard + soft - soft.detach()
    return mask, soft

def subgraph_consistency_loss(aux, logits_base, model, data, labels, m_tr, gene_mask, xtalk, gextra, hps, device):
    lam = float(getattr(hps, "subgraph_lambda", 0.0) or 0.0)
    if lam <= 0 or not isinstance(aux, dict):
        return torch.tensor(0.0, device=device, requires_grad=True)
    for k in ("z_me","z_h","z_c"):
        if k not in aux:
            return torch.tensor(0.0, device=device, requires_grad=True)
    sel = ((labels>=0) & m_tr & gene_mask)
    if sel.sum() < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    gene_sel = sel[gene_mask]
    z_me = aux["z_me"][gene_sel]; z_h = aux["z_h"][gene_sel]; z_c = aux["z_c"][gene_sel]
    s_me = (z_me.pow(2).sum(-1).sqrt().mean()).unsqueeze(0)
    s_h  = (z_h .pow(2).sum(-1).sqrt().mean()).unsqueeze(0)
    s_c  = (z_c .pow(2).sum(-1).sqrt().mean()).unsqueeze(0)
    scores = torch.cat([s_me, s_h, s_c], dim=0)  # (3,)
    k = int(getattr(hps, "subgraph_k", 2) or 2)
    tau = float(getattr(hps, "subgraph_tau", 0.5) or 0.5)
    mask_hard, soft = _straight_through_topk(scores, k=k, tau=tau)  # (3,)
    keep = (mask_hard > 0.5)
    ablate = {"me": not bool(keep[0].item()), "h": not bool(keep[1].item()), "circ": not bool(keep[2].item())}
    out_sub = fwd_full(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra, ablate=ablate)
    logits_sub = out_sub[0] if isinstance(out_sub, tuple) else out_sub
    sel_idx = ((labels>=0) & m_tr)
    yb = torch.sigmoid(logits_base[sel_idx])
    ys = torch.sigmoid(logits_sub[sel_idx])
    cons = torch.mean((yb - ys).pow(2))
    elam = float(getattr(hps, "subgraph_entropy_lambda", 0.0) or 0.0)
    entropy = -(soft.clamp_min(1e-8).log() * soft).sum()
    return lam * cons + elam * entropy

# ---------------- 新增：保存最终节点特征（分类器输入之前的表示） ----------------
def save_node_repr(fold_id, aux, labels, m_tr, m_va, gene_mask, out_dir, ids_map):
    """
    保存每一折下，训练集 / 验证集中节点的最终特征（输入分类器之前的表示）到 CSV。

    假设:
      - aux["z_rel_g"] 为 gene-level 最终嵌入，shape = (Ng, H)
      - gene_mask 为 bool 向量，标记哪些节点是 gene
    输出:
      - fold{fold_id}_train_node_repr.csv
      - fold{fold_id}_val_node_repr.csv
    每个文件列:
      - gene_unified_id, gene_index, label, f0..f{H-1}
    """
    if not isinstance(aux, dict) or ("z_rel_g" not in aux):
        print(f"[warn] fold{fold_id}: aux 中未找到 'z_rel_g'，跳过节点特征保存")
        return

    # gene-level 表示
    Z = aux["z_rel_g"].detach().cpu().numpy()  # (Ng, H)

    gene_mask_cpu = gene_mask.detach().cpu().numpy().astype(bool)
    labels_cpu = labels.detach().cpu().numpy()
    m_tr_cpu = m_tr.detach().cpu().numpy().astype(bool)
    m_va_cpu = m_va.detach().cpu().numpy().astype(bool)

    # unified node id 中，基因节点的全局 index
    gene_unified_ids = np.where(gene_mask_cpu)[0].astype(int)
    # global -> local (在 Z 中的行号)
    global2local = {int(g): i for i, g in enumerate(gene_unified_ids)}

    def _save(split_name, split_mask_cpu):
        sel = (labels_cpu >= 0) & split_mask_cpu & gene_mask_cpu
        idx_global = np.where(sel)[0].astype(int)
        if idx_global.size == 0:
            print(f"[warn] fold{fold_id} {split_name}: 无标注节点，跳过节点特征保存")
            return
        # 映射到 Z 的行号
        local_idx = [global2local[int(g)] for g in idx_global]
        feats = Z[local_idx, :]  # (n_sel, H)
        gene_index = [int(ids_map.get(int(g), -1)) for g in idx_global]
        labels_sel = labels_cpu[idx_global].astype(int)

        df = pd.DataFrame({
            "gene_unified_id": idx_global,
            "gene_index": gene_index,
            "label": labels_sel
        })
        for j in range(feats.shape[1]):
            df[f"f{j}"] = feats[:, j]
        out_path = out_dir / f"fold{fold_id}_{split_name}_node_repr.csv"
        df.to_csv(out_path, index=False)
        print(f"[info] 保存节点特征: {out_path} (n={len(df)}, dim={feats.shape[1]})")

    _save("train", m_tr_cpu)
    _save("val",   m_va_cpu)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".", help="Directory containing the CSVs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="Results", help="Output folder inside data_dir")
    ap.add_argument("--hps_json", type=str, default=None, help="Path to a JSON overriding HyperParams")
    ap.add_argument("--tag", type=str, default=None, help="Optional subfolder/tag under out_dir")
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    import os; os.chdir(args.data_dir)
    # Allow subfolder/tag per run
    results_dir = Path(args.out_dir)
    if args.tag:
        results_dir = results_dir / args.tag
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TrainConfig()
    # --- Apply HyperParams overrides from JSON if provided ---
    if args.hps_json is not None:
        with open(args.hps_json, "r") as f:
            _ov = json.load(f)
        for k, v in _ov.items():
            if hasattr(cfg.hparams, k):
                setattr(cfg.hparams, k, v)

    # Save the hyperparams actually used for this run
    with open(results_dir / "hparams_used.json", "w") as _f:
        json.dump({k: getattr(cfg.hparams, k) for k in vars(cfg.hparams)}, _f, indent=2)
    # quick visibility
    hps = cfg.hparams
    print(f"[HPS] focal={hps.focal_gamma}, rdrop={hps.rdrop_lambda}, contrast=({hps.contrast_lambda},{hps.contrast_tau}), "
          f"diff={hps.diffusion_lambda}[{hps.diffusion_sigma_min},{hps.diffusion_sigma_max}], irm={hps.irm_lambda}, ns={hps.ns_lambda}, "
          f"subgraph=({hps.subgraph_lambda}, k={hps.subgraph_k}, tau={hps.subgraph_tau}, ent={hps.subgraph_entropy_lambda}), "
          f"smooth={hps.smooth_lambda}, head_aux={hps.head_aux_lambda}, dropout={hps.dropout}, edge_drop={hps.edge_drop_rate}")

    loaded = load_csv_graph(cfg.paths)
    data = loaded.data
    data.x = data.x.to(device).float()
    data.edge_index = data.edge_index.to(device).long()
    data.edge_type  = data.edge_type.to(device).long()
    data.edge_weight= data.edge_weight.to(device).float()
    gene_mask = loaded.gene_mask.bool().to(device)
    labels = loaded.label.float().to(device)

    xtalk = torch.zeros(data.num_nodes, 7, device=device)
    xtalk[gene_mask] = torch.from_numpy(loaded.xtalk_counts).to(device)
    gextra_np = getattr(loaded, "gene_extra", None)
    if gextra_np is None or gextra_np.shape[1]==0:
        gextra = torch.zeros(data.num_nodes, 0, device=device)
        gene_extra_dim = 0
    else:
        gene_extra_dim = int(gextra_np.shape[1])
        gextra = torch.zeros(data.num_nodes, gene_extra_dim, device=device)
        gextra[gene_mask] = torch.from_numpy(gextra_np).to(device)

    unify2orig = {int(u):int(o) for u,o in zip(loaded.node_ids["gene_unified_ids"], loaded.node_ids["gene_index"])}

    gene_ids = torch.where(gene_mask)[0].detach().cpu().numpy()
    y_gene = labels[gene_mask].detach().cpu().numpy()
    labeled = y_gene >= 0
    labeled_gene_ids = gene_ids[labeled]
    y_lab = y_gene[labeled].astype(int)

    if len(np.unique(y_lab)) < 2:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(labeled_gene_ids)
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(labeled_gene_ids, y_lab)

    rows = []
    fold_id = 0
    hps = cfg.hparams

    for tr_idx, va_idx in splitter:
        fold_id += 1
        tr_ids = labeled_gene_ids[tr_idx]
        va_ids = labeled_gene_ids[va_idx]

        try:
            model = XTalkRGCN(hps, gene_extra_dim=gene_extra_dim).to(device)
        except TypeError:
            model = XTalkRGCN(in_dim=64, hidden_dim=hps.dim, num_relations=len(RELATION_LIST),
                              num_bases=hps.basis, dropout=hps.dropout,
                              use_xtalk_gate=hps.enable_xtalk_gate).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hps.epochs, eta_min=hps.lr/50.0)

        m_tr = torch.zeros_like(gene_mask); m_tr[tr_ids]=True
        m_va = torch.zeros_like(gene_mask); m_va[va_ids]=True

        best = {"pr": -1, "state": None}
        bad = 0

        for ep in range(1, hps.epochs+1):
            model.train(); opt.zero_grad(set_to_none=True)

            # two stochastic forward passes for R-Drop & robust gradients
            out1 = fwd_full(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra)
            out2 = fwd_full(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra)
            if isinstance(out1, tuple): logits1, aux1 = out1
            else: logits1, aux1 = out1, {}
            if isinstance(out2, tuple): logits2, aux2 = out2
            else: logits2, aux2 = out2, {}

            # base supervised loss: focal BCE (gamma from hps; falls back to BCE if 0)
            base_loss1, posw = bce_pos_weight(logits1, labels, m_tr)
            gamma = float(getattr(hps, "focal_gamma", 0.0) or 0.0)
            sup_loss = focal_bce_with_logits(logits1, labels, m_tr, gamma=gamma, pos_weight=posw)

            # auxiliary heads & smoothing
            lam_h = getattr(hps, "head_aux_lambda", 0.2)
            lam_s = getattr(hps, "smooth_lambda", 1e-3)
            loss = sup_loss
            if isinstance(aux1, dict):
                if "smooth" in aux1:
                    loss = loss + aux1["smooth"]
                loss = loss + aux_heads_loss(aux1, labels, m_tr, gene_mask, head_aux_lambda=lam_h)

            # relation-contrastive InfoNCE across (me,h,circ) per-gene channels
            lam_c = getattr(hps, "contrast_lambda", 0.0)
            tau_c = getattr(hps, "contrast_tau", 0.2)
            if lam_c > 0 and isinstance(aux1, dict):
                loss = loss + lam_c * relation_contrastive_loss(aux1, labels, m_tr, gene_mask, tau=tau_c)

            # structural modular losses (optional)
            loss = loss + diffusion_loss(aux1, labels, gene_mask, m_tr, hps, device, model)
            loss = loss + irm_invariance_loss(model, data, labels, m_tr, gene_mask, xtalk, gextra, hps, device)
            loss = loss + ns_logic_loss(aux1, logits1, labels, gene_mask, m_tr, hps, device, model)
            loss = loss + subgraph_consistency_loss(aux1, logits1, model, data, labels, m_tr, gene_mask, xtalk, gextra, hps, device)

            # R-Drop KL between two stochastic passes on labeled training genes
            lam_r = getattr(hps, "rdrop_lambda", 0.0)
            train_gene_mask = (labels>=0) & m_tr & gene_mask
            if lam_r > 0:
                loss = loss + lam_r * rdrop_kl(logits1, logits2, train_gene_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sch.step()

            # ---- validation (PR-AUC) ----
            model.eval()
            with torch.no_grad():
                outv = fwd_full(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra)
                if isinstance(outv, tuple): logits_v = outv[0]
                else: logits_v = outv
                pv = torch.sigmoid(logits_v)
                mv = (labels>=0) & m_va
                if mv.sum()>0:
                    y_true = labels[mv].detach().cpu().numpy().astype(int)
                    y_prob = pv[mv].detach().cpu().numpy()
                    try: pr = average_precision_score(y_true, y_prob)
                    except: pr = float("nan")
                else: pr = float("nan")
            cur = pr if not np.isnan(pr) else 0.0
            if cur > best["pr"]:
                best = {"pr": cur, "state": {k:v.detach().cpu() for k,v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
            if bad >= 10:
                print(f"[EarlyStop] fold={fold_id}, ep={ep}, best_val_aupr={best['pr']:.4f}")
                break

        if best["state"] is not None:
            model.load_state_dict(best["state"])
        model.eval()

        with torch.no_grad():
            out = fwd_full(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra)
            if isinstance(out, tuple): logits, aux = out
            else: logits, aux = out, {}

        metrics_train = eval_and_dump("train", logits, labels, m_tr, fold_id, results_dir, unify2orig)
        metrics_val   = eval_and_dump("val",   logits, labels, m_va, fold_id, results_dir, unify2orig)
        rows.append({"fold":fold_id,"split":"train",**metrics_train})
        rows.append({"fold":fold_id,"split":"val",**metrics_val})

        if isinstance(aux, dict) and "rel_attn" in aux:
            save_interpret(fold_id, "val", aux, m_va, labels, results_dir, unify2orig)

        # ★ 新增：保存该折训练集 / 验证集的最终节点特征
        save_node_repr(fold_id, aux, labels, m_tr, m_va, gene_mask, results_dir, unify2orig)

        # channel ablation analysis (ME/H/C) on val set
        with torch.no_grad():
            base = torch.sigmoid(logits)
            def to_prob(o):
                if isinstance(o, tuple): o=o[0]
                return torch.sigmoid(o)
            out_me0 = fwd_full(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra, ablate={"me":True})
            out_h0  = fwd_full(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra, ablate={"h":True})
            out_c0  = fwd_full(model, data.x, data.edge_index, data.edge_type, data.edge_weight, gene_mask, xtalk, gextra, ablate={"circ":True})
            p_me0 = to_prob(out_me0); p_h0=to_prob(out_h0); p_c0=to_prob(out_c0)

            mask_np = ((labels>=0) & m_va).detach().cpu().numpy()
            idx = np.where(mask_np)[0].astype(int)
            df_ab = pd.DataFrame({
                "gene_unified_id": idx,
                "gene_index": [int(unify2orig.get(int(i),-1)) for i in idx],
                "prob_base": base[(labels>=0) & m_va].detach().cpu().numpy(),
                "prob_no_me": p_me0[(labels>=0) & m_va].detach().cpu().numpy(),
                "prob_no_h":  p_h0[(labels>=0) & m_va].detach().cpu().numpy(),
                "prob_no_c":  p_c0[(labels>=0) & m_va].detach().cpu().numpy(),
            })
            for col in ["me","h","c"]:
                df_ab[f"delta_{col}"] = df_ab["prob_base"] - df_ab[f"prob_no_{col}"]
            df_ab.to_csv(results_dir / f"fold{fold_id}_val_ablation.csv", index=False)

    df = pd.DataFrame(rows)
    order = ["fold","split","roc_auc","pr_auc","accuracy","f1","precision","recall","specificity"]
    df = df[order]
    df.to_csv(results_dir / "cv5_metrics_per_fold.csv", index=False)
    def agg(group):
        g = group.drop(columns=["fold","split"], errors="ignore")
        return pd.Series({**{f"mean_{c}": g[c].mean() for c in g.columns},
                          **{f"std_{c}":  g[c].std()  for c in g.columns}})
    summary = df.groupby("split").apply(agg).reset_index()
    summary.to_csv(results_dir / "cv5_metrics_summary.csv", index=False)

if __name__=="__main__":
    main()
