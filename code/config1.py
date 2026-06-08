from dataclasses import dataclass, field
from typing import Dict, List

# ========= 选择预设 =========
# 可选项: "S1" | "S2" | "S3"
# S1: 稳健提升（推荐默认）; S2: 更强正则（小数据/过拟合明显）; S3: 轻量快速（验证方向）
PRESET: str = "S3"

# ========= Relations & Node types =========
REL_MEQTL   = "meqtl"
REL_HQTL    = "hqtl"
REL_CIRCQTL = "circqtl"
REL_LD      = "ld"
REL_PPI     = "ppi"

REL_X_ME_H      = "x(me,h)"
REL_X_ME_CIRC   = "x(me,circ)"
REL_X_H_CIRC    = "x(h,circ)"
REL_X_ME_H_CIRC = "x(me,h,circ)"

RELATION_LIST: List[str] = [
    REL_MEQTL, REL_HQTL, REL_CIRCQTL, REL_LD, REL_PPI,
    REL_X_ME_H, REL_X_ME_CIRC, REL_X_H_CIRC, REL_X_ME_H_CIRC,
]
REL2ID: Dict[str, int] = {name: i for i, name in enumerate(RELATION_LIST)}

NODE_SNP_ME   = 2
NODE_SNP_H    = 3
NODE_SNP_CIRC = 4
NODE_GENE     = 5

# ========= Paths =========
@dataclass
class Paths:
    snp_nodes_me: str = "../ASD_Data/1.meQTL_SNP_nodes_feature.csv"
    snp_nodes_h:  str = "../ASD_Data/2.hQTL_SNP_nodes_feature.csv"
    snp_nodes_c:  str = "../ASD_Data/3.circQTL_SNP_nodes_feature.csv"
    gene_nodes:   str = "../ASD_Data/4.genes_proteins_nodes_feature.csv"
    edge_snp_gene: str = "../ASD_Data/0.SNP_gene_edges.csv"
    edge_snp_snp:  str = "../ASD_Data/0.SNP_SNP_edges.csv"
    edge_gene_gene:str = "../ASD_Data/0.gene_gene_edges.csv"

# ========= Hyper-parameters (ALL switches live here) =========
@dataclass
class HyperParams:
    # ---- 模型维度与深度 ----
    dim: int = 64
    num_layers: int = 4
    basis: int = 8

    # ---- 优化器 ----
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    seed: int = 42

    # ---- 正则与采样 ----
    dropout: float = 0.1
    edge_drop_rate: float = 0.1

    # ---- 模块开关 ----
    enable_xtalk_gate: bool = True
    attn_heads: int = 4
    appnp_K: int = 10
    appnp_alpha: float = 0.1
    aux_lambda: float = 0.1                 # reserved
    smooth_lambda: float = 1e-3             # Laplacian smoothing regularizer
    head_aux_lambda: float = 0.2            # auxiliary heads BCE loss

    # ---- 监督增强 / 可选（与你现有代码保持一致）----
    focal_gamma: float = 1.0                # 0 => vanilla BCE
    contrast_lambda: float = 0.2
    contrast_tau: float = 0.2
    rdrop_lambda: float = 0.5
    diffusion_lambda: float = 0.2
    diffusion_sigma_min: float = 0.1
    diffusion_sigma_max: float = 0.5
    irm_lambda: float = 1.0
    ns_lambda: float = 0.5
    subgraph_lambda: float = 0.3
    subgraph_k: int = 2
    subgraph_tau: float = 0.5
    subgraph_entropy_lambda: float = 0.4

    # ---- 新增：激活/标准化选择（配合 patched 模型）----
    # activation: "relu" | "silu" | "gelu"
    activation: str = "silu"
    # norm: "layer" | "batch" | "none"
    norm: str = "layer"
    # Pre-LN (True) 或 Post-LN (False)
    pre_ln: bool = True

# ========= 三套可选预设（仅覆盖上面的同名键） =========
PRESETS: Dict[str, Dict] = {
    # S1: 稳健提升（首选）
    "S1": {
        "num_layers": 6,
        "dim": 64,
        "activation": "silu",
        "norm": "layer",
        "pre_ln": True,
        "dropout": 0.2,
        "edge_drop_rate": 0.1,
        "appnp_K": 15,
        "appnp_alpha": 0.10,
        "lr": 7e-4,
        "weight_decay": 0.01,
        "focal_gamma": 1.0,
    },
    # S2: 更强正则（小数据/过拟合明显）
    "S2": {
        "num_layers": 3,
        "dim": 64,
        "activation": "gelu",
        "norm": "batch",
        "pre_ln": False,
        "dropout": 0.3,
        "edge_drop_rate": 0.2,
        "appnp_K": 20,
        "appnp_alpha": 0.15,
        "lr": 8e-5,
        "weight_decay": 0.02,
        "focal_gamma": 1.5,
    },
    # S3: 轻量快速（验证方向）
    "S3": {
        "num_layers": 6,
        "dim": 64,
        "activation": "prelu",
        "norm": "layer",
        "pre_ln": True,
        "dropout": 0.3,
        "edge_drop_rate": 0.2,
        "appnp_K": 1,
        "appnp_alpha": 0.10,
        "lr": 5e-3,
        "weight_decay": 0.01,
        "focal_gamma": 0.0,
    },
}

# 供 TrainConfig 默认使用的超参工厂：依据 PRESET 返回 HyperParams
def _hp_factory() -> HyperParams:
    override = PRESETS.get(PRESET, {})
    return HyperParams(**override)

# ========= Train config (data splits etc.; DO NOT place module switches here) =========
@dataclass
class TrainConfig:
    paths: Paths = field(default_factory=Paths)
    # 关键：默认用预设工厂生成超参（从 PRESET 读取）
    hparams: HyperParams = field(default_factory=_hp_factory)
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    stratify_by_label: bool = True
    topk: int = 100

# ======== 使用提示 ========
# 1) 只需修改文件顶部的 PRESET="S1"/"S2"/"S3" 即可切换整套参数；
# 2) 仍可通过命令行/JSON 在运行时覆盖个别键值（cv_5fold.py 会在载入后覆盖）。
