from dataclasses import dataclass, field
from typing import Dict, List


REL_RQTL = "rQTL"
REL_HQTL = "hQTL"
REL_CIRCQTL = "circQTL"
REL_LD = "LD"
REL_PPI = "PPI"
REL_R_H = "r-h"
REL_R_CIRC = "r-circ"
REL_H_CIRC = "h-circ"
REL_R_H_CIRC = "r-h-circ"

RELATION_LIST: List[str] = [
    REL_RQTL,
    REL_HQTL,
    REL_CIRCQTL,
    REL_LD,
    REL_PPI,
    REL_R_H,
    REL_R_CIRC,
    REL_H_CIRC,
    REL_R_H_CIRC,
]
REL2ID: Dict[str, int] = {
    relation_name: relation_id
    for relation_id, relation_name in enumerate(RELATION_LIST)
}

NODE_SNP_RQTL = 2
NODE_SNP_HQTL = 3
NODE_SNP_CIRCQTL = 4
NODE_GENE = 5


@dataclass
class Paths:
    snp_nodes_r: str = "data/rqtl_snp_nodes.csv"
    snp_nodes_h: str = "data/hqtl_snp_nodes.csv"
    snp_nodes_circ: str = "data/circqtl_snp_nodes.csv"
    gene_nodes: str = "data/gene_nodes.csv"
    edge_snp_gene: str = "data/snp_gene_association_edges.csv"
    edge_snp_snp: str = "data/snp_snp_ld_edges.csv"
    edge_gene_gene: str = "data/gene_gene_ppi_edges.csv"


@dataclass
class HyperParams:
    dim: int = 64
    num_layers: int = 6
    basis: int = 8
    lr: float = 5e-4
    weight_decay: float = 1e-2
    epochs: int = 100
    seed: int = 42
    dropout: float = 0.3
    edge_drop_rate: float = 0.2
    enable_convergence_gate: bool = True
    attn_heads: int = 4
    appnp_K: int = 1
    appnp_alpha: float = 0.1
    smooth_lambda: float = 1e-3
    focal_gamma: float = 1.0
    consistency_lambda: float = 0.5
    stability_lambda: float = 0.3
    irm_lambda: float = 1.0
    contrast_lambda: float = 0.2
    contrast_tau: float = 0.2
    pair_lambda: float = 0.1
    pair_margin: float = 0.2
    activation: str = "prelu"
    norm: str = "layer"
    pre_ln: bool = True


@dataclass
class TrainConfig:
    paths: Paths = field(default_factory=Paths)
    hparams: HyperParams = field(default_factory=HyperParams)
    val_ratio: float = 0.2
    stratify_by_label: bool = True
    topk: int = 100
