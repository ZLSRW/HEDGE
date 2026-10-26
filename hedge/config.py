from dataclasses import dataclass
from pathlib import Path
REL_RQTL, REL_HQTL, REL_CIRCQTL, REL_LD, REL_PPI = ('rqtl', 'hqtl', 'circqtl', 'ld', 'ppi')
RELATION_LIST = [REL_RQTL, REL_HQTL, REL_CIRCQTL, REL_LD, REL_PPI] + ['unused_slot_%d' % i for i in range(5, 9)]
REL2ID = {r: i for i, r in enumerate(RELATION_LIST)}
DATA_ROOT = Path(__file__).resolve().parents[1] / 'data'

@dataclass
class Paths:
    snp_nodes_r: str = str(DATA_ROOT / 'rqtl_snp_nodes.csv')
    snp_nodes_h: str = str(DATA_ROOT / 'hqtl_snp_nodes.csv')
    snp_nodes_c: str = str(DATA_ROOT / 'circqtl_snp_nodes.csv')
    gene_nodes: str = str(DATA_ROOT / 'gene_nodes.csv')
    edge_snp_gene: str = str(DATA_ROOT / 'snp_gene_association_edges.csv')
    edge_snp_snp: str = str(DATA_ROOT / 'snp_snp_ld_edges.csv')
    edge_gene_gene: str = str(DATA_ROOT / 'gene_gene_ppi_edges.csv')

@dataclass
class HyperParams:
    dim: int = 64
    num_layers: int = 6
    basis: int = 8
    lr: float = 0.005
    weight_decay: float = 0.01
    epochs: int = 100
    seed: int = 42
    dropout: float = 0.3
    edge_drop_rate: float = 0.2
    enable_convergence_gate: bool = True
    attn_heads: int = 4
    appnp_K: int = 2
    appnp_alpha: float = 0.5
    smooth_lambda: float = 0.001
    rdrop_lambda: float = 0.5
    diffusion_lambda: float = 0.05
    diffusion_sigma_min: float = 0.1
    diffusion_sigma_max: float = 0.5
    irm_lambda: float = 1.0
    activation: str = 'prelu'
    norm: str = 'layer'
    pre_ln: bool = True
