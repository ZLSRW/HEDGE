# HEDGE

HEDGE is a heterogeneous epigenetic diffusion-and-gating encoder for autism spectrum disorder (ASD) risk-gene identification. It integrates cortical regulatory QTL (rQTL), histone QTL (hQTL), circRNA QTL (circQTL), linkage disequilibrium (LD), and protein–protein interaction (PPI) evidence in a typed SNP–gene graph.

This repository accompanies the manuscript **“A Heterogeneous Epigenetic Diffusion-and-Gating Encoder for Autism Spectrum Disorder Risk Gene Identification.”**

## Graph definition

The graph contains rQTL, hQTL, and circQTL SNP nodes and gene nodes. Its three structural edge families are SNP–SNP LD, SNP–gene association, and gene–gene PPI. Every structural edge is stored in both orientations during graph loading, producing symmetric adjacency matrices for undirected message passing.

HEDGE uses nine operational relations:

`rQTL`, `hQTL`, `circQTL`, `LD`, `PPI`, `r-h`, `r-circ`, `h-circ`, and `r-h-circ`.

The four cross-channel relations are derived from the complete set of QTL channels incident on each target gene and are added as operational labels to the corresponding SNP–gene associations. The original structural association tables remain unchanged.

## Repository contents

```text
HEDGE/
├── code/
│   ├── config.py
│   ├── cv_5fold.py
│   ├── cv_5fold_ratio.py
│   ├── data_io.py
│   ├── grid_search.py
│   ├── hedge_model.py
│   ├── model_layers.py
│   ├── training.py
│   └── validate_data.py
├── data/
│   ├── README.md
│   ├── rqtl_snp_nodes.csv
│   ├── hqtl_snp_nodes.csv
│   ├── circqtl_snp_nodes.csv
│   ├── gene_nodes.csv
│   ├── snp_snp_ld_edges.csv
│   ├── snp_gene_association_edges.csv
│   └── gene_gene_ppi_edges.csv
├── manuscript/HEDGE_manuscript.pdf
├── CITATION.cff
├── DATA_SOURCES.md
├── REPOSITORY_MANIFEST.md
├── data_validation_report.json
├── environment.yml
├── main.py
├── MANIFEST.sha256
└── requirements.txt
```

## Environment

The manuscript environment is defined in `environment.yml` with Python 3.8.2 and PyTorch 1.8.0. Compatibility validation of this release was completed with Python 3.8.20, NumPy 1.24.4, pandas 2.0.3, scikit-learn 1.3.2, PyTorch 2.3.0, and PyTorch Geometric 2.6.1.

Create the manuscript environment with:

```bash
conda env create -f environment.yml
conda activate hedge
```

Install the compatibility-tested Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Data validation

Run the complete schema, identifier, feature, weight, endpoint, and relation audit before training:

```bash
python main.py validate-data --data_dir .
```

The validated package contains 6,699 SNP nodes, 8,848 unique gene nodes, 75,365 LD structural edges, 19,784 SNP–gene structural associations, and 72,348 PPI structural edges. The machine-readable validation result is stored in `data_validation_report.json`.

## Model evaluation

Run the five-fold cross-validation protocol:

```bash
python main.py train --data_dir . --out_dir results/cross_validation
```

Run the reduced-supervision analysis:

```bash
python main.py ratio --data_dir . --out_dir results/reduced_supervision
```

Run the hyperparameter grid search reported for APPNP propagation depth, R-GCN depth, and convergence gating:

```bash
python main.py search --data_dir . --out_dir results/hyperparameter_search --strategy grid
```

Training uses RMSProp, the manuscript-defined five-fold stratified splits, validation AUPR for checkpoint selection, focal loss, stochastic consistency, channel stability, invariant-risk, prior-alignment, channel-pair, and PPI smoothing terms. The default HEDGE configuration uses six relation-aware layers and one APPNP propagation step.

## Outputs

The cross-validation workflow writes fold-level predictions, channel gates, nine-relation attention weights, static channel matrices, eight channel-perturbation profiles, per-fold metrics, summary statistics, and the complete hyperparameter record. Generated results are excluded from Git tracking by `.gitignore`.

## Data provenance

The packaged CSV files are processed graph inputs. Their public and controlled-access source resources, identifiers, and manuscript mappings are listed in `DATA_SOURCES.md`. Table schemas and numeric codes are documented in `data/README.md`.
