# HEDGE

Code and processed data for **A Heterogeneous Epigenetic Diffusion-and-Gating Encoder for Autism Spectrum Disorder Risk Gene Identification**.

HEDGE integrates rQTL, hQTL and circQTL evidence with LD and PPI connectivity. Relation-specific message passing, convergence-aware channel mixing, PPI diffusion and three prediction heads produce gene-level ASD risk scores.

## Installation

Use Python 3.8 and install the tested dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The tested environment uses Python 3.8.20, PyTorch 2.3.0, PyTorch Geometric 2.6.1, NumPy 1.24.4, pandas 2.0.3, SciPy 1.10.1 and scikit-learn 1.3.2. The commands below run from the repository root. HEDGE selects CPU by default; use `--device cuda` for CUDA training.

## Data

The repository includes the processed 64-dimensional node features and weighted graph edges needed to run HEDGE. Training starts from these features; sequence embedding models and external database downloads are not required.

| File in `data/` | Contents |
| --- | --- |
| `rqtl_snp_nodes.csv` | rQTL SNP identifiers and features |
| `hqtl_snp_nodes.csv` | hQTL SNP identifiers and features |
| `circqtl_snp_nodes.csv` | circQTL SNP identifiers and features |
| `gene_nodes.csv` | Gene identifiers, labels and features |
| `snp_gene_association_edges.csv` | Weighted regulatory associations |
| `snp_snp_ld_edges.csv` | LD edges |
| `gene_gene_ppi_edges.csv` | PPI edges |

HEDGE merges repeated gene identifiers into graph nodes and defines cross-validation folds over 8,848 unique genes: 4,933 positive and 3,915 negative. The resulting graph has 15,547 nodes, including 6,699 SNP nodes. Edges are stored in both orientations during loading.

`data/baselines/` contains the feature and edge tables used by the comparison methods. Their loaders retain gene-table records, matching the baseline evaluation protocol described in the Supplementary Information.

`reference_data/` contains numerical Source Data and Supplementary Data tables supplied with the manuscript. These files are preserved byte for byte. `results/` contains saved validation predictions and analysis inputs used to recompute reported results. The input checksums are in `MANIFEST.sha256`.

```bash
python main.py validate-data
sha256sum -c MANIFEST.sha256 --quiet
```

## Training and prediction

```bash
python main.py train --config configs/main.json --out-dir outputs/main --device cpu
python scripts/predict.py --checkpoint outputs/main/fold1_best.pt --out-dir outputs/prediction
```

The main configuration uses 64-dimensional representations, six R-GCN layers, eight bases, four relation-attention heads, two APPNP propagation steps and restart weight 0.5. Optimization uses AdamW with learning rate 0.005 and weight decay 0.01 for up to 100 epochs. Validation AUPR selects the saved checkpoint. Predictions use a probability threshold of 0.5 for binary classification.

The five training-objective terms are supervised binary cross-entropy, PPI smoothness, stochastic probability consistency, two-environment invariance and noise reconstruction. The noise-reconstruction mapping remains fixed while gradients pass through it to HEDGE.

`--feature-mode standardized` applies per-dimension normalization across graph nodes and is the default. `--feature-mode input` uses the supplied feature values directly. Each checkpoint stores its feature mode, model configuration and channel intervention; inference restores those settings.

Each fold exports a checkpoint, epoch history, training and validation probabilities, gene-level relation attention, channel affinities, channel-mixing weights and relation embeddings. Relation-attention slots 0–4 represent rQTL, hQTL, circQTL, LD and PPI. Slots 5–8 contain zero messages and participate in the nine-slot attention normalization. A zero message can have a nonzero attention weight.

## Experiments

```bash
python main.py ratios --train-fractions 0,.1,.2,.3,.4,.5 --repeats 5 --out-dir outputs/ratios
python main.py channel-ablation --out-dir outputs/channels
python main.py component-ablation --out-dir outputs/components
python main.py grid --config configs/grid.json --out-dir outputs/grid
```

The channel experiment independently fits eight conditions: all channels, no regulatory channels, each channel alone and each channel removed. Nodes, LD and PPI remain present. The component experiment includes the complete model and removal of the convergence gate, graph propagation, APPNP, dropout or auxiliary losses. Its `component` optimization protocol uses Adam, plateau scheduling and the component-study objective. Main training uses the `main` protocol. `--protocol` selects either protocol explicitly.

The grid crosses five APPNP depths, four R-GCN depths and two gate settings, giving 40 configurations. `configs/grid.json` supplies the fixed settings for this experiment. Main benchmark predictions, component ablations and grid results are separate experiments.

## Comparison methods

The comparison implementations use the manuscript's task-specific adaptations and processed inputs.

| Manuscript name | Python module | Model |
| --- | --- | --- |
| DeepASDPred | `baselines.deepasdpred` | Feature-based CNN |
| ASiD | `baselines.asid` | Linear logistic classifier |
| GrID-Net | `baselines.gridnet` | Directed graph classifier |
| Mantis-ml | `baselines.mantis_ml` | ExtraTrees, random forest, SVC and gradient-boosting probability ensemble |
| DeepND | `baselines.deepnd` | Single-task graph classifier |
| GraphSAGE | `baselines.graphsage` | GraphSAGE classifier |
| VBASS | `baselines.vbass` | Shared MLP, two-class prediction head and Gaussian encoder |

```bash
python -m baselines.deepasdpred.train --out_dir outputs/baselines/DeepASDPred
python -m baselines.asid.train --out_dir outputs/baselines/ASiD
python -m baselines.gridnet.train --out_dir outputs/baselines/GrID-Net
python -m baselines.mantis_ml.train --out_dir outputs/baselines/Mantis-ml
python -m baselines.deepnd.train --out_dir outputs/baselines/DeepND
python -m baselines.graphsage.train --out_dir outputs/baselines/GraphSAGE
python -m baselines.vbass.train --out_dir outputs/baselines/VBASS
```

Each module also provides `ratios` with the same `--out_dir` argument for the reduced-label experiments. Neural baseline training accepts `--epochs` and all baseline training commands accept `--folds` and `--seed`.

## Reported results

```bash
python main.py reproduce --out-dir outputs/reported_results
```

This command recomputes results from the supplied predictions and numerical tables. Training commands fit new models and write their outputs separately.

The calculation covers eight models across five validation folds, metric-specific percentage-point gains, 40 grid configurations, evidence tiers, risk groups, candidate channel counts, relation attention, regulatory groups, comorbidity modules and UMAP coordinate correspondence. Each gain uses the strongest baseline for that metric. Full-precision gains and gains calculated from the displayed rounded values are exported separately.

The supplied results contain 234 discovery genes and 28 highlighted functional candidates, including INSYN2B. Candidate channel counts are 23 hQTL, four rQTL and one circQTL. Level 5 contains 4,406 unique genes, with 3,817 correct classifications and 589 incorrect classifications. Positive-gene relation-attention distributions pool 4,933 genes across five fitted models and nine slots; model means are calculated per gene when required for downstream summaries.

## Downstream analyses

```bash
python scripts/analyze.py effects --input results/effect_space/per_gene_probs_raw.csv --out-dir outputs/effects
python scripts/analyze.py modules --input results/module_features/modules_gene_level_features.csv --out-dir outputs/modules
```

The effect analysis calculates seven channel-perturbation effects for positive genes, applies min–max normalization and standardization, fits an eight-component Gaussian mixture and evaluates bootstrap NMI. `--bootstraps` defaults to 20. The module analysis uses risk score, nine phenotype-category indicators, three affinities and three gates, with silhouette-based selection among two to six clusters.

`python scripts/analyze.py disease-sets --input scores.csv --sets disease_sets.json` runs rank-sum tests within the positive-gene universe and applies Benjamini–Hochberg correction across the supplied category's disease sets. The JSON maps disease names to lists of gene symbols. The score table requires `gene_symbol` and `risk_score`; an optional `label` column selects positive genes.

`python scripts/analyze.py enrichment-map --input enrichment.csv --sets categories.json` summarizes a module/library enrichment table using the supplied category regular expressions. It uses up to 30 terms with adjusted P ≤ 0.5. Cell labels count matched terms and cell values sum −log10 adjusted P. Terms may contribute to more than one category. Enrichment inputs require `Term` and `Adjusted P-value` columns. Gene membership tables and enrichment results are included in `reference_data/Supplementary Data/`.

In the saved numerical analysis tables, the channel key `me` denotes rQTL; `r` denotes the same channel in HEDGE's exported affinity tables. `h` and `circ` denote hQTL and circQTL.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover nine-slot attention, gate mixing, objective gradients, empty training masks, channel interventions, component removal, multiple-testing adjustment, enrichment summaries, effect normalization, grid configuration and UMAP coordinate matching.
