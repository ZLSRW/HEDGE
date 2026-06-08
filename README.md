# ASD Crosstalk Manuscript Submission Package

This folder is a self-contained submission package prepared from `/data/liguodong/0ASD_codes` for the manuscript `ASD_Crosstalk_SN.pdf`.

## What was included

- `manuscript/ASD_Crosstalk_SN.pdf`: manuscript to submit.
- `code/`: core source files traced from the actual training entry scripts.
- `ASD_Data/`: the CSV files required by the core training pipeline.
- `main.py`: unified entrypoint added for packaging and reproducibility.
- `requirements.txt`: Python dependencies for the core pipeline.
- `SCRIPT_MAP.md`: dependency tree and rationale for each included script.
- `hps_example.json`: example hyperparameter override file.

## Important note about the original root `main.py`

The original `/data/liguodong/0ASD_codes/main.py` is only a PyCharm sample script and is not connected to the manuscript pipeline. For this package, the real runnable entrypoint is the new `main.py` in this folder, which wraps the actual manuscript scripts under `code/`.

## Directory layout

```text
ASD_Crosstalk_SN_submission/
├── ASD_Data/
├── code/
├── manuscript/
├── hps_example.json
├── main.py
├── README.md
├── requirements.txt
└── SCRIPT_MAP.md
```

## Core entry commands

Run from this folder:

```bash
python main.py train --data_dir . --out_dir Results
python main.py ratio --data_dir . --out_dir Results
python main.py search --data_dir . --out_root Results_Search --strategy grid-mod
```

## What each command does

- `train`: runs the standard 5-fold cross-validation pipeline from `code/cv_5fold.py`.
- `ratio`: runs robustness experiments across multiple train:validation ratios from `code/cv_5fold_ratio.py`.
- `search`: runs the reduced hyperparameter search from `code/grid_search.py`.

## Data path convention

The copied code keeps its original relative path design:

- scripts live in `code/`
- data live in `ASD_Data/`

Because `code/config.py` and `code/config1.py` point to `../ASD_Data/...`, this packaged layout remains compatible without editing the original research scripts.

## Hyperparameter override example

You can override fields from `HyperParams` with:

```bash
python main.py train --data_dir . --out_dir Results --hps_json hps_example.json
```

## Expected outputs

- `train`: writes fold-level predictions, metrics, interpretation tables, and `hparams_used.json`.
- `ratio`: writes results into `*_ratio/` subdirectories for each train:val setting.
- `search`: writes one folder per parameter combination plus summary CSV files.

## Environment notes

- Python 3.9+ is recommended.
- A CUDA GPU is optional; the code falls back to CPU if CUDA is unavailable.
- `torch-geometric` must match the installed PyTorch version.

## Included core data files

- `0.SNP_SNP_edges.csv`
- `0.SNP_gene_edges.csv`
- `0.gene_gene_edges.csv`
- `1.meQTL_SNP_nodes_feature.csv`
- `2.hQTL_SNP_nodes_feature.csv`
- `3.circQTL_SNP_nodes_feature.csv`
- `4.genes_proteins_nodes_feature.csv`

Additional documentation from the original repository was also copied:

- `ASD_Data/Data_README.md`
- `ASD_Data/Model_XTalk-RGCN-ASD_proposal.md`
