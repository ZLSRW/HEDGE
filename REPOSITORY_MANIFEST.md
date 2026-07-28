# Repository manifest

| Component | Function |
|---|---|
| `main.py` | Unified command-line entry point |
| `environment.yml` | Manuscript Python and PyTorch environment |
| `requirements.txt` | Compatibility-tested dependency versions |
| `code/config.py` | Nine relation names, data paths, and HEDGE hyperparameters |
| `code/data_io.py` | Strict table parsing and symmetric graph construction |
| `code/model_layers.py` | Relation-specific graph convolution, relation attention, and convergence gating |
| `code/hedge_model.py` | HEDGE architecture, PPI diffusion, and per-gene multi-head fusion |
| `code/training.py` | RMSProp optimization, manuscript-defined objectives, metrics, and exports |
| `code/cv_5fold.py` | Five-fold cross-validation and channel perturbation analysis |
| `code/cv_5fold_ratio.py` | Reduced-supervision evaluation |
| `code/grid_search.py` | APPNP depth, R-GCN depth, and convergence-gate search |
| `code/validate_data.py` | Full data and relation validation |
| `data/` | Processed node and structural-edge tables |
| `data_validation_report.json` | Machine-readable validation result |
| `MANIFEST.sha256` | SHA-256 checksums for all deposited files |
| `manuscript/HEDGE_manuscript.pdf` | Matching manuscript PDF |
