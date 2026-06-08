# Script Map

This document records the actual code chain included in the submission package.

## Entry chain

```text
main.py
├── train  -> code/cv_5fold.py
├── ratio  -> code/cv_5fold_ratio.py
└── search -> code/grid_search.py
```

## Dependency tree

```text
code/cv_5fold.py
├── code/config.py
├── code/data_io.py
└── code/model_xtalk_rgcn.py
    ├── code/config1.py
    └── code/model_layers.py

code/cv_5fold_ratio.py
├── code/config.py
├── code/data_io.py
└── code/model_xtalk_rgcn.py
    ├── code/config1.py
    └── code/model_layers.py

code/grid_search.py
├── code/config.py
├── code/data_io.py
├── code/model_xtalk_rgcn.py
└── code/cv_5fold.py
    ├── reuses training losses and metrics helpers
    └── shares the same lower-level dependencies listed above
```

## Why these files were included

- `code/cv_5fold.py`: standard manuscript training and evaluation entry script.
- `code/cv_5fold_ratio.py`: robustness analysis across train:validation ratios.
- `code/grid_search.py`: hyperparameter search used to organize parameter sweeps.
- `code/data_io.py`: loads the manuscript CSV graph data and builds the graph object.
- `code/model_xtalk_rgcn.py`: defines the XTalk-RGCN model used by the training scripts.
- `code/model_layers.py`: shared R-GCN, channel attention, and crosstalk gate layers.
- `code/config.py`: top-level training configuration and hyperparameters.
- `code/config1.py`: relation definitions and data path schema used by lower-level modules.

## Files intentionally not included

- Original repository root `main.py`: only a PyCharm demo, not part of the manuscript pipeline.
- Large historical result folders under `models/`: outputs, not source dependencies.
- Plotting and downstream analysis scripts outside the direct import chain: relevant for extended analysis, but not required to run the core model pipeline traced from the actual entry scripts.
