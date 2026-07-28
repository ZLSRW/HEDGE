import argparse
import itertools
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from config import TrainConfig
from data_io import load_csv_graph
from training import compute_metrics, prepare_tensors, set_seed, train_split


SEARCH_SPACE = {
    "appnp_K": (1, 2, 4, 6, 8),
    "num_layers": (2, 3, 4, 6),
    "enable_convergence_gate": (False, True),
}


def parameter_combinations(strategy, trials, seed):
    names = tuple(SEARCH_SPACE)
    combinations = [
        dict(zip(names, values))
        for values in itertools.product(*(SEARCH_SPACE[name] for name in names))
    ]
    if strategy == "grid":
        return combinations
    generator = random.Random(seed)
    generator.shuffle(combinations)
    return combinations[: min(trials, len(combinations))]


def evaluate_configuration(loaded, hparams, folds, seed, device):
    tensors = prepare_tensors(loaded, device)
    gene_ids = torch.where(tensors["gene_mask"])[0].cpu().numpy()
    gene_labels = tensors["labels"][tensors["gene_mask"]].cpu().numpy()
    labeled_selection = gene_labels >= 0
    labeled_gene_ids = gene_ids[labeled_selection]
    labeled_labels = gene_labels[labeled_selection].astype(int)
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    rows = []
    for fold_id, (train_index, validation_index) in enumerate(
        splitter.split(labeled_gene_ids, labeled_labels), start=1
    ):
        _, logits, _, _, validation_mask = train_split(
            tensors,
            hparams,
            labeled_gene_ids[train_index],
            labeled_gene_ids[validation_index],
            seed=seed + fold_id,
        )
        selected = ((tensors["labels"] >= 0) & validation_mask).cpu().numpy()
        probabilities = torch.sigmoid(logits).cpu().numpy()[selected]
        targets = tensors["labels"].cpu().numpy()[selected].astype(int)
        rows.append({"fold": fold_id, **compute_metrics(targets, probabilities)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for HEDGE."
    )
    parser.add_argument("--data_dir", default=".", help="HEDGE repository root")
    parser.add_argument("--out_dir", default="results/hyperparameter_search")
    parser.add_argument("--strategy", choices=("grid", "random"), default="grid")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repository_root = Path(args.data_dir).resolve()
    if not (repository_root / "data").is_dir():
        raise FileNotFoundError(f"Data directory not found: {repository_root / 'data'}")
    os.chdir(repository_root)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    base_config = TrainConfig()
    loaded = load_csv_graph(base_config.paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combinations = parameter_combinations(args.strategy, args.trials, args.seed)
    (output_dir / "search_space.json").write_text(
        json.dumps(
            {
                name: list(values)
                for name, values in SEARCH_SPACE.items()
            },
            indent=2,
        )
    )

    summary_rows = []
    all_fold_rows = []
    for run_id, parameters in enumerate(combinations, start=1):
        hparams = deepcopy(base_config.hparams)
        for name, value in parameters.items():
            setattr(hparams, name, value)
        fold_metrics = evaluate_configuration(
            loaded, hparams, args.folds, args.seed, device
        )
        run_name = (
            f"K{parameters['appnp_K']}_L{parameters['num_layers']}_"
            f"G{int(parameters['enable_convergence_gate'])}"
        )
        run_dir = output_dir / run_name
        run_dir.mkdir(exist_ok=True)
        fold_metrics.to_csv(run_dir / "validation_metrics_per_fold.csv", index=False)
        (run_dir / "hyperparameters.json").write_text(
            json.dumps(vars(hparams), indent=2)
        )
        fold_metrics.insert(0, "run", run_name)
        all_fold_rows.append(fold_metrics)
        summary = {
            "run": run_name,
            **parameters,
        }
        for metric in (
            "roc_auc",
            "pr_auc",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "specificity",
        ):
            summary[f"mean_{metric}"] = fold_metrics[metric].mean()
            summary[f"std_{metric}"] = fold_metrics[metric].std()
        summary_rows.append(summary)
        print(f"Completed search run {run_id}/{len(combinations)}: {run_name}")

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "search_summary.csv", index=False
    )
    pd.concat(all_fold_rows, ignore_index=True).to_csv(
        output_dir / "all_validation_metrics.csv", index=False
    )
    print(f"HEDGE hyperparameter search results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
