import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from config import TrainConfig
from cv_5fold import apply_hyperparameter_overrides
from data_io import load_csv_graph
from training import (
    compute_metrics,
    prepare_tensors,
    save_fold_outputs,
    save_hparams,
    set_seed,
    summarize_metrics,
    train_split,
)


SUPERVISION_SETTINGS = (
    ("zero_supervision", 0.0),
    ("train_10_percent", 0.1),
    ("train_20_percent", 0.2),
    ("train_30_percent", 0.3),
    ("train_40_percent", 0.4),
    ("train_50_percent", 0.5),
)


def main():
    parser = argparse.ArgumentParser(
        description="Reduced-supervision evaluation for HEDGE."
    )
    parser.add_argument("--data_dir", default=".", help="HEDGE repository root")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/reduced_supervision")
    parser.add_argument("--hps_json")
    args = parser.parse_args()

    repository_root = Path(args.data_dir).resolve()
    if not (repository_root / "data").is_dir():
        raise FileNotFoundError(f"Data directory not found: {repository_root / 'data'}")
    os.chdir(repository_root)
    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    config = TrainConfig()
    apply_hyperparameter_overrides(config.hparams, args.hps_json)
    loaded = load_csv_graph(config.paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = prepare_tensors(loaded, device)

    gene_ids = torch.where(tensors["gene_mask"])[0].cpu().numpy()
    gene_labels = tensors["labels"][tensors["gene_mask"]].cpu().numpy()
    labeled_selection = gene_labels >= 0
    labeled_gene_ids = gene_ids[labeled_selection]
    labeled_labels = gene_labels[labeled_selection].astype(int)
    all_indices = np.arange(len(labeled_gene_ids))

    for setting_name, train_fraction in SUPERVISION_SETTINGS:
        output_dir = output_root / setting_name
        output_dir.mkdir(parents=True, exist_ok=True)
        save_hparams(config.hparams, output_dir)
        metric_rows = []

        if train_fraction == 0.0:
            splits = [
                (np.array([], dtype=int), all_indices.copy())
                for _ in range(args.folds)
            ]
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=args.folds,
                train_size=train_fraction,
                random_state=args.seed,
            )
            splits = list(splitter.split(all_indices, labeled_labels))

        for fold_id, (train_index, validation_index) in enumerate(splits, start=1):
            train_ids = labeled_gene_ids[train_index]
            validation_ids = labeled_gene_ids[validation_index]
            _, logits, auxiliary, train_mask, validation_mask = train_split(
                tensors,
                config.hparams,
                train_ids,
                validation_ids,
                seed=args.seed + fold_id,
            )
            for split_name, mask in (
                ("train", train_mask),
                ("validation", validation_mask),
            ):
                selected = ((tensors["labels"] >= 0) & mask).cpu().numpy()
                if selected.sum() == 0:
                    continue
                probabilities = torch.sigmoid(logits).cpu().numpy()[selected]
                targets = tensors["labels"].cpu().numpy()[selected].astype(int)
                metric_rows.append(
                    {
                        "fold": fold_id,
                        "split": split_name,
                        **compute_metrics(targets, probabilities),
                    }
                )
            save_fold_outputs(
                fold_id,
                logits,
                auxiliary,
                tensors["labels"],
                train_mask,
                validation_mask,
                loaded,
                output_dir,
            )
        summarize_metrics(metric_rows, output_dir)

    print(f"HEDGE reduced-supervision results: {output_root.resolve()}")


if __name__ == "__main__":
    main()
