import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from config import TrainConfig
from data_io import load_csv_graph
from training import (
    compute_metrics,
    forward_model,
    prepare_tensors,
    save_fold_outputs,
    save_hparams,
    set_seed,
    summarize_metrics,
    train_split,
)


def apply_hyperparameter_overrides(hparams, json_path):
    if json_path is None:
        return
    overrides = json.loads(Path(json_path).read_text())
    unknown = sorted(set(overrides) - set(vars(hparams)))
    if unknown:
        raise ValueError(f"Unknown hyperparameters: {', '.join(unknown)}")
    for name, value in overrides.items():
        setattr(hparams, name, value)


def save_perturbation_outputs(
    fold_id,
    model,
    tensors,
    validation_mask,
    loaded,
    output_dir,
):
    settings = {
        "all_kept": {},
        "all_removed": {"rqtl": True, "hqtl": True, "circqtl": True},
        "only_rQTL": {"hqtl": True, "circqtl": True},
        "only_hQTL": {"rqtl": True, "circqtl": True},
        "only_circQTL": {"rqtl": True, "hqtl": True},
        "remove_rQTL": {"rqtl": True},
        "remove_hQTL": {"hqtl": True},
        "remove_circQTL": {"circqtl": True},
    }
    selected = (
        (tensors["labels"] >= 0) & validation_mask
    ).detach().cpu().numpy()
    global_ids = np.where(selected)[0].astype(int)
    identity = {
        int(unified): (int(original), str(symbol))
        for unified, original, symbol in zip(
            loaded.node_ids["gene_unified_ids"],
            loaded.node_ids["gene_index"],
            loaded.node_ids["gene_symbol"],
        )
    }
    output = {
        "gene_unified_id": global_ids,
        "gene_index": [identity[value][0] for value in global_ids],
        "gene_symbol": [identity[value][1] for value in global_ids],
        "label": tensors["labels"].detach().cpu().numpy()[selected].astype(int),
    }
    model.eval()
    with torch.no_grad():
        for setting_name, ablation in settings.items():
            logits, _ = forward_model(model, tensors, ablate=ablation)
            output[setting_name] = torch.sigmoid(logits).detach().cpu().numpy()[selected]
    pd.DataFrame(output).to_csv(
        Path(output_dir) / f"fold_{fold_id}_validation_channel_perturbations.csv",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Five-fold cross-validation for HEDGE."
    )
    parser.add_argument("--data_dir", default=".", help="HEDGE repository root")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/cross_validation")
    parser.add_argument("--hps_json")
    args = parser.parse_args()

    repository_root = Path(args.data_dir).resolve()
    if not (repository_root / "data").is_dir():
        raise FileNotFoundError(f"Data directory not found: {repository_root / 'data'}")
    os.chdir(repository_root)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    config = TrainConfig()
    apply_hyperparameter_overrides(config.hparams, args.hps_json)
    save_hparams(config.hparams, output_dir)

    loaded = load_csv_graph(config.paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = prepare_tensors(loaded, device)
    gene_ids = torch.where(tensors["gene_mask"])[0].cpu().numpy()
    gene_labels = tensors["labels"][tensors["gene_mask"]].cpu().numpy()
    labeled_selection = gene_labels >= 0
    labeled_gene_ids = gene_ids[labeled_selection]
    labeled_labels = gene_labels[labeled_selection].astype(int)

    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    metric_rows = []
    for fold_id, (train_index, validation_index) in enumerate(
        splitter.split(labeled_gene_ids, labeled_labels), start=1
    ):
        train_ids = labeled_gene_ids[train_index]
        validation_ids = labeled_gene_ids[validation_index]
        model, logits, auxiliary, train_mask, validation_mask = train_split(
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
        save_perturbation_outputs(
            fold_id,
            model,
            tensors,
            validation_mask,
            loaded,
            output_dir,
        )

    summarize_metrics(metric_rows, output_dir)
    print(f"HEDGE cross-validation results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
