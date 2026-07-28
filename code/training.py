import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import RELATION_LIST
from hedge_model import HEDGE


CHANNEL_KEYS = ("rqtl", "hqtl", "circqtl")
CHANNEL_PAIRS = ((0, 1), (0, 2), (1, 2))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_probability, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_probability = np.asarray(y_probability, dtype=float)
    y_prediction = (y_probability >= threshold).astype(int)
    metrics = {}
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_probability))
        if np.unique(y_true).size == 2
        else float("nan")
    )
    metrics["pr_auc"] = (
        float(average_precision_score(y_true, y_probability))
        if y_true.size
        else float("nan")
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_prediction, average="binary", zero_division=0
    )
    metrics.update(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float((y_prediction == y_true).mean()),
    )
    tn, fp, _, _ = confusion_matrix(
        y_true, y_prediction, labels=[0, 1]
    ).ravel()
    metrics["specificity"] = float(tn / (tn + fp)) if tn + fp else float("nan")
    return metrics


def focal_loss(logits, labels, mask, gamma):
    selected = (labels >= 0) & mask
    if selected.sum() == 0:
        return logits.sum() * 0.0
    targets = labels[selected]
    selected_logits = logits[selected]
    positive_count = targets.sum()
    total_count = torch.tensor(
        targets.numel(), dtype=targets.dtype, device=targets.device
    )
    alpha = (total_count - positive_count) / total_count
    probability = torch.sigmoid(selected_logits).clamp(1e-7, 1.0 - 1e-7)
    positive_term = alpha * (1.0 - probability).pow(gamma) * targets * torch.log(probability)
    negative_term = (1.0 - alpha) * probability.pow(gamma) * (1.0 - targets) * torch.log(1.0 - probability)
    return -(positive_term + negative_term).mean()


def bernoulli_symmetric_kl(logits_a, logits_b, mask):
    if mask.sum() == 0:
        return logits_a.sum() * 0.0
    probability_a = torch.sigmoid(logits_a[mask]).clamp(1e-7, 1.0 - 1e-7)
    probability_b = torch.sigmoid(logits_b[mask]).clamp(1e-7, 1.0 - 1e-7)

    def kl(first, second):
        return (
            first * torch.log(first / second)
            + (1.0 - first) * torch.log((1.0 - first) / (1.0 - second))
        ).mean()

    return 0.5 * (kl(probability_a, probability_b) + kl(probability_b, probability_a))


def prior_alignment_loss(auxiliary, train_gene_mask, temperature):
    gene_selection = train_gene_mask[auxiliary["gene_ids"].to(train_gene_mask.device)]
    if gene_selection.sum() < 2:
        return auxiliary["z_prior"].sum() * 0.0
    fused = F.normalize(auxiliary["z_fused"][gene_selection], dim=1)
    prior = F.normalize(auxiliary["z_prior"][gene_selection], dim=1)
    similarity = fused @ prior.T / temperature
    target = torch.arange(similarity.size(0), device=similarity.device)
    return 0.5 * (
        F.cross_entropy(similarity, target)
        + F.cross_entropy(similarity.T, target)
    )


def pair_coactivation_loss(auxiliary, train_gene_mask, margin):
    gene_ids = auxiliary["gene_ids"].to(train_gene_mask.device)
    gene_selection = train_gene_mask[gene_ids]
    if gene_selection.sum() == 0:
        return auxiliary["gates_train"].sum() * 0.0
    gates = auxiliary["gates_train"][gene_selection]
    penalties = [
        torch.relu(margin - (gates[:, first] * gates[:, second]).mean())
        for first, second in CHANNEL_PAIRS
    ]
    return torch.stack(penalties).mean()


def forward_model(model, tensors, ablate=None):
    return model(
        tensors["x"],
        tensors["edge_index"],
        tensors["edge_type"],
        tensors["edge_weight"],
        tensors["gene_mask"],
        tensors["convergence_counts"],
        tensors["gene_extra"],
        ablate=ablate,
        return_aux=True,
    )


def prepare_tensors(loaded, device):
    graph = loaded.data
    gene_mask = loaded.gene_mask.bool().to(device)
    convergence_counts = torch.zeros(graph.num_nodes, 7, device=device)
    convergence_counts[gene_mask] = torch.from_numpy(
        loaded.convergence_counts
    ).to(device)
    return {
        "x": graph.x.to(device).float(),
        "edge_index": graph.edge_index.to(device).long(),
        "edge_type": graph.edge_type.to(device).long(),
        "edge_weight": graph.edge_weight.to(device).float(),
        "gene_mask": gene_mask,
        "labels": loaded.label.to(device).float(),
        "convergence_counts": convergence_counts,
        "gene_extra": torch.zeros(graph.num_nodes, 0, device=device),
    }


def _environment_losses(model, tensors, labels, train_mask, base_logits):
    environment_logits = []
    for channel in CHANNEL_KEYS:
        logits, _ = forward_model(model, tensors, ablate={channel: True})
        environment_logits.append(logits)

    selected = (labels >= 0) & train_mask & tensors["gene_mask"]
    stability = torch.stack(
        [
            (torch.sigmoid(base_logits[selected]) - torch.sigmoid(logits[selected]))
            .square()
            .mean()
            for logits in environment_logits
        ]
    ).mean()

    scale = torch.tensor(1.0, device=labels.device, requires_grad=True)
    irm_penalties = []
    for logits in environment_logits:
        environment_loss = F.binary_cross_entropy_with_logits(
            scale * logits[selected], labels[selected]
        )
        gradient = torch.autograd.grad(
            environment_loss, scale, create_graph=True
        )[0]
        irm_penalties.append(gradient.square())
    return stability, torch.stack(irm_penalties).sum()


def _state_to_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _selected_arrays(logits, labels, mask):
    selected = ((labels >= 0) & mask).detach().cpu().numpy()
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()[selected]
    targets = labels.detach().cpu().numpy()[selected].astype(int)
    global_ids = np.where(selected)[0].astype(int)
    return targets, probabilities, global_ids


def train_split(
    tensors,
    hparams,
    train_ids,
    validation_ids,
    seed,
    patience=10,
):
    set_seed(seed)
    device = tensors["x"].device
    gene_mask = tensors["gene_mask"]
    labels = tensors["labels"]
    train_mask = torch.zeros_like(gene_mask)
    validation_mask = torch.zeros_like(gene_mask)
    train_mask[train_ids] = True
    validation_mask[validation_ids] = True

    model = HEDGE(hparams=hparams, gene_extra_dim=0).to(device)
    if len(train_ids) == 0:
        model.eval()
        with torch.no_grad():
            logits, auxiliary = forward_model(model, tensors)
        return model, logits, auxiliary, train_mask, validation_mask

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay
    )
    best_aupr = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, hparams.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits_a, auxiliary_a = forward_model(model, tensors)
        logits_b, _ = forward_model(model, tensors)

        supervised_mask = (labels >= 0) & train_mask & gene_mask
        loss = focal_loss(
            logits_a, labels, supervised_mask, hparams.focal_gamma
        )
        loss = loss + auxiliary_a["smooth"]
        loss = loss + hparams.consistency_lambda * bernoulli_symmetric_kl(
            logits_a, logits_b, supervised_mask
        )
        loss = loss + hparams.contrast_lambda * prior_alignment_loss(
            auxiliary_a, supervised_mask, hparams.contrast_tau
        )
        loss = loss + hparams.pair_lambda * pair_coactivation_loss(
            auxiliary_a, supervised_mask, hparams.pair_margin
        )
        stability, invariance = _environment_losses(
            model, tensors, labels, train_mask, logits_a
        )
        loss = loss + hparams.stability_lambda * stability
        loss = loss + hparams.irm_lambda * invariance

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits, _ = forward_model(model, tensors)
            targets, probabilities, _ = _selected_arrays(
                validation_logits, labels, validation_mask
            )
            validation_aupr = (
                float(average_precision_score(targets, probabilities))
                if targets.size
                else float("nan")
            )
        score = validation_aupr if np.isfinite(validation_aupr) else -1.0
        if score > best_aupr:
            best_aupr = score
            best_state = _state_to_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, auxiliary = forward_model(model, tensors)
    return model, logits, auxiliary, train_mask, validation_mask


def save_fold_outputs(
    fold_id,
    logits,
    auxiliary,
    labels,
    train_mask,
    validation_mask,
    loaded,
    output_dir,
):
    output_dir = Path(output_dir)
    unified_ids = loaded.node_ids["gene_unified_ids"].astype(int)
    original_ids = loaded.node_ids["gene_index"].astype(int)
    symbols = loaded.node_ids["gene_symbol"]
    identity = {
        int(unified): (int(original), str(symbol))
        for unified, original, symbol in zip(unified_ids, original_ids, symbols)
    }

    for split_name, mask in (("train", train_mask), ("validation", validation_mask)):
        targets, probabilities, global_ids = _selected_arrays(logits, labels, mask)
        table = pd.DataFrame(
            {
                "gene_unified_id": global_ids,
                "gene_index": [identity[value][0] for value in global_ids],
                "gene_symbol": [identity[value][1] for value in global_ids],
                "label": targets,
                "probability": probabilities,
                "prediction": (probabilities >= 0.5).astype(int),
            }
        )
        table.to_csv(
            output_dir / f"fold_{fold_id}_{split_name}_predictions.csv",
            index=False,
        )

    validation_global_ids = np.where(
        ((labels >= 0) & validation_mask).detach().cpu().numpy()
    )[0]
    gene_ids = auxiliary["gene_ids"].numpy().astype(int)
    local_index = {global_id: position for position, global_id in enumerate(gene_ids)}
    gate_array = auxiliary["gates"].numpy()
    relation_attention = (
        auxiliary["rel_attn"][:, :, validation_global_ids]
        .mean(dim=0)
        .transpose(0, 1)
        .numpy()
    )
    interpretation = pd.DataFrame(
        {
            "gene_unified_id": validation_global_ids,
            "gene_index": [identity[value][0] for value in validation_global_ids],
            "gene_symbol": [identity[value][1] for value in validation_global_ids],
            "gate_rQTL": [gate_array[local_index[value], 0] for value in validation_global_ids],
            "gate_hQTL": [gate_array[local_index[value], 1] for value in validation_global_ids],
            "gate_circQTL": [gate_array[local_index[value], 2] for value in validation_global_ids],
        }
    )
    for relation_id, relation_name in enumerate(RELATION_LIST):
        interpretation[f"attention_{relation_name}"] = relation_attention[:, relation_id]
    interpretation.to_csv(
        output_dir / f"fold_{fold_id}_validation_interpretation.csv",
        index=False,
    )
    (output_dir / f"fold_{fold_id}_static_channel_matrix.json").write_text(
        json.dumps(auxiliary["A_static"].numpy().tolist(), indent=2)
    )


def summarize_metrics(rows, output_dir):
    metrics = pd.DataFrame(rows)
    ordered_columns = [
        "fold",
        "split",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
    ]
    metrics = metrics[ordered_columns]
    metrics.to_csv(Path(output_dir) / "metrics_per_fold.csv", index=False)
    numeric_columns = ordered_columns[2:]
    summary = (
        metrics.groupby("split")[numeric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column[0] if column[1] == "" else f"{column[1]}_{column[0]}"
        for column in summary.columns
    ]
    summary.to_csv(Path(output_dir) / "metrics_summary.csv", index=False)
    return metrics, summary


def save_hparams(hparams, output_dir):
    Path(output_dir, "hyperparameters.json").write_text(
        json.dumps(asdict(hparams), indent=2)
    )
