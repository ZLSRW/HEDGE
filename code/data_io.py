from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from config import (
    Paths,
    REL2ID,
    REL_CIRCQTL,
    REL_HQTL,
    REL_H_CIRC,
    REL_LD,
    REL_PPI,
    REL_RQTL,
    REL_R_CIRC,
    REL_R_H,
    REL_R_H_CIRC,
)


def _parse_feature64(series: pd.Series, table_name: str) -> np.ndarray:
    vectors = []
    for row_number, value in enumerate(series.astype(str), start=2):
        tokens = value.split(",")
        if len(tokens) != 64:
            raise ValueError(
                f"{table_name}:{row_number} contains {len(tokens)} feature values; expected 64"
            )
        try:
            vector = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(
                f"{table_name}:{row_number} contains a non-numeric feature value"
            ) from exc
        if not np.isfinite(vector).all():
            raise ValueError(
                f"{table_name}:{row_number} contains a non-finite feature value"
            )
        vectors.append(vector)
    return np.asarray(vectors, dtype=np.float32)


@dataclass
class LoadedGraph:
    data: Data
    gene_mask: torch.Tensor
    snp_mask: torch.Tensor
    label: torch.Tensor
    node_ids: Dict[str, np.ndarray]
    convergence_counts: np.ndarray


def load_csv_graph(paths: Paths) -> LoadedGraph:
    rqtl = pd.read_csv(paths.snp_nodes_r, dtype=str, keep_default_na=False)
    hqtl = pd.read_csv(paths.snp_nodes_h, dtype=str, keep_default_na=False)
    circqtl = pd.read_csv(paths.snp_nodes_circ, dtype=str, keep_default_na=False)
    genes = pd.read_csv(paths.gene_nodes, dtype=str, keep_default_na=False)

    x_r = _parse_feature64(rqtl["feature_64"], paths.snp_nodes_r)
    x_h = _parse_feature64(hqtl["feature_64"], paths.snp_nodes_h)
    x_circ = _parse_feature64(circqtl["feature_64"], paths.snp_nodes_circ)
    x_gene = _parse_feature64(genes["feature_64"], paths.gene_nodes)

    idx_r = rqtl["global_idx"].astype(int).to_numpy()
    idx_h = hqtl["global_idx"].astype(int).to_numpy()
    idx_circ = circqtl["global_idx"].astype(int).to_numpy()
    gene_index = genes["gene_index"].astype(int).to_numpy()

    all_snp_indices = np.concatenate([idx_r, idx_h, idx_circ])
    if len(np.unique(all_snp_indices)) != len(all_snp_indices):
        raise ValueError("SNP global_idx values must be unique across the three QTL channels")
    if len(np.unique(gene_index)) != len(gene_index):
        raise ValueError("gene_index values must be unique")

    n_snp = int(all_snp_indices.max()) + 1
    expected_snp_indices = np.arange(n_snp)
    if not np.array_equal(np.sort(all_snp_indices), expected_snp_indices):
        raise ValueError("SNP global_idx values must form a contiguous range beginning at zero")

    sorted_gene_indices = sorted(gene_index.tolist())
    gene_to_unified = {
        original_id: n_snp + local_id
        for local_id, original_id in enumerate(sorted_gene_indices)
    }
    unified_gene_ids = np.asarray(
        [gene_to_unified[original_id] for original_id in gene_index], dtype=np.int64
    )
    n_total = n_snp + len(gene_to_unified)

    node_features = np.zeros((n_total, 64), dtype=np.float32)
    node_features[idx_r] = x_r
    node_features[idx_h] = x_h
    node_features[idx_circ] = x_circ
    node_features[unified_gene_ids] = x_gene

    snp_mask = np.zeros(n_total, dtype=bool)
    snp_mask[:n_snp] = True
    gene_mask = np.zeros(n_total, dtype=bool)
    gene_mask[unified_gene_ids] = True

    labels = np.full(n_total, -1.0, dtype=np.float32)
    labels[unified_gene_ids] = genes["label"].astype(float).to_numpy()

    sg_table = pd.read_csv(paths.edge_snp_gene, dtype=str, keep_default_na=False)
    ld_table = pd.read_csv(paths.edge_snp_snp, dtype=str, keep_default_na=False)
    ppi_table = pd.read_csv(paths.edge_gene_gene, dtype=str, keep_default_na=False)

    def edge_weights(table: pd.DataFrame) -> np.ndarray:
        if "val" not in table.columns:
            return np.ones(len(table), dtype=float)
        values = pd.to_numeric(table["val"], errors="raise").astype(float).to_numpy()
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Edge weights must be finite and non-negative")
        return values

    all_snp_set = set(all_snp_indices.tolist())
    r_set = set(idx_r.tolist())
    h_set = set(idx_h.tolist())
    circ_set = set(idx_circ.tolist())

    structural_sg_records = []
    channels_by_gene = {}
    skipped_sg = 0
    sg_weights = edge_weights(sg_table)
    for row_id, (snp_id_raw, gene_id_raw) in enumerate(
        zip(sg_table["row"], sg_table["col"])
    ):
        snp_id = int(snp_id_raw)
        gene_id = int(gene_id_raw)
        unified_gene_id = gene_to_unified.get(gene_id)
        if unified_gene_id is None:
            skipped_sg += 1
            continue
        if snp_id in r_set:
            relation = REL_RQTL
        elif snp_id in h_set:
            relation = REL_HQTL
        elif snp_id in circ_set:
            relation = REL_CIRCQTL
        else:
            raise ValueError(f"Unknown SNP index in SNP-gene table: {snp_id}")
        structural_sg_records.append(
            (snp_id, unified_gene_id, relation, float(sg_weights[row_id]))
        )
        channels_by_gene.setdefault(unified_gene_id, set()).add(relation)

    combination_to_relation = {
        frozenset((REL_RQTL, REL_HQTL)): REL_R_H,
        frozenset((REL_RQTL, REL_CIRCQTL)): REL_R_CIRC,
        frozenset((REL_HQTL, REL_CIRCQTL)): REL_H_CIRC,
        frozenset((REL_RQTL, REL_HQTL, REL_CIRCQTL)): REL_R_H_CIRC,
    }

    sg_edges = []
    for snp_id, unified_gene_id, relation, weight in structural_sg_records:
        relation_id = REL2ID[relation]
        sg_edges.extend(
            [
                (snp_id, unified_gene_id, relation_id, weight),
                (unified_gene_id, snp_id, relation_id, weight),
            ]
        )
        cross_relation = combination_to_relation.get(
            frozenset(channels_by_gene[unified_gene_id])
        )
        if cross_relation is not None:
            cross_relation_id = REL2ID[cross_relation]
            sg_edges.extend(
                [
                    (snp_id, unified_gene_id, cross_relation_id, weight),
                    (unified_gene_id, snp_id, cross_relation_id, weight),
                ]
            )

    ld_edges = []
    ld_weights = edge_weights(ld_table)
    for row_id, (source_raw, target_raw) in enumerate(
        zip(ld_table["row"], ld_table["col"])
    ):
        source = int(source_raw)
        target = int(target_raw)
        if source not in all_snp_set or target not in all_snp_set:
            raise ValueError("LD edge references an unknown SNP index")
        relation_id = REL2ID[REL_LD]
        weight = float(ld_weights[row_id])
        ld_edges.extend(
            [(source, target, relation_id, weight), (target, source, relation_id, weight)]
        )

    ppi_edges = []
    skipped_ppi = 0
    ppi_weights = edge_weights(ppi_table)
    for row_id, (source_raw, target_raw) in enumerate(
        zip(ppi_table["row"], ppi_table["col"])
    ):
        source = gene_to_unified.get(int(source_raw))
        target = gene_to_unified.get(int(target_raw))
        if source is None or target is None:
            skipped_ppi += 1
            continue
        relation_id = REL2ID[REL_PPI]
        weight = float(ppi_weights[row_id])
        ppi_edges.extend(
            [(source, target, relation_id, weight), (target, source, relation_id, weight)]
        )

    all_edges = sg_edges + ld_edges + ppi_edges
    if not all_edges:
        raise RuntimeError("The graph contains no edges")

    source, target, relation, weight = zip(*all_edges)
    graph = Data(
        x=torch.from_numpy(node_features),
        edge_index=torch.tensor([source, target], dtype=torch.long),
        edge_type=torch.tensor(relation, dtype=torch.long),
        edge_weight=torch.tensor(weight, dtype=torch.float32),
        num_nodes=n_total,
    )

    counts = np.zeros((len(gene_to_unified), 7), dtype=np.float32)
    for _, unified_gene_id, relation, _ in structural_sg_records:
        local_gene_id = unified_gene_id - n_snp
        if relation == REL_RQTL:
            counts[local_gene_id, 0] += 1
        elif relation == REL_HQTL:
            counts[local_gene_id, 1] += 1
        elif relation == REL_CIRCQTL:
            counts[local_gene_id, 2] += 1
    present = (counts[:, :3] > 0).astype(np.float32)
    counts[:, 3] = present[:, 0] * present[:, 1]
    counts[:, 4] = present[:, 0] * present[:, 2]
    counts[:, 5] = present[:, 1] * present[:, 2]
    counts[:, 6] = present[:, 0] * present[:, 1] * present[:, 2]

    print(
        f"[data] SNP-gene structural associations={len(structural_sg_records)}, "
        f"stored directed operational edges={len(sg_edges)}, skipped={skipped_sg}"
    )
    print(
        f"[data] PPI structural edges={len(ppi_table) - skipped_ppi}, "
        f"stored directed edges={len(ppi_edges)}, skipped={skipped_ppi}"
    )

    return LoadedGraph(
        data=graph,
        gene_mask=torch.from_numpy(gene_mask),
        snp_mask=torch.from_numpy(snp_mask),
        label=torch.from_numpy(labels),
        node_ids={
            "gene_unified_ids": unified_gene_ids,
            "gene_index": gene_index,
            "gene_symbol": genes["gene_symbol"].to_numpy(),
            "snp_n": np.asarray(n_snp),
        },
        convergence_counts=counts,
    )
