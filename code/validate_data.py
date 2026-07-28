import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import RELATION_LIST, TrainConfig
from data_io import load_csv_graph


EXPECTED_COLUMNS = {
    "rqtl_snp_nodes.csv": ["global_idx", "type", "snp", "flank_101nt", "feature_64"],
    "hqtl_snp_nodes.csv": ["global_idx", "type", "snp", "flank_101nt", "feature_64"],
    "circqtl_snp_nodes.csv": ["global_idx", "type", "snp", "flank_101nt", "feature_64"],
    "gene_nodes.csv": [
        "gene_index",
        "type",
        "gene_symbol",
        "source",
        "protein_ids",
        "label",
        "label_source",
        "feature_64",
    ],
    "snp_snp_ld_edges.csv": ["row", "col", "val", "type"],
    "snp_gene_association_edges.csv": ["row", "col", "val", "type"],
    "gene_gene_ppi_edges.csv": ["row", "col", "type"],
}


def validate_feature_column(table, table_name):
    lengths = table["feature_64"].astype(str).str.split(",").str.len()
    if not (lengths == 64).all():
        raise ValueError(f"{table_name} contains a feature vector that is not 64-dimensional")
    values = np.asarray(
        [[float(value) for value in row.split(",")] for row in table["feature_64"]],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{table_name} contains non-finite feature values")


def main():
    parser = argparse.ArgumentParser(description="Validate the HEDGE data package.")
    parser.add_argument("--data_dir", default=".", help="HEDGE repository root")
    parser.add_argument("--report", default="data_validation_report.json")
    args = parser.parse_args()

    repository_root = Path(args.data_dir).resolve()
    os.chdir(repository_root)
    data_dir = repository_root / "data"
    tables = {}
    for filename, expected_columns in EXPECTED_COLUMNS.items():
        table = pd.read_csv(data_dir / filename, dtype=str, keep_default_na=False)
        if list(table.columns) != expected_columns:
            raise ValueError(
                f"{filename} columns are {list(table.columns)}; expected {expected_columns}"
            )
        if table.duplicated().any():
            raise ValueError(f"{filename} contains duplicate rows")
        tables[filename] = table
        if "feature_64" in table.columns:
            validate_feature_column(table, filename)

    gene_table = tables["gene_nodes.csv"]
    if gene_table["gene_index"].duplicated().any():
        raise ValueError("gene_nodes.csv contains duplicate gene_index values")
    if gene_table["gene_symbol"].duplicated().any():
        raise ValueError("gene_nodes.csv contains duplicate gene_symbol values")
    if set(gene_table["label"]) - {"0", "1"}:
        raise ValueError("Gene labels must be 0 or 1")

    snp_tables = [
        tables["rqtl_snp_nodes.csv"],
        tables["hqtl_snp_nodes.csv"],
        tables["circqtl_snp_nodes.csv"],
    ]
    all_snp_ids = [int(value) for table in snp_tables for value in table["global_idx"]]
    if len(all_snp_ids) != len(set(all_snp_ids)):
        raise ValueError("SNP global indices are not unique")
    if sorted(all_snp_ids) != list(range(max(all_snp_ids) + 1)):
        raise ValueError("SNP global indices are not contiguous")

    snp_id_set = set(all_snp_ids)
    gene_id_set = set(gene_table["gene_index"].astype(int))
    ld_table = tables["snp_snp_ld_edges.csv"]
    sg_table = tables["snp_gene_association_edges.csv"]
    ppi_table = tables["gene_gene_ppi_edges.csv"]
    for filename, table in (("snp_snp_ld_edges.csv", ld_table), ("snp_gene_association_edges.csv", sg_table)):
        weights = table["val"].astype(float).to_numpy()
        if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
            raise ValueError(f"{filename} weights must lie in [0, 1]")
    if not set(ld_table["row"].astype(int)).union(ld_table["col"].astype(int)) <= snp_id_set:
        raise ValueError("The LD table references an unknown SNP")
    if not set(sg_table["row"].astype(int)) <= snp_id_set:
        raise ValueError("The SNP-gene table references an unknown SNP")
    if not set(sg_table["col"].astype(int)) <= gene_id_set:
        raise ValueError("The SNP-gene table references an unknown gene")
    if not set(ppi_table["row"].astype(int)).union(ppi_table["col"].astype(int)) <= gene_id_set:
        raise ValueError("The PPI table references an unknown gene")
    for filename in ("snp_snp_ld_edges.csv", "gene_gene_ppi_edges.csv"):
        table = tables[filename]
        if (table["row"] == table["col"]).any():
            raise ValueError(f"{filename} contains self-loops")

    loaded = load_csv_graph(TrainConfig().paths)
    relation_counts = Counter(
        RELATION_LIST[int(relation_id)]
        for relation_id in loaded.data.edge_type.tolist()
    )
    missing_relations = [
        relation for relation in RELATION_LIST if relation_counts[relation] == 0
    ]
    if missing_relations:
        raise ValueError(
            f"Operational relations without edges: {', '.join(missing_relations)}"
        )

    report = {
        "status": "passed",
        "file_rows": {filename: int(len(table)) for filename, table in tables.items()},
        "node_counts": {
            "rQTL_SNP": int(len(tables["rqtl_snp_nodes.csv"])),
            "hQTL_SNP": int(len(tables["hqtl_snp_nodes.csv"])),
            "circQTL_SNP": int(len(tables["circqtl_snp_nodes.csv"])),
            "gene": int(len(gene_table)),
        },
        "gene_label_counts": {
            label: int(count)
            for label, count in gene_table["label"].value_counts().sort_index().items()
        },
        "stored_directed_operational_edge_counts": dict(relation_counts),
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
