# HEDGE data dictionary

## File inventory

| File | Rows | Role |
|---|---:|---|
| `rqtl_snp_nodes.csv` | 314 | Cortical rQTL SNP nodes |
| `hqtl_snp_nodes.csv` | 5,408 | Histone QTL SNP nodes |
| `circqtl_snp_nodes.csv` | 977 | circRNA QTL SNP nodes |
| `gene_nodes.csv` | 8,848 | Unique gene nodes, labels, and 64-dimensional features |
| `snp_snp_ld_edges.csv` | 75,365 | SNP–SNP LD structural edges |
| `snp_gene_association_edges.csv` | 19,784 | SNP–gene structural associations |
| `gene_gene_ppi_edges.csv` | 72,348 | Gene–gene PPI structural edges |

Each structural edge appears once in its CSV file. `data_io.py` stores both orientations for symmetric message passing.

## Node tables

The three SNP tables contain:

- `global_idx`: global SNP-node index; the combined range is `0–6698`.
- `type`: node code; `2` denotes rQTL, `3` denotes hQTL, and `4` denotes circQTL.
- `snp`: variant identifier.
- `flank_101nt`: centered 101-nt sequence window.
- `feature_64`: comma-separated 64-dimensional SNP representation.

`gene_nodes.csv` contains:

- `gene_index`: stable index used by both gene-edge tables.
- `type`: node code `5` for genes.
- `gene_symbol`: gene symbol.
- `source`: feature-construction source.
- `protein_ids`: UniProt entry identifiers separated by semicolons.
- `label`: binary ASD label (`1` positive; `0` putative negative).
- `label_source`: `SPARK/SFARI`, `QTL_map`, or `None`.
- `feature_64`: comma-separated 64-dimensional gene representation.

The positive class comprises curated ASD genes and genes carrying ASD-associated regulatory evidence. Putative negatives are the remaining genes in the PPI universe after excluding both positive groups.

## Structural edge tables

All edge tables use `row` and `col` as endpoint indices.

- `snp_snp_ld_edges.csv`: `val` is the LD edge weight in `[0,1]`; `type=4`.
- `snp_gene_association_edges.csv`: `val` is the SNP–gene association weight in `[0,1]`; `type=1`, `2`, and `3` denote rQTL, hQTL, and circQTL source channels.
- `gene_gene_ppi_edges.csv`: PPI edges are equally weighted during loading; `type=5`.

## Operational relation construction

The three single-channel SNP–gene relations are assigned from the source SNP node set. LD and PPI retain their structural relation names. For a gene receiving two or three QTL channels, the loader adds the corresponding `r-h`, `r-circ`, `h-circ`, or `r-h-circ` operational label to its SNP–gene associations. This produces the nine relation-specific adjacency matrices used by HEDGE without adding new structural associations to the deposited CSV tables.
