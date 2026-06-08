# Data README

本说明覆盖当前目录下 7 个表的含义、列名解释与计数。

## 文件清单与行数

- **0.gene_gene_edges.csv** — 行数：72348, 列：`row, col, type`，`type` 分布：{'5': 72348}
- **0.SNP_gene_edges.csv** — 行数：19784, 列：`row, col, val`
- **0.SNP_SNP_edges.csv** — 行数：75365, 列：`row, col, val`
- **1.meQTL_SNP_nodes_feature.csv** — 行数：314, 列：`global_idx, type, snp, flank_101nt, feature_64`，`type` 分布：{'2': 314}
- **2.hQTL_SNP_nodes_feature.csv** — 行数：5408, 列：`global_idx, type, snp, flank_101nt, feature_64`，`type` 分布：{'3': 5408}
- **3.circQTL_SNP_nodes_feature.csv** — 行数：977, 列：`global_idx, type, snp, flank_101nt, feature_64`，`type` 分布：{'4': 977}
- **4.genes_proteins_nodes_feature.csv** — 行数：8873, 列：`gene_index, type, gene_symbol, source, protein_ids, label, label_source, feature_64`，`type` 分布：{'5': 8873}

## 节点索引范围（global_idx）

- meQTL SNP：`0..313`（共 314 个；对应 `type=2`）
- hQTL SNP：`314..5721`（共 5408 个；对应 `type=3`）
- circQTL SNP：`5722..6698`（共 977 个；对应 `type=4`）

据此，`0.SNP_gene_edges.csv` 中 `row` 的分段规则为：`row≤313`→type=1；`313<row≤5721`→type=2；否则 type=3。

---

## 表结构与字段释义

### 1) 0.gene_gene_edges.csv
- **row**：基因节点索引（`gene_index`），与 `4.genes_proteins_nodes_feature.csv` 的 `gene_index` 一致。
- **col**：基因节点索引（同上）。
- **type**：边类型，固定为 `5`（基因–基因边）。
> 说明：该表不含权重列，默认等权（或在构图时赋值 1）。

### 2) 0.SNP_gene_edges.csv
- **row**：SNP 节点全局索引（`global_idx`）。
- **col**：基因节点索引（`gene_index`）。
- **val**：边权重（例如 1 表示映射命中，或根据统计量加权）。
- **type**：边类型，按 `row` 所属区间划分：
  - `row ≤ 313` → `1`（meQTL SNP → gene）
  - `313 < row ≤ 5721` → `2`（hQTL SNP → gene）
  - 其余 → `3`（circQTL SNP → gene）

### 3) 0.SNP_SNP_edges.csv
- **row**：SNP 节点全局索引（`global_idx`）。
- **col**：SNP 节点全局索引（`global_idx`）。
- **val**：边权重（例如 LD 的 $r^2$ 或“同位点=1.0”）。
- **type**：边类型，固定为 `4`（SNP–SNP 边）。

### 4) 1.meQTL_SNP_nodes_feature.csv / 2.hQTL_SNP_nodes_feature.csv / 3.circQTL_SNP_nodes_feature.csv
- **global_idx**：SNP 节点的全局索引，用于与边表对齐。
- **type**：SNP 节点类型：`2`=meQTL、`3`=hQTL、`4`=circQTL。
- **snp**：SNP 文本 ID（形如 `chr11:3015094` 或带 ALT 的 `chr7_ki270803v1_alt:487981`）。
- **flank_101nt**：SNP 周围 ±50bp 的 101 nt 序列（若参考不含 contig 则可能填充 `N`）。
- **feature_64**：SNP 的 64 维表征（逗号分隔的 64 个浮点）。

### 5) 4.genes_proteins_nodes_feature.csv
- **gene_index**：基因节点索引（用于与边表 `col` 对齐）。
- **type**：节点类型，固定为 `5`（基因）。
- **gene_symbol**：基因符号（大写）。
- **source**：向量来源（如 `ELM(ESM)` / `ELM(kmer)` / `centroid` / `genehash`）。
- **protein_ids**：对应 UniProt Entry（多值以分号分隔）。
- **label**：ASD风险基因标签标签（1表示是/0表示不是）。
- **label_source**：标签来源（如 `SPARK/SFARI` 或 `QTL_map`）。
- **feature_64**：基因的 64 维表征（逗号分隔 64 个浮点）。

---

## 编码约定（type）
- **节点**：`2`=meQTL SNP，`3`=hQTL SNP，`4`=circQTL SNP，`5`=gene。
- **边**：
  - `0.SNP_SNP_edges.csv`：`type=4`（SNP–SNP）。
  - `0.SNP_gene_edges.csv`：`type` 随 `row` 所属 QTL 分区而定（见上）。
  - `0.gene_gene_edges.csv`：`type=5`（gene–gene）。

## 方案设计