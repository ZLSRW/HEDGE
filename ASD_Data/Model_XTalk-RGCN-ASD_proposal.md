# XTalk-RGCN-ASD：基于修饰串扰的 ASD 风险基因预测方案（R-GCN 框架）

> 以你当前的数据表为输入，面向 **meQTL / hQTL / circQTL × Gene × PPI/LD** 的异质图学习，显式建模**修饰串扰（crosstalk）**对基因风险的影响，并给出可解释结果。

---

## 0. 目标与输入 / 输出

### 目标
- 在综合 meQTL / hQTL / circQTL 与基因层面的网络背景下，**预测基因是否为 ASD 风险基因**；
- 显式度量 **不同修饰及其“串扰组合”** 对预测的贡献（可解释）；
- 在真实 QTL 与网络证据上取得**可复现**的性能提升。

### 最小输入（与你现有文件一一对应）
- **节点（含 64 维向量 + 类型）**
  - `1.meQTL_SNP_nodes_feature.csv`：`global_idx, type(=2), snp, feature_64, ...`
  - `2.hQTL_SNP_nodes_feature.csv`：`type=3`
  - `3.circQTL_SNP_nodes_feature.csv`：`type=4`
  - `4.genes_proteins_nodes_feature.csv`：`gene_index, type(=5), gene_symbol, feature_64, label(1/0), label_source, ...`
- **边**
  - `0.SNP_SNP_edges.csv`：`row(global_idx), col(global_idx), val`（LD 等；也可用 *_typed 版本）
  - `0.SNP_gene_edges.csv`：`row(global_idx), col(gene_index), val`（三类 QTL 的映射边；也可用 *_typed 版本）
  - `0.gene_gene_edges.csv`：`row(gene_index), col(gene_index)`（PPI 或其它基因间连边）

> **注**：SNP 可包含 ALT contig（例如 `chr7_ki270803v1_alt:487981`），已在前序数据准备阶段保留，不影响建图与特征。

### 输出
- `asdrisk_predictions.csv`：每个基因 `gene_index, gene_symbol, risk_prob`
- `asdrisk_explanations.jsonl`：每个基因一行，包含 **按修饰/串扰组合** 的贡献分解、Top-K 证据路径、关键关系权重
- `xtalk_matrix.csv`/`npy`：学习到的 **串扰重要性矩阵**（可热图展示）

---

## 1. 图建模（数据 → 异质图）

### 1.1 节点与关系类型
- **节点类型**
  - `Gene`（type=5）
  - `SNP`（统一一种节点类型，但带属性 `snp_type ∈ {me,h,circ}` 分别对应 2/3/4）
- **关系类型（R-GCN 的 relation id）**
  - `SNP —[meQTL]→ Gene`
  - `SNP —[hQTL]→ Gene`
  - `SNP —[circQTL]→ Gene`
  - `SNP —[ld]→ SNP`（LD/同位点）
  - `Gene —[ppi]→ Gene`(无边权)
- **边权 `val`**：若存在，归一化到 [0,1]；缺失时置 1。

### 1.2 “串扰”先验与增广
- **(A) Gene 端的修饰汇聚计数向量（7 维）** 
  对每个基因统计：`c_me, c_h, c_circ` 与两两/三者共现 `c_me∧h, c_me∧circ, c_h∧circ, c_me∧h∧circ`，归一化后与基因 `feature_64` **拼接**（71 维），再经一层线性变换回 64 维。

- **(B) 串扰“虚拟关系”的软注入（不改原文件，考虑这个）** 
  不新增边文件，仅在模型端把“**共现**”转化为少量 **x-relations**：`x(me,h)`, `x(me,circ)`, `x(h,circ)`, `x(me,h,circ)`，以 **basis 分解**的方式注入 R-GCN（参数共享、开销小）。

---

## 2. 模型：XTalk‑RGCN‑ASD

### 2.1 主干（R-GCN with basis）
- R-GCN 2–3 层，隐藏 64，**basis 分解**：
  - 常规关系：`{meQTL, hQTL, circQTL, ld, ppi}`
  - 串扰基：`{x(me,h), x(me,circ), x(h,circ), x(me,h,circ)}`
  - 每个关系的权重矩阵写作 `W_r = Σ_b a_{r,b} · B_b`

### 2.2 串扰交互门（Crosstalk Gate）
在 **Gene 节点**聚合来自三条 QTL 关系的消息时，引入**关系对关系的门控**：

$$
m_v = \sum_{r \in \{me,h,circ\}} \alpha_r(v)\, h_r(v),\quad
\alpha(v) = \mathrm{softmax}\big([h_{me},h_{h},h_{circ}]^\top G\big) \odot g(v)
$$

- `G ∈ ℝ^{3×3}` 为可学习的**串扰矩阵**（对称或近似对称，**串扰矩阵**，这个号）；  
- `g(v)` 由 7 维汇聚计数生成的**动态门**；  
- 该机制能显式学习“me↔h”等**互作的增强/抑制**，并可导出热图解释。

### 2.3 证据感知的消息权重
- 将边的 `val`（如 LD 的 \(r^2\)、QTL 统计）经过分布校准（分位数归一）作为消息权；
- 对同一基因若存在 **多类 QTL 的汇聚**，对相应的 `SNP→Gene` 消息做**可学习加成**（多证据更强）。

### 2.4 多任务与弱监督（可选）
- **主任务**：`Gene` 的二分类（ASD 风险，BCE/Focal Loss）
- **辅任务**：预测该基因是否被 `{me,h,circ}` 支持（多标签 BCE），促使表示显式承载修饰信息；
- **SNP 对比学习**（选做）：LD 近邻聚合，跨染色体分散，稳定 SNP 向量。

---

## 3. 训练与评估

### 3.1 数据拆分
- **按基因分层的 5 折交叉验证**（避免信息泄漏）；
- 额外 **chr-leave-one** 评估稳健性；
- 负样本为未标注 1 的基因（可用 PU 学习权重）。

### 3.2 训练细节
- AdamW，lr=1e-3；层间残差 + LayerNorm；关系 Dropout 0.1–0.2；
- R-GCN 层数 2–3，basis 数 8–12。

### 3.3 指标
- 主：PR-AUC / ROC-AUC；
- Top-K 命中（在 SFARI 高置信集）；
- **消融**：去掉任意一类 QTL 或所有 x-relations，性能下降幅度量化“串扰的增益”。

---

## 4. 可解释性（面向“串扰”的证据拆解）

1. **关系/串扰贡献分解**  
   - 导出 Gene 节点最终表示对 `{me,h,circ}` 的注意力权重 `α_r(v)` 与 `G` 的热图；
   - 输出到 `asdrisk_explanations.jsonl`：  
     `{"gene":..., "prob":..., "rel_contrib": {"me":..., "h":..., "circ":..., "me×h":..., ...}}`。

2. **证据路径 Top‑K**  
   - 用“边权×注意力×消息范数”近似每条路径（SNP→Gene）的贡献，选择前 K 个作为**解释路径**；
   - 路径包含：SNP id（含是否 ALT）、QTL 类型、LD/QTL 强度等。

3. **反事实分析（结构化 Dropout）**  
   - 依次移除某类关系或某个**串扰基**，重前向得 `Δp`，作为必要性证据；
   - 形成“若无 me↔h 串扰，该基因风险概率将下降 X%”的**可检验结论**。

---

## 5. **创新点**（≥3 项）

1. **串扰基的低秩注入（x‑relations with basis）**  
   用极少量共享 basis 在 R‑GCN 中表示“修饰共现”的互作，不改边文件、几乎不增参，显式建模**crosstalk**。

2. **关系对关系的“串扰门”**  
   以可学习矩阵 `G` + 动态门 `g(v)` 在 Gene 端融合 me/h/circ 消息，支持**增强/抑制**型交互，并产出**可解释热图**。

3. **证据感知的消息权重 + 多证据汇聚加成**  
   把 `val` 融入消息传递，同时对多路 QTL 汇聚做加成，使强证据主导、弱证据降权，提升鲁棒性与可解释性。

4. （可选）**多任务将“修饰归因”写入表示**  
   让 Gene 表示同时预测“受哪类修饰支持”，增强外推能力并提升解释的可信度。

5. （可选）**反事实结构 Dropout**  
   以删关系/删串扰基的 `Δp` 作为**必要性**证据，提供更贴近因果的问题回答。

---

## 6. 实施要点（PyTorch Geometric / DGL）

- **装载**：
  - 读取 4 个节点表，`feature_64` 解析为 `FloatTensor[64]`；
  - 读取 3 个边表，生成关系分图与边权；
  - 统计 7 维串扰计数向量并拼接到 Gene 特征（或独立 MLP 后再并入）。

- **模型接口**：
  - `forward(graph) -> gene_logits`（只在 Gene 上计算 BCE）；
  - 导出 `G`、`α_r(v)`、Top‑K 路径贡献用于解释输出。

- **超参建议**：
  - 层数 2–3，隐藏 64，basis=8–12，dropout=0.1–0.2；
  - 早停看 PR‑AUC；Focal Loss 处理正负不均。

---

## 7. 结果产物（标准化导出）

- `asdrisk_predictions.csv`：`gene_index,gene_symbol,risk_prob`
- `asdrisk_explanations.jsonl`：关系/串扰贡献、Top‑K 路径、反事实 `Δp`
- `xtalk_matrix.csv`：`G` 或其对称化版本（3×3），可直接画热图
- 训练日志与消融报告：记录移除各关系/串扰基后的性能变化

---

### 备注（与现有数据的契合）
- **ALT 位点**已在节点阶段保留，构图与特征均适配；
- **缺特征**的基因/SNP 已通过邻居均值/哈希兜底补齐，不阻塞训练；
- **标签稀疏**可用 focal loss 或 PU‑weighting；  
- 以上模块均**不要求**额外改动你已有的 CSV，只在模型端做“串扰”注入与门控。

