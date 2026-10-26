from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

def _auto_detect_edge_cols(df_edges: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df_edges.columns)
    cand = [c for c in cols if 'gene' in c.lower() or 'index' in c.lower()]
    if len(cand) >= 2:
        return (cand[0], cand[1])
    return (cols[0], cols[1])

def _parse_feature_64(col: pd.Series, expected_dim: int=64) -> np.ndarray:
    feats = []
    for i, v in enumerate(col):
        if isinstance(v, (list, np.ndarray)):
            arr = np.asarray(v, dtype=float)
        else:
            txt = str(v).strip()
            if txt.startswith('[') and txt.endswith(']'):
                txt = txt[1:-1]
            txt = txt.replace('\t', ' ')
            txt = txt.replace(';', ' ')
            txt = txt.replace(',', ' ')
            parts = [p for p in txt.split() if p != '']
            try:
                arr = np.array([float(p) for p in parts], dtype=float)
            except Exception as e:
                raise ValueError(f'第 {i} 行 feature_64 无法解析为浮点数: {v}') from e
        if arr.size != expected_dim:
            raise ValueError(f'feature_64 期望长度 = {expected_dim}, 但在第 {i} 行得到 {arr.size} (原始值: {v})')
        feats.append(arr)
    return np.stack(feats, axis=0)

def _load_gene_part(data_root: Path, feat_dim: int=64):
    gene_feat_path = data_root / 'gene_nodes.csv'
    if not gene_feat_path.exists():
        raise FileNotFoundError(f'找不到基因特征文件: {gene_feat_path}')
    df_gene = pd.read_csv(gene_feat_path)
    cols = df_gene.columns.tolist()
    if 'gene_index' not in cols:
        raise ValueError("gene_nodes.csv 中必须包含 'gene_index' 列")
    if 'feature_64' not in cols:
        raise ValueError("gene_nodes.csv 中必须包含 'feature_64' 列")
    if 'label' not in cols:
        print("[WARN] 未在基因特征文件中找到 'label' 列, 默认全部设为 -1 (未标注)")
        df_gene['label'] = -1
    gene_index = df_gene['gene_index'].to_numpy()
    if 'gene_symbol' in cols:
        gene_symbol = df_gene['gene_symbol'].to_numpy()
    else:
        gene_symbol = np.array([f'G{i}' for i in gene_index])
    feat_mat = _parse_feature_64(df_gene['feature_64'], expected_dim=feat_dim)
    x_gene = torch.from_numpy(feat_mat).float()
    y_gene = df_gene['label'].to_numpy().astype(float)
    return (gene_index, gene_symbol, x_gene, y_gene)

def _load_snp_features(data_root: Path, feat_dim: int=64, filenames: List[str]=None) -> Dict[str, np.ndarray]:
    if filenames is None:
        filenames = ['rqtl_snp_nodes.csv', 'hqtl_snp_nodes.csv', 'circqtl_snp_nodes.csv']
    snp_feat_dict: Dict[str, np.ndarray] = {}
    for fname in filenames:
        path = data_root / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        cols = df.columns.tolist()
        id_col = None
        for c in cols:
            lc = c.lower()
            if 'snp' in lc and ('id' in lc or 'index' in lc):
                id_col = c
                break
        if id_col is None:
            id_col = cols[0]
        snp_ids = df[id_col].astype(str).tolist()
        if 'feature_64' in cols:
            feats_mat = _parse_feature_64(df['feature_64'], expected_dim=feat_dim)
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                feats_mat = np.zeros((df.shape[0], feat_dim), dtype=float)
            else:
                arr = df[num_cols].to_numpy(dtype=float)
                if arr.shape[1] > feat_dim:
                    arr = arr[:, :feat_dim]
                elif arr.shape[1] < feat_dim:
                    pad = np.zeros((arr.shape[0], feat_dim - arr.shape[1]), dtype=float)
                    arr = np.concatenate([arr, pad], axis=1)
                feats_mat = arr
        for sid, vec in zip(snp_ids, feats_mat):
            if sid in snp_feat_dict:
                snp_feat_dict[sid] = (snp_feat_dict[sid] + vec) / 2.0
            else:
                snp_feat_dict[sid] = vec
    return snp_feat_dict

def _collect_snp_ids_from_edges(data_root: Path) -> List[str]:
    snp_ids = set()
    path_sg = data_root / 'snp_gene_association_edges.csv'
    if path_sg.exists():
        df = pd.read_csv(path_sg)
        cols = df.columns.tolist()
        snp_col = None
        for c in cols:
            if 'snp' in c.lower():
                snp_col = c
                break
        if snp_col is None:
            snp_col = cols[0]
        snp_ids.update(df[snp_col].astype(str).tolist())
    path_ss = data_root / 'snp_snp_ld_edges.csv'
    if path_ss.exists():
        df = pd.read_csv(path_ss)
        cols = df.columns.tolist()
        snp_cols = [c for c in cols if 'snp' in c.lower()]
        if len(snp_cols) >= 2:
            c1, c2 = (snp_cols[0], snp_cols[1])
        elif len(cols) >= 2:
            c1, c2 = (cols[0], cols[1])
        else:
            c1 = c2 = cols[0]
        snp_ids.update(df[c1].astype(str).tolist())
        snp_ids.update(df[c2].astype(str).tolist())
    return sorted(list(snp_ids))

def load_asd_data(data_root: Path, feat_dim: int=64) -> Dict:
    data_root = Path(data_root)
    gene_index, gene_symbol, x_gene, y_gene = _load_gene_part(data_root, feat_dim=feat_dim)
    n_gene = x_gene.size(0)
    print(f'[data_io] 读取基因节点: {n_gene} 个')
    gene_index2pos = {int(g): i for i, g in enumerate(gene_index)}
    snp_ids_edges = _collect_snp_ids_from_edges(data_root)
    print(f'[data_io] 从边文件收集到 SNP ID 数量: {len(snp_ids_edges)}')
    snp_feat_dict = _load_snp_features(data_root, feat_dim=feat_dim)
    print(f'[data_io] 从特征文件中读取到 SNP 特征数量: {len(snp_feat_dict)}')
    all_snp_ids = set(snp_ids_edges).union(set(snp_feat_dict.keys()))
    all_snp_ids = sorted(list(all_snp_ids))
    n_snp = len(all_snp_ids)
    print(f'[data_io] SNP 节点总数: {n_snp}')
    snp_id2pos = {sid: n_gene + i for i, sid in enumerate(all_snp_ids)}
    if n_snp > 0:
        snp_feats = np.zeros((n_snp, feat_dim), dtype=float)
        for sid, pos in snp_id2pos.items():
            local_idx = pos - n_gene
            if sid in snp_feat_dict:
                snp_feats[local_idx] = snp_feat_dict[sid]
        x_snp = torch.from_numpy(snp_feats).float()
    else:
        x_snp = torch.zeros((0, feat_dim), dtype=torch.float32)
    N = n_gene + n_snp
    x_all = torch.zeros((N, feat_dim), dtype=torch.float32)
    x_all[0:n_gene] = x_gene
    if n_snp > 0:
        x_all[n_gene:] = x_snp
    y_all = np.full((N,), -1.0, dtype=float)
    y_all[0:n_gene] = y_gene
    node_gene_index = np.full((N,), -1, dtype=int)
    node_gene_symbol = np.array([''] * N, dtype=object)
    node_gene_index[0:n_gene] = gene_index
    node_gene_symbol[0:n_gene] = gene_symbol
    for sid, pos in snp_id2pos.items():
        node_gene_symbol[pos] = sid
    node_type = np.zeros((N,), dtype=int)
    if n_snp > 0:
        node_type[n_gene:] = 1
    all_src = []
    all_dst = []
    path_gg = data_root / 'gene_gene_ppi_edges.csv'
    if path_gg.exists():
        df_gg = pd.read_csv(path_gg)
        if not df_gg.empty:
            ucol, vcol = _auto_detect_edge_cols(df_gg)
            u = df_gg[ucol].to_numpy()
            v = df_gg[vcol].to_numpy()
            for uu, vv in zip(u, v):
                if int(uu) in gene_index2pos and int(vv) in gene_index2pos:
                    i = gene_index2pos[int(uu)]
                    j = gene_index2pos[int(vv)]
                    all_src.append(i)
                    all_dst.append(j)
                    all_src.append(j)
                    all_dst.append(i)
    else:
        print(f'[data_io] 警告: 未找到 {path_gg}, 将没有 gene-gene 边')
    path_sg = data_root / 'snp_gene_association_edges.csv'
    if path_sg.exists():
        df_sg = pd.read_csv(path_sg)
        if not df_sg.empty:
            cols = df_sg.columns.tolist()
            snp_col = None
            gene_col = None
            for c in cols:
                if 'snp' in c.lower():
                    snp_col = c
                if 'gene' in c.lower():
                    gene_col = c
            if snp_col is None or gene_col is None:
                if len(cols) >= 2:
                    snp_col, gene_col = (cols[0], cols[1])
                else:
                    snp_col = gene_col = cols[0]
            snp_ids = df_sg[snp_col].astype(str).to_numpy()
            gene_ids = df_sg[gene_col].to_numpy()
            for sid, gid in zip(snp_ids, gene_ids):
                if sid in snp_id2pos and int(gid) in gene_index2pos:
                    i = snp_id2pos[sid]
                    j = gene_index2pos[int(gid)]
                    all_src.append(i)
                    all_dst.append(j)
                    all_src.append(j)
                    all_dst.append(i)
    else:
        print(f'[data_io] 警告: 未找到 {path_sg}, 将没有 SNP-gene 边')
    path_ss = data_root / 'snp_snp_ld_edges.csv'
    if path_ss.exists():
        df_ss = pd.read_csv(path_ss)
        if not df_ss.empty:
            cols = df_ss.columns.tolist()
            snp_cols = [c for c in cols if 'snp' in c.lower()]
            if len(snp_cols) >= 2:
                c1, c2 = (snp_cols[0], snp_cols[1])
            elif len(cols) >= 2:
                c1, c2 = (cols[0], cols[1])
            else:
                c1 = c2 = cols[0]
            s1 = df_ss[c1].astype(str).to_numpy()
            s2 = df_ss[c2].astype(str).to_numpy()
            for a, b in zip(s1, s2):
                if a in snp_id2pos and b in snp_id2pos:
                    i = snp_id2pos[a]
                    j = snp_id2pos[b]
                    all_src.append(i)
                    all_dst.append(j)
                    all_src.append(j)
                    all_dst.append(i)
    else:
        print(f'[data_io] 警告: 未找到 {path_ss}, 将没有 SNP-SNP 边')
    all_src = np.array(all_src, dtype=np.int64)
    all_dst = np.array(all_dst, dtype=np.int64)
    if all_src.size == 0:
        print('[data_io] 警告: 没有任何边, 将返回空图')
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
    else:
        edge_index = torch.from_numpy(np.vstack([all_src, all_dst])).long()
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    print(f'[data_io] 边总数(无向计数): {edge_index.size(1)}')
    return dict(x=x_all, y=y_all, edge_indices_list=[edge_index], edge_weights_list=[edge_weight], gene_index=node_gene_index, gene_symbol=node_gene_symbol, node_type=node_type)

def default_data_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2] / 'data' / 'baselines'
