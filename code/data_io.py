from typing import Dict, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from config1 import (
    REL2ID, REL_MEQTL, REL_HQTL, REL_CIRCQTL, REL_LD, REL_PPI,
    Paths
)

def _parse_feature64(series: pd.Series) -> np.ndarray:
    vecs = []
    for s in series.astype(str).tolist():
        toks = s.split(',')
        vals = []
        for t in toks:
            try:
                vals.append(float(t))
            except Exception:
                vals.append(0.0)
        vals = (vals + [0.0]*64)[:64]
        vecs.append(vals)
    return np.array(vecs, dtype=np.float32)

@dataclass
class LoadedGraph:
    data: Data
    gene_mask: torch.Tensor
    snp_mask: torch.Tensor
    label: torch.Tensor
    node_ids: Dict[str, np.ndarray]
    xtalk_counts: np.ndarray

def load_csv_graph(paths: Paths) -> LoadedGraph:
    me = pd.read_csv(paths.snp_nodes_me, dtype=str, keep_default_na=False)
    h  = pd.read_csv(paths.snp_nodes_h,  dtype=str, keep_default_na=False)
    c  = pd.read_csv(paths.snp_nodes_c,  dtype=str, keep_default_na=False)
    g  = pd.read_csv(paths.gene_nodes,   dtype=str, keep_default_na=False)

    x_me = _parse_feature64(me['feature_64'])
    x_h  = _parse_feature64(h['feature_64'])
    x_c  = _parse_feature64(c['feature_64'])
    x_g  = _parse_feature64(g['feature_64'])

    idx_me = me['global_idx'].astype(int).to_numpy()
    idx_h  = h['global_idx'].astype(int).to_numpy()
    idx_c  = c['global_idx'].astype(int).to_numpy()
    gene_index = g['gene_index'].astype(int).to_numpy()

    n_snp = max(idx_me.max() if len(idx_me) else -1,
                idx_h.max() if len(idx_h) else -1,
                idx_c.max() if len(idx_c) else -1) + 1

    g2u = {gi: (n_snp + i) for i, gi in enumerate(sorted(set(gene_index)))}
    unified_gene_ids = np.array([g2u[gi] for gi in gene_index], dtype=np.int64)
    n_total = n_snp + len(g2u)

    # helper sets for robust mapping
    unified_gene_set = set(unified_gene_ids.tolist())

    def safe_map_gene(x):
        """Return unified gene id if possible, else None.
        Accepts either original gene_index (found in g2u) or an already unified id (in unified_gene_ids).
        """
        try:
            v = int(x)
        except Exception:
            return None
        if v in g2u:              # original gene_index
            return g2u[v]
        if v >= n_snp and v in unified_gene_set:  # already unified id
            return v
        return None

    X = np.zeros((n_total, 64), dtype=np.float32)
    X[idx_me, :] = x_me
    X[idx_h,  :] = x_h
    X[idx_c,  :] = x_c
    X[unified_gene_ids, :] = x_g

    snp_mask = np.zeros(n_total, dtype=bool); snp_mask[:n_snp] = True
    gene_mask = np.zeros(n_total, dtype=bool); gene_mask[unified_gene_ids] = True

    lab = np.full(n_total, -1.0, dtype=np.float32)
    if 'label' in g.columns:
        lab[unified_gene_ids] = g['label'].astype(float).to_numpy()

    e_sg = pd.read_csv(paths.edge_snp_gene, dtype=str, keep_default_na=False)
    e_ss = pd.read_csv(paths.edge_snp_snp,  dtype=str, keep_default_na=False)
    e_gg = pd.read_csv(paths.edge_gene_gene, dtype=str, keep_default_na=False)

    def _get_w(df):
        if 'val' in df.columns:
            w = pd.to_numeric(df['val'], errors='coerce').fillna(1.0).astype(float).to_numpy()
        else:
            w = np.ones(len(df), dtype=float)
        return np.clip(w, 0.0, 1e6)

    me_set = set(idx_me.tolist()); h_set = set(idx_h.tolist()); c_set = set(idx_c.tolist())

    def build_sg(df: pd.DataFrame):
        edges = []; skipped_sg = 0
        w = _get_w(df)
        s = pd.to_numeric(df['row'], errors='coerce').astype('Int64').to_numpy()
        t = df['col'].astype(int).tolist()
        for i, u in enumerate(s):
            if pd.isna(u): 
                continue
            u = int(u)
            v = safe_map_gene(t[i])
            if v is None:
                skipped_sg += 1
                continue
            if u in me_set: rid = REL2ID[REL_MEQTL]
            elif u in h_set: rid = REL2ID[REL_HQTL]
            elif u in c_set: rid = REL2ID[REL_CIRCQTL]
            else: rid = REL2ID[REL_MEQTL]
            edges.append((u, v, rid, float(w[i]))); edges.append((v, u, rid, float(w[i])))
        return edges, {"n": len(edges), "skipped_gene_missing": skipped_sg}

    def build_ss(df: pd.DataFrame):
        edges = []
        w = _get_w(df)
        s = pd.to_numeric(df['row'], errors='coerce').astype('Int64').to_numpy()
        t = pd.to_numeric(df['col'], errors='coerce').astype('Int64').to_numpy()
        for i, (u, v) in enumerate(zip(s, t)):
            if pd.isna(u) or pd.isna(v): 
                continue
            u = int(u); v = int(v)
            rid = REL2ID[REL_LD]
            edges.append((u, v, rid, float(w[i]))); edges.append((v, u, rid, float(w[i])))
        return edges

    def build_gg(df: pd.DataFrame):
        edges = []; skipped_gg = 0
        w = _get_w(df)
        s = df['row'].astype(int).to_numpy(); t = df['col'].astype(int).to_numpy()
        for i, (u_local, v_local) in enumerate(zip(s, t)):
            u = safe_map_gene(u_local); v = safe_map_gene(v_local)
            if u is None or v is None:
                skipped_gg += 1
                continue
            rid = REL2ID[REL_PPI]
            edges.append((u, v, rid, float(w[i]) if i < len(w) else 1.0))
            edges.append((v, u, rid, float(w[i]) if i < len(w) else 1.0))
        return edges, {"n": len(edges), "skipped_gene_missing": skipped_gg}

    edges_sg, sg_stats = build_sg(e_sg)
    edges_ss = build_ss(e_ss)
    edges_gg, gg_stats = build_gg(e_gg)

    all_edges = edges_sg + edges_ss + edges_gg
    try:
        print(f"[data_io] SNP->Gene edges: built={len(edges_sg)}, skipped_missing_gene={sg_stats.get('skipped_gene_missing',0)}")
        print(f"[data_io] Gene->Gene edges: built={len(edges_gg)}, skipped_missing_gene={gg_stats.get('skipped_gene_missing',0)}")
    except Exception:
        pass

    if len(all_edges) == 0:
        raise RuntimeError("No edges found; please check CSV paths")

    src, dst, rel, ww = zip(*all_edges)
    data = Data(
        x=torch.from_numpy(X),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_type=torch.tensor(rel, dtype=torch.long),
        edge_weight=torch.tensor(ww, dtype=torch.float32),
        num_nodes=n_snp + len(g2u),
    )

    counts = np.zeros((len(g2u), 7), dtype=np.float32)
    for (u, v, rid, _) in edges_sg:
        row = v - n_snp
        if row < 0 or row >= len(counts): continue
        if rid == REL2ID[REL_MEQTL]: counts[row,0]+=1
        elif rid == REL2ID[REL_HQTL]: counts[row,1]+=1
        elif rid == REL2ID[REL_CIRCQTL]: counts[row,2]+=1
    present = (counts[:,:3] > 0).astype(np.float32)
    counts[:,3] = present[:,0]*present[:,1]
    counts[:,4] = present[:,0]*present[:,2]
    counts[:,5] = present[:,1]*present[:,2]
    counts[:,6] = present[:,0]*present[:,1]*present[:,2]

    return LoadedGraph(
        data=data,
        gene_mask=torch.from_numpy(gene_mask),
        snp_mask=torch.from_numpy(snp_mask),
        label=torch.from_numpy(lab),
        node_ids={'gene_unified_ids': unified_gene_ids, 'gene_index': gene_index, 'snp_n': np.array(n_snp)},
        xtalk_counts=counts
    )