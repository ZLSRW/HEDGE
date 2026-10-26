import torch

def validate(loaded, feature_mode):
    d = loaded.data
    gm = loaded.gene_mask
    labels = loaded.label[gm]
    assert tuple(d.x.shape) == (15547, 64)
    assert int(gm.sum()) == 8848 and int((labels == 1).sum()) == 4933 and (int((labels == 0).sum()) == 3915)
    assert set(d.edge_type.tolist()) == set(range(5))
    assert int(d.edge_index.min()) >= 0 and int(d.edge_index.max()) < d.num_nodes
    assert torch.isfinite(d.x).all() and torch.isfinite(d.edge_weight).all()
    if feature_mode == 'standardized':
        assert float(d.x.mean(0).abs().max()) < 0.0001
    unique = {tuple(x) for x in zip(d.edge_index[0].tolist(), d.edge_index[1].tolist(), d.edge_type.tolist(), d.edge_weight.tolist())}
    assert all(((v, u, r, w) in unique for u, v, r, w in unique))
    return {'pass': True, 'feature_mode': feature_mode, 'nodes': int(d.num_nodes), 'SNPs': int(loaded.snp_mask.sum()), 'genes': int(gm.sum()), 'positive': int((labels == 1).sum()), 'negative': int((labels == 0).sum()), 'gene_input_records': len(loaded.node_ids['gene_index']), 'relations_with_edges': sorted(set(d.edge_type.tolist())), 'attention_slots': 9, 'zero_message_slots': [5, 6, 7, 8], 'edges_both_orientations': int(d.edge_index.shape[1]), 'max_abs_feature_mean': float(d.x.mean(0).abs().max())}
