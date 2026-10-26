from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from hedge import HEDGE, HyperParams, Paths
from hedge.data import load_csv_graph
from hedge.training import CHANNEL_CONFIGS, graph_intervention, interpretation

def main():
    parser = argparse.ArgumentParser(description='HEDGE checkpoint inference')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/prediction'))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    names = {'snp_nodes_r': 'rqtl_snp_nodes.csv', 'snp_nodes_h': 'hqtl_snp_nodes.csv', 'snp_nodes_c': 'circqtl_snp_nodes.csv', 'gene_nodes': 'gene_nodes.csv', 'edge_snp_gene': 'snp_gene_association_edges.csv', 'edge_snp_snp': 'snp_snp_ld_edges.csv', 'edge_gene_gene': 'gene_gene_ppi_edges.csv'}
    paths = Paths(**{k: str(args.data_dir / v) for k, v in names.items()})
    loaded = load_csv_graph(paths, checkpoint['feature_mode'])
    data = loaded.data.to(args.device)
    gm = loaded.gene_mask.to(args.device)
    counts = torch.zeros(data.num_nodes, 7, device=args.device)
    counts[gm] = torch.from_numpy(loaded.convergence_counts).to(args.device)
    data, counts, _ = graph_intervention(data, counts, CHANNEL_CONFIGS[checkpoint.get('channel', 'all_kept')])
    model = HEDGE(HyperParams(**checkpoint['hparams'])).to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    interventions = checkpoint.get('ablate', {})
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index, data.edge_type, data.edge_weight, gm, counts, ablate=interventions)
    ids = torch.where(gm)[0].cpu().numpy()
    lookup = dict(zip(loaded.node_ids['gene_unified_ids'], loaded.node_ids['gene_index']))
    table = pd.DataFrame({'gene_unified_id': ids, 'gene_index': [lookup[i] for i in ids], 'risk_score': logits[gm].sigmoid().cpu().numpy()})
    table['pred_label'] = (table.risk_score >= 0.5).astype(int)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / 'gene_predictions.csv', index=False)

    class InferenceModel(torch.nn.Module):

        def forward(self, *values):
            return model(*values, ablate=interventions)
    interpretation(InferenceModel(), data, counts, gm, loaded.label, lookup, args.out_dir, 'prediction')
    (args.out_dir / 'settings.json').write_text(json.dumps({'feature_mode': checkpoint['feature_mode'], 'channel': checkpoint.get('channel', 'all_kept'), 'best_epoch': checkpoint['best_epoch']}, indent=2) + '\n')
if __name__ == '__main__':
    main()
