from pathlib import Path
from dataclasses import fields
import argparse, json, copy
from hedge.config import Paths, HyperParams

def main():
    ap = argparse.ArgumentParser(description='HEDGE training and analysis')
    ap.add_argument('command', choices=['validate-data', 'train', 'ratios', 'channel-ablation', 'component-ablation', 'grid', 'reproduce'])
    ap.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    ap.add_argument('--config', type=Path, default=None)
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    ap.add_argument('--device', default='cpu', help='cpu or cuda; choose explicitly')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--feature-mode', choices=['standardized', 'input'], default='standardized')
    ap.add_argument('--train-fractions', default='0,.1,.2,.3,.4,.5')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--channel', default='all_kept')
    ap.add_argument('--component', default='full')
    ap.add_argument('--protocol', choices=['main', 'component'], default=None)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--patience', type=int, default=25)
    args = ap.parse_args()
    if args.command == 'reproduce':
        from hedge.reference import reproduce
        reproduce(Path(__file__).resolve().parent / 'reference_data', args.out_dir)
        return
    from hedge.data import load_csv_graph
    names = {'snp_nodes_r': 'rqtl_snp_nodes.csv', 'snp_nodes_h': 'hqtl_snp_nodes.csv', 'snp_nodes_c': 'circqtl_snp_nodes.csv', 'gene_nodes': 'gene_nodes.csv', 'edge_snp_gene': 'snp_gene_association_edges.csv', 'edge_snp_snp': 'snp_snp_ld_edges.csv', 'edge_gene_gene': 'gene_gene_ppi_edges.csv'}
    paths = Paths(**{k: str(args.data_dir / v) for k, v in names.items()})
    loaded = load_csv_graph(paths, args.feature_mode)
    if args.command == 'validate-data':
        from hedge.validation import validate
        print(json.dumps(validate(loaded, args.feature_mode), indent=2))
        return
    config = args.config or Path(__file__).resolve().parent / ('configs/grid.json' if args.command == 'grid' else 'configs/main.json')
    raw = json.loads(config.read_text())
    valid = {f.name for f in fields(HyperParams)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError('Inactive or unknown parameters in active config: ' + str(sorted(unknown)))
    hp = HyperParams(**raw)
    if args.epochs is not None:
        hp.epochs = args.epochs
    if args.seed is not None:
        hp.seed = args.seed
    if hp.dim != 64 or hp.num_layers < 1 or hp.epochs < 1:
        raise ValueError('This processed64-dimensional graph requires dim=64, num_layers>=1 and epochs>=1')
    from hedge.training import run, CHANNEL_CONFIGS, COMPONENTS
    kwargs = dict(device=args.device, folds=args.folds, feature_mode=args.feature_mode, protocol=args.protocol or ('component' if args.command == 'component-ablation' else 'main'), patience=args.patience)
    if args.command == 'train':
        run(loaded, hp, args.out_dir, channel=args.channel, component=args.component, **kwargs)
    elif args.command == 'ratios':
        for frac in map(float, args.train_fractions.split(',')):
            run(loaded, hp, args.out_dir / f'fraction_{frac:g}', train_fraction=frac, repeats=args.repeats, **kwargs)
    elif args.command == 'channel-ablation':
        for channel in CHANNEL_CONFIGS:
            run(loaded, hp, args.out_dir / channel, channel=channel, **kwargs)
    elif args.command == 'component-ablation':
        for component in COMPONENTS:
            run(loaded, hp, args.out_dir / component, component=component, **kwargs)
    elif args.command == 'grid':
        for k in [1, 2, 4, 6, 8]:
            for layers in [2, 3, 4, 6]:
                for gate in [False, True]:
                    trial = copy.deepcopy(hp)
                    trial.appnp_K = k
                    trial.num_layers = layers
                    trial.enable_convergence_gate = gate
                    run(loaded, trial, args.out_dir / f'{k}-{layers}-{int(gate)}', **kwargs)
if __name__ == '__main__':
    main()
