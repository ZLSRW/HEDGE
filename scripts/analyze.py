from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from hedge.analysis import comorbidity_modules, disease_set_tests, mapped_enrichment, regulatory_groups

def main():
    parser = argparse.ArgumentParser(description='HEDGE downstream analysis')
    parser.add_argument('analysis', choices=['effects', 'modules', 'disease-sets', 'enrichment-map'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--sets', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/analysis'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstraps', type=int, default=20)
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.analysis == 'effects':
        result, report = regulatory_groups(table, seed=args.seed, bootstraps=args.bootstraps)
    elif args.analysis == 'modules':
        if 'label' in table:
            table = table[table.label == 1]
        result, report = comorbidity_modules(table, seed=args.seed)
    else:
        if args.sets is None:
            parser.error('--sets is required for this analysis')
        specification = json.loads(args.sets.read_text())
        if args.analysis == 'disease-sets':
            if 'label' in table:
                table = table[table.label == 1]
            result = disease_set_tests(table, specification)
        else:
            result = mapped_enrichment(table, specification)
        report = {'rows': len(result)}
    result.to_csv(args.out_dir / (args.analysis + '.csv'), index=False)
    (args.out_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
if __name__ == '__main__':
    main()
