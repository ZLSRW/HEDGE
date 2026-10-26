from pathlib import Path
import json
import numpy as np
import pandas as pd
from .training import metrics

def require(condition, message):
    if not condition:
        raise ValueError(message)

def verify_umap_coordinate_sets(partition, main, tolerance=1e-06):
    cols = ['umap_1', 'umap_2']
    left = {k: v[cols].drop_duplicates().to_numpy() for k, v in partition.groupby('gene_index')}
    right = {k: v[cols].drop_duplicates().to_numpy() for k, v in main.groupby('gene_index')}
    require(set(left) == set(right), 'UMAP gene sets differ')
    maximum = 0.0
    for gene, a in left.items():
        b = right[gene]
        distance = np.abs(a[:, None, :] - b[None, :, :]).max(axis=2)
        nearest = max(float(distance.min(axis=1).max()), float(distance.min(axis=0).max()))
        require(nearest <= tolerance, 'UMAP coordinates differ for gene %s' % gene)
        maximum = max(maximum, nearest)
    return {'pass': True, 'unique_genes': len(left), 'partition_records': len(partition), 'main_records': len(main), 'max_abs_matching_coordinate_difference': maximum, 'coordinates_match_within_export_precision': True}

def reproduce(root, out):
    root, out = (Path(root), Path(out))
    out.mkdir(parents=True, exist_ok=True)
    source = root / 'Source Data'
    supplementary = root / 'Supplementary Data'
    results = root.parent / 'results'
    rows = []
    for folder in sorted((results / 'benchmarks').iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(folder.glob('fold*_val_prob.csv'))
        require(len(files) == 5, 'Expected five validation folds: ' + folder.name)
        for fold, path in enumerate(files, 1):
            d = pd.read_csv(path)
            rows.append({'model': folder.name, 'fold': fold, 'n_records': len(d), 'n_unique_genes': d.gene_index.nunique(), **metrics(d.label.to_numpy(), d.prob.to_numpy())})
    folds = pd.DataFrame(rows)
    folds.to_csv(out / 'benchmark_per_fold.csv', index=False)
    names = ['AUC', 'AUPR', 'ACC', 'Precision']
    means = folds.groupby('model')[names].mean()
    means.to_csv(out / 'benchmark_means.csv')
    published = pd.read_csv(source / 'Figure 2/metrics_means.csv', index_col=0).rename(columns={'Prec': 'Precision'})
    published = published.rename(index={'OURS': 'HEDGE', 'DeepASDPred-CNN': 'DeepASDPred', 'Mantis-ML': 'Mantis-ml'})
    require(set(means.index) == set(published.index), 'Benchmark model names differ')
    require(np.allclose(means.values, published.loc[means.index, names].values, atol=5.1e-05, rtol=0), 'Benchmark means differ from Source Data')
    gains = []
    for metric in names:
        baseline = means.drop('HEDGE')[metric].idxmax()
        gains.append({'metric': metric, 'strongest_baseline': baseline, 'gain_percentage_points': 100 * (means.loc['HEDGE', metric] - means.loc[baseline, metric]), 'gain_percentage_points_from_displayed_values': 100 * (published.loc['HEDGE', metric] - published.loc[baseline, metric])})
    pd.DataFrame(gains).to_csv(out / 'metric_specific_gains.csv', index=False)
    detail = pd.read_csv(source / 'Figure 6/level_prediction_detail.csv')
    require(len(detail) == detail.gene_index.nunique() == 8848, 'Evidence tiers require one row per gene')
    require(np.array_equal(detail.correct.astype(bool), detail.pred_label == detail.label), 'Incorrect tier correctness flags')
    tiers = detail.groupby('level_label').agg(n_genes=('gene_index', 'size'), n_correct=('correct', 'sum'))
    tiers['n_incorrect'] = tiers.n_genes - tiers.n_correct
    tiers['accuracy'] = tiers.n_correct / tiers.n_genes
    saved = pd.read_csv(source / 'Figure 6/level_prediction_summary.csv').set_index('level_label')
    require(np.allclose(tiers.values, saved.loc[tiers.index, tiers.columns].values, atol=1e-10), 'Evidence tier summary differs')
    tiers.to_csv(out / 'evidence_tiers.csv')
    risk = pd.read_csv(source / 'Figure 6/risk_score_by_group_unique_genes.csv')
    require(risk.gene_index.nunique() == len(risk) == 8848, 'Risk groups require unique genes')
    risk_summary = risk.groupby('group_asd_discovery').risk_score.agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    saved_risk = pd.read_csv(source / 'Figure 6/risk_score_by_group_summary.csv').set_index('group_asd_discovery')
    require(np.allclose(risk_summary.values, saved_risk.loc[risk_summary.index, risk_summary.columns].values, atol=1e-10), 'Risk group summary differs')
    risk_summary.to_csv(out / 'risk_groups.csv')
    pool = pd.read_csv(supplementary / 'Supplementary Data 6/G_new_candidates_per_chr_topK.csv')
    candidates = pd.read_csv(source / 'Figure 6/lollipop_candidates_regions.csv')
    require(len(pool) == pool.gene_symbol.nunique() == 234, 'Expected 234 discovery genes')
    require(len(candidates) == candidates.gene_symbol.nunique() == 28, 'Expected 28 functional candidates')
    require(set(candidates.gene_symbol) <= set(pool.gene_symbol), 'Highlighted candidates must belong to discovery pool')
    require('INSYN2B' in set(candidates.gene_symbol), 'INSYN2B missing')
    affinity = pd.read_csv(source / 'Figure 3/gene_affinity_complete_8848.csv')
    selected = affinity[affinity.gene_symbol.isin(candidates.gene_symbol)].copy()
    selected['dominant_channel'] = selected[['affinity_r', 'affinity_h', 'affinity_circ']].idxmax(axis=1).map({'affinity_r': 'rQTL', 'affinity_h': 'hQTL', 'affinity_circ': 'circQTL'})
    require(len(selected) == 28, 'Candidate affinities missing')
    actual = selected.groupby('dominant_channel').size()
    channel = pd.read_csv(source / 'Figure 6/channel_dominance_potential_vs_other_candidates.csv')
    saved_channel = channel[channel.group == 'Potential_ASD'].set_index('dominant_channel')
    require(np.array_equal(actual.values, saved_channel.loc[actual.index, 'n_genes'].values), 'Candidate channel counts differ')
    selected.to_csv(out / 'functional_candidates.csv', index=False)
    channel.to_csv(out / 'candidate_channel_counts.csv', index=False)
    attention = pd.read_csv(source / 'Figure 3/relation_attention_density_Positive.csv')
    require(len(attention) == 4933 * 5 * 9, 'Expected 4933 genes by five fitted models by nine slots')
    sizes = attention.groupby(['gene_unified_id', 'fitted_model']).size()
    require(sizes.eq(9).all() and np.allclose(attention.groupby(['gene_unified_id', 'fitted_model']).AttentionWeight.sum(), 1, atol=1e-06), 'Nine-slot attention normalization differs')
    relation_summary = attention.groupby('slot_id').AttentionWeight.agg(['size', 'mean', 'std', 'median', 'min', 'max'])
    reference = pd.read_csv(source / 'Figure 3/relation_attention_summary_Positive.csv').set_index('slot_id')
    require(np.allclose(relation_summary['mean'], reference.loc[relation_summary.index, 'mean'], atol=1e-10), 'Relation attention means differ')
    wide = attention.pivot(index=['gene_unified_id', 'fitted_model'], columns='slot_id', values='AttentionWeight')
    require(all((np.allclose(wide[5], wide[i], atol=1e-10) for i in [6, 7, 8])), 'Zero-message slots differ')
    relation_summary.to_csv(out / 'relation_attention.csv')
    excluded = affinity[~affinity.included_in_raincloud.astype(bool)]
    require(len(excluded) == 28, 'Affinity display exclusions differ')
    grid = pd.read_csv(supplementary / 'Supplementary Data 1/hyperparameter_grid_per_fold.csv')
    keys = ['appnp_K', 'num_layers', 'enable_xtalk_gate']
    grid = grid[grid.split == 'val'] if 'split' in grid else grid
    require(len(grid.groupby(keys)) == 40, 'Expected 40 grid configurations')
    grid.to_csv(out / 'hyperparameter_grid_per_fold.csv', index=False)
    groups = pd.read_csv(results / 'effect_space/cluster_assignments.csv')
    require(len(groups) == 4933 and groups.cluster.nunique() == 8, 'Regulatory groups differ')
    groups.groupby('cluster').size().rename('n_genes').to_csv(out / 'regulatory_group_sizes.csv')
    modules = pd.read_csv(results / 'module_features/modules_gene_level_features.csv')
    require(len(modules) == 590 and modules.risk_score.eq(1).all(), 'Comorbidity gene selection differs')
    require(modules.groupby('module').size().sort_index().tolist() == [87, 409, 71, 23], 'Comorbidity modules differ')
    modules.groupby('module').size().rename('n_genes').to_csv(out / 'comorbidity_module_sizes.csv')
    disease = pd.read_csv(results / 'disease_set_analysis/all_categories_summary.csv')
    significant = disease[disease.q_value < 0.05]
    require(len(significant) == 332, 'Significant disease-set count differs')
    umap = verify_umap_coordinate_sets(pd.read_csv(supplementary / 'Supplementary Data 2/Supplementary Data 2.csv'), pd.read_csv(results / 'main_scatter_umap_coordinates.csv'))
    report = {'pass': True, 'models': len(means), 'fold_prediction_files': len(folds), 'recalculated_metrics': int(means.size), 'unique_genes': len(detail), 'level5_correct': int(tiers.loc['Level5', 'n_correct']), 'level5_incorrect': int(tiers.loc['Level5', 'n_incorrect']), 'level5_accuracy': float(tiers.loc['Level5', 'accuracy']), 'candidate_pool': len(pool), 'functional_candidates': len(candidates), 'candidate_channel_counts': {k: int(v) for k, v in actual.items()}, 'attention_slots': 9, 'attention_observations': len(attention), 'grid_configurations': 40, 'regulatory_groups': 8, 'comorbidity_genes': 590, 'comorbidity_modules': 4, 'significant_disease_sets': len(significant), 'umap': umap}
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report
