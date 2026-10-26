import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import ranksums
EFFECTS = ['effect_me', 'effect_h', 'effect_circ', 'drop_all', 'lift_only_me', 'lift_only_h', 'lift_only_circ']

def effect_space(probabilities):
    d = probabilities[probabilities.label == 1].copy()
    if d.gene_unified_id.duplicated().any():
        raise ValueError('Use one mean validation probability per gene/condition')
    for c in ['me', 'h', 'circ']:
        d['effect_' + c] = d.p_all_kept - d['p_remove_' + c]
        d['lift_only_' + c] = d['p_only_' + c] - d.p_all_removed
    d['drop_all'] = d.p_all_kept - d.p_all_removed
    normalized = d.copy()
    for c in EFFECTS:
        values = d[c].to_numpy()
        span = values.max() - values.min()
        normalized[c] = np.round((values - values.min()) / span, 4) if span > 1e-12 else 0.0
    return (d, normalized, StandardScaler().fit_transform(normalized[EFFECTS]))

def regulatory_groups(probabilities, seed=42, bootstraps=20):
    raw, norm, z = effect_space(probabilities)

    def fit(x, seed):
        return GaussianMixture(8, covariance_type='full', random_state=seed, n_init=10, max_iter=1000, tol=0.0001, reg_covar=1e-06).fit(x)
    model = fit(z, seed)
    labels = model.predict(z)
    norm['cluster'] = labels
    rng = np.random.default_rng(seed)
    scores = []
    for rep in range(bootstraps):
        idx = rng.choice(len(z), len(z), replace=True)
        other = fit(z[idx], 100 + rep)
        scores.append(normalized_mutual_info_score(labels, other.predict(z)))
    return (norm, {'silhouette': float(silhouette_score(z, labels)), 'bootstrap_NMI_mean': float(np.mean(scores)) if scores else None, 'bootstrap_replicates': bootstraps, 'BIC': float(model.bic(z))})

def bh_adjust(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return q

def disease_set_tests(positive_scores, sets):
    d = positive_scores.drop_duplicates('gene_symbol')
    universe = set(d.gene_symbol)
    rows = []
    for disease, genes in sets.items():
        mask = d.gene_symbol.isin(set(genes) & universe)
        a = d.loc[mask, 'risk_score'].to_numpy()
        b = d.loc[~mask, 'risk_score'].to_numpy()
        if len(a) < 5 or len(b) < 20:
            continue
        stat, p = ranksums(a, b)
        rows.append({'disease': disease, 'N_in_set': len(a), 'N_out_set': len(b), 'mean_in': a.mean(), 'mean_out': b.mean(), 'rank_sum_stat': stat, 'p_value': p})
    table = pd.DataFrame(rows)
    if len(table):
        table['q_value'] = bh_adjust(table.p_value)
    return table

def category_jaccard(sets, positive_universe):
    names = list(sets)
    projected = {k: set(v) & set(positive_universe) for k, v in sets.items()}
    return pd.DataFrame([[len(projected[a] & projected[b]) / max(1, len(projected[a] | projected[b])) for b in names] for a in names], index=names, columns=names)

def comorbidity_modules(features, k_candidates=range(2, 7), seed=42):
    d = features[features.risk_score >= features.risk_score.quantile(0.9)].copy()
    tags = [c for c in d if c.startswith('tag_')]
    columns = ['risk_score'] + tags + ['affinity_me', 'affinity_h', 'affinity_circ', 'gate_me', 'gate_h', 'gate_circ']
    if len(columns) != 16:
        raise ValueError('Expected risk+9 category indicators+3 affinities+3 gates')
    z = StandardScaler().fit_transform(d[columns])
    candidates = []
    for k in k_candidates:
        if 2 <= k < len(d):
            labels = KMeans(k, random_state=seed, n_init=20).fit_predict(z)
            candidates.append((silhouette_score(z, labels), k, labels))
    score, k, labels = max(candidates, key=lambda x: x[0])
    d['module'] = labels
    return (d, {'K': k, 'silhouette': float(score), 'features': columns})

def mapped_enrichment(enrichment, category_patterns, q_limit=0.5, top_n=30):
    terms = enrichment[enrichment['Adjusted P-value'] <= q_limit].sort_values('Adjusted P-value').head(top_n)
    rows = []
    for key, pattern in category_patterns.items():
        matched = terms[terms.Term.str.contains(pattern, case=False, regex=True, na=False)]
        rows.append({'category_key': key, 'n_terms': len(matched), 'sum_neglog10_q': float(-np.log10(matched['Adjusted P-value'].clip(lower=1e-300)).sum())})
    return pd.DataFrame(rows)
