"""
Aggregation and statistical tests of the stage-wise structural evaluation (step 6).

- Dispersion: mean, standard deviation and 95% t-confidence interval over the folds.
- Stage contrasts: paired t-tests over the folds, Holm-corrected over the labeling models of the same
  dataset, recommender, contrast and measure.
- Flatness over the trade-off weight: Friedman test over lambda in {0.1, ..., 1.0} (blocks = folds) and a
  paired TOST equivalence test between C@0.1 and C@1.0 with margin EQUIVALENCE_MARGIN.
- Model comparison: Friedman test over the models (blocks = dataset x recommender) and the Nemenyi
  critical difference (Demsar, 2006).
- Reproducibility of the reference structure: ARI between the PREF labelings of every pair of folds.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from evaluations.stage_wise_evaluation import PREF_STAGE, CAND_STAGE

ALPHA = 0.05
EQUIVALENCE_MARGIN = 0.05

MEASURES = [
    'SILHOUETTE', 'ARI', 'AMI', 'NMI', 'BCUBED_PRECISION', 'BCUBED_RECALL', 'BCUBED_F1', 'PAIR_JACCARD',
    'RETENTION', 'KAPPA', 'GROUP_JACCARD', 'FLAG_JACCARD', 'FLAG_JACCARD_CHANCE', 'N_GROUPS',
    'EFFECTIVE_GROUPS', 'LARGEST_GROUP_SHARE', 'FLAGGED_SHARE', 'INDEPENDENT_LABEL_JACCARD', 'INDEPENDENT_ARI',
    'INDEPENDENT_N_GROUPS', 'PROJECTION_FIDELITY_ARI', 'FIT_TIME', 'PROJECTION_TIME', 'SILHOUETTE_TIME',
]

# (name, first stage, second stage). The difference is "second - first".
CONTRASTS = [
    ('PREF->CAND', PREF_STAGE, CAND_STAGE),
    ('CAND->C@0.0', CAND_STAGE, 'C@0.0'),
    ('C@0.0->C@0.1', 'C@0.0', 'C@0.1'),
    ('C@0.0->C@1.0', 'C@0.0', 'C@1.0'),
    ('C@0.1->C@1.0', 'C@0.1', 'C@1.0'),
    ('PREF->C@1.0', PREF_STAGE, 'C@1.0'),
]
TESTED_MEASURES = ['SILHOUETTE', 'ARI', 'AMI', 'BCUBED_F1', 'KAPPA']
CALIBRATED_WEIGHTS = ['C@0.1', 'C@0.2', 'C@0.3', 'C@0.4', 'C@0.5', 'C@0.6', 'C@0.7', 'C@0.8', 'C@0.9', 'C@1.0']


# ------------------------------------------------------------------------------------------- #
# Basic statistics
# ------------------------------------------------------------------------------------------- #
def mean_ci(values, confidence: float = 0.95) -> tuple:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = v.mean()
    if n == 1:
        return mean, np.nan, np.nan, np.nan, 1
    sd = v.std(ddof=1)
    half = stats.t.ppf((1 + confidence) / 2, n - 1) * sd / np.sqrt(n)
    return mean, sd, mean - half, mean + half, n


def paired_t(first, second) -> tuple:
    """
    Paired t-test of second - first. Returns (mean difference, CI low, CI high, p-value, n).
    """
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = b[keep] - a[keep]
    mean, sd, low, high, n = mean_ci(d)
    if n < 2:
        return mean, low, high, np.nan, n
    if sd == 0:
        return mean, mean, mean, (1.0 if mean == 0 else 0.0), n
    p_value = stats.ttest_1samp(d, 0.0).pvalue
    return mean, low, high, float(p_value), n


def tost_paired(first, second, margin: float = EQUIVALENCE_MARGIN) -> float:
    """
    Two one-sided t-tests for the equivalence of second and first within +/- margin.
    A p-value below ALPHA means that the difference is inside the margin.
    """
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = b[keep] - a[keep]
    n = len(d)
    if n < 2:
        return np.nan
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(mean) < margin else 1.0
    p_lower = 1 - stats.t.cdf((mean + margin) / se, n - 1)
    p_upper = stats.t.cdf((mean - margin) / se, n - 1)
    return float(max(p_lower, p_upper))


def holm(p_values) -> np.ndarray:
    """
    Holm-Bonferroni adjusted p-values (NaN values are ignored).
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    if len(valid) == 0:
        return adjusted
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def friedman(matrix: np.ndarray) -> tuple:
    """
    Friedman test. Rows are blocks and columns are treatments.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < 2 or matrix.shape[1] < 3 or np.isnan(matrix).any():
        return np.nan, np.nan
    if np.allclose(matrix, matrix[:, [0]]):
        return 0.0, 1.0
    statistic, p_value = stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    if np.isnan(p_value):
        return 0.0, 1.0
    return float(statistic), float(p_value)


def nemenyi_cd(k: int, n_blocks: int, alpha: float = ALPHA) -> float:
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


# ------------------------------------------------------------------------------------------- #
# Aggregations
# ------------------------------------------------------------------------------------------- #
def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ['DATASET', 'MODEL', 'FAMILY', 'RECOMMENDER', 'STAGE', 'STAGE_ORDER', 'LABELS']
    rows = []
    for key, group in raw.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        for measure in MEASURES:
            if measure not in group:
                continue
            mean, sd, low, high, n = mean_ci(group[measure])
            row[measure] = mean
            row[measure + '_SD'] = sd
            row[measure + '_CI_LOW'] = low
            row[measure + '_CI_HIGH'] = high
            row[measure + '_N'] = n
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def _fold_values(data: pd.DataFrame, stage: str, measure: str) -> pd.Series:
    return data[data['STAGE'] == stage].set_index('FOLD')[measure].sort_index()


def contrasts(raw: pd.DataFrame) -> pd.DataFrame:
    projected = raw[raw['LABELS'] == 'PROJECTED']
    rows = []
    for (dataset, recommender, model), data in projected.groupby(['DATASET', 'RECOMMENDER', 'MODEL']):
        for name, first, second in CONTRASTS:
            for measure in TESTED_MEASURES:
                a = _fold_values(data, first, measure)
                b = _fold_values(data, second, measure)
                a, b = a.align(b, join='inner')
                mean, low, high, p_value, n = paired_t(a.values, b.values)
                rows.append({
                    'DATASET': dataset, 'RECOMMENDER': recommender, 'MODEL': model,
                    'FAMILY': data['FAMILY'].iloc[0], 'CONTRAST': name, 'MEASURE': measure,
                    'FIRST_MEAN': np.nanmean(a.values) if len(a) else np.nan,
                    'SECOND_MEAN': np.nanmean(b.values) if len(b) else np.nan,
                    'DIFFERENCE': mean, 'CI_LOW': low, 'CI_HIGH': high, 'P_VALUE': p_value, 'N': n,
                    'P_TOST': tost_paired(a.values, b.values),
                })
    result = pd.DataFrame(rows)
    result['P_HOLM'] = np.nan
    for _, index in result.groupby(['DATASET', 'RECOMMENDER', 'CONTRAST', 'MEASURE']).groups.items():
        result.loc[index, 'P_HOLM'] = holm(result.loc[index, 'P_VALUE'].values)
    result['SIGNIFICANT'] = result['P_HOLM'] < ALPHA
    result['EQUIVALENT'] = result['P_TOST'] < ALPHA
    return result


def flatness(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Is the measure constant over lambda in {0.1, ..., 1.0}?
    """
    projected = raw[raw['LABELS'] == 'PROJECTED']
    rows = []
    for (dataset, recommender, model), data in projected.groupby(['DATASET', 'RECOMMENDER', 'MODEL']):
        for measure in TESTED_MEASURES:
            table = data[data['STAGE'].isin(CALIBRATED_WEIGHTS)].pivot_table(
                index='FOLD', columns='STAGE', values=measure, aggfunc='first'
            ).reindex(columns=CALIBRATED_WEIGHTS)
            statistic, p_value = friedman(table.values)
            fold_means = table.mean(axis=0)
            rows.append({
                'DATASET': dataset, 'RECOMMENDER': recommender, 'MODEL': model,
                'FAMILY': data['FAMILY'].iloc[0], 'MEASURE': measure,
                'FRIEDMAN_STATISTIC': statistic, 'FRIEDMAN_P': p_value,
                'RANGE_OVER_LAMBDA': float(fold_means.max() - fold_means.min()),
                'P_TOST_C0.1_C1.0': tost_paired(table['C@0.1'].values, table['C@1.0'].values),
            })
    result = pd.DataFrame(rows)
    result['FRIEDMAN_P_HOLM'] = np.nan
    for _, index in result.groupby(['DATASET', 'RECOMMENDER', 'MEASURE']).groups.items():
        result.loc[index, 'FRIEDMAN_P_HOLM'] = holm(result.loc[index, 'FRIEDMAN_P'].values)
    result['VARIES_WITH_LAMBDA'] = result['FRIEDMAN_P_HOLM'] < ALPHA
    result['EQUIVALENT_C0.1_C1.0'] = result['P_TOST_C0.1_C1.0'] < ALPHA
    return result


def model_ranking(raw: pd.DataFrame, stages: list, measures: list) -> tuple:
    """
    Average rank of every model (rank 1 = highest value) over the dataset x recommender blocks, with the
    Friedman test and the Nemenyi critical difference. Undefined values (e.g., a silhouette with a single
    group) receive the worst rank.
    """
    projected = raw[raw['LABELS'] == 'PROJECTED']
    fold_means = projected.groupby(['DATASET', 'RECOMMENDER', 'MODEL', 'FAMILY', 'STAGE'])[measures].mean().reset_index()
    ranks_rows, tests_rows = [], []
    for stage in stages:
        for measure in measures:
            table = fold_means[fold_means['STAGE'] == stage].pivot_table(
                index=['DATASET', 'RECOMMENDER'], columns='MODEL', values=measure, aggfunc='first', dropna=False
            )
            # Models that were not evaluated in some block (no eligible configuration) are left out, because
            # the Friedman test needs complete blocks.
            present = fold_means[fold_means['STAGE'] == stage].groupby('MODEL')[['DATASET', 'RECOMMENDER']].apply(
                lambda g: len(g.drop_duplicates()))
            table = table[[m for m in table.columns if present.get(m, 0) == table.shape[0]]]
            filled = table.fillna(-np.inf)
            ranks = filled.rank(axis=1, ascending=False, method='average')
            statistic, p_value = friedman(ranks.values)
            k, n_blocks = table.shape[1], table.shape[0]
            cd = nemenyi_cd(k, n_blocks)
            tests_rows.append({'STAGE': stage, 'MEASURE': measure, 'N_MODELS': k, 'N_BLOCKS': n_blocks,
                               'FRIEDMAN_STATISTIC': statistic, 'FRIEDMAN_P': p_value, 'NEMENYI_CD': cd})
            average = ranks.mean(axis=0)
            best = average.min()
            for model, rank in average.items():
                ranks_rows.append({
                    'STAGE': stage, 'MEASURE': measure, 'MODEL': model,
                    'FAMILY': fold_means.loc[fold_means['MODEL'] == model, 'FAMILY'].iloc[0],
                    'AVERAGE_RANK': rank, 'MEAN_VALUE': float(table[model].mean()),
                    'WITHIN_CD_OF_BEST': bool(rank - best <= cd),
                })
    return pd.DataFrame(ranks_rows), pd.DataFrame(tests_rows)


def reproducibility(labels_by_fold: dict) -> pd.DataFrame:
    """
    :param labels_by_fold: {(dataset, model): {fold: DataFrame[USER_ID, FIT_LABEL, REFERENCE_LABEL]}}
    """
    rows = []
    for (dataset, model), folds in labels_by_fold.items():
        for column in ['FIT_LABEL', 'REFERENCE_LABEL']:
            values = []
            for fa, fb in itertools.combinations(sorted(folds), 2):
                a = folds[fa].set_index('USER_ID')[column]
                b = folds[fb].set_index('USER_ID')[column]
                a, b = a.align(b, join='inner')
                values.append(adjusted_rand_score(a.values, b.values))
            mean, sd, low, high, n = mean_ci(values)
            rows.append({'DATASET': dataset, 'MODEL': model, 'LABELS': column, 'CROSS_FOLD_ARI': mean,
                         'CROSS_FOLD_ARI_SD': sd, 'CROSS_FOLD_ARI_MIN': np.min(values) if values else np.nan,
                         'CROSS_FOLD_ARI_MAX': np.max(values) if values else np.nan, 'N_PAIRS': n})
    return pd.DataFrame(rows)


def calibration_summary(raw: pd.DataFrame) -> pd.DataFrame:
    measures = ['KL_MISCALIBRATION', 'EMANON2_MISCALIBRATION', 'GENRES_PER_USER', 'ITEM_JACCARD_WITH_C@0.0',
                'ITEM_JACCARD_WITH_C@0.1', 'SAME_ITEMS_AS_C@0.1', 'SAME_GENRES_AS_C@0.1', 'GENRE_SHIFT_FROM_C@0.1',
                'PRECISION@10', 'MAP@10']
    rows = []
    for key, group in raw.groupby(['DATASET', 'RECOMMENDER', 'STAGE', 'STAGE_ORDER']):
        row = dict(zip(['DATASET', 'RECOMMENDER', 'STAGE', 'STAGE_ORDER'], key))
        for measure in measures:
            mean, sd, low, high, n = mean_ci(group[measure])
            row.update({measure: mean, measure + '_SD': sd, measure + '_CI_LOW': low, measure + '_CI_HIGH': high})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['DATASET', 'RECOMMENDER', 'STAGE_ORDER'])
