"""
Stage-wise structural evaluation protocol.

For a dataset, fold and labeling model:
1. the model is fitted once on the users' preference distribution (PREF), the reference structure;
2. the users of every later stage are projected onto the reference groups:
   CAND (the 100 candidate items), C@0.0 (the uncalibrated top-10 list) and C@0.1 ... C@1.0
   (the calibrated top-10 lists);
3. the projected labels of each stage are compared with the projected labels of PREF with
   measures that do not depend on how the groups are numbered, and the cohesion of every stage
   is measured with the silhouette coefficient.

All stages are aligned by user and by genre before any comparison.
"""
import logging
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, cohen_kappa_score,
    jaccard_score
)
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix
from scikit_pierre.distributions.compute_distribution import computer_users_distribution_pandas

from datasets.registred_datasets import RegisteredDataset
from evaluations.labeling_models import (
    ReferenceStructure, family_of, silhouette, group_size_descriptors, outlier_share,
    DETECTOR_FAMILY, DENSITY_FAMILY, OUTLIER_LABEL, NOISE_LABEL
)
from settings.labels import Label
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)

PREF_STAGE = "PREF"
CAND_STAGE = "CAND"
# Stages where the labeling model is also fitted independently, to reproduce the comparison of the
# first version of the paper (independent fits compared with the Jaccard score of the raw labels).
INDEPENDENT_FIT_STAGES = [CAND_STAGE, "C@0.0", "C@1.0"]

CWS_FLOOR = 0.00001


def stage_order(stage: str) -> float:
    if stage == PREF_STAGE:
        return -2.0
    if stage == CAND_STAGE:
        return -1.0
    return float(stage.split("@")[1])


# ------------------------------------------------------------------------------------------- #
# Distributions
# ------------------------------------------------------------------------------------------- #
class GenreSpace:
    """
    Item-genre matrices used to compute the Class Weighted Strategy (CWS) distribution of many users
    at once. The result is identical to scikit_pierre's class_weighted_strategy followed by fillna(0),
    which is checked on a sample of users every time a stage is computed.
    """

    def __init__(self, items: pd.DataFrame):
        self.items = items
        self.genres = sorted({g for genres in items[Label.CLASS_GENRE] for g in genres.split('|')})
        genre_ix = {g: k for k, g in enumerate(self.genres)}
        self.item_ix = {int(i): k for k, i in enumerate(items[Label.ITEM_ID])}
        rows, cols, ratio = [], [], []
        for item_id, genres in zip(items[Label.ITEM_ID], items[Label.CLASS_GENRE]):
            split = genres.split('|')
            for g in split:
                rows.append(self.item_ix[int(item_id)])
                cols.append(genre_ix[g])
                ratio.append(1.0 / len(split))
        shape = (len(self.item_ix), len(self.genres))
        self.genre_ratio = sparse.csr_matrix((ratio, (rows, cols)), shape=shape)
        self.genre_indicator = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)

    def cws(self, interactions: pd.DataFrame, users: list) -> np.ndarray:
        """
        CWS distribution of each user in `users` (rows) over the genres (columns).
        """
        data = interactions.drop_duplicates(subset=[Label.USER_ID, Label.ITEM_ID], keep='last')
        user_ix = {u: k for k, u in enumerate(users)}
        data = data[data[Label.USER_ID].isin(user_ix)]
        rows = data[Label.USER_ID].map(user_ix).to_numpy()
        cols = data[Label.ITEM_ID].astype(int).map(self.item_ix).to_numpy()
        scores = data[Label.TRANSACTION_VALUE].astype(float).to_numpy()
        presence = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(users), len(self.item_ix)))
        weights = sparse.csr_matrix((scores, (rows, cols)), shape=(len(users), len(self.item_ix)))

        numerator = (weights @ self.genre_ratio).toarray()
        denominator = (weights @ self.genre_indicator).toarray()
        present = (presence @ self.genre_indicator).toarray() > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            value = np.where((denominator > 0) & (numerator > 0), numerator / denominator, CWS_FLOOR)
        return np.where(present, value, 0.0)

    def check_against_library(self, interactions: pd.DataFrame, users: list, values: np.ndarray,
                              n_users: int = 15, seed: int = 0):
        """
        It compares the vectorized CWS with scikit_pierre on a random sample of users.
        """
        rng = np.random.default_rng(seed)
        sample = list(rng.choice(users, size=min(n_users, len(users)), replace=False))
        subset = interactions[interactions[Label.USER_ID].isin(sample)].copy()
        reference = computer_users_distribution_pandas(
            users_preference_set=subset, items_df=self.items, distribution="CWS"
        ).fillna(0)
        reference.index = reference.index.astype(int)
        reference = reference.reindex(index=sample, columns=self.genres).fillna(0)
        rows = [users.index(u) for u in sample]
        if not np.allclose(reference.to_numpy(), values[rows], atol=1e-9):
            raise ValueError("The vectorized CWS distribution differs from scikit_pierre.")


class StageData:
    """
    Distributions of all stages of one dataset, recommender and fold, aligned by user and genre.
    """

    def __init__(self, experiment_name: str, split_methodology: str, dataset: str, trial: int, fold: int,
                 distribution: str, distribution_class: str, tradeoff: str, fairness: str, relevance: str,
                 selector: str, weights: list):
        self.dataset_instance = RegisteredDataset.load_dataset(dataset)
        self.dataset_instance.set_environment(experiment_name=experiment_name, split_methodology=split_methodology)
        self.dataset_name = self.dataset_instance.get_dataset_name()
        self.space = GenreSpace(self.dataset_instance.get_items())

        self.experiment_name = experiment_name
        self.split_methodology = split_methodology
        self.trial = trial
        self.fold = fold
        self.distribution = distribution
        self.distribution_class = distribution_class
        self.tradeoff = tradeoff
        self.fairness = fairness
        self.relevance = relevance
        self.selector = selector
        self.weights = weights

        pref = SaveAndLoad.load_user_preference_distribution(
            experiment_name=experiment_name, split_methodology=split_methodology, dataset=dataset,
            trial=trial, fold=fold, distribution=distribution, distribution_class=distribution_class
        )
        pref.index = pref.index.astype(int)
        unknown = set(pref.columns) - set(self.space.genres)
        if unknown:
            raise ValueError(f"The preference distribution has classes that are not genres: {sorted(unknown)[:5]}")
        self.pref = pref.reindex(columns=self.space.genres).fillna(0).sort_index()
        self.users = self.pref.index.tolist()

    def preference(self) -> np.ndarray:
        return self.pref.to_numpy(dtype=float)

    def _stage(self, interactions: pd.DataFrame, check: bool) -> np.ndarray:
        interactions = interactions.copy()
        interactions[Label.USER_ID] = interactions[Label.USER_ID].astype(int)
        missing = set(self.users) - set(interactions[Label.USER_ID].unique())
        if missing:
            raise ValueError(f"{len(missing)} users of the preference distribution have no items in a stage.")
        values = self.space.cws(interactions, self.users)
        if check:
            self.space.check_against_library(interactions, self.users, values)
        return values

    def candidates(self, recommender: str, check: bool = True) -> np.ndarray:
        cand = SaveAndLoad.load_candidate_items(
            dataset=self.dataset_name, fold=self.fold, trial=self.trial, algorithm=recommender,
            split_methodology=self.split_methodology, experiment_name=self.experiment_name
        )
        return self._stage(cand, check)

    def recommendation_list(self, recommender: str, weight: str, check: bool = True) -> pd.DataFrame:
        return SaveAndLoad.load_recommendation_lists(
            dataset=self.dataset_name, recommender=recommender, trial=self.trial, fold=self.fold,
            tradeoff=self.tradeoff, distribution=self.distribution, fairness=self.fairness,
            relevance=self.relevance, tradeoff_weight=weight, select_item=self.selector,
            split_methodology=self.split_methodology, distribution_class=self.distribution_class,
            experiment_name=self.experiment_name
        )

    def stages(self, recommender: str, check: bool = True) -> dict:
        """
        {stage name: (users x genres) matrix} for CAND and every trade-off weight.
        """
        stages = {CAND_STAGE: self.candidates(recommender, check)}
        for weight in self.weights:
            rec_list = self.recommendation_list(recommender, weight)
            stages[weight] = self._stage(rec_list, check)
        return stages


# ------------------------------------------------------------------------------------------- #
# Agreement between two labelings of the same users
# ------------------------------------------------------------------------------------------- #
def bcubed(reference: np.ndarray, labels: np.ndarray) -> tuple:
    """
    B-cubed precision, recall and F1 (Amigo et al., 2009), from the contingency table.
    """
    table = contingency_matrix(reference, labels, sparse=True).astype(float)
    n = table.sum()
    squared = table.multiply(table)
    ref_sizes = np.asarray(table.sum(axis=1)).ravel()
    lab_sizes = np.asarray(table.sum(axis=0)).ravel()
    precision = (np.asarray(squared.sum(axis=0)).ravel() / lab_sizes).sum() / n
    recall = (np.asarray(squared.sum(axis=1)).ravel() / ref_sizes).sum() / n
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)


def pair_jaccard(reference: np.ndarray, labels: np.ndarray) -> float:
    """
    Pair-counting Jaccard index: pairs of users grouped together in both labelings over the pairs
    grouped together in at least one of them. It does not depend on the group numbering.
    """
    (_, c01), (c10, c11) = pair_confusion_matrix(reference, labels)
    denominator = c11 + c10 + c01
    return float(c11 / denominator) if denominator > 0 else 1.0


def flagged_set_jaccard(reference: np.ndarray, labels: np.ndarray, flag: int) -> tuple:
    """
    Jaccard of the sets of flagged users (atypical users or noise) and its value expected by chance
    for two independent sets with the same sizes.
    """
    a = reference == flag
    b = labels == flag
    union = (a | b).sum()
    if union == 0:
        return np.nan, np.nan
    observed = (a & b).sum() / union
    n = len(a)
    expected_intersection = a.sum() * b.sum() / n
    expected = expected_intersection / (a.sum() + b.sum() - expected_intersection)
    return float(observed), float(expected)


def agreement(model_name: str, reference: np.ndarray, labels: np.ndarray) -> dict:
    """
    Agreement between the reference (PREF) labels and the labels of a stage.
    Both labelings share the group identities, because they come from the same fitted model.
    """
    precision, recall, f1 = bcubed(reference, labels)
    group_labels = np.union1d(reference, labels)
    result = {
        'ARI': adjusted_rand_score(reference, labels),
        'AMI': adjusted_mutual_info_score(reference, labels),
        'NMI': normalized_mutual_info_score(reference, labels),
        'BCUBED_PRECISION': precision,
        'BCUBED_RECALL': recall,
        'BCUBED_F1': f1,
        'PAIR_JACCARD': pair_jaccard(reference, labels),
        'RETENTION': float((reference == labels).mean()),
        'KAPPA': cohen_kappa_score(reference, labels) if len(group_labels) > 1 else np.nan,
        'GROUP_JACCARD': jaccard_score(reference, labels, labels=group_labels, average='macro', zero_division=0),
        'FLAG_JACCARD': np.nan,
        'FLAG_JACCARD_CHANCE': np.nan,
    }
    family = family_of(model_name)
    if family in [DETECTOR_FAMILY, DENSITY_FAMILY]:
        flag = OUTLIER_LABEL if family == DETECTOR_FAMILY else NOISE_LABEL
        result['FLAG_JACCARD'], result['FLAG_JACCARD_CHANCE'] = flagged_set_jaccard(reference, labels, flag)
    return result


def independent_fit_agreement(reference_fit_labels: np.ndarray, independent_labels: np.ndarray) -> dict:
    """
    Comparison used in the first version of the paper: a second model is fitted on the stage and its
    raw labels are compared with the Jaccard score averaged over the label values (sklearn
    jaccard_score, average='macro'). The ARI of the same pair of labelings is given for contrast.
    """
    return {
        'INDEPENDENT_LABEL_JACCARD': jaccard_score(
            reference_fit_labels, independent_labels, average='macro', zero_division=0
        ),
        'INDEPENDENT_ARI': adjusted_rand_score(reference_fit_labels, independent_labels),
    }


# ------------------------------------------------------------------------------------------- #
# Calibration manipulation check
# ------------------------------------------------------------------------------------------- #
def kl_miscalibration(p: np.ndarray, q: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    KL(p || (1 - alpha) q + alpha p) per user, after normalizing p and q to sum to one (Steck, 2018).
    """
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    q_tilde = (1 - alpha) * q + alpha * p
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(p > 0, p * np.log(p / q_tilde), 0.0)
    return terms.sum(axis=1)


def emanon2_miscalibration(p: np.ndarray, q: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    The calibration objective used by the post-processing (VICIS_EMANON2 between p and the tilde q).
    """
    q_tilde = (1 - alpha) * q + alpha * p
    p_a = np.where(p == 0, CWS_FLOOR, p)
    q_b = np.where(q_tilde == 0, CWS_FLOOR, q_tilde)
    return (((p_a - q_b) ** 2) / (np.minimum(p_a, q_b) ** 2)).sum(axis=1)


# ------------------------------------------------------------------------------------------- #
# Protocol
# ------------------------------------------------------------------------------------------- #
class StageWiseEvaluation:
    """
    It runs the protocol for one dataset, fold and labeling model, over all recommenders.
    """

    def __init__(self, experiment_name: str, split_methodology: str, dataset: str, trial: int, fold: int,
                 model_name: str, recommenders: list, weights: list, distribution: str,
                 distribution_class: str, tradeoff: str, fairness: str, relevance: str, selector: str):
        self.experiment_name = experiment_name
        self.split_methodology = split_methodology
        self.dataset = dataset
        self.trial = trial
        self.fold = fold
        self.model_name = model_name
        self.recommenders = recommenders
        self.weights = weights
        self.distribution = distribution
        self.distribution_class = distribution_class
        self.data = StageData(
            experiment_name=experiment_name, split_methodology=split_methodology, dataset=dataset, trial=trial,
            fold=fold, distribution=distribution, distribution_class=distribution_class, tradeoff=tradeoff,
            fairness=fairness, relevance=relevance, selector=selector, weights=weights
        )
        selection = SaveAndLoad.load_hyperparameters_conformity(
            experiment_name=experiment_name, dataset=dataset, split_methodology=split_methodology,
            cluster=model_name, distribution=distribution, distribution_class=distribution_class
        )
        self.params = selection["params"]

    def _row(self, recommender: str, stage: str, x: np.ndarray, labels: np.ndarray, reference: np.ndarray,
             fidelity: float, fit_time: float) -> dict:
        start = time.perf_counter()
        cohesion = silhouette(self.model_name, x, labels)
        silhouette_time = time.perf_counter() - start
        descriptors = group_size_descriptors(self.model_name, labels)
        row = {
            'DATASET': self.dataset, 'FOLD': self.fold, 'TRIAL': self.trial,
            'MODEL': self.model_name, 'FAMILY': family_of(self.model_name),
            'RECOMMENDER': recommender, 'STAGE': stage, 'STAGE_ORDER': stage_order(stage),
            'N_USERS': x.shape[0], 'N_GENRES': x.shape[1],
            'SILHOUETTE': cohesion,
            'N_GROUPS': descriptors['n_groups'], 'EFFECTIVE_GROUPS': descriptors['effective_groups'],
            'LARGEST_GROUP_SHARE': descriptors['largest_group_share'],
            'FLAGGED_SHARE': outlier_share(self.model_name, labels),
            'PROJECTION_FIDELITY_ARI': fidelity, 'FIT_TIME': fit_time, 'SILHOUETTE_TIME': silhouette_time,
        }
        row.update(agreement(self.model_name, reference, labels))
        return row

    def run(self) -> tuple:
        """
        :return: (stage-wise measures, reference labels of the users)
        """
        x_pref = self.data.preference()
        structure = ReferenceStructure(self.model_name, self.params).fit(x_pref)
        fit_labels = structure.fit_labels_
        reference = structure.project(x_pref)
        fidelity = adjusted_rand_score(fit_labels, reference)

        rows = []
        # Descriptive structure of PREF, with the labels found by the model.
        pref_row = self._row("NONE", PREF_STAGE, x_pref, fit_labels, fit_labels, fidelity, structure.fit_time_)
        pref_row['LABELS'] = 'FIT'
        rows.append(pref_row)

        for recommender in self.recommenders:
            stages = {PREF_STAGE: x_pref}
            stages.update(self.data.stages(recommender))
            for stage, x_stage in stages.items():
                start = time.perf_counter()
                labels = structure.project(x_stage)
                projection_time = time.perf_counter() - start
                row = self._row(recommender, stage, x_stage, labels, reference, fidelity, structure.fit_time_)
                row['LABELS'] = 'PROJECTED'
                row['PROJECTION_TIME'] = projection_time
                if stage in INDEPENDENT_FIT_STAGES:
                    independent = ReferenceStructure(self.model_name, self.params).fit(x_stage)
                    row.update(independent_fit_agreement(fit_labels, independent.fit_labels_))
                    row['INDEPENDENT_N_GROUPS'] = group_size_descriptors(
                        self.model_name, independent.fit_labels_)['n_groups']
                rows.append(row)

        labels_df = pd.DataFrame({
            Label.USER_ID: self.data.users, 'FIT_LABEL': fit_labels, 'REFERENCE_LABEL': reference
        })
        return pd.DataFrame(rows), labels_df


def calibration_check(experiment_name: str, split_methodology: str, dataset: str, trial: int, fold: int,
                      recommender: str, weights: list, distribution: str, distribution_class: str,
                      tradeoff: str, fairness: str, relevance: str, selector: str) -> pd.DataFrame:
    """
    Miscalibration of the candidate set and of every list with respect to PREF, and the overlap of
    every list with the uncalibrated list (C@0.0). It shows whether the post-processing calibrates
    the lists towards the genre preferences.
    Every list is also compared directly with the first calibrated list (C@0.1): item overlap, share of
    users with the same items, share of users with the same genres, and the per-user change of the genre
    distribution. It shows whether a larger trade-off weight changes the lists or only their items.
    """
    data = StageData(
        experiment_name=experiment_name, split_methodology=split_methodology, dataset=dataset, trial=trial,
        fold=fold, distribution=distribution, distribution_class=distribution_class, tradeoff=tradeoff,
        fairness=fairness, relevance=relevance, selector=selector, weights=weights
    )
    x_pref = data.preference()
    stages = data.stages(recommender, check=False)
    rec_lists = {weight: data.recommendation_list(recommender, weight) for weight in weights}
    item_sets = {
        weight: rec_list.groupby(Label.USER_ID)[Label.ITEM_ID].apply(set) for weight, rec_list in rec_lists.items()
    }
    base = item_sets[weights[0]]
    calibrated = item_sets[weights[1]]
    x_calibrated = stages[weights[1]]

    def jaccard(sets: pd.Series, reference: pd.Series) -> float:
        return float(np.mean([len(sets[u] & reference[u]) / len(sets[u] | reference[u]) for u in reference.index]))

    # Relevance: binary relevance of the test interactions (feedback 1).
    test = data.dataset_instance.get_test_transactions(trial=trial, fold=fold)
    relevant = test[test[Label.TRANSACTION_VALUE] > 0].groupby(Label.USER_ID)[Label.ITEM_ID].apply(set)

    def relevance(rec_list: pd.DataFrame) -> tuple:
        precision, average_precision = [], []
        for user_id, user_list in rec_list.sort_values(Label.ORDER).groupby(Label.USER_ID):
            hits = np.array([item in relevant.get(user_id, set()) for item in user_list[Label.ITEM_ID]])
            precision.append(hits.mean())
            cumulative = np.cumsum(hits) / np.arange(1, len(hits) + 1)
            average_precision.append((cumulative * hits).sum() / max(1, min(len(relevant.get(user_id, ())), len(hits))))
        return float(np.mean(precision)), float(np.mean(average_precision))

    rows = []
    for stage, x_stage in stages.items():
        overlap, precision, map_value = np.nan, np.nan, np.nan
        overlap_calibrated, same_items, same_genres, genre_shift = np.nan, np.nan, np.nan, np.nan
        if stage in item_sets:
            overlap = jaccard(item_sets[stage], base)
            overlap_calibrated = jaccard(item_sets[stage], calibrated)
            same_items = float(np.mean([item_sets[stage][u] == calibrated[u] for u in calibrated.index]))
            same_genres = float(((x_stage > 0) == (x_calibrated > 0)).all(axis=1).mean())
            genre_shift = float(np.linalg.norm(x_stage - x_calibrated, axis=1).mean())
            precision, map_value = relevance(rec_lists[stage])
        rows.append({
            'DATASET': dataset, 'FOLD': fold, 'RECOMMENDER': recommender, 'STAGE': stage,
            'STAGE_ORDER': stage_order(stage),
            'KL_MISCALIBRATION': float(kl_miscalibration(x_pref, x_stage).mean()),
            'EMANON2_MISCALIBRATION': float(np.median(emanon2_miscalibration(x_pref, x_stage))),
            'GENRES_PER_USER': float((x_stage > 0).sum(axis=1).mean()),
            'ITEM_JACCARD_WITH_C@0.0': overlap,
            'ITEM_JACCARD_WITH_C@0.1': overlap_calibrated,
            'SAME_ITEMS_AS_C@0.1': same_items,
            'SAME_GENRES_AS_C@0.1': same_genres,
            'GENRE_SHIFT_FROM_C@0.1': genre_shift,
            'PRECISION@10': precision,
            'MAP@10': map_value,
        })
    return pd.DataFrame(rows)


def stage_wise_dir(experiment_name: str, dataset: str, split_methodology: str, distribution_class: str,
                   distribution: str, *parts) -> str:
    from settings.path_dir_file import PathDirFile
    path = "/".join([
        PathDirFile.DATA_DIR, experiment_name, "metrics", dataset, split_methodology, "STAGEWISE",
        distribution_class, distribution, *parts
    ])
    os.makedirs(path, exist_ok=True)
    return path
