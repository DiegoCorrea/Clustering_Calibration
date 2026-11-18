"""
Interpretation of the reference structure (step 6, option STAGEWISE).

For the K-Means configuration selected for each dataset, it finds the genre that separates the groups the most
(largest size-weighted variance of the group centroids), measures how much cohesion remains without that genre,
and follows the share of users that have the genre at every stage of the pipeline.

It also measures the dispersion of the users' distributions at every stage (mean Euclidean distance between users),
which shows directly whether the users become closer to each other; the group measures only show whether they keep
their reference group.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_chunked, silhouette_score

from evaluations.labeling_models import ReferenceStructure
from evaluations.stage_wise_evaluation import StageData, CAND_STAGE, PREF_STAGE
from settings.labels import Label
from settings.save_and_load import SaveAndLoad


def dominant_genre(x: np.ndarray, labels: np.ndarray, genres: list) -> tuple:
    groups, sizes = np.unique(labels, return_counts=True)
    centroids = np.vstack([x[labels == g].mean(axis=0) for g in groups])
    weights = sizes / sizes.sum()
    overall = weights @ centroids
    spread = (weights[:, None] * (centroids - overall) ** 2).sum(axis=0)
    j = int(np.argmax(spread))
    return j, genres[j], centroids[:, j], spread[j] / spread.sum()


def structure_interpretation(experiment_name: str, split_methodology: str, dataset: str, trial: int, folds: list,
                             recommenders: list, weights: list, distribution: str, distribution_class: str,
                             tradeoff: str, fairness: str, relevance: str, selector: str,
                             model_name: str = Label.KMEANS) -> pd.DataFrame:
    params = SaveAndLoad.load_hyperparameters_conformity(
        experiment_name=experiment_name, dataset=dataset, split_methodology=split_methodology,
        cluster=model_name, distribution=distribution, distribution_class=distribution_class
    )["params"]
    rows = []
    for fold in folds:
        data = StageData(
            experiment_name=experiment_name, split_methodology=split_methodology, dataset=dataset, trial=trial,
            fold=fold, distribution=distribution, distribution_class=distribution_class, tradeoff=tradeoff,
            fairness=fairness, relevance=relevance, selector=selector, weights=weights
        )
        x = data.preference()
        genres = data.space.genres
        labels = ReferenceStructure(model_name, params).fit(x).fit_labels_
        j, genre, centroid_values, spread_share = dominant_genre(x, labels, genres)
        x_without = np.delete(x, j, axis=1)
        labels_without = ReferenceStructure(model_name, params).fit(x_without).fit_labels_

        profile = data.dataset_instance.get_full_train_transactions(trial=trial, fold=fold).groupby(
            Label.USER_ID).size().reindex(data.users).to_numpy()
        has_genre = x[:, j] > 0
        row = {
            'DATASET': dataset, 'FOLD': fold, 'MODEL': model_name, 'PARAMS': params, 'DOMINANT_GENRE': genre,
            'SPREAD_SHARE': spread_share,
            'CENTROID_MAX': float(centroid_values.max()), 'CENTROID_MIN': float(centroid_values.min()),
            'SILHOUETTE': float(silhouette_score(x, labels)),
            'SILHOUETTE_WITHOUT_GENRE': float(silhouette_score(x_without, labels_without)),
            'PROFILE_WITH_GENRE': float(profile[has_genre].mean()),
            'PROFILE_WITHOUT_GENRE': float(profile[~has_genre].mean()),
            'RECOMMENDER': 'NONE', 'STAGE': PREF_STAGE, 'SHARE_WITH_GENRE': float(has_genre.mean()),
        }
        rows.append(row)
        for recommender in recommenders:
            for stage, x_stage in data.stages(recommender, check=False).items():
                rows.append({**row, 'RECOMMENDER': recommender, 'STAGE': stage,
                             'SHARE_WITH_GENRE': float((x_stage[:, j] > 0).mean())})
    return pd.DataFrame(rows)


def mean_distance_between_users(x: np.ndarray) -> float:
    """
    Mean Euclidean distance over all pairs of users, computed in chunks to bound the memory.
    """
    n = len(x)
    total = sum(float(chunk.sum()) for chunk in pairwise_distances_chunked(x))
    return total / (n * (n - 1))


def stage_dispersion(experiment_name: str, split_methodology: str, dataset: str, trial: int, folds: list,
                     recommenders: list, weights: list, distribution: str, distribution_class: str,
                     tradeoff: str, fairness: str, relevance: str, selector: str) -> pd.DataFrame:
    rows = []
    for fold in folds:
        data = StageData(
            experiment_name=experiment_name, split_methodology=split_methodology, dataset=dataset, trial=trial,
            fold=fold, distribution=distribution, distribution_class=distribution_class, tradeoff=tradeoff,
            fairness=fairness, relevance=relevance, selector=selector, weights=weights
        )
        rows.append({'DATASET': dataset, 'FOLD': fold, 'RECOMMENDER': 'NONE', 'STAGE': PREF_STAGE,
                     'MEAN_DISTANCE': mean_distance_between_users(data.preference())})
        for recommender in recommenders:
            for stage, x_stage in data.stages(recommender, check=False).items():
                rows.append({'DATASET': dataset, 'FOLD': fold, 'RECOMMENDER': recommender, 'STAGE': stage,
                             'MEAN_DISTANCE': mean_distance_between_users(x_stage)})
    return pd.DataFrame(rows)
