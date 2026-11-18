"""
Run-time benchmark of the labeling models used by the stage-wise protocol.

The benchmark resamples the users' preference distributions of a dataset to increasing numbers of
users (bootstrap of the users plus a small jitter on the non-zero values, so that users are not
duplicated) and widens the genre space by appending permuted copies of the genre columns.
Each model uses the configuration selected for the dataset. A model is not run on the next size when
its expected time (current time times the growth factor of the size) exceeds the time budget.
"""
import gc
import time

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from evaluations.labeling_models import ReferenceStructure, family_of, excluded_mask, RANDOM_STATE
from settings.labels import Label
from settings.save_and_load import SaveAndLoad

# Models whose memory grows with the square of the number of users (dense affinity/distance matrices).
QUADRATIC_MEMORY_MODELS = [Label.AGGLOMERATIVE, Label.SPECTRAL]
QUADRATIC_MEMORY_MAX_USERS = 16000
SILHOUETTE_MAX_USERS = 16000
# Configuration used to time a model when no configuration was eligible in the search.
REFERENCE_PARAMS = {Label.OPTICS: {'min_samples': 10, 'metric': 'euclidean'}}


def resample_users(x: np.ndarray, n_users: int, rng: np.random.Generator, jitter: float = 0.01) -> np.ndarray:
    rows = x[rng.integers(0, x.shape[0], size=n_users)]
    noise = rng.normal(0.0, jitter, size=rows.shape) * (rows > 0)
    return np.clip(rows + noise, 0.0, 1.0)


def widen_genres(x: np.ndarray, factor: int, rng: np.random.Generator, jitter: float = 0.01) -> np.ndarray:
    blocks = [x]
    for _ in range(factor - 1):
        block = x[:, rng.permutation(x.shape[1])]
        blocks.append(np.clip(block + rng.normal(0.0, jitter, size=block.shape) * (block > 0), 0.0, 1.0))
    return np.hstack(blocks)


def _time_model(model_name: str, params: dict, x_fit: np.ndarray, x_project: np.ndarray) -> dict:
    structure = ReferenceStructure(model_name, params).fit(x_fit)
    start = time.perf_counter()
    labels = structure.project(x_project)
    projection_time = time.perf_counter() - start

    silhouette_time = np.nan
    if x_project.shape[0] <= SILHOUETTE_MAX_USERS:
        keep = ~excluded_mask(model_name, labels)
        if len(np.unique(labels[keep])) > 1:
            start = time.perf_counter()
            silhouette_score(x_project[keep], labels[keep])
            silhouette_time = time.perf_counter() - start
    return {'FIT_TIME': structure.fit_time_, 'PROJECTION_TIME': projection_time, 'SILHOUETTE_TIME': silhouette_time}


def scalability_benchmark(experiment_name: str, split_methodology: str, dataset: str, distribution: str,
                          distribution_class: str, models: list, user_sizes: list, genre_factors: list,
                          genre_users: int, time_budget: float) -> pd.DataFrame:
    pref = SaveAndLoad.load_user_preference_distribution(
        experiment_name=experiment_name, split_methodology=split_methodology, dataset=dataset,
        trial=1, fold=1, distribution=distribution, distribution_class=distribution_class
    )
    x_base = pref.reindex(columns=sorted(pref.columns)).fillna(0).sort_index().to_numpy(dtype=float)
    rows = []

    for model_name in models:
        params = SaveAndLoad.load_hyperparameters_conformity(
            experiment_name=experiment_name, dataset=dataset, split_methodology=split_methodology,
            cluster=model_name, distribution=distribution, distribution_class=distribution_class
        )["params"]
        note = ""
        if params is None:
            # No configuration was eligible in the search: the run time is measured with a reference one.
            params = REFERENCE_PARAMS[model_name]
            note = "reference configuration"

        # Number of users.
        previous_time, previous_size = None, None
        for n_users in user_sizes:
            reason = ""
            if model_name in QUADRATIC_MEMORY_MODELS and n_users > QUADRATIC_MEMORY_MAX_USERS:
                reason = "memory"
            elif previous_time is not None and previous_time * (n_users / previous_size) ** 2 > time_budget:
                reason = "time budget"
            row = {'DATASET': dataset, 'MODEL': model_name, 'FAMILY': family_of(model_name),
                   'BENCHMARK': 'USERS', 'N_USERS': n_users, 'N_GENRES': x_base.shape[1], 'SKIPPED': reason,
                   'NOTE': note}
            if reason == "":
                rng = np.random.default_rng(RANDOM_STATE + n_users)
                x_fit = resample_users(x_base, n_users, rng)
                x_project = resample_users(x_base, n_users, rng)
                row.update(_time_model(model_name, params, x_fit, x_project))
                previous_time, previous_size = row['FIT_TIME'] + row['PROJECTION_TIME'], n_users
                del x_fit, x_project
                gc.collect()
            rows.append(row)
            print(row, flush=True)

        # Number of genres.
        for factor in genre_factors:
            rng = np.random.default_rng(RANDOM_STATE + 1000 * factor)
            x_users = resample_users(x_base, genre_users, rng)
            x_fit = widen_genres(x_users, factor, rng)
            x_project = widen_genres(resample_users(x_base, genre_users, rng), factor, rng)
            row = {'DATASET': dataset, 'MODEL': model_name, 'FAMILY': family_of(model_name),
                   'BENCHMARK': 'GENRES', 'N_USERS': genre_users, 'N_GENRES': x_fit.shape[1], 'SKIPPED': "",
                   'NOTE': note}
            row.update(_time_model(model_name, params, x_fit, x_project))
            rows.append(row)
            print(row, flush=True)
            gc.collect()

    return pd.DataFrame(rows)
