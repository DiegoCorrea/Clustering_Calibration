import logging
import random
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from evaluations.labeling_models import (
    ReferenceStructure, parameter_grid, family_of, silhouette, group_size_descriptors,
    DENSITY_FAMILY, MAX_NOISE_SHARE, RANDOM_STATE
)
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class ManualConformityAlgorithmSearch:
    """
    Model selection of the labeling (unsupervised) algorithms on the users' preference distributions.

    Each configuration is fitted on the preference distribution of every fold and scored with the
    signed silhouette coefficient. The selected configuration maximizes the mean silhouette over the
    folds; ties are broken in favor of fewer groups. Configurations with fewer than two groups in some
    fold, or (density-based models) with more than MAX_NOISE_SHARE of noise users, are not eligible.
    The whole search trace is saved next to the selected configuration.
    """

    def __init__(
        self,
        experiment_name: str, split_methodology: str,
        dataset_name: str, distribution_list: list, distribution_class_list: str,
        n_jobs: int, fold: int, trial: int, n_inter: int
    ):
        self.dataset_name = dataset_name
        self.n_jobs = n_jobs
        self.fold = fold
        self.trial = trial
        self.n_inter = n_inter

        self.experiment_name = experiment_name
        self.split_methodology = split_methodology

        self.distribution_list = distribution_list
        self.distribution_class_list = distribution_class_list

    def preparing_data(self, distribution, distribution_class) -> list:
        """
        It loads the preference distribution of every fold, with the genres in a fixed order.
        """
        users_distribution_list = []
        for trial in range(1, self.trial + 1):
            for fold in range(1, self.fold + 1):
                dist_df = SaveAndLoad.load_user_preference_distribution(
                    dataset=self.dataset_name, trial=trial, fold=fold,
                    distribution=distribution, experiment_name=self.experiment_name,
                    split_methodology=self.split_methodology, distribution_class=distribution_class
                )
                dist_df = dist_df.reindex(columns=sorted(dist_df.columns)).fillna(0).sort_index()
                users_distribution_list.append(dist_df.to_numpy(dtype=float))
        return users_distribution_list

    @staticmethod
    def search(params: dict, conformity_str: str, users_distribution_list: list) -> dict:
        """
        It fits one configuration on every fold and returns its scores.
        """
        silhouette_list = []
        n_groups_list = []
        excluded_list = []
        fit_time_list = []
        error = ""
        for users_dist in users_distribution_list:
            try:
                structure = ReferenceStructure(model_name=conformity_str, params=params).fit(users_dist)
            except Exception as e:  # e.g., undefined distances for a metric
                error = f"{type(e).__name__}: {e}"
                break
            descriptors = group_size_descriptors(conformity_str, structure.fit_labels_)
            silhouette_list.append(silhouette(conformity_str, users_dist, structure.fit_labels_))
            n_groups_list.append(descriptors['n_groups'])
            excluded_list.append(descriptors['excluded_share'])
            fit_time_list.append(structure.fit_time_)

        complete = len(silhouette_list) == len(users_distribution_list) and error == ""
        eligible = complete and not np.isnan(silhouette_list).any()
        if eligible and family_of(conformity_str) == DENSITY_FAMILY:
            eligible = max(excluded_list) <= MAX_NOISE_SHARE

        return {
            "params": params,
            "silhouette": float(np.mean(silhouette_list)) if complete else np.nan,
            "silhouette_std": float(np.std(silhouette_list, ddof=1)) if complete and len(silhouette_list) > 1 else np.nan,
            "silhouette_per_fold": silhouette_list,
            "n_groups": float(np.mean(n_groups_list)) if n_groups_list else np.nan,
            "excluded_share": float(np.mean(excluded_list)) if excluded_list else np.nan,
            "fit_time": float(np.mean(fit_time_list)) if fit_time_list else np.nan,
            "eligible": bool(eligible),
            "error": error,
        }

    def get_params_to_use(self, conformity_str: str) -> list:
        """
        The partition and mixture grids are evaluated in full. Larger grids (density-based models)
        are sampled with a fixed seed.
        """
        params_list = parameter_grid(conformity_str)
        if self.n_inter < len(params_list):
            return random.Random(RANDOM_STATE).sample(params_list, self.n_inter)
        return params_list

    @staticmethod
    def select_best(payload: list):
        eligible = [p for p in payload if p["eligible"]]
        if not eligible:
            return None
        return sorted(eligible, key=lambda p: (-p["silhouette"], p["n_groups"]))[0]

    def run(self, conformity_str: str):
        """
        Start to run the Manual Grid Search for Unsupervised Learning Clustering Algorithms.
        """

        params_to_use = self.get_params_to_use(conformity_str=conformity_str)

        for distribution in self.distribution_list:
            for distribution_class in self.distribution_class_list:
                users_distribution_list = self.preparing_data(
                    distribution=distribution, distribution_class=distribution_class
                )
                start = time.perf_counter()
                payload = Parallel(n_jobs=self.n_jobs, verbose=0)(
                    delayed(ManualConformityAlgorithmSearch.search)(
                        params=params, conformity_str=conformity_str,
                        users_distribution_list=users_distribution_list
                    ) for params in params_to_use
                )
                search_time = time.perf_counter() - start

                best_param = self.select_best(payload)
                if best_param is None:
                    # The model is reported as "no eligible configuration" and is not evaluated in step 5.
                    print(f"No eligible configuration for {conformity_str} on {self.dataset_name}")
                    best_param = {"params": None, "silhouette": np.nan, "eligible": False}
                best_param = dict(best_param)
                best_param["n_evaluated"] = len(payload)
                best_param["n_eligible"] = int(sum(p["eligible"] for p in payload))
                best_param["search_time"] = search_time

                SaveAndLoad.save_hyperparameters_conformity(
                    best_params=best_param, dataset=self.dataset_name,
                    cluster=conformity_str, distribution=distribution,
                    experiment_name=self.experiment_name, split_methodology=self.split_methodology,
                    distribution_class=distribution_class
                )

                trace = pd.DataFrame([
                    {**{"param_" + k: v for k, v in p["params"].items()},
                     **{k: v for k, v in p.items() if k not in ["params", "silhouette_per_fold"]}}
                    for p in payload
                ])
                SaveAndLoad.save_conformity_search_trace(
                    data=trace, dataset=self.dataset_name,
                    cluster=conformity_str, distribution=distribution,
                    experiment_name=self.experiment_name, split_methodology=self.split_methodology,
                    distribution_class=distribution_class
                )
                print(f"{self.dataset_name} {conformity_str}: {best_param['params']} "
                      f"silhouette={best_param['silhouette']:.4f} ({search_time:.0f}s)", flush=True)
