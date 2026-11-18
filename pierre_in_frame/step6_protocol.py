import itertools
import logging
import os
from statistics import mean, mode

import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame

from evaluations import stage_wise_statistics as sws
from evaluations.metrics_comprises import MetricComprises
from evaluations.stage_wise_evaluation import stage_wise_dir
from evaluations.structure_interpretation import structure_interpretation, stage_dispersion
from settings.path_dir_file import PathDirFile
from settings.constants import Constants
from settings.labels import Label
from settings.save_and_load import SaveAndLoad
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep6(Step):
    """
    TODO: Docstring
    """

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = Input.step6()

    @staticmethod
    def set_the_logfile_by_instance(
            dataset: str, recommender: str, trial: int, fold: int, tradeoff: str, distribution: str,
            fairness: str, relevance: str, tradeoff_weight: str, select_item: str
    ):
        """
        TODO: Docstring
        """
        # Setup Log configuration
        # setup_logging(
        #     log_error="error.log", log_info="info.log",
        #     save_path=PathDirFile.set_decision_file(
        #         dataset=dataset, recommender=recommender, trial=trial, fold=fold, tradeoff=tradeoff,
        #         distribution=distribution, fairness=fairness, relevance=relevance, tradeoff_weight=tradeoff_weight,
        #         select_item=select_item
        #     )
        # )
        pass

    def print_basic_info_by_instance(self, **kwargs):
        """
        TODO: Docstring
        """

        logger.info("$" * 50)
        logger.info("$" * 50)
        # Logging machine data
        self.machine_information()
        logger.info("-" * 50)

        # Logging the experiment setup
        logger.info("[METRIC STEP]")
        logger.info(kwargs)
        logger.info("$" * 50)
        logger.info("$" * 50)

    def main(self):
        """
        TODO: Docstring
        """
        if self.experimental_settings['opt'] == Label.CONFORMITY:
            self.conformity_parallelization()
        elif self.experimental_settings['opt'] == Label.EVALUATION_METRICS:
            self.recommender_metrics_parallelization()
        elif self.experimental_settings['opt'] == Label.STAGEWISE:
            self.stage_wise_compilation()
        else:
            print(f"Option {self.experimental_settings['opt']} is not registered!")

    # Stage-wise protocol
    def stage_wise_compilation(self):
        """
        It gathers the fold results of the stage-wise protocol (step 5, option STAGEWISE) of all datasets,
        and computes the dispersion over the folds and the statistical tests.
        The folds and trials come from the environment file (the legacy CONFORMITY compilation used
        Constants.K_FOLDS_VALUE = 3 and ignored folds 4 and 5).
        """
        settings = self.experimental_settings
        distribution = settings['distribution'][0]
        distribution_class = settings['distribution_class'][0]
        out_dir = "/".join([PathDirFile.RESULTS_DIR, "stagewise"])
        os.makedirs(out_dir, exist_ok=True)

        raw_list, calibration_list, selection_rows, scalability_list, trace_list = [], [], [], [], []
        labels_by_fold = {}
        missing = []
        for dataset in settings['dataset']:
            for model in settings['conformity']:
                model_dir = stage_wise_dir(
                    settings['experiment_name'], dataset, settings['split_methodology'], distribution_class,
                    distribution, model
                )
                for trial in settings['trial']:
                    for fold in settings['fold']:
                        metric_file = os.path.join(model_dir, f"trial-{trial}_fold-{fold}.csv")
                        labels_file = os.path.join(model_dir, f"trial-{trial}_fold-{fold}_labels.csv")
                        if not os.path.exists(metric_file):
                            missing.append(metric_file)
                            continue
                        raw_list.append(pd.read_csv(metric_file))
                        labels_by_fold.setdefault((dataset, model), {})[fold] = pd.read_csv(labels_file)

                selection = SaveAndLoad.load_hyperparameters_conformity(
                    experiment_name=settings['experiment_name'], dataset=dataset,
                    split_methodology=settings['split_methodology'], cluster=model,
                    distribution=distribution, distribution_class=distribution_class
                )
                selection_rows.append({
                    'DATASET': dataset, 'MODEL': model, 'PARAMS': selection['params'],
                    'SEARCH_SILHOUETTE': selection.get('silhouette'),
                    'SEARCH_SILHOUETTE_SD': selection.get('silhouette_std'),
                    'N_EVALUATED': selection.get('n_evaluated'), 'N_ELIGIBLE': selection.get('n_eligible'),
                    'SEARCH_TIME': selection.get('search_time'),
                })
                try:
                    trace = SaveAndLoad.load_conformity_search_trace(
                        experiment_name=settings['experiment_name'], dataset=dataset,
                        split_methodology=settings['split_methodology'], cluster=model,
                        distribution=distribution, distribution_class=distribution_class
                    )
                except FileNotFoundError:
                    missing.append(f"search trace of {model} on {dataset}")
                    continue
                trace.insert(0, 'MODEL', model)
                trace.insert(0, 'DATASET', dataset)
                trace_list.append(trace)

            calibration_dir = stage_wise_dir(
                settings['experiment_name'], dataset, settings['split_methodology'], distribution_class,
                distribution, "CALIBRATION_CHECK"
            )
            for recommender in settings['recommender']:
                for trial in settings['trial']:
                    for fold in settings['fold']:
                        path = os.path.join(calibration_dir, f"{recommender}_trial-{trial}_fold-{fold}.csv")
                        if os.path.exists(path):
                            calibration_list.append(pd.read_csv(path))
                        else:
                            missing.append(path)

            scalability_file = os.path.join(stage_wise_dir(
                settings['experiment_name'], dataset, settings['split_methodology'], distribution_class,
                distribution, "SCALABILITY"
            ), "scalability.csv")
            if os.path.exists(scalability_file):
                scalability_list.append(pd.read_csv(scalability_file))

        if missing:
            print(f"WARNING: {len(missing)} result files are missing, e.g. {missing[:3]}")

        raw = pd.concat(raw_list, ignore_index=True)
        raw.to_csv(os.path.join(out_dir, "stagewise_raw.csv"), index=False)
        sws.summarize(raw).to_csv(os.path.join(out_dir, "stagewise_summary.csv"), index=False)
        sws.contrasts(raw).to_csv(os.path.join(out_dir, "stagewise_contrasts.csv"), index=False)
        sws.flatness(raw).to_csv(os.path.join(out_dir, "stagewise_flatness.csv"), index=False)
        ranks, tests = sws.model_ranking(
            raw, stages=['CAND', 'C@0.0', 'C@0.1', 'C@1.0'], measures=['ARI', 'AMI', 'BCUBED_F1', 'SILHOUETTE']
        )
        ranks.to_csv(os.path.join(out_dir, "stagewise_model_ranks.csv"), index=False)
        tests.to_csv(os.path.join(out_dir, "stagewise_model_friedman.csv"), index=False)
        sws.reproducibility(labels_by_fold).to_csv(os.path.join(out_dir, "stagewise_reproducibility.csv"), index=False)
        pd.DataFrame(selection_rows).to_csv(os.path.join(out_dir, "labeling_selection.csv"), index=False)
        pd.concat(trace_list, ignore_index=True).to_csv(os.path.join(out_dir, "labeling_search_trace.csv"), index=False)
        if calibration_list:
            calibration = pd.concat(calibration_list, ignore_index=True)
            calibration.to_csv(os.path.join(out_dir, "calibration_check_raw.csv"), index=False)
            sws.calibration_summary(calibration).to_csv(
                os.path.join(out_dir, "calibration_check_summary.csv"), index=False
            )
        if scalability_list:
            pd.concat(scalability_list, ignore_index=True).to_csv(
                os.path.join(out_dir, "scalability.csv"), index=False
            )
        interpretation = pd.concat([
            structure_interpretation(
                experiment_name=settings['experiment_name'], split_methodology=settings['split_methodology'],
                dataset=dataset, trial=settings['trial'][0], folds=settings['fold'],
                recommenders=settings['recommender'], weights=settings['weight'], distribution=distribution,
                distribution_class=distribution_class, tradeoff=settings['tradeoff'][0],
                fairness=settings['fairness'][0], relevance=settings['relevance'][0], selector=settings['selector'][0]
            ) for dataset in settings['dataset']
        ], ignore_index=True)
        interpretation.to_csv(os.path.join(out_dir, "structure_interpretation.csv"), index=False)
        dispersion = pd.concat([
            stage_dispersion(
                experiment_name=settings['experiment_name'], split_methodology=settings['split_methodology'],
                dataset=dataset, trial=settings['trial'][0], folds=settings['fold'],
                recommenders=settings['recommender'], weights=settings['weight'], distribution=distribution,
                distribution_class=distribution_class, tradeoff=settings['tradeoff'][0],
                fairness=settings['fairness'][0], relevance=settings['relevance'][0], selector=settings['selector'][0]
            ) for dataset in settings['dataset']
        ], ignore_index=True)
        dispersion.to_csv(os.path.join(out_dir, "stage_dispersion.csv"), index=False)
        print(f"Stage-wise results saved in {out_dir}")

    # Conformity parallelization
    def load_conformity_metric_jaccard(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
            experiment_name: str, split_methodology: str, distribution_class: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    experiment_name=experiment_name, split_methodology=split_methodology,
                    distribution_class=distribution_class,
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.JACCARD_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_cand_items_list.append(metric_df.iloc[0][Label.JACCARD_SCORE])
                users_rec_lists_list.append(metric_df.iloc[1][Label.JACCARD_SCORE])

        merged_metrics_df = DataFrame([
            [mean(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mean(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]
        ],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_jaccard_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
            experiment_name: str, split_methodology: str, distribution_class: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        jaccard_df = self.load_conformity_metric_jaccard(
            experiment_name=experiment_name, split_methodology=split_methodology,
            distribution_class=distribution_class,
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        jaccard_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return jaccard_df

    def load_conformity_metric_silhouette(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
            experiment_name: str, split_methodology: str, distribution_class: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []
        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    experiment_name=experiment_name, split_methodology=split_methodology,
                    distribution_class=distribution_class,
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.SILHOUETTE_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(metric_df.iloc[0][Label.SILHOUETTE_SCORE])
                users_cand_items_list.append(metric_df.iloc[1][Label.SILHOUETTE_SCORE])
                users_rec_lists_list.append(metric_df.iloc[2][Label.SILHOUETTE_SCORE])

        merged_metrics_df = DataFrame([
            [mean(users_pref_list), Label.USERS_PREF,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mean(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mean(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_silhouette_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
            experiment_name: str, split_methodology: str, distribution_class: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        silhouette_df = self.load_conformity_metric_silhouette(
            experiment_name=experiment_name, split_methodology=split_methodology,
            distribution_class=distribution_class,
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        silhouette_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return silhouette_df

    def load_conformity_metric_label(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
            experiment_name: str, split_methodology: str, distribution_class: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []
        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    experiment_name=experiment_name, split_methodology=split_methodology,
                    distribution_class=distribution_class,
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.LABEL_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(metric_df.iloc[0][Label.LABEL_SCORE])
                users_cand_items_list.append(metric_df.iloc[1][Label.LABEL_SCORE])
                users_rec_lists_list.append(metric_df.iloc[2][Label.LABEL_SCORE])

        merged_metrics_df = DataFrame([
            [mode(users_pref_list), Label.USERS_PREF,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_labels_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
            experiment_name: str, split_methodology: str, distribution_class: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        label_df = self.load_conformity_metric_label(
            experiment_name=experiment_name, split_methodology=split_methodology,
            distribution_class=distribution_class,
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        label_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return label_df

    def load_conformity_metric_davies(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
            experiment_name: str, split_methodology: str, distribution_class: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []
        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    experiment_name=experiment_name, split_methodology=split_methodology,
                    distribution_class=distribution_class,
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.DAVIES_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(metric_df.iloc[0][Label.DAVIES_SCORE])
                users_cand_items_list.append(metric_df.iloc[1][Label.DAVIES_SCORE])
                users_rec_lists_list.append(metric_df.iloc[2][Label.DAVIES_SCORE])

        merged_metrics_df = DataFrame([
            [mode(users_pref_list), Label.USERS_PREF,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_davies_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
            experiment_name: str, split_methodology: str, distribution_class: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        label_df = self.load_conformity_metric_davies(
            experiment_name=experiment_name, split_methodology=split_methodology,
            distribution_class=distribution_class,
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        label_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return label_df

    def load_conformity_metric_calinski(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
            experiment_name: str, split_methodology: str, distribution_class: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []
        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    experiment_name=experiment_name, split_methodology=split_methodology,
                    distribution_class=distribution_class,
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.CALINSKI_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(metric_df.iloc[0][Label.CALINSKI_SCORE])
                users_cand_items_list.append(metric_df.iloc[1][Label.CALINSKI_SCORE])
                users_rec_lists_list.append(metric_df.iloc[2][Label.CALINSKI_SCORE])

        merged_metrics_df = DataFrame([
            [mode(users_pref_list), Label.USERS_PREF,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_calinski_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
            experiment_name: str, split_methodology: str, distribution_class: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        label_df = self.load_conformity_metric_calinski(
            experiment_name=experiment_name, split_methodology=split_methodology,
            distribution_class=distribution_class,
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        label_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return label_df

    def conformity_parallelization(self):
        """
        TODO: Docstring
        """
        for dataset in self.experimental_settings['dataset']:
            combination = [
                [self.experimental_settings['experiment_name']], [self.experimental_settings["split_methodology"]],
                self.experimental_settings['recommender'],
                self.experimental_settings['distribution'], self.experimental_settings['fairness'],
                self.experimental_settings['relevance'], self.experimental_settings['weight'],
                self.experimental_settings['tradeoff'], self.experimental_settings['selector'],
                self.experimental_settings["distribution_class"]
            ]

            # Jaccard
            jaccard_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_jaccard_metric)(
                    experiment_name=experiment_name, split_methodology=split_methodology,
                    distribution_class=distribution_class,
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for experiment_name, split_methodology, recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector, distribution_class
                in list(itertools.product(*combination))
            )
            jaccard_results = pd.concat(jaccard_output)
            print(jaccard_results)
            SaveAndLoad.save_compiled_metric(
                data=jaccard_results, dataset=dataset, metric=Label.JACCARD_SCORE
            )

            # Silhouette
            silhouette_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_silhouette_metric)(
                    experiment_name=experiment_name, split_methodology=split_methodology,
                    distribution_class=distribution_class,
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for experiment_name, split_methodology, recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector, distribution_class
                in list(itertools.product(*combination))
            )
            silhouette_results = pd.concat(silhouette_output)
            print(silhouette_results)
            SaveAndLoad.save_compiled_metric(
                data=silhouette_results, dataset=dataset, metric=Label.SILHOUETTE_SCORE
            )

            # Labels
            label_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_labels_metric)(
                    experiment_name=experiment_name, split_methodology=split_methodology,
                    distribution_class=distribution_class,
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for experiment_name, split_methodology, recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector, distribution_class
                in list(itertools.product(*combination))
            )
            label_results = pd.concat(label_output)
            print(label_results)
            SaveAndLoad.save_compiled_metric(
                data=label_results, dataset=dataset, metric=Label.LABEL_SCORE
            )

            # Davies
            davies_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_davies_metric)(
                    experiment_name=experiment_name, split_methodology=split_methodology,
                    distribution_class=distribution_class,
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for experiment_name, split_methodology, recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector, distribution_class
                in list(itertools.product(*combination))
            )
            davies_results = pd.concat(davies_output)
            print(davies_results)
            SaveAndLoad.save_compiled_metric(
                data=davies_results, dataset=dataset, metric=Label.DAVIES_SCORE
            )

            # Calinski
            calinski_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_calinski_metric)(
                    experiment_name=experiment_name, split_methodology=split_methodology,
                    distribution_class=distribution_class,
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for experiment_name, split_methodology, recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector, distribution_class
                in list(itertools.product(*combination))
            )
            calinski_results = pd.concat(calinski_output)
            print(calinski_results)
            SaveAndLoad.save_compiled_metric(
                data=calinski_results, dataset=dataset, metric=Label.CALINSKI_SCORE
            )

    # Metrics Parallelization

    def recommender_metrics_parallelization(self):
        """
        TODO: Docstring
        """
        for dataset in self.experimental_settings['dataset']:
            for metric in self.experimental_settings['metric']:
                combination = [
                    [self.experimental_settings['experiment_name']], [self.experimental_settings["split_methodology"]],
                    self.experimental_settings['recommender'],
                    self.experimental_settings['distribution'], self.experimental_settings['fairness'],
                    self.experimental_settings['relevance'], self.experimental_settings['weight'],
                    self.experimental_settings['tradeoff'], self.experimental_settings['selector'],
                    self.experimental_settings["distribution_class"]
                ]
                process_combination = list(itertools.product(*combination))
                print(f"Dataset {dataset} with metric {metric} starts {len(process_combination)} processes.")
                output = Parallel(
                    n_jobs=self.experimental_settings['n_jobs'], verbose=10, batch_size=1,
                    backend="multiprocessing", prefer="processes"
                )(
                    delayed(MetricComprises.it_comprises_recommender_metric)(
                        experiment_name=experiment_name, split_methodology=split_methodology,
                        recommender=recommender, dataset=dataset, metric=metric,
                        trial=self.experimental_settings['trial'], fold=self.experimental_settings['fold'],
                        distribution=distribution, fairness=fairness, relevance=relevance,
                        weight=weight, tradeoff=tradeoff, selector=selector,
                        distribution_class=distribution_class
                    ) for experiment_name, split_methodology, recommender, distribution, fairness, relevance, weight, tradeoff, selector, distribution_class
                    in process_combination
                )

                results = pd.concat(output)

                SaveAndLoad.save_compiled_metric(
                    data=results, dataset=dataset, metric=metric
                )


if __name__ == '__main__':
    """
    Starting the decision protocol
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep6()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
