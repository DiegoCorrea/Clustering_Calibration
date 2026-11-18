import logging
import os

import pandas as pd

from evaluations.hypothesis import welch
from graphics import stage_wise_charts, stage_wise_tables
from settings.path_dir_file import PathDirFile
from graphics.conformity import ConformityGraphics
from graphics.recommender import SingleRecommenderGraphics, WelchHypothesisTestGraphics
# from evaluations.best_worst import best_and_worst_systems, best_and_worst_fairness_measure
# from evaluations.hypothesis import welch
# from graphics.research_questions.perspectives import components_box_graphic, fairness_box_graphic
from settings.labels import Label

from settings.save_and_load import SaveAndLoad

from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep7(Step):
    """
    TODO: Docstring
    """

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = Input.step7()
        logger.info(self.experimental_settings)

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

    def main(self):
        """
        TODO: Docstring
        """

        if self.experimental_settings['opt'] == Label.STAGEWISE:
            self.stage_wise_outputs()
        elif self.experimental_settings['opt'] == Label.CONFORMITY and self.experimental_settings['view'] == Label.DATASET_CHART:
            self.conformity_charts()
        elif self.experimental_settings['opt'] == Label.CONFORMITY and self.experimental_settings['view'] == Label.DATASET_ANALYZE:
            self.conformity_analyses()
        elif self.experimental_settings['opt'] == Label.EVALUATION_METRICS and self.experimental_settings['view'] == Label.DATASET_CHART:
            self.recommender_charts()
        elif self.experimental_settings['opt'] == Label.EVALUATION_METRICS and self.experimental_settings['view'] == Label.DATASET_ANALYZE:
            self.recommender()
        else:
            print(f"Option or View is not allowed to use yet! ... {self.experimental_settings['opt']} ... {self.experimental_settings['view']}")

    def stage_wise_outputs(self):
        """
        Figures and LaTeX tables of the stage-wise protocol, from the files compiled by step 6.
        """
        settings = self.experimental_settings
        source = os.path.join(PathDirFile.RESULTS_DIR, "stagewise")
        target = os.path.join(PathDirFile.RESULTS_DIR, settings.get('output_dir', 'paper_revision'))
        figures = os.path.join(target, "figures")
        tables = os.path.join(target, "tables")
        datasets = settings['dataset']
        recommenders = settings['recommender']

        def load(name):
            path = os.path.join(source, name)
            return pd.read_csv(path) if os.path.exists(path) else None

        raw = load("stagewise_raw.csv")
        summary = load("stagewise_summary.csv")
        reproducibility = load("stagewise_reproducibility.csv")
        selection = load("labeling_selection.csv")
        families = {
            family: [m for m in stage_wise_tables.MODEL_ORDER if (raw["MODEL"] == m).any()
                     and raw.loc[raw["MODEL"] == m, "FAMILY"].iloc[0] == family]
            for family in stage_wise_charts.FAMILY_ORDER
        }

        # Figures
        stage_wise_charts.stage_lines_by_family(
            raw, "ARI", "ARI with PREF groups", os.path.join(figures, "stage_ari_family.pdf"), datasets, recommenders,
            reference_line=0.0
        )
        stage_wise_charts.stage_lines_by_family(
            raw, "SILHOUETTE", "Silhouette", os.path.join(figures, "stage_silhouette_family.pdf"), datasets,
            recommenders, reference_line=0.0
        )
        stage_wise_charts.stage_lines_by_family(
            raw, "BCUBED_F1", "B-cubed F1", os.path.join(figures, "stage_bcubed_family.pdf"), datasets, recommenders
        )
        for dataset in datasets:
            for measure, label in [("ARI", "ARI with PREF groups"), ("SILHOUETTE", "Silhouette")]:
                stage_wise_charts.stage_lines_by_model(
                    raw, measure, label, os.path.join(figures, f"stage_{measure.lower()}_models_{dataset}.pdf"),
                    dataset, recommenders, families
                )
        stage_wise_charts.description_vs_stability(
            summary, reproducibility, os.path.join(figures, "description_vs_stability.pdf"), datasets
        )
        stage_wise_charts.independent_vs_projected(raw, os.path.join(figures, "independent_label_jaccard.pdf"))
        trace = load("labeling_search_trace.csv")
        if trace is not None:
            stage_wise_charts.search_trace(
                trace, os.path.join(figures, "search_silhouette.pdf"), datasets,
                families["PARTITION"] + families["MIXTURE"]
            )
        calibration = load("calibration_check_summary.csv")
        if calibration is not None:
            stage_wise_charts.calibration_check(
                calibration, os.path.join(figures, "calibration_check.pdf"), datasets, recommenders
            )
        scalability = load("scalability.csv")
        if scalability is not None:
            stage_wise_charts.scalability(scalability, os.path.join(figures, "scalability.pdf"), families)

        # Tables
        stage_wise_tables.structure_table(raw, selection, reproducibility, datasets,
                                          os.path.join(tables, "structure.tex"))
        stage_wise_tables.stage_agreement_table(raw, datasets, os.path.join(tables, "stage_ari.tex"))
        stage_wise_tables.family_stage_table(
            raw, "SILHOUETTE", datasets, os.path.join(tables, "stage_silhouette_family.tex"),
            "tab:stage_silhouette",
            "Silhouette of every stage with the reference groups (mean over the models of each family and the "
            "three recommenders $\\pm$ half-width of the 95\\% confidence interval over the five folds)."
        )
        stage_wise_tables.contrast_table(load("stagewise_contrasts.csv"), datasets, os.path.join(tables, "contrasts.tex"))
        stage_wise_tables.flatness_table(load("stagewise_flatness.csv"), datasets, os.path.join(tables, "flatness.tex"))
        stage_wise_tables.ranking_table(load("stagewise_model_ranks.csv"), load("stagewise_model_friedman.csv"),
                                        os.path.join(tables, "ranking.tex"))
        if calibration is not None:
            stage_wise_tables.calibration_table(calibration, datasets, recommenders,
                                                os.path.join(tables, "calibration_check.tex"))
        if scalability is not None:
            stage_wise_tables.scalability_table(scalability, os.path.join(tables, "scalability.tex"))
        interpretation = load("structure_interpretation.csv")
        if interpretation is not None:
            stage_wise_tables.interpretation_table(interpretation, datasets, recommenders,
                                                   os.path.join(tables, "interpretation.tex"))
        print(f"Figures and tables saved in {target}")

    def conformity_charts(self):
        for dataset_name in self.experimental_settings['dataset']:
            print("|" * 100)
            print("-"*45, " ", dataset_name, " ", "-"*45)
            print("|" * 100)

            print("Silhouette Lines: Weight by Conformity Algorithms")
            results = SaveAndLoad.load_compiled_metric(
                dataset=dataset_name, metric=Label.SILHOUETTE_SCORE
            )
            for recommender in self.experimental_settings['recommender']:
                results_by_recommender = results[results['RECOMMENDER'] == recommender].copy()
                ConformityGraphics.weight_by_metric_line(
                    data=results_by_recommender, dataset_name=dataset_name, conformity_algos=Label.REGISTERED_UNSUPERVISED,
                    y_label="Silhouette Value", metric_name=Label.SILHOUETTE_SCORE, recommender=recommender
                )

            print("Davis Lines: Weight by Conformity Algorithms")
            results = SaveAndLoad.load_compiled_metric(
                dataset=dataset_name, metric=Label.DAVIES_SCORE
            )
            for recommender in self.experimental_settings['recommender']:
                results_by_recommender = results[results['RECOMMENDER'] == recommender].copy()
                ConformityGraphics.weight_by_metric_line(
                    data=results_by_recommender, dataset_name=dataset_name, conformity_algos=Label.REGISTERED_UNSUPERVISED,
                    y_label="Davis-Bouldin Value", metric_name=Label.DAVIES_SCORE, recommender=recommender
                )

            print("Calinski-Harabasz Lines: Weight by Conformity Algorithms")
            results = SaveAndLoad.load_compiled_metric(
                dataset=dataset_name, metric=Label.CALINSKI_SCORE
            )
            for recommender in self.experimental_settings['recommender']:
                results_by_recommender = results[results['RECOMMENDER'] == recommender].copy()
                ConformityGraphics.weight_by_metric_line(
                    data=results_by_recommender, dataset_name=dataset_name, conformity_algos=Label.REGISTERED_UNSUPERVISED,
                    y_label="Calinski-Harabasz Value", metric_name=Label.CALINSKI_SCORE, recommender=recommender
                )

            print("Jaccard Lines: Weight by Conformity Algorithms")

            jaccard_results = SaveAndLoad.load_compiled_metric(
                dataset=dataset_name, metric=Label.JACCARD_SCORE
            )
            for recommender in self.experimental_settings['recommender']:
                results_by_recommender = jaccard_results[jaccard_results['RECOMMENDER'] == recommender].copy()
                ConformityGraphics.weight_by_jaccard_line(
                    data=results_by_recommender[results_by_recommender[Label.CONFORMITY_DIST_MEANING] == Label.USERS_REC_LISTS],
                    dataset_name=dataset_name, conformity_algos=Label.REGISTERED_UNSUPERVISED,
                    rule=Label.USERS_REC_LISTS, recommender=recommender
                )

    def conformity_analyses(self):
        for dataset_name in self.experimental_settings['dataset']:
            print("|" * 100)
            print("-"*10, " ", dataset_name, " ", "-"*10)
            print("|" * 100)

            jaccard_results = SaveAndLoad.load_compiled_metric(
                dataset=dataset_name, metric=Label.JACCARD_SCORE
            )
            print(jaccard_results)

            silhlouete_results = SaveAndLoad.load_compiled_metric(
                dataset=dataset_name, metric=Label.SILHOUETTE_SCORE
            )
            print(silhlouete_results)

    def distribution_as_goal(self, data, dataset, metric, graphic_name):
        for recommender in self.experimental_settings['recommender']:
            for tradeoff in self.experimental_settings['tradeoff']:
                for fairness in self.experimental_settings['fairness']:
                    for relevance in self.experimental_settings['relevance']:
                        for selector in self.experimental_settings['selector']:
                            for graphic in self.experimental_settings['type']:
                                temp_graphic_name = "_".join([graphic, graphic_name, recommender, tradeoff, fairness, relevance, selector])
                                print(temp_graphic_name)
                                filtered_data = data[
                                    (data[Label.RECOMMENDER] == recommender) &
                                    (data[Label.TRADEOFF] == tradeoff) &
                                    (data[Label.CALIBRATION_MEASURE_LABEL] == fairness) &
                                    (data[Label.RELEVANCE] == relevance) &
                                    (data[Label.SELECTOR_LABEL] == selector)
                                ].copy()
                                if len(filtered_data) > 0:
                                    if graphic == Label.GRAPHIC_LINE_TYPE:
                                        SingleRecommenderGraphics.line_generic_weight_by_metric(
                                            data=filtered_data, y_label=metric + " Values", metric=metric, dataset=dataset,
                                            goal_label=Label.DISTRIBUTION_LABEL,
                                            goal_list=self.experimental_settings['distribution'],
                                            graphic_name=temp_graphic_name
                                        )
                                    elif graphic == Label.GRAPHIC_HEAT_WELCH_TYPE:
                                        WelchHypothesisTestGraphics.heatmap(
                                            data=filtered_data, metric=metric, dataset=dataset,
                                            goal_label=Label.DISTRIBUTION_LABEL,
                                            goal_list=self.experimental_settings['distribution'],
                                            graphic_name=temp_graphic_name
                                        )
                                    else:
                                        pass

    def recommender_charts(self):
        for dataset in self.experimental_settings['dataset']:
            for metric in self.experimental_settings['metric']:
                for goal in self.experimental_settings['goal']:
                    graphic_name = "_".join([dataset, metric, goal])
                    data = SaveAndLoad.load_compiled_metric(
                        dataset=dataset, metric=metric
                    )
                    if goal == Label.DISTRIBUTION_LABEL:
                        self.distribution_as_goal(
                            data=data, dataset=dataset, metric=metric, graphic_name=graphic_name
                        )
                    else:
                        pass

    def old_recommender_charts(self):
        for metric in [Label.MAP, Label.MRR, Label.MACE]:
            all_results = {}
            for dataset_name in self.experimental_settings['dataset']:
                print("|" * 100)
                print("-"*45, " ", dataset_name, " ", "-"*45)
                print("|" * 100)

                print("Recommender dataset: Weight by Conformity Algorithms")

                results = SaveAndLoad.load_compiled_metric(
                    dataset=dataset_name, metric=metric
                )
                all_results[dataset_name] = results[results[Label.TRADEOFF_WEIGHT_LABEL].isin(Label.CONST_WEIGHT)]
                SingleRecommenderGraphics.weight_by_metric_histogram(
                    data=results[results[Label.TRADEOFF_WEIGHT_LABEL].isin(Label.CONST_WEIGHT)],
                    dataset_name=dataset_name, metric=metric
                )
            SingleRecommenderGraphics.weight_by_metric_all_datasets(
                data=all_results, y_label=metric + " Values", metric=metric
            )
            SingleRecommenderGraphics.weight_by_metric_all_datasets(
                data=all_results, y_label=metric + " Values", metric=metric
            )

    #     if setup_config['opt'] == "CHART":
    #         charts(setup_config)
    #     elif setup_config['opt'] == "ANALYZE":
    #         analyses(setup_config)
    #     else:
    #         pass
    #
    # def charts(self):
    #     for dataset_name in setup_config['dataset']:
    #         data = pd.read_csv(PathDirFile.get_decision_file(dataset_name))
    #
    #         # Ranking
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MAP")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MAP")
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRR")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRR")
    #
    #         # Calibration
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MACE")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MACE")
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRMC")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRMC")
    #
    #         # Coefficients
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CCE")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CCE")
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CMC")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CMC")
    #
    #         # Coefficients
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="PERFORMANCE")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="PERFORMANCE")

    # def analyses(self):
    #     for dataset_name in setup_config['dataset']:
    #         print("|" * 100)
    #         print("-"*10, " ", dataset_name, " ", "-"*10)
    #         print("|" * 100)
    #
    #         results = pd.read_csv(PathDirFile.get_decision_file(dataset_name))
    #         results.fillna(0.0)
    #
    #         # execution_time_analyze(data=results)
    #         best_and_worst_systems(data=results, order_by_metric='MAP', ascending=False)
    #         best_and_worst_systems(data=results, order_by_metric='MRR', ascending=False)
    #         best_and_worst_systems(data=results, order_by_metric='MACE', ascending=True)
    #         best_and_worst_systems(data=results, order_by_metric='CCE', ascending=True)
    #
    #         best_and_worst_fairness_measure(data=results, order_by_metric='MAP', ascending=False)
    #         best_and_worst_fairness_measure(data=results, order_by_metric='MRR', ascending=False)
    #         best_and_worst_fairness_measure(data=results, order_by_metric='MACE', ascending=True)
    #         best_and_worst_fairness_measure(data=results, order_by_metric='CCE', ascending=True)

            welch(data=results, order_by_metric='MAP', ascending=False)
            welch(data=results, order_by_metric='MRR', ascending=False)
            welch(data=results, order_by_metric='MACE', ascending=True)
            welch(data=results, order_by_metric='CCE', ascending=True)

            print("|" * 100)


if __name__ == '__main__':
    """
    Starting the Charts, Tables and H-Test production
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep7()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
