"""
Same runs as shell/step4_genre_fix.sh (environment/step4/movies_cross_genre_fix.json), executed dataset by
dataset and skipping the lists that were already regenerated after the step-1 fix (newer than the
regenerated preference distributions). The C@0.0 lists are also skipped: with lambda = 0 the
list only depends on the relevance, not on the target distribution, and the regenerated C@0.0 lists
were verified to be identical to the original ones. Run from the project root:
    OMP_NUM_THREADS=1 python3 shell/step4_genre_fix_ordered.py
"""
import json
import multiprocessing
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(ROOT, "pierre_in_frame"))
sys.path.insert(0, os.getcwd())

from settings.path_dir_file import PathDirFile  # noqa: E402
from settings.save_and_load import SaveAndLoad  # noqa: E402
from step4_postprocessing import starting_postprocessing  # noqa: E402

if __name__ == '__main__':
    cfg = SaveAndLoad.load_step_file(step="step4", file_name="movies_cross_genre_fix")
    # Modification time of the regenerated (genre) preference distribution of the first dataset.
    cutoff = os.path.getmtime(PathDirFile.dataset_distribution_path(
        dataset=cfg["dataset"][0], experiment_name=cfg["experiment_name"],
        split_methodology=cfg["split_methodology"], fold=1, trial=1,
        distribution_class=cfg["distribution_class"][0], filename=cfg["distribution"][0] + ".csv"
    ))
    tasks = []
    for dataset in cfg["dataset"]:
        for recommender in cfg["recommender"]:
            for fold in cfg["fold"]:
                for weight in cfg["weight"]:
                    if weight == "C@0.0":
                        continue
                    path = PathDirFile.get_recommendation_list_file(
                        experiment_name=cfg["experiment_name"], split_methodology=cfg["split_methodology"],
                        dataset=dataset, recommender=recommender, trial=1, fold=fold,
                        tradeoff=cfg["tradeoff"][0], distribution=cfg["distribution"][0],
                        fairness=cfg["fairness"][0], relevance=cfg["relevance"][0],
                        tradeoff_weight=weight, select_item=cfg["selector"][0],
                        distribution_class=cfg["distribution_class"][0]
                    )
                    if os.path.exists(path) and os.path.getmtime(path) > cutoff:
                        continue
                    tasks.append((
                        cfg["experiment_name"], cfg["split_methodology"], recommender, fold, 1, dataset,
                        cfg["tradeoff"][0], cfg["distribution"][0], cfg["fairness"][0], cfg["relevance"][0],
                        weight, cfg["selector"][0], cfg["list_size"][0], cfg["alpha"][0], cfg["d"][0],
                        "NO", cfg["distribution_class"][0]
                    ))
    print(f"Lists to regenerate: {len(tasks)}", flush=True)
    with multiprocessing.Pool(processes=cfg["n_jobs"]) as pool:
        pool.starmap(starting_postprocessing, tasks, chunksize=1)
