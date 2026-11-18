"""
LaTeX tables (booktabs) of the stage-wise structural evaluation (step 7, option STAGEWISE).
Every number comes from the files compiled by step 6 in results/stagewise/.
"""
import ast
import os

import numpy as np
import pandas as pd

from evaluations.stage_wise_statistics import mean_ci, ALPHA
from graphics.stage_wise_charts import DATASET_LABEL, MODEL_LABEL, FAMILY_STYLE, FAMILY_ORDER

MODEL_ORDER = ["KMEANS", "BISECTING", "AGGLO", "SPECTRAL", "BIRCH", "FCM", "GAUSSIAN", "BAYESIAN",
               "DBSCAN", "OPTICS", "IF", "LOF", "OSVM", "SGD", "ENVELOPE"]
PARAM_LABEL = {"n_clusters": "k", "n_components": "k", "eps": "\\varepsilon", "min_samples": "m",
               "metric": "", "n_estimators": "T", "contamination": "c", "nu": "\\nu", "n_neighbors": "n"}
METRIC_SHORT = {"euclidean": "eucl.", "manhattan": "manh.", "chebyshev": "cheb.", "cosine": "cos.",
                "braycurtis": "bray.", "canberra": "canb.", "correlation": "corr."}


def write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def fmt(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.{digits}f}".replace("-", "$-$") if value < 0 else f"{value:.{digits}f}"


def fmt_ci(values, digits: int = 2) -> str:
    mean, _, low, high, n = mean_ci(values)
    if n == 0:
        return "--"
    if n == 1 or np.isnan(low):
        return fmt(mean, digits)
    return f"{fmt(mean, digits)}$\\pm${(high - mean):.{digits}f}"


def fmt_params(params) -> str:
    if isinstance(params, str):
        params = ast.literal_eval(params)
    parts = []
    for key, value in params.items():
        if key == "metric":
            parts.append(METRIC_SHORT.get(value, value))
        elif key == "contamination" or key == "nu":
            parts.append(f"${PARAM_LABEL[key]}{{=}}{value:g}$")
        else:
            parts.append(f"${PARAM_LABEL.get(key, key)}{{=}}{value:g}$")
    return ", ".join(parts)


def recommender_mean_per_fold(raw: pd.DataFrame, stage: str, measure: str, dataset: str, model: str) -> np.ndarray:
    data = raw[(raw["LABELS"] == "PROJECTED") & (raw["STAGE"] == stage) & (raw["DATASET"] == dataset)
               & (raw["MODEL"] == model)]
    return data.groupby("FOLD")[measure].mean().values


def structure_table(raw: pd.DataFrame, selection: pd.DataFrame, reproducibility: pd.DataFrame, datasets: list,
                    path: str):
    """
    RQ1: structure found on the preference distribution (labels found by each model).
    """
    pref = raw[(raw["LABELS"] == "FIT") & (raw["STAGE"] == "PREF")]
    repro = reproducibility[reproducibility["LABELS"] == "FIT_LABEL"].set_index(["DATASET", "MODEL"])
    sel = selection.set_index(["DATASET", "MODEL"])
    lines = [
        "\\begin{table*}[!t]", "\\centering",
        "\\caption{Structure of the preference distributions (PREF) found by each labeling model with the selected "
        "configuration. \\#G: number of groups (noise excluded); Eff.: effective number of groups "
        "($\\exp$ of the entropy of the group sizes); Max: share of the largest group; Flag.: share of users "
        "flagged as noise (density) or atypical (detectors); SC: silhouette coefficient; Rep.: reproducibility, "
        "the ARI between the PREF labelings of two folds (mean over the 10 pairs of folds). Only SC is reported as mean "
        "$\\pm$ half-width of the 95\\% confidence interval over the five folds, approximate because the folds share "
        "interactions (Section~\\ref{sec:statistics}); the other columns are means only.}",
        "\\label{tab:structure}", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Model & Configuration & \\#G & Eff. & Max & Flag. & SC & Rep. \\\\", "\\midrule",
    ]
    for dataset in datasets:
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{DATASET_LABEL[dataset]}}}}} \\\\")
        for model in MODEL_ORDER:
            data = pref[(pref["DATASET"] == dataset) & (pref["MODEL"] == model)]
            if data.empty:
                continue
            rep = repro.loc[(dataset, model), "CROSS_FOLD_ARI"] if (dataset, model) in repro.index else np.nan
            lines.append(" & ".join([
                MODEL_LABEL[model], fmt_params(sel.loc[(dataset, model), "PARAMS"]),
                fmt(data["N_GROUPS"].mean(), 1), fmt(data["EFFECTIVE_GROUPS"].mean(), 2),
                fmt(data["LARGEST_GROUP_SHARE"].mean(), 2), fmt(data["FLAGGED_SHARE"].mean(), 2),
                fmt_ci(data["SILHOUETTE"].values), fmt(rep, 2),
            ]) + " \\\\")
        lines.append("\\midrule" if dataset != datasets[-1] else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table*}", ""]
    write(path, "\n".join(lines))


def stage_agreement_table(raw: pd.DataFrame, datasets: list, path: str):
    """
    RQ2: group preservation (ARI) at the candidate stage, the uncalibrated list and the calibrated list (lambda = 0.1;
    C@1.0 is equivalent, see the flatness table), mean over the three recommenders.
    """
    stages = ["CAND", "C@0.0", "C@0.1"]
    width = len(stages)
    lines = [
        "\\begin{table*}[!t]", "\\centering",
        "\\caption{Group preservation between PREF and each later stage: ARI (chance-corrected; 0 is the agreement "
        "expected by chance). The users of every stage are projected onto the groups fitted on PREF, so all stages share "
        "the group identities. C@1.0 is equivalent to C@0.1 (Table~\\ref{tab:flatness}). Values are means over the "
        "three recommenders $\\pm$ half-width of the 95\\% confidence interval over the five folds, approximate because the folds "
        "share interactions (Section~\\ref{sec:statistics}); --: no eligible configuration.}",
        "\\label{tab:stage_ari}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l" + "rrr" * len(datasets) + "}", "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{{width}}}{{c}}{{{DATASET_LABEL[d]}}}" for d in datasets) + " \\\\",
        " ".join(f"\\cmidrule(lr){{{2 + width * i}-{1 + width * (i + 1)}}}" for i in range(len(datasets))),
        "Model & " + " & ".join([" & ".join(stages) for _ in datasets]) + " \\\\",
        "\\midrule",
    ]
    for family in FAMILY_ORDER:
        for model in MODEL_ORDER:
            if raw.loc[raw["MODEL"] == model, "FAMILY"].iloc[0] != family:
                continue
            cells = [MODEL_LABEL[model]]
            for dataset in datasets:
                for stage in stages:
                    cells.append(fmt_ci(recommender_mean_per_fold(raw, stage, "ARI", dataset, model)))
            lines.append(" & ".join(cells) + " \\\\")
        if family != FAMILY_ORDER[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(path, "\n".join(lines))


def family_stage_table(raw: pd.DataFrame, measure: str, datasets: list, path: str, label: str, caption: str):
    """
    Mean of the measure over the models of each family, per stage (mean over recommenders, CI over folds).
    """
    stages = ["PREF", "CAND", "C@0.0", "C@0.1", "C@0.5", "C@1.0"]
    projected = raw[raw["LABELS"] == "PROJECTED"]
    lines = [
        "\\begin{table}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{ll" + "r" * len(stages) + "}", "\\toprule",
        "Dataset & Family & " + " & ".join(stages) + " \\\\", "\\midrule",
    ]
    for dataset in datasets:
        for i, family in enumerate(FAMILY_ORDER):
            cells = [DATASET_LABEL[dataset] if i == 0 else "", FAMILY_STYLE[family]["label"]]
            for stage in stages:
                data = projected[(projected["DATASET"] == dataset) & (projected["FAMILY"] == family)
                                 & (projected["STAGE"] == stage)]
                cells.append(fmt_ci(data.groupby("FOLD")[measure].mean().values))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule" if dataset != datasets[-1] else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    write(path, "\n".join(lines))


def contrast_table(contrasts: pd.DataFrame, datasets: list, path: str):
    """
    Number of (model, recommender) cases with a significant decrease/increase (paired t-test over the folds,
    Holm-corrected over the models) and with a difference within +/-0.05 (TOST), per contrast.
    """
    order = ["PREF->CAND", "CAND->C@0.0", "C@0.0->C@0.1", "C@0.1->C@1.0"]
    names = {"PREF->CAND": "PREF$\\rightarrow$CAND", "CAND->C@0.0": "CAND$\\rightarrow$C@0.0",
             "C@0.0->C@0.1": "C@0.0$\\rightarrow$C@0.1", "C@0.1->C@1.0": "C@0.1$\\rightarrow$C@1.0"}
    measures = [("SILHOUETTE", "Silhouette"), ("ARI", "ARI")]
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Statistical evidence for the change between consecutive stages, over the 45 (model, recommender) "
        "cases of each dataset. $\\downarrow$/$\\uparrow$: significant decrease/increase (paired $t$-test over the five "
        "folds, Holm correction over the 15 models, $\\alpha=0.05$); $\\approx$: equivalent within $\\pm 0.05$ (paired "
        "TOST, $\\alpha=0.05$). The remaining cases are inconclusive. The tests are approximate because the folds share "
        "interactions (Section~\\ref{sec:statistics}).}",
        "\\label{tab:contrasts}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{ll" + "rrr" * len(measures) + "}", "\\toprule",
        " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{m[1]}}}" for m in measures) + " \\\\",
        " ".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(measures))),
        "Dataset & Contrast & " + " & ".join("$\\downarrow$ & $\\uparrow$ & $\\approx$" for _ in measures) + " \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        for i, contrast in enumerate(order):
            cells = [DATASET_LABEL[dataset] if i == 0 else "", names[contrast]]
            for measure, _ in measures:
                data = contrasts[(contrasts["DATASET"] == dataset) & (contrasts["CONTRAST"] == contrast)
                                 & (contrasts["MEASURE"] == measure)]
                decrease = ((data["P_HOLM"] < ALPHA) & (data["DIFFERENCE"] < 0)).sum()
                increase = ((data["P_HOLM"] < ALPHA) & (data["DIFFERENCE"] > 0)).sum()
                equivalent = (data["P_TOST"] < ALPHA).sum()
                cells += [str(decrease), str(increase), str(equivalent)]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule" if dataset != datasets[-1] else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    write(path, "\n".join(lines))


def flatness_table(flatness: pd.DataFrame, datasets: list, path: str):
    measures = [("SILHOUETTE", "Silhouette"), ("ARI", "ARI")]
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Does the structure change with the trade-off weight beyond $\\lambda=0.1$? For each of the 45 "
        "(model, recommender) cases: Var.: the Friedman test over $\\lambda\\in\\{0.1,\\dots,1.0\\}$ (blocks: folds; "
        "Holm correction over the models) rejects constancy; Eq.: C@0.1 and C@1.0 are equivalent within $\\pm 0.05$ "
        "(paired TOST); Range: largest difference between the fold means of two values of $\\lambda$ "
        "(median over the cases). The tests are approximate because the folds share interactions "
        "(Section~\\ref{sec:statistics}).}",
        "\\label{tab:flatness}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "rrr" * len(measures) + "}", "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{m[1]}}}" for m in measures) + " \\\\",
        " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(measures))),
        "Dataset & " + " & ".join("Var. & Eq. & Range" for _ in measures) + " \\\\", "\\midrule",
    ]
    for dataset in datasets:
        cells = [DATASET_LABEL[dataset]]
        for measure, _ in measures:
            data = flatness[(flatness["DATASET"] == dataset) & (flatness["MEASURE"] == measure)]
            cells += [str(int(data["VARIES_WITH_LAMBDA"].sum())), str(int(data["EQUIVALENT_C0.1_C1.0"].sum())),
                      fmt(data["RANGE_OVER_LAMBDA"].median(), 3)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(path, "\n".join(lines))


def ranking_table(ranks: pd.DataFrame, tests: pd.DataFrame, path: str):
    columns = [("CAND", "ARI"), ("C@0.0", "ARI"), ("C@1.0", "ARI"), ("C@0.0", "BCUBED_F1"), ("C@0.0", "SILHOUETTE")]
    header = ["CAND", "C@0.0", "C@1.0", "C@0.0", "C@0.0"]
    test_rows = [tests[(tests["STAGE"] == s) & (tests["MEASURE"] == m)].iloc[0] for s, m in columns]
    cd = test_rows[0]["NEMENYI_CD"]
    n_blocks = int(test_rows[0]["N_BLOCKS"])
    names = ["ARI CAND", "ARI C@0.0", "ARI C@1.0", "B$^3$ C@0.0", "SC C@0.0"]
    p_values = ", ".join(
        f"{h}: $p<0.001$" if r['FRIEDMAN_P'] < 0.001 else f"{h}: $p={r['FRIEDMAN_P']:.2g}$"
        for h, r in zip(names, test_rows)
    )
    excluded = [MODEL_LABEL[m] for m in MODEL_ORDER if m not in set(ranks["MODEL"])]
    if excluded:
        p_values += (f". Not ranked ({', '.join(excluded)}): no eligible configuration in some dataset, and the "
                     "Friedman test requires complete blocks")
    lines = [
        "\\begin{table}[!t]", "\\centering",
        f"\\caption{{Average rank of the labeling models (1 = highest value) over the {n_blocks} dataset $\\times$ "
        f"recommender blocks. Friedman tests: {p_values}. Bold: ranks within the Nemenyi critical difference "
        f"(CD $={cd:.2f}$) of the best model of the column. Ranks are computed from the fold means, without "
        "intervals. B$^3$ (B-cubed F1) is not corrected for chance and is high for labelings with a dominant group "
        "(Section~\\ref{sec:protocol}).}",
        "\\label{tab:ranking}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "r" * len(columns) + "}", "\\toprule",
        " & \\multicolumn{3}{c}{ARI} & B$^3$ & SC \\\\", "\\cmidrule(lr){2-4} \\cmidrule(lr){5-5} \\cmidrule(lr){6-6}",
        "Model & " + " & ".join(header) + " \\\\", "\\midrule",
    ]
    previous_family = None
    for model in MODEL_ORDER:
        cells = [MODEL_LABEL[model]]
        family = ranks.loc[ranks["MODEL"] == model, "FAMILY"]
        if family.empty:
            continue
        if previous_family is not None and family.iloc[0] != previous_family:
            lines.append("\\midrule")
        previous_family = family.iloc[0]
        for stage, measure in columns:
            row = ranks[(ranks["MODEL"] == model) & (ranks["STAGE"] == stage) & (ranks["MEASURE"] == measure)].iloc[0]
            value = f"{row['AVERAGE_RANK']:.1f}"
            cells.append(f"\\textbf{{{value}}}" if row["WITHIN_CD_OF_BEST"] else value)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(path, "\n".join(lines))


def calibration_table(summary: pd.DataFrame, datasets: list, recommenders: list, path: str):
    stages = ["CAND", "C@0.0", "C@0.1", "C@1.0"]
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Manipulation check of the calibration. KL: mean miscalibration $\\mathrm{KL}(P\\|\\tilde{Q})$ between "
        "the genre preference distribution and each stage (distributions normalized to sum to one, $\\alpha=0.01$). "
        "Item overlap: Jaccard similarity between the items of C@0.0 and C@0.1 (0.0/0.1) and between the items of the "
        "calibrated lists C@0.1 and C@1.0 (0.1/1.0). MAP: MAP@10 on the test interactions. "
        "Means over the five folds only, without intervals.}",
        "\\label{tab:calibration_check}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{l" + "r" * (len(stages) + 4) + "}", "\\toprule",
        " & \\multicolumn{" + str(len(stages)) + "}{c}{KL} & \\multicolumn{2}{c}{Item overlap} "
        "& \\multicolumn{2}{c}{MAP} \\\\",
        "\\cmidrule(lr){2-" + str(1 + len(stages)) + "} \\cmidrule(lr){" + str(2 + len(stages)) + "-"
        + str(3 + len(stages)) + "} \\cmidrule(lr){" + str(4 + len(stages)) + "-" + str(5 + len(stages)) + "}",
        "Rec. & " + " & ".join(stages) + " & 0.0/0.1 & 0.1/1.0 & C@0.0 & C@0.1 \\\\", "\\midrule",
    ]
    for dataset in datasets:
        lines.append("\\multicolumn{" + str(len(stages) + 5) + "}{l}{\\textit{" + DATASET_LABEL[dataset] + "}} \\\\")
        for recommender in recommenders:
            data = summary[(summary["DATASET"] == dataset) & (summary["RECOMMENDER"] == recommender)].set_index("STAGE")
            cells = [recommender]
            cells += [fmt(data.loc[s, "KL_MISCALIBRATION"], 3) if s in data.index else "--" for s in stages]
            cells.append(fmt(data.loc["C@0.1", "ITEM_JACCARD_WITH_C@0.0"], 2) if "C@0.1" in data.index else "--")
            cells.append(fmt(data.loc["C@1.0", "ITEM_JACCARD_WITH_C@0.1"], 2)
                         if "C@1.0" in data.index and "ITEM_JACCARD_WITH_C@0.1" in data else "--")
            cells += [fmt(data.loc[s, "MAP@10"], 3) if s in data.index and "MAP@10" in data else "--"
                      for s in ["C@0.0", "C@0.1"]]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule" if dataset != datasets[-1] else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    write(path, "\n".join(lines))


def scalability_table(data: pd.DataFrame, path: str):
    users = data[data["BENCHMARK"] == "USERS"].copy()
    users["TIME"] = users["FIT_TIME"] + users["PROJECTION_TIME"]
    genres = data[data["BENCHMARK"] == "GENRES"].copy()
    genres["TIME"] = genres["FIT_TIME"] + genres["PROJECTION_TIME"]
    sizes = sorted(users["N_USERS"].unique())
    shown = [s for s in [4000, 16000, 64000] if s in sizes]
    genre_sizes = sorted(genres["N_GENRES"].unique())
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Run time (seconds, one CPU thread) of fitting a labeling model on the preference distributions and "
        "projecting the same number of users, for resampled Twitter Movies users; $b$: slope of $\\log t$ on "
        "$\\log |U|$ (empirical complexity exponent); last columns: time with "
        f"{genres['N_USERS'].iloc[0] if not genres.empty else '--'} users and wider genre spaces. "
        "--: not run (memory limit or time budget). $^\\dagger$: no configuration was eligible in the search; timed "
        "with $m{=}10$ and the Euclidean distance. Single measurements, without repetitions or intervals.}",
        "\\label{tab:scalability}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{l" + "r" * (len(shown) + 1 + len(genre_sizes)) + "}", "\\toprule",
        "Model & " + " & ".join(f"$|U|{{=}}${s // 1000}k" for s in shown) + " & $b$ & "
        + " & ".join(f"$|G|{{=}}{g}$" for g in genre_sizes) + " \\\\", "\\midrule",
    ]
    for model in MODEL_ORDER:
        series = users[(users["MODEL"] == model)].set_index("N_USERS")
        if series.empty:
            continue
        reference = "NOTE" in series and series["NOTE"].fillna("").str.len().gt(0).any()
        cells = [MODEL_LABEL[model] + ("$^\\dagger$" if reference else "")]
        for s in shown:
            value = series.loc[s, "TIME"] if s in series.index else np.nan
            cells.append(fmt(value, 2) if not np.isnan(value) else "--")
        valid = series[series["TIME"].notna() & (series.index >= 2000)]
        slope = np.polyfit(np.log(valid.index.values), np.log(valid["TIME"].values), 1)[0] if len(valid) >= 3 else np.nan
        cells.append(fmt(slope, 2))
        g = genres[genres["MODEL"] == model].set_index("N_GENRES")
        cells += [fmt(g.loc[n, "TIME"], 2) if n in g.index else "--" for n in genre_sizes]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(path, "\n".join(lines))


def interpretation_table(data: pd.DataFrame, datasets: list, recommenders: list, path: str):
    """
    What separates the groups of the selected K-Means configuration, and how the share of users with the dominant
    genre changes along the pipeline.
    """
    stages = ["PREF", "CAND", "C@0.0", "C@0.1", "C@1.0"]
    lines = [
        "\\begin{table*}[!t]", "\\centering",
        "\\caption{Interpretation of the K-Means groups. Genre: the genre with the largest spread of the group "
        "centroids (Spread: its share of the total spread); SC: Silhouette with all genres and without that genre "
        "(same $k$); Users with the genre: share of users whose distribution contains the genre at each stage (mean over "
        "folds and recommenders). Means over the five folds only, without intervals.}",
        "\\label{tab:interpretation}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{llrrr" + "r" * len(stages) + "}", "\\toprule",
        " & & & \\multicolumn{2}{c}{SC} & \\multicolumn{" + str(len(stages)) + "}{c}{Users with the genre} \\\\",
        "\\cmidrule(lr){4-5} \\cmidrule(lr){6-" + str(5 + len(stages)) + "}",
        "Dataset & Genre & Spread & All & W/o & " + " & ".join(stages) + " \\\\", "\\midrule",
    ]
    for dataset in datasets:
        d = data[data["DATASET"] == dataset]
        if d.empty:
            continue
        genre = d["DOMINANT_GENRE"].mode().iloc[0]
        pref = d[d["STAGE"] == "PREF"]
        cells = [DATASET_LABEL[dataset], genre, fmt(pref["SPREAD_SHARE"].mean(), 2), fmt(pref["SILHOUETTE"].mean(), 2),
                 fmt(pref["SILHOUETTE_WITHOUT_GENRE"].mean(), 2)]
        cells += [fmt(d[d["STAGE"] == s]["SHARE_WITH_GENRE"].mean(), 2) for s in stages]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(path, "\n".join(lines))
