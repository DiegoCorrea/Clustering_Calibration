"""
Figures of the stage-wise structural evaluation (step 7, option STAGEWISE).

Figures are sized for an IEEE two-column paper and saved as PDF. Identity is never carried by color
alone: every family/model also has its own marker and line style, so the figures survive grayscale
printing.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluations.stage_wise_statistics import mean_ci  # noqa: E402

DOUBLE_COLUMN = 7.16
SINGLE_COLUMN = 3.5

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Families: the three validated all-pairs slots of the reference palette and a dark neutral.
FAMILY_STYLE = {
    "PARTITION": {"color": "#2a78d6", "marker": "o", "linestyle": "-", "label": "Partition"},
    "MIXTURE": {"color": "#eb6834", "marker": "s", "linestyle": "--", "label": "Mixture"},
    "DENSITY": {"color": "#1baf7a", "marker": "^", "linestyle": "-.", "label": "Density"},
    "DETECTOR": {"color": "#52514e", "marker": "D", "linestyle": ":", "label": "Outlier detector"},
}
FAMILY_ORDER = ["PARTITION", "MIXTURE", "DENSITY", "DETECTOR"]

# Categorical order of the reference palette (line charts, adjacent pairs).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SERIES_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

DATASET_LABEL = {"ml-1m": "MovieLens-1M", "yahoo-movies": "Yahoo Movies", "twitter_movies": "Twitter Movies"}
MODEL_LABEL = {
    "KMEANS": "K-Means", "BISECTING": "BKM", "AGGLO": "Agglomerative", "SPECTRAL": "Spectral",
    "BIRCH": "Birch", "FCM": "FCM", "GAUSSIAN": "GM", "BAYESIAN": "BGM", "DBSCAN": "DBSCAN",
    "OPTICS": "OPTICS", "IF": "IF", "LOF": "LOF", "OSVM": "OCSVM", "SGD": "OCSVM-SGD", "ENVELOPE": "EE",
}
STAGE_TICK = {"PREF": "PREF", "CAND": "CAND"}


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
        "axes.edgecolor": INK_SECONDARY, "axes.linewidth": 0.5,
        "axes.labelcolor": INK, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2, "lines.markersize": 3.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


def stage_axis(stages: list) -> tuple:
    ticks = [STAGE_TICK[s] if s in STAGE_TICK else s.split("@")[1] for s in stages]
    return list(range(len(stages))), ticks


def ordered_stages(data: pd.DataFrame) -> list:
    return data.drop_duplicates("STAGE").sort_values("STAGE_ORDER")["STAGE"].tolist()


def family_fold_values(raw: pd.DataFrame, measure: str) -> pd.DataFrame:
    """
    Per fold, the mean of the measure over the models of each family (projected labels).
    """
    projected = raw[raw["LABELS"] == "PROJECTED"]
    return projected.groupby(
        ["DATASET", "RECOMMENDER", "FAMILY", "STAGE", "STAGE_ORDER", "FOLD"]
    )[measure].mean().reset_index()


def stage_lines_by_family(raw: pd.DataFrame, measure: str, y_label: str, path: str, datasets: list,
                          recommenders: list, y_limits: tuple = None, reference_line: float = None):
    """
    Small multiples (datasets x recommenders): mean of each family over the folds with its 95% CI band.
    """
    set_style()
    values = family_fold_values(raw, measure)
    stages = ordered_stages(values)
    x, ticks = stage_axis(stages)
    fig, axes = plt.subplots(len(datasets), len(recommenders), figsize=(DOUBLE_COLUMN, 1.6 * len(datasets)),
                             sharex=True, sharey=True, squeeze=False)
    for i, dataset in enumerate(datasets):
        for j, recommender in enumerate(recommenders):
            ax = axes[i][j]
            panel = values[(values["DATASET"] == dataset) & (values["RECOMMENDER"] == recommender)]
            for family in FAMILY_ORDER:
                style = FAMILY_STYLE[family]
                fam = panel[panel["FAMILY"] == family]
                if fam.empty:
                    continue
                stats_rows = [mean_ci(fam[fam["STAGE"] == s][measure]) for s in stages]
                mean = np.array([r[0] for r in stats_rows])
                low = np.array([r[2] for r in stats_rows])
                high = np.array([r[3] for r in stats_rows])
                ax.fill_between(x, low, high, color=style["color"], alpha=0.15, linewidth=0)
                ax.plot(x, mean, color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                        label=style["label"], markeredgecolor="white", markeredgewidth=0.4)
            if reference_line is not None:
                ax.axhline(reference_line, color=INK_SECONDARY, linewidth=0.6)
            ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)} · {recommender}", color=INK)
            ax.set_xticks(x)
            ax.set_xticklabels(ticks, rotation=90)
            if y_limits:
                ax.set_ylim(*y_limits)
            if j == 0:
                ax.set_ylabel(y_label)
            if i == len(datasets) - 1:
                ax.set_xlabel("Stage (λ of the top-10 list)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, path)


def stage_lines_by_model(raw: pd.DataFrame, measure: str, y_label: str, path: str, dataset: str,
                         recommenders: list, families: dict, y_limits: tuple = None):
    """
    Supplementary figure: one row per family, one column per recommender, one line per model.
    """
    set_style()
    projected = raw[(raw["LABELS"] == "PROJECTED") & (raw["DATASET"] == dataset)]
    stages = ordered_stages(projected)
    x, ticks = stage_axis(stages)
    fig, axes = plt.subplots(len(families), len(recommenders), figsize=(DOUBLE_COLUMN, 1.6 * len(families)),
                             sharex=True, sharey=True, squeeze=False)
    for i, (family, models) in enumerate(families.items()):
        for j, recommender in enumerate(recommenders):
            ax = axes[i][j]
            for k, model in enumerate(models):
                data = projected[(projected["MODEL"] == model) & (projected["RECOMMENDER"] == recommender)]
                if data.empty:
                    continue
                stats_rows = [mean_ci(data[data["STAGE"] == s][measure]) for s in stages]
                mean = np.array([r[0] for r in stats_rows])
                low = np.array([r[2] for r in stats_rows])
                high = np.array([r[3] for r in stats_rows])
                ax.fill_between(x, low, high, color=SERIES_COLORS[k], alpha=0.12, linewidth=0)
                ax.plot(x, mean, color=SERIES_COLORS[k], marker=SERIES_MARKERS[k], linestyle=SERIES_LINESTYLES[k],
                        label=MODEL_LABEL.get(model, model), markeredgecolor="white", markeredgewidth=0.4)
            ax.set_title(f"{FAMILY_STYLE[family]['label']} · {recommender}", color=INK)
            ax.set_xticks(x)
            ax.set_xticklabels(ticks, rotation=90)
            if y_limits:
                ax.set_ylim(*y_limits)
            if j == 0:
                ax.set_ylabel(y_label)
            if j == len(recommenders) - 1:
                ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
            if i == len(families) - 1:
                ax.set_xlabel("Stage (λ of the top-10 list)")
    fig.suptitle(DATASET_LABEL.get(dataset, dataset), color=INK, y=1.0)
    fig.tight_layout()
    save(fig, path)


def _repel_labels(ax, points: list, labels: list, step: float = 2.0, iterations: int = 400):
    """
    Places one text label next to each point. Labels are pushed vertically away from other labels and from the
    markers (in points, the unit of the offsets) until nothing overlaps; long moves get a leader line.
    """
    fig = ax.figure
    fig.canvas.draw()
    to_points = 72.0 / fig.dpi
    anchors = np.array([ax.transData.transform(p) for p in points]) * to_points
    offsets = np.tile(np.array([4.0, 1.0]), (len(points), 1))
    width = np.array([2.6 * len(lab) + 2 for lab in labels])
    height = 6.5
    marker = 3.5

    def overlap(a0, a1, b0, b1):
        return min(a1, b1) - max(a0, b0)

    for _ in range(iterations):
        moved = False
        pos = anchors + offsets
        for a in range(len(points)):
            # other labels
            for b in range(len(points)):
                if a == b:
                    continue
                if overlap(pos[a, 0], pos[a, 0] + width[a], pos[b, 0], pos[b, 0] + width[b]) > 0 and \
                        overlap(pos[a, 1], pos[a, 1] + height, pos[b, 1], pos[b, 1] + height) > 0:
                    offsets[a, 1] += step if (pos[a, 1], a) >= (pos[b, 1], b) else -step
                    moved = True
            # markers (including its own)
            for b in range(len(points)):
                if overlap(pos[a, 0], pos[a, 0] + width[a], anchors[b, 0] - marker, anchors[b, 0] + marker) > 0 and \
                        overlap(pos[a, 1], pos[a, 1] + height, anchors[b, 1] - marker, anchors[b, 1] + marker) > 0:
                    offsets[a, 1] += step if pos[a, 1] + height / 2 >= anchors[b, 1] else -step
                    moved = True
        if not moved:
            break
    for point, label, offset in zip(points, labels, offsets):
        ax.annotate(label, point, xytext=tuple(offset), textcoords="offset points", fontsize=5.5,
                    color=INK_SECONDARY, va="bottom",
                    arrowprops=dict(arrowstyle="-", color="#b5b4ae", linewidth=0.4, shrinkA=0, shrinkB=2)
                    if abs(offset[1] - 1.0) > 7 else None)


def description_vs_stability(summary: pd.DataFrame, reproducibility: pd.DataFrame, path: str, datasets: list,
                             stability_stage: str = "C@1.0"):
    """
    x: cross-fold reproducibility of the reference structure (ARI between the PREF labelings of two folds);
    y: stability through the pipeline (ARI between PREF and the stage, mean over recommenders).
    Hollow markers: near-trivial structures (effective number of groups below 1.5).
    """
    set_style()
    fig, axes = plt.subplots(1, len(datasets), figsize=(DOUBLE_COLUMN, 2.5), sharey=True, squeeze=False)
    pref = summary[(summary["STAGE"] == "PREF") & (summary["LABELS"] == "FIT")]
    stage = summary[(summary["STAGE"] == stability_stage) & (summary["LABELS"] == "PROJECTED")]
    stability = stage.groupby(["DATASET", "MODEL"])["ARI"].mean()
    repro = reproducibility[reproducibility["LABELS"] == "FIT_LABEL"].set_index(["DATASET", "MODEL"])["CROSS_FOLD_ARI"]
    for j, dataset in enumerate(datasets):
        ax = axes[0][j]
        points, labels = [], []
        for _, row in pref[pref["DATASET"] == dataset].iterrows():
            model = row["MODEL"]
            style = FAMILY_STYLE[row["FAMILY"]]
            trivial = row["EFFECTIVE_GROUPS"] < 1.5
            xv = repro.get((dataset, model), np.nan)
            yv = stability.get((dataset, model), np.nan)
            ax.scatter(xv, yv, s=22, marker=style["marker"], linewidths=0.9,
                       facecolors="none" if trivial else style["color"], edgecolors=style["color"], zorder=3)
            points.append((xv, yv))
            labels.append(MODEL_LABEL.get(model, model))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.1, 1.0)
        _repel_labels(ax, points, labels)
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.5)
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.5)
        ax.set_title(DATASET_LABEL.get(dataset, dataset), color=INK)
        if j == 0:
            ax.set_ylabel(f"Stability PREF→{stability_stage} (ARI)")
    handles = [plt.Line2D([], [], marker=FAMILY_STYLE[f]["marker"], color=FAMILY_STYLE[f]["color"], linestyle="",
                          label=FAMILY_STYLE[f]["label"]) for f in FAMILY_ORDER]
    handles.append(plt.Line2D([], [], marker="o", color=INK_SECONDARY, markerfacecolor="none", linestyle="",
                              label="Near-trivial structure"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.supxlabel("Reproducibility of the PREF structure (cross-fold ARI)", fontsize=7.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


def independent_vs_projected(raw: pd.DataFrame, path: str):
    """
    The comparison of the first version (independent fits, Jaccard of the raw label values) against the
    ARI of the same two labelings. Each point is a dataset x recommender x model x fold x stage.
    """
    set_style()
    data = raw[raw["INDEPENDENT_LABEL_JACCARD"].notna()] if "INDEPENDENT_LABEL_JACCARD" in raw else raw.iloc[0:0]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.6))
    for family in FAMILY_ORDER:
        fam = data[data["FAMILY"] == family]
        style = FAMILY_STYLE[family]
        ax.scatter(fam["INDEPENDENT_ARI"], fam["INDEPENDENT_LABEL_JACCARD"], s=7, marker=style["marker"],
                   color=style["color"], alpha=0.55, linewidths=0, label=style["label"])
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.5)
    ax.set_xlabel("ARI between the two labelings")
    ax.set_ylabel("Jaccard of raw labels (first version)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handletextpad=0.2,
              columnspacing=0.8, markerscale=1.6)
    fig.tight_layout()
    save(fig, path)


def calibration_check(summary: pd.DataFrame, path: str, datasets: list, recommenders: list):
    set_style()
    fig, axes = plt.subplots(2, len(datasets), figsize=(DOUBLE_COLUMN, 3.4), sharex=True, squeeze=False)
    for j, dataset in enumerate(datasets):
        for k, recommender in enumerate(recommenders):
            data = summary[(summary["DATASET"] == dataset) & (summary["RECOMMENDER"] == recommender)]
            data = data.sort_values("STAGE_ORDER")
            stages = data["STAGE"].tolist()
            x, ticks = stage_axis(stages)
            for i, (measure, label) in enumerate([("KL_MISCALIBRATION", "Miscalibration (KL)"),
                                                  ("GENRES_PER_USER", "Genres per user")]):
                ax = axes[i][j]
                ax.fill_between(x, data[measure + "_CI_LOW"], data[measure + "_CI_HIGH"], color=SERIES_COLORS[k],
                                alpha=0.15, linewidth=0)
                ax.plot(x, data[measure], color=SERIES_COLORS[k], marker=SERIES_MARKERS[k],
                        linestyle=SERIES_LINESTYLES[k], label=recommender, markeredgecolor="white",
                        markeredgewidth=0.4)
                ax.set_xticks(x)
                ax.set_xticklabels(ticks, rotation=90)
                if j == 0:
                    ax.set_ylabel(label)
                if i == 0:
                    ax.set_title(DATASET_LABEL.get(dataset, dataset), color=INK)
                else:
                    ax.set_xlabel("Stage (λ of the top-10 list)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(recommenders), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)


def search_trace(trace: pd.DataFrame, path: str, datasets: list, models: list):
    """
    Mean silhouette over the folds for every evaluated number of groups (partition and mixture models).
    """
    set_style()
    fig, axes = plt.subplots(1, len(datasets), figsize=(DOUBLE_COLUMN, 2.2), sharey=True, squeeze=False)
    for j, dataset in enumerate(datasets):
        ax = axes[0][j]
        for k, model in enumerate(models):
            data = trace[(trace["DATASET"] == dataset) & (trace["MODEL"] == model)].copy()
            if data.empty:
                continue
            column = "param_n_clusters" if data["param_n_clusters"].notna().any() else "param_n_components"
            data = data.sort_values(column)
            ax.plot(data[column], data["silhouette"], color=SERIES_COLORS[k], marker=SERIES_MARKERS[k],
                    linestyle=SERIES_LINESTYLES[k], markersize=2.4, label=MODEL_LABEL.get(model, model),
                    markeredgecolor="white", markeredgewidth=0.3)
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100])
        ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
        ax.set_title(DATASET_LABEL.get(dataset, dataset), color=INK)
        ax.set_xlabel("Number of groups (k)")
        if j == 0:
            ax.set_ylabel("Mean silhouette (5 folds)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, path)


def scalability(data: pd.DataFrame, path: str, families: dict):
    """
    Fit + projection time versus the number of users (log-log), one panel per family.
    """
    set_style()
    users = data[(data["BENCHMARK"] == "USERS") & (data["SKIPPED"].isna() | (data["SKIPPED"] == ""))].copy()
    users["TIME"] = users["FIT_TIME"] + users["PROJECTION_TIME"]
    fig, axes = plt.subplots(1, len(families), figsize=(DOUBLE_COLUMN, 2.1), sharey=True, squeeze=False)
    for j, (family, models) in enumerate(families.items()):
        ax = axes[0][j]
        for k, model in enumerate(models):
            series = users[users["MODEL"] == model].sort_values("N_USERS")
            ax.plot(series["N_USERS"], series["TIME"], color=SERIES_COLORS[k], marker=SERIES_MARKERS[k],
                    linestyle=SERIES_LINESTYLES[k], label=MODEL_LABEL.get(model, model),
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(FAMILY_STYLE[family]["label"], color=INK)
        ax.set_xlabel("Users")
        if j == 0:
            ax.set_ylabel("Fit + projection time (s)")
        ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    fig.tight_layout()
    save(fig, path)
