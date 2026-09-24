"""Figure style shared by every notebook, so one entity keeps one colour across the whole project."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve

FIGURES = Path(__file__).resolve().parent.parent / "reports" / "figures"

# Colour follows the entity, never its position in a chart. Blue and orange were checked with the
# dataviz palette validator (colour-blind separation, contrast); the reference line is a neutral grey.
COLORS = {"xgboost": "#2a78d6", "logistic": "#eb6834", "permutation": "#1baf7a", "baseline": "#8a8983"}
LABELS = {"xgboost": "XGBoost", "logistic": "Logistic regression"}
MARKERS = {"xgboost": "o", "logistic": "s", "permutation": "D"}  # second channel besides colour, for greyscale print

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#ffffff"
N_BINS = 10  # calibration bins: the figure and the number quoted next to it must use the same ones


def use_style():
    """Recessive grid and axes, text in ink (never in the series colour)."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED, "text.color": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "font.size": 10,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, name):
    """Write a figure to reports/figures/ (the folder the report reads from)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)


def calibration_bins(y_true, proba):
    """Observed churn rate and mean predicted probability in equal-count bins."""
    return calibration_curve(y_true, proba, n_bins=N_BINS, strategy="quantile")


def calibration_plot(y_true, probas, ax):
    """Observed churn rate against mean predicted probability."""
    ax.plot([0, 1], [0, 1], color=COLORS["baseline"], lw=1.2, ls=(0, (4, 3)), label="Perfect calibration")
    for name, proba in probas.items():
        observed, predicted = calibration_bins(y_true, proba)
        ax.plot(predicted, observed, color=COLORS[name], marker=MARKERS[name], lw=2, ms=6.5,
                mec=SURFACE, mew=1.5, label=LABELS[name])
    ax.set(xlabel="Predicted churn probability", ylabel="Observed churn rate",
           xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.legend(loc="upper left")


def importance_plot(share, labels, ax):
    """One dot per method for each feature: the share of that method's total importance it receives.

    `share` has the features as rows (top first) and one column per method, named by a key of COLORS.
    """
    y = np.arange(len(share))[::-1]
    ax.hlines(y, share.min(axis=1), share.max(axis=1), color=GRID, lw=1.5, zorder=1)
    for key in share.columns:
        ax.scatter(share[key], y, color=COLORS[key], marker=MARKERS[key], s=46, edgecolor=SURFACE,
                   linewidth=1.2, zorder=2, label=labels[key])
    ax.set_yticks(y, share.index)
    ax.set(xlabel="Share of the method's total importance", xlim=(0, None))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")


def effect_plot(x, shap_values, linear_values, ax, xlabel):
    """What each model does with one numeric feature, in log-odds: SHAP dots (XGBoost) and a line (logistic)."""
    x, linear_values = np.asarray(x), np.asarray(linear_values)
    order = np.argsort(x)
    ax.scatter(x, shap_values, color=COLORS["xgboost"], s=8, alpha=0.35, linewidths=0, label="XGBoost (SHAP)")
    ax.plot(x[order], linear_values[order], color=COLORS["logistic"], lw=2, label="Logistic regression")
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.set(xlabel=xlabel)


def deletion_plot(results, labels, ax):
    """AUC lost after shuffling the k top-ranked features of each ranking, against k random features.

    `results` has one row per (ranking, k) with the columns ranking, k, top_k_drop, random_median,
    random_low and random_high (the random sets are the same for every ranking).
    """
    random = results[results["ranking"] == results["ranking"].iloc[0]].set_index("k")
    ax.fill_between(random.index, random["random_low"], random["random_high"], color=GRID, label="Random features (5-95 %)")
    ax.plot(random.index, random["random_median"], color=COLORS["baseline"], lw=1.5, ls=(0, (4, 3)),
            label="Random features (median)")
    for key in labels:
        rows = results[results["ranking"] == key].set_index("k")
        ax.plot(rows.index, rows["top_k_drop"], color=COLORS[key], marker=MARKERS[key], lw=2, ms=6.5,
                mec=SURFACE, mew=1.5, label=labels[key])
    ax.set_xticks(random.index)
    ax.set(xlabel="k = number of features shuffled", ylabel="Fall in test ROC-AUC")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))


def local_plot(contributions, values, ax, top=5):
    """The `top` largest contributions of one customer to the log-odds of churn, and the sum of the rest."""
    order = contributions.abs().sort_values(ascending=False).index
    shown = contributions[order[:top]]
    bars = np.append(shown.to_numpy(), contributions[order[top:]].sum())
    labels = [f"{name} = {values[name]}" for name in shown.index] + [f"{len(order) - top} other features"]
    y = np.arange(len(bars))[::-1]
    ax.barh(y, bars, color=COLORS["xgboost"], height=0.62)
    ax.axvline(0, color=INK_MUTED, lw=1)
    for yi, value in zip(y, bars):
        ax.text(value + np.sign(value) * 0.04, yi, f"{value:+.2f}", va="center", fontsize=8, color=INK,
                ha="left" if value >= 0 else "right")
    ax.set_yticks(y, labels)
    ax.margins(x=0.18)
    ax.grid(axis="y", visible=False)
