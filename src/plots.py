"""Figure style shared by every notebook, so one entity keeps one colour across the whole project."""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

FIGURES = Path(__file__).resolve().parent.parent / "reports" / "figures"

# Colour follows the entity, never its position in a chart. Blue and orange were checked with the
# dataviz palette validator (colour-blind separation, contrast); the reference line is a neutral grey.
COLORS = {"xgboost": "#2a78d6", "logistic": "#eb6834", "baseline": "#8a8983"}
LABELS = {"xgboost": "XGBoost", "logistic": "Logistic regression"}
MARKERS = {"xgboost": "o", "logistic": "s"}  # second channel besides colour, for greyscale print

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
