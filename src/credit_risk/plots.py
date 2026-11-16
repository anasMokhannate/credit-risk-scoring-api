"""Evaluation charts logged to MLflow with each training run."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from credit_risk.metrics import decile_table, evaluate, ks_statistic

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
MODEL = "#2a78d6"  # the model is always this blue
REFERENCE = "#8a8983"  # chance level, perfect calibration, base rate


def _axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return fig, ax


def _note(ax, x, y, text, **kwargs):
    ax.text(x, y, text, color=MUTED, fontsize=9, **kwargs)


def roc_figure(y_true, y_score) -> Figure:
    metrics = evaluate(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    best = int(np.argmax(tpr - fpr))
    fig, ax = _axes(
        f"ROC curve: AUC {metrics['auc']:.3f}, Gini {metrics['gini']:.3f}",
        "False positive rate (good clients flagged)",
        "True positive rate (defaulters caught)",
    )
    ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1, linestyle="--")
    ax.plot(fpr, tpr, color=MODEL, linewidth=2)
    # The KS statistic is the largest vertical gap between the curve and the diagonal.
    ax.plot([fpr[best]] * 2, [fpr[best], tpr[best]], color=INK, linewidth=1)
    _note(ax, fpr[best] + 0.02, (fpr[best] + tpr[best]) / 2, f"KS {metrics['ks']:.3f}", va="center")
    _note(ax, 0.72, 0.66, "chance", rotation=38)
    ax.set(xlim=(0, 1), ylim=(0, 1.005))
    fig.tight_layout()
    return fig


def precision_recall_figure(y_true, y_score) -> Figure:
    metrics = evaluate(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    base_rate = float(np.mean(y_true))
    fig, ax = _axes(
        f"Precision-recall curve: PR AUC {metrics['pr_auc']:.3f}",
        "Recall (share of defaulters caught)",
        "Precision (share of flagged clients who default)",
    )
    ax.axhline(base_rate, color=REFERENCE, linewidth=1, linestyle="--")
    ax.plot(recall, precision, color=MODEL, linewidth=2)
    _note(ax, 0.99, base_rate + 0.015, f"default rate {base_rate:.1%}", ha="right")
    ax.set(xlim=(0, 1), ylim=(0, 1.005))
    fig.tight_layout()
    return fig


def calibration_figure(y_true, y_score, bins: int = 10) -> Figure:
    observed, predicted = calibration_curve(y_true, y_score, n_bins=bins, strategy="quantile")
    limit = max(observed.max(), predicted.max()) * 1.08
    fig, ax = _axes(
        f"Calibration: Brier score {evaluate(y_true, y_score)['brier']:.4f}",
        "Mean predicted probability of default",
        "Observed default rate",
    )
    ax.plot([0, limit], [0, limit], color=REFERENCE, linewidth=1, linestyle="--")
    ax.plot(predicted, observed, color=MODEL, linewidth=2, marker="o", markersize=6)
    _note(ax, limit * 0.80, limit * 0.86, "perfect", rotation=38)
    ax.set(xlim=(0, limit), ylim=(0, limit))
    fig.tight_layout()
    return fig


def decile_figure(y_true, y_score) -> Figure:
    table = decile_table(y_true, y_score)
    base_rate = float(np.mean(y_true))
    fig, ax = _axes(
        f"Default rate by score band: {table['default_rate'].iloc[-1]:.0%} in the riskiest tenth",
        "Score band (1 = safest tenth of applicants, 10 = riskiest)",
        "Observed default rate",
    )
    ax.grid(axis="x", visible=False)
    bars = ax.bar(table["band"], table["default_rate"], color=MODEL, width=0.72)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in table["default_rate"]], color=MUTED, fontsize=8)
    ax.axhline(base_rate, color=REFERENCE, linewidth=1, linestyle="--")
    _note(ax, 0.6, base_rate + table["default_rate"].max() * 0.015, f"average {base_rate:.1%}")
    ax.set_xticks(table["band"])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    return fig


def evaluation_figures(y_true, y_score) -> dict[str, Figure]:
    return {
        "roc_curve": roc_figure(y_true, y_score),
        "precision_recall_curve": precision_recall_figure(y_true, y_score),
        "calibration": calibration_figure(y_true, y_score),
        "default_rate_by_band": decile_figure(y_true, y_score),
    }


__all__ = ["evaluation_figures", "ks_statistic"]
