"""Credit risk evaluation metrics. Pure functions on arrays: no Spark, no MLflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_score) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and the score threshold where it is reached.

    KS is the largest gap between the cumulative score distributions of defaulters
    and non-defaulters, which equals max(TPR - FPR) over all thresholds.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best = int(np.argmax(tpr - fpr))
    return float(tpr[best] - fpr[best]), float(thresholds[best])


def evaluate(y_true, y_score) -> dict[str, float]:
    """Ranking metrics (AUC, Gini, KS, PR AUC) and probability metrics (Brier, log loss)."""
    auc = float(roc_auc_score(y_true, y_score))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, y_score)[0],
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score)),
    }


def decile_table(y_true, y_score, bins: int = 10) -> pd.DataFrame:
    """Default rate by score band, from the safest band (1) to the riskiest.

    ``cum_default_share`` is read from the riskiest band down: the share of all
    defaults caught by refusing that band and every riskier one.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    # Rank first so that tied scores still give equal-sized bands.
    frame["band"] = pd.qcut(frame["score"].rank(method="first"), bins, labels=False) + 1
    table = frame.groupby("band").agg(
        applicants=("y", "size"),
        defaults=("y", "sum"),
        default_rate=("y", "mean"),
        mean_score=("score", "mean"),
        min_score=("score", "min"),
    )
    table["cum_default_share"] = table["defaults"][::-1].cumsum()[::-1] / table["defaults"].sum()
    return table.reset_index()
