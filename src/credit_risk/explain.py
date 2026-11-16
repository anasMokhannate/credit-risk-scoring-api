"""Explainability with SHAP: what drives risk overall, and why one applicant scored as they did.

SHAP splits each prediction into one contribution per feature. For this model the
contributions are in log-odds: ``base value + sum of contributions`` is the model's
raw score, and the sigmoid of that is the probability of default. A positive
contribution pushes the applicant's risk up, a negative one pulls it down.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure

from credit_risk.config import Config, load_config
from credit_risk.model import load_champion, load_labelled, prepare, setup_mlflow
from credit_risk.plots import GRID, INK, MODEL, MUTED, SURFACE

ADDITIVITY_TOLERANCE = 1e-6

# Plain-language names for the features most likely to appear in a reason code.
FEATURE_LABELS = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "EXT_SOURCE_MEAN": "Average external credit score",
    "EXT_SOURCE_MIN": "Lowest external credit score",
    "EXT_SOURCE_KNOWN_COUNT": "Number of external credit scores available",
    "CREDIT_TERM": "Share of the loan repaid each year",
    "CREDIT_INCOME_RATIO": "Loan amount relative to income",
    "ANNUITY_INCOME_RATIO": "Annual repayment relative to income",
    "GOODS_CREDIT_RATIO": "Price of the goods relative to the loan amount",
    "EMPLOYED_AGE_RATIO": "Share of adult life in the current job",
    "AGE_YEARS": "Age",
    "DAYS_BIRTH": "Age",
    "DAYS_EMPLOYED": "Time in the current job",
    "DAYS_ID_PUBLISH": "Time since the identity document was issued",
    "DAYS_REGISTRATION": "Time since the registration was changed",
    "DAYS_LAST_PHONE_CHANGE": "Time since the phone number was changed",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Annual repayment",
    "AMT_GOODS_PRICE": "Price of the goods",
    "AMT_INCOME_TOTAL": "Income",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_CONTRACT_TYPE": "Type of loan",
    "NAME_INCOME_TYPE": "Type of income",
    "NAME_FAMILY_STATUS": "Family status",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer type",
    "OWN_CAR_AGE": "Age of the applicant's car",
    "FLAG_OWN_CAR": "Car ownership",
    "REGION_RATING_CLIENT_W_CITY": "Rating of the region of residence",
    "INST_LATE_SHARE": "Share of past instalments paid late",
    "INST_LATE_SHARE_12M": "Share of instalments paid late in the last 12 months",
    "INST_DAYS_LATE_MEAN": "Average delay on past instalments",
    "INST_DAYS_LATE_MAX": "Longest delay on a past instalment",
    "INST_PAYMENT_RATIO_MEAN": "Share of each instalment actually paid",
    "INST_UNDERPAID_SUM": "Total underpaid on past instalments",
    "INST_AMT_PAYMENT_SUM": "Total repaid on past loans",
    "PREV_REFUSED_SHARE": "Share of past applications refused",
    "PREV_REFUSED_COUNT": "Number of past applications refused",
    "PREV_REFUSED_COUNT_12M": "Applications refused in the last 12 months",
    "PREV_APPROVED_SHARE": "Share of past applications approved",
    "PREV_CNT_PAYMENT_MEAN": "Average duration of past loans",
    "PREV_DAYS_DECISION_MAX": "Time since the last application",
    "BUREAU_DEBT_CREDIT_RATIO": "Outstanding debt relative to credit granted elsewhere",
    "BUREAU_ACTIVE_COUNT": "Number of active credits elsewhere",
    "BUREAU_ACTIVE_SHARE": "Share of credits elsewhere still active",
    "BUREAU_ACTIVE_DEBT_SUM": "Outstanding debt on active credits elsewhere",
    "BUREAU_DAYS_CREDIT_MAX": "Time since the most recent credit elsewhere",
    "BUREAU_DAYS_CREDIT_MEAN": "Average age of credits elsewhere",
    "BUREAU_DAYS_ENDDATE_MAX": "Time until the last credit elsewhere ends",
    "BUREAU_OVERDUE_SUM": "Amount overdue elsewhere",
    "BUREAU_BB_LATE_SHARE_MEAN": "Share of late months on credits elsewhere",
    "POS_DPD_MAX": "Longest delay on a point-of-sale loan",
    "POS_DPD_SHARE": "Share of months in arrears on point-of-sale loans",
    "POS_INSTALMENT_FUTURE_MEAN": "Instalments left on point-of-sale loans",
    "CC_UTILISATION_MEAN": "Average share of the credit card limit used",
    "CC_UTILISATION_MAX": "Highest share of the credit card limit used",
    "CC_BALANCE_MEAN": "Average credit card balance",
    "CC_DRAWINGS_MEAN": "Average monthly credit card drawings",
    "CC_MIN_PAYMENT_MISSED_SHARE": "Share of months with the minimum card payment missed",
}
# Fallback for a feature without its own label: name the part of the file it comes from.
GROUP_LABELS = {
    "BUREAU_": "Credit history at other lenders",
    "PREV_": "Previous applications",
    "INST_": "Past instalment payments",
    "POS_": "Point-of-sale loan history",
    "CC_": "Credit card history",
}


class ExplanationError(RuntimeError):
    """Raised when SHAP contributions do not add up to the model's score."""


def feature_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for prefix, label in GROUP_LABELS.items():
        if feature.startswith(prefix):
            return f"{label} ({feature})"
    return f"Application data ({feature})"


def check_additivity(contributions, base_value: float, raw_scores) -> None:
    """Base value + contributions must rebuild the model's raw score for every row."""
    gap = np.abs(np.asarray(contributions).sum(axis=1) + base_value - np.asarray(raw_scores))
    if gap.max() > ADDITIVITY_TOLERANCE:
        raise ExplanationError(
            f"SHAP values do not add up to the model output (largest gap {gap.max():.2e}): "
            "the explanations cannot be trusted."
        )


def explain(model, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """SHAP contributions (one column per feature, in log-odds) and the base value.

    Uses TreeSHAP in its path-dependent form: exact for tree ensembles, and it
    needs no background dataset.
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older SHAP versions: one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    base_value = float(np.ravel(explainer.expected_value)[-1])
    check_additivity(values, base_value, model.predict(X, raw_score=True))
    return pd.DataFrame(values, columns=X.columns, index=X.index), base_value


def global_importance(contributions: pd.DataFrame) -> pd.DataFrame:
    """Rank features by their average absolute contribution over many applicants."""
    importance = contributions.abs().mean().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(importance) + 1),
            "feature": importance.index,
            "label": [feature_label(f) for f in importance.index],
            "mean_abs_shap": importance.to_numpy(),
            "share": (importance / importance.sum()).to_numpy(),
        }
    )


def reason_codes(ids, X: pd.DataFrame, contributions: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """For each applicant, the features that raised their risk the most.

    Only positive contributions are reasons: a feature that lowered the risk does
    not explain a refusal. An applicant with no positive contribution gets no row.
    """
    values = contributions.to_numpy()
    order = np.argsort(-values, axis=1)[:, :top_n]
    rows = np.repeat(np.arange(len(values)), order.shape[1])
    columns = order.ravel()
    picked = values[rows, columns]
    keep = picked > 0
    rows, columns, picked = rows[keep], columns[keep], picked[keep]
    features = contributions.columns.to_numpy()[columns]
    raw = X.to_numpy(dtype=object)[rows, columns]
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.asarray(ids)[rows],
            "rank": (np.tile(np.arange(1, order.shape[1] + 1), len(values)))[keep],
            "feature": features,
            "feature_value": [None if pd.isna(v) else str(v) for v in raw],
            "shap_value": picked,
            "reason": [feature_label(f) for f in features],
        }
    )


# -------------------------------------------------------------------------- plots


def importance_figure(importance: pd.DataFrame, top: int = 20) -> Figure:
    """Bar chart of the features with the largest average impact on the score."""
    shown = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(shown["feature"], shown["mean_abs_shap"], color=MODEL, height=0.72)
    ax.bar_label(
        bars,
        labels=[f"{v:.3f}" for v in shown["mean_abs_shap"]],
        color=MUTED,
        fontsize=8,
        padding=3,
    )
    fig.suptitle(
        f"What drives the score: top {len(shown)} features by average impact",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    ax.set_xlabel("Mean absolute SHAP value (log-odds)", color=MUTED, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.margins(x=0.08, y=0.01)
    fig.tight_layout()
    return fig


def beeswarm_figure(contributions: pd.DataFrame, X: pd.DataFrame, top: int = 20) -> Figure:
    """SHAP's summary plot: one dot per applicant, coloured by the feature's value."""
    # A category has no "high" or "low" value: blank it so its dots are drawn grey.
    numeric = X.apply(
        lambda column: (
            pd.Series(np.nan, index=column.index) if column.dtype == "category" else column
        )
    )
    # summary_plot draws on the current figure: give it its own.
    fig = plt.figure(facecolor=SURFACE)
    shap.summary_plot(
        contributions.to_numpy(), numeric, max_display=top, show=False, plot_size=(9, 7)
    )
    fig.suptitle(
        "Direction of each feature's effect on risk (grey: categories)",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.axes[0].set_xlabel("SHAP value (log-odds): right raises risk, left lowers it")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- run


def run(spark, config: Config, sample_size: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Explain the champion on a sample of the holdout and store the global view.

    The importance table and plots are added to the champion's MLflow run, and the
    table is also written to ``gold.shap_global_importance`` for the monitoring step.
    """
    setup_mlflow(config, spark)
    model, spec, version = load_champion(config)
    holdout = load_labelled(spark, config).query("split == 'holdout'")
    sample = holdout.sample(min(sample_size, len(holdout)), random_state=seed)
    X = prepare(sample, spec)

    contributions, base_value = explain(model, X)
    importance = global_importance(contributions)

    with mlflow.start_run(run_id=version.run_id):
        mlflow.log_metric("shap_base_value", base_value)
        mlflow.log_param("shap_sample_size", len(X))
        mlflow.log_table(importance, "shap/global_importance.json")
        for name, figure in {
            "shap_importance": importance_figure(importance),
            "shap_beeswarm": beeswarm_figure(contributions, X),
        }.items():
            mlflow.log_figure(figure, f"plots/{name}.png")
            plt.close(figure)

    stored = importance.assign(
        model_name=config.model_name, model_version=str(version.version), base_value=base_value
    )
    (
        spark.createDataFrame(stored)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(config.table("gold", "shap_global_importance"))
    )
    print(
        f"explain {config.model_name} v{version.version}: {len(X):,} applicants explained, "
        f"base value {base_value:.3f}, top feature {importance['feature'].iloc[0]}"
    )
    return importance


if __name__ == "__main__":
    from credit_risk.spark import get_spark

    top = run(get_spark(), load_config())
    print(top.head(20).to_string(index=False))
