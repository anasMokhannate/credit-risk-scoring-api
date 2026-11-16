"""Batch scoring: apply the champion model to a batch and keep every score.

The dataset has no calendar dates, so scoring batches are simulated from the
holdout (applicants the model never trained on). The first batches are left
untouched; later ones receive a known, deliberate change. The monitoring step
must then detect those changes, which proves it works.

Tables written, all in gold:

* ``scoring_batches``: the simulated input, one row per applicant with its batch.
* ``scores_history``: append-only history of scores, with the model version used.
* ``score_explanations``: the top reasons behind each score (SHAP).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from credit_risk.config import Config, load_config
from credit_risk.explain import explain, reason_codes
from credit_risk.metrics import evaluate
from credit_risk.model import TARGET, load_champion, load_labelled, prepare, setup_mlflow

# Batch -> what was done to it. Batches are monthly, starting January 2026.
SCENARIOS = {
    1: "none",
    2: "none",
    3: "none",
    4: "income_down_15pct",
    5: "external_scores_down_15pct",
    6: "external_scores_down_15pct_and_30pct_missing",
}

# Illustrative credit policy: approve under 5% probability of default, refuse from
# 15%, send the rest to manual review.
APPROVE_BELOW = 0.05
REFUSE_FROM = 0.15

EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

SCORES_SCHEMA = (
    "SK_ID_CURR bigint, batch_id int, score_date date, drift_scenario string, pd_score double, "
    "risk_band string, decision string, observed_default int, model_name string, "
    "model_version string, scored_at timestamp"
)
EXPLANATIONS_SCHEMA = (
    "SK_ID_CURR bigint, batch_id int, model_version string, base_value double, rank int, "
    "feature string, feature_value string, shap_value double, reason string"
)


# --------------------------------------------------------------------- simulation


def assign_batches(ids, n_batches: int = len(SCENARIOS)) -> np.ndarray:
    """Spread applicants over batches 1..n by a hash of their id (stable across runs)."""
    return (pd.util.hash_array(np.asarray(ids, dtype="int64")) % n_batches + 1).astype(int)


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the features derived from income and external scores after a change.

    Mirrors ``features.application_features``; a test checks the two stay in step.
    """
    df = df.copy()
    income = df["AMT_INCOME_TOTAL"].where(df["AMT_INCOME_TOTAL"] != 0)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].where(
        df["CNT_FAM_MEMBERS"] != 0
    )
    df["EXT_SOURCE_MEAN"] = df[EXT_SOURCES].mean(axis=1)
    df["EXT_SOURCE_MIN"] = df[EXT_SOURCES].min(axis=1)
    df["EXT_SOURCE_KNOWN_COUNT"] = df[EXT_SOURCES].notna().sum(axis=1)
    return df


def apply_scenario(df: pd.DataFrame, scenario: str, seed: int = 0) -> pd.DataFrame:
    """Apply a named, deliberate change to a batch of applicants."""
    if scenario == "none":
        return df
    df = df.copy()
    if scenario == "income_down_15pct":
        # A different population: applicants declare lower incomes.
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"] * 0.85
    elif scenario.startswith("external_scores_down_15pct"):
        # A riskier population: external scores are lower across the board.
        df[EXT_SOURCES] = df[EXT_SOURCES] * 0.85
        if scenario.endswith("30pct_missing"):
            # A data incident: a provider stops sending two of the scores for some applicants.
            lost = np.random.default_rng(seed).random(len(df)) < 0.30
            df.loc[lost, ["EXT_SOURCE_2", "EXT_SOURCE_3"]] = np.nan
    else:
        raise ValueError(f"Unknown scenario {scenario!r}")
    return recompute_derived(df)


def make_batches(holdout: pd.DataFrame) -> pd.DataFrame:
    """Turn the holdout into dated batches, with the scenario of each batch applied."""
    holdout = holdout.drop(columns=["split"], errors="ignore")
    batch_ids = assign_batches(holdout["SK_ID_CURR"])
    batches = []
    for batch_id, scenario in SCENARIOS.items():
        batch = apply_scenario(holdout[batch_ids == batch_id], scenario, seed=batch_id)
        batches.append(
            batch.assign(
                batch_id=batch_id,
                score_date=date(2026, batch_id, 1),
                drift_scenario=scenario,
            )
        )
    return pd.concat(batches, ignore_index=True)


def simulate_batches(spark, config: Config) -> dict[int, int]:
    holdout = load_labelled(spark, config).query("split == 'holdout'")
    batches = make_batches(holdout)
    (
        spark.createDataFrame(batches)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(config.table("gold", "scoring_batches"))
    )
    sizes = batches["batch_id"].value_counts().sort_index().to_dict()
    print(
        f"scoring {config.table('gold', 'scoring_batches')}: {len(batches):,} applicants in "
        f"{len(sizes)} batches"
    )
    return sizes


# ------------------------------------------------------------------------ scoring


def risk_band(scores) -> np.ndarray:
    scores = np.asarray(scores)
    return np.select([scores < APPROVE_BELOW, scores < REFUSE_FROM], ["low", "medium"], "high")


DECISIONS = {"low": "approve", "medium": "review", "high": "refuse"}


def score(model, spec: dict, batch: pd.DataFrame, top_n: int = 5):
    """Score a batch and explain it. Pure pandas: returns (scores, explanations)."""
    X = prepare(batch, spec)
    probabilities = model.predict_proba(X)[:, 1]
    bands = risk_band(probabilities)
    scores = pd.DataFrame(
        {
            "SK_ID_CURR": batch["SK_ID_CURR"].to_numpy(),
            "pd_score": probabilities,
            "risk_band": bands,
            "decision": [DECISIONS[b] for b in bands],
        }
    )
    contributions, base_value = explain(model, X)
    explanations = reason_codes(batch["SK_ID_CURR"].to_numpy(), X, contributions, top_n)
    return scores, explanations.assign(base_value=base_value)


def _replace_batch(spark, frame: pd.DataFrame, schema: str, table: str, batch_id: int) -> None:
    """Write one batch so that re-running it replaces its rows instead of duplicating them."""
    columns = [field.split()[0] for field in schema.split(", ")]
    writer = spark.createDataFrame(frame[columns], schema).write.format("delta")
    if spark.catalog.tableExists(table):
        writer.mode("overwrite").option("replaceWhere", f"batch_id = {batch_id}").saveAsTable(table)
    else:
        writer.mode("append").saveAsTable(table)


def score_batch(spark, config: Config, batch_id: int, champion=None) -> dict:
    """Score one batch with the champion and store its scores and explanations."""
    model, spec, version = champion or load_champion(config)
    batch = (
        spark.table(config.table("gold", "scoring_batches"))
        .where(f"batch_id = {int(batch_id)}")
        .toPandas()
    )
    if batch.empty:
        raise ValueError(f"Batch {batch_id} does not exist in gold.scoring_batches")
    scores, explanations = score(model, spec, batch)

    scores = scores.assign(
        batch_id=batch_id,
        score_date=batch["score_date"].to_numpy(),
        drift_scenario=batch["drift_scenario"].to_numpy(),
        # Known here because batches are simulated from labelled data. In production
        # the outcome arrives months later and is joined to the stored score then.
        observed_default=batch[TARGET].astype(int).to_numpy(),
        model_name=config.model_name,
        model_version=str(version.version),
        scored_at=pd.Timestamp.now(tz="UTC").tz_localize(None),
    )
    explanations = explanations.assign(batch_id=batch_id, model_version=str(version.version))
    _replace_batch(spark, scores, SCORES_SCHEMA, config.table("gold", "scores_history"), batch_id)
    _replace_batch(
        spark,
        explanations,
        EXPLANATIONS_SCHEMA,
        config.table("gold", "score_explanations"),
        batch_id,
    )

    stats = {
        "batch_id": batch_id,
        "scenario": batch["drift_scenario"].iloc[0],
        "applicants": len(scores),
        "mean_score": float(scores["pd_score"].mean()),
        "approve_rate": float((scores["decision"] == "approve").mean()),
        "refuse_rate": float((scores["decision"] == "refuse").mean()),
        "auc": evaluate(scores["observed_default"], scores["pd_score"])["auc"],
    }
    print(
        f"scoring batch {batch_id} ({stats['scenario']}): {stats['applicants']:,} applicants, "
        f"mean score {stats['mean_score']:.4f}, approved {stats['approve_rate']:.1%}, "
        f"refused {stats['refuse_rate']:.1%}, AUC {stats['auc']:.4f}"
    )
    return stats


def run(spark, config: Config) -> list[dict]:
    """Simulate the batches, then score each one with the champion."""
    setup_mlflow(config, spark)
    simulate_batches(spark, config)
    champion = load_champion(config)
    return [score_batch(spark, config, batch_id, champion) for batch_id in SCENARIOS]


if __name__ == "__main__":
    from credit_risk.spark import get_spark

    run(get_spark(), load_config())
