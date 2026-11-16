import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from credit_risk.config import Config
from credit_risk.features import application_features
from credit_risk.model import select_features
from credit_risk.scoring import (
    SCENARIOS,
    apply_scenario,
    assign_batches,
    make_batches,
    recompute_derived,
    risk_band,
    score,
    score_batch,
)
from credit_risk.spark import ensure_schemas

DERIVED = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "INCOME_PER_PERSON",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_KNOWN_COUNT",
]


@pytest.fixture(scope="module")
def applicants():
    rng = np.random.default_rng(11)
    n = 1200
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + n),
            "AMT_INCOME_TOTAL": rng.uniform(50_000, 300_000, n),
            "AMT_CREDIT": rng.uniform(100_000, 900_000, n),
            "AMT_ANNUITY": rng.uniform(5_000, 50_000, n),
            "CNT_FAM_MEMBERS": rng.integers(1, 5, n).astype(float),
            "EXT_SOURCE_1": np.where(rng.random(n) < 0.5, np.nan, rng.random(n)),
            "EXT_SOURCE_2": rng.random(n),
            "EXT_SOURCE_3": np.where(rng.random(n) < 0.2, np.nan, rng.random(n)),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        }
    )
    frame = recompute_derived(frame)
    risk = 1 / (1 + np.exp(2.0 + 4 * (frame["EXT_SOURCE_MEAN"] - 0.5)))
    return frame.assign(TARGET=(rng.random(n) < risk).astype(int), split="holdout")


@pytest.fixture(scope="module")
def champion(applicants):
    spec = select_features(applicants)
    X = applicants[spec["features"]].copy()
    X["NAME_CONTRACT_TYPE"] = pd.Categorical(
        X["NAME_CONTRACT_TYPE"], categories=spec["categorical"]["NAME_CONTRACT_TYPE"]
    )
    model = lgb.LGBMClassifier(n_estimators=40, num_leaves=7, min_child_samples=5, verbose=-1)
    return model.fit(X, applicants["TARGET"]), spec


# --------------------------------------------------------------------- simulation


def test_batches_are_stable_and_cover_everyone(applicants):
    ids = applicants["SK_ID_CURR"]
    batches = assign_batches(ids)

    assert set(batches) == set(SCENARIOS)
    assert (assign_batches(ids.sample(frac=1, random_state=0).sort_index()) == batches).all()
    assert np.bincount(batches)[1:].min() > len(ids) / len(SCENARIOS) * 0.7


def test_derived_features_match_the_gold_layer(spark, applicants):
    """The pandas recomputation must give what the Spark feature code gives."""
    raw = applicants.drop(columns=[*DERIVED, "TARGET", "split"]).head(200)
    raw = raw.assign(AMT_GOODS_PRICE=1.0, DAYS_EMPLOYED=-1, DAYS_BIRTH=-9000)
    gold = application_features(spark.createDataFrame(raw)).toPandas().sort_values("SK_ID_CURR")
    recomputed = recompute_derived(raw).sort_values("SK_ID_CURR")

    for column in DERIVED:
        np.testing.assert_allclose(
            recomputed[column].to_numpy(dtype=float),
            gold[column].to_numpy(dtype=float),
            err_msg=column,
        )


def test_income_scenario_changes_income_and_its_ratios_only(applicants):
    drifted = apply_scenario(applicants, "income_down_15pct")

    np.testing.assert_allclose(drifted["AMT_INCOME_TOTAL"], applicants["AMT_INCOME_TOTAL"] * 0.85)
    np.testing.assert_allclose(
        drifted["CREDIT_INCOME_RATIO"], applicants["CREDIT_INCOME_RATIO"] / 0.85
    )
    pd.testing.assert_series_equal(drifted["EXT_SOURCE_MEAN"], applicants["EXT_SOURCE_MEAN"])
    pd.testing.assert_series_equal(drifted["AMT_CREDIT"], applicants["AMT_CREDIT"])


def test_external_score_scenario_lowers_scores_and_loses_some(applicants):
    drifted = apply_scenario(applicants, "external_scores_down_15pct_and_30pct_missing", seed=6)

    lost = drifted["EXT_SOURCE_2"].isna()
    assert 0.25 < lost.mean() < 0.35
    np.testing.assert_allclose(
        drifted.loc[~lost, "EXT_SOURCE_2"], applicants.loc[~lost, "EXT_SOURCE_2"] * 0.85
    )
    # Derived columns follow: fewer known scores, and a mean rebuilt from what is left.
    assert drifted["EXT_SOURCE_KNOWN_COUNT"].mean() < applicants["EXT_SOURCE_KNOWN_COUNT"].mean()
    only_first = lost & drifted["EXT_SOURCE_1"].notna()
    np.testing.assert_allclose(
        drifted.loc[only_first, "EXT_SOURCE_MEAN"], drifted.loc[only_first, "EXT_SOURCE_1"]
    )


def test_unknown_scenario_is_rejected(applicants):
    with pytest.raises(ValueError, match="Unknown scenario"):
        apply_scenario(applicants, "meteor_strike")


def test_make_batches_keeps_every_applicant_once(applicants):
    batches = make_batches(applicants)

    assert len(batches) == len(applicants)
    assert batches["SK_ID_CURR"].is_unique
    assert batches.groupby("batch_id")["drift_scenario"].first().to_dict() == SCENARIOS
    untouched = batches[batches["batch_id"] <= 3].set_index("SK_ID_CURR")["AMT_INCOME_TOTAL"]
    original = applicants.set_index("SK_ID_CURR")["AMT_INCOME_TOTAL"]
    pd.testing.assert_series_equal(
        untouched.sort_index(), original.loc[untouched.index].sort_index()
    )


# ------------------------------------------------------------------------ scoring


def test_risk_bands_follow_the_policy():
    assert risk_band([0.01, 0.0499, 0.05, 0.1499, 0.15, 0.9]).tolist() == [
        "low",
        "low",
        "medium",
        "medium",
        "high",
        "high",
    ]


def test_score_returns_decisions_and_reasons(applicants, champion):
    model, spec = champion
    scores, explanations = score(model, spec, applicants.head(300), top_n=3)

    assert len(scores) == 300
    assert scores["pd_score"].between(0, 1).all()
    assert set(scores["decision"]) <= {"approve", "review", "refuse"}
    assert (scores.loc[scores["pd_score"] >= 0.15, "decision"] == "refuse").all()
    # At most 3 reasons per applicant, each one raising the risk.
    assert explanations.groupby("SK_ID_CURR").size().max() <= 3
    assert (explanations["shap_value"] > 0).all()
    assert set(explanations["SK_ID_CURR"]) <= set(scores["SK_ID_CURR"])


def test_rescoring_a_batch_replaces_its_rows(spark, applicants, champion):
    cfg = Config(raw_path="unused", schema_prefix="scoretest_")
    ensure_schemas(spark, cfg)
    spark.createDataFrame(make_batches(applicants)).write.format("delta").mode(
        "overwrite"
    ).saveAsTable(cfg.table("gold", "scoring_batches"))

    class Version:
        version = 7

    first = score_batch(spark, cfg, 1, (*champion, Version))
    score_batch(spark, cfg, 2, (*champion, Version))
    score_batch(spark, cfg, 1, (*champion, Version))  # run batch 1 a second time

    history = spark.table(cfg.table("gold", "scores_history"))
    assert history.where("batch_id = 1").count() == first["applicants"]
    assert history.count() == history.select("SK_ID_CURR").distinct().count()
    assert {r["model_version"] for r in history.select("model_version").distinct().collect()} == {
        "7"
    }
    reasons = spark.table(cfg.table("gold", "score_explanations"))
    assert (
        reasons.where("batch_id = 1")
        .groupBy("SK_ID_CURR", "rank")
        .count()
        .where("count > 1")
        .count()
        == 0
    )

    with pytest.raises(ValueError, match="does not exist"):
        score_batch(spark, cfg, 99, (*champion, Version))
