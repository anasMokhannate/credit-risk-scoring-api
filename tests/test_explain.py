import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from credit_risk.explain import (
    ExplanationError,
    check_additivity,
    explain,
    feature_label,
    global_importance,
    reason_codes,
)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(3)
    n = 800
    X = pd.DataFrame(
        {
            "EXT_SOURCE_2": np.where(rng.random(n) < 0.2, np.nan, rng.random(n)),
            "INST_LATE_SHARE": rng.random(n),
            "NAME_CONTRACT_TYPE": pd.Categorical(rng.choice(["Cash loans", "Revolving loans"], n)),
            "NOISE": rng.random(n),
        }
    )
    risk = 0.1 + 0.6 * X["INST_LATE_SHARE"] - 0.3 * X["EXT_SOURCE_2"].fillna(0.5)
    y = (rng.random(n) < risk.clip(0.02, 0.98)).astype(int)
    model = lgb.LGBMClassifier(n_estimators=40, num_leaves=7, min_child_samples=5, verbose=-1)
    return model.fit(X, y), X


def test_contributions_add_up_to_the_score(fitted):
    model, X = fitted
    contributions, base_value = explain(model, X)

    assert list(contributions.columns) == list(X.columns)
    raw = model.predict(X, raw_score=True)
    np.testing.assert_allclose(contributions.sum(axis=1) + base_value, raw, atol=1e-9)
    # The sigmoid of the rebuilt score is the predicted probability.
    probability = 1 / (1 + np.exp(-(contributions.sum(axis=1) + base_value)))
    np.testing.assert_allclose(probability, model.predict_proba(X)[:, 1], atol=1e-9)


def test_the_informative_feature_ranks_above_noise(fitted):
    model, X = fitted
    importance = global_importance(explain(model, X)[0])

    assert importance["feature"].iloc[0] == "INST_LATE_SHARE"
    assert importance.set_index("feature")["rank"]["NOISE"] > 1
    assert importance["share"].sum() == pytest.approx(1.0)


def test_broken_contributions_are_rejected():
    with pytest.raises(ExplanationError, match="do not add up"):
        check_additivity(np.array([[0.2, 0.1]]), base_value=-1.0, raw_scores=np.array([-0.5]))
    check_additivity(np.array([[0.2, 0.1]]), base_value=-1.0, raw_scores=np.array([-0.7]))


def test_global_importance_by_hand():
    contributions = pd.DataFrame({"A": [0.1, -0.3], "B": [-1.0, 1.0], "C": [0.0, 0.0]})
    importance = global_importance(contributions)

    assert importance["feature"].tolist() == ["B", "A", "C"]
    assert importance["mean_abs_shap"].tolist() == pytest.approx([1.0, 0.2, 0.0])
    assert importance["rank"].tolist() == [1, 2, 3]


def test_reason_codes_keep_only_what_raised_the_risk():
    X = pd.DataFrame(
        {
            "EXT_SOURCE_2": [0.1, 0.9, 0.8],
            "INST_LATE_SHARE": [0.7, 0.0, 0.0],
            "NAME_CONTRACT_TYPE": ["Cash loans", None, "Cash loans"],
        }
    )
    contributions = pd.DataFrame(
        {
            "EXT_SOURCE_2": [0.9, -0.5, -0.1],
            "INST_LATE_SHARE": [0.4, -0.2, -0.3],
            "NAME_CONTRACT_TYPE": [-0.1, 0.05, -0.2],
        }
    )
    reasons = reason_codes([101, 102, 103], X, contributions, top_n=2)

    risky = reasons[reasons["SK_ID_CURR"] == 101]
    assert risky["feature"].tolist() == ["EXT_SOURCE_2", "INST_LATE_SHARE"]  # largest first
    assert risky["rank"].tolist() == [1, 2]
    assert risky["shap_value"].tolist() == [0.9, 0.4]
    assert risky["feature_value"].tolist() == ["0.1", "0.7"]
    assert risky["reason"].iloc[1] == "Share of past instalments paid late"

    # One positive contribution only, and its value is missing.
    single = reasons[reasons["SK_ID_CURR"] == 102]
    assert single["feature"].tolist() == ["NAME_CONTRACT_TYPE"]
    assert single["feature_value"].tolist() == [None]

    # Nothing raised this applicant's risk: no reason at all.
    assert reasons[reasons["SK_ID_CURR"] == 103].empty


def test_feature_labels_fall_back_to_the_source():
    assert feature_label("EXT_SOURCE_2") == "External credit score 2"
    assert (
        feature_label("BUREAU_PROLONG_SUM")
        == "Credit history at other lenders (BUREAU_PROLONG_SUM)"
    )
    assert feature_label("FLAG_DOCUMENT_3") == "Application data (FLAG_DOCUMENT_3)"
