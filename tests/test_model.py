import mlflow
import numpy as np
import pandas as pd
import pytest

from credit_risk.config import Config
from credit_risk.metrics import decile_table, evaluate, ks_statistic
from credit_risk.model import (
    ModelQualityError,
    load_champion,
    prepare,
    select_features,
    setup_mlflow,
    train,
)

FAST = {"n_estimators": 60, "num_leaves": 7, "min_child_samples": 5, "n_jobs": 1}


# ------------------------------------------------------------------------ metrics


def test_perfect_ranking():
    metrics = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert (metrics["auc"], metrics["gini"], metrics["ks"]) == (1.0, 1.0, 1.0)


def test_ks_by_hand():
    # Cutting at 0.4 catches all 3 defaulters (TPR 1) and flags 1 of 4 good
    # clients (FPR 0.25): the largest gap, 0.75.
    ks, threshold = ks_statistic([0, 0, 0, 1, 0, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9])
    assert ks == pytest.approx(0.75)
    assert threshold == 0.4


def test_gini_is_twice_auc_minus_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    metrics = evaluate(y, rng.random(200))
    assert metrics["gini"] == pytest.approx(2 * metrics["auc"] - 1)


def test_decile_table():
    y = [0] * 80 + [1] * 20
    table = decile_table(y, np.linspace(0, 1, 100))

    assert table["applicants"].tolist() == [10] * 10
    assert table["defaults"].tolist() == [0] * 8 + [10, 10]
    # The riskiest band alone holds half the defaults; the top two hold all of them.
    assert table["cum_default_share"].tolist()[-2:] == [1.0, 0.5]
    assert table["cum_default_share"].iloc[0] == 1.0


# ----------------------------------------------------------------------- features


def test_select_features_drops_non_features_and_empty_columns():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "CODE_GENDER": ["F", "M", "F"],
            "AMT_CREDIT": [1.0, 2.0, 3.0],
            "CONSTANT": [7, 7, 7],
            "EMPTY": [None, None, None],
            "NAME_CONTRACT_TYPE": ["Cash loans", None, "Revolving loans"],
        }
    )
    spec = select_features(df)

    assert spec["features"] == ["AMT_CREDIT", "NAME_CONTRACT_TYPE"]
    assert spec["dropped"] == ["CONSTANT", "EMPTY"]
    assert spec["categorical"] == {"NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"]}


def test_prepare_uses_training_categories():
    spec = {"features": ["B", "A"], "categorical": {"B": ["x", "y"]}, "dropped": []}
    X = prepare(pd.DataFrame({"A": [1.0, 2.0], "B": ["y", "never seen"], "C": [0, 0]}), spec)

    assert list(X.columns) == ["B", "A"]
    assert X["B"].cat.categories.tolist() == ["x", "y"]
    assert X["B"].cat.codes.tolist() == [1, -1]  # unseen category becomes missing


def test_prepare_fails_on_a_missing_feature():
    spec = {"features": ["A", "B"], "categorical": {}, "dropped": []}
    with pytest.raises(KeyError):
        prepare(pd.DataFrame({"A": [1.0]}), spec)


# ----------------------------------------------------------------------- training


@pytest.fixture(scope="module")
def applicants():
    """Synthetic applicants whose default risk truly depends on the features."""
    rng = np.random.default_rng(7)
    n = 3000
    ext = rng.random(n)
    late = rng.random(n)
    contract = rng.choice(["Cash loans", "Revolving loans"], n)
    risk = 1 / (
        1 + np.exp(-(-2.5 - 3 * (ext - 0.5) + 3 * (late - 0.5) + (contract == "Cash loans")))
    )
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "TARGET": rng.random(n) < risk,
            "split": np.where(np.arange(n) % 5 == 0, "holdout", "dev"),
            "CODE_GENDER": rng.choice(["F", "M"], n),
            "EXT_SOURCE_1": np.where(rng.random(n) < 0.3, np.nan, ext),
            "INST_LATE_SHARE": late,
            "NAME_CONTRACT_TYPE": contract,
            "NOISE": rng.random(n),
        }
    )


@pytest.fixture(scope="module")
def tracking(tmp_path_factory):
    setup_mlflow(Config(raw_path="unused"), local_dir=tmp_path_factory.mktemp("mlflow"))
    return Config(raw_path="unused")


def test_training_logs_registers_and_promotes(applicants, tracking):
    result = train(applicants, tracking.model_name, FAST, n_splits=3, early_stopping_rounds=20)

    assert result["promoted"] is True
    assert result["n_features"] == 4  # gender is excluded
    assert len(result["fold_aucs"]) == 3
    assert result["cv"]["auc"] > 0.65 and result["test"]["auc"] > 0.65

    logged = mlflow.get_run(result["run_id"]).data
    assert logged.metrics["test_auc"] == result["test"]["auc"]
    assert logged.metrics["baseline_cv_auc"] > 0.5
    assert logged.params["num_leaves"] == "7"

    model, spec, version = load_champion(tracking)
    assert str(version.version) == result["model_version"]
    assert "CODE_GENDER" not in spec["features"]
    scores = model.predict_proba(prepare(applicants, spec))[:, 1]
    assert ((scores >= 0) & (scores <= 1)).all()


def test_same_seed_gives_the_same_model(applicants, tracking):
    first = train(applicants, "repeat_model", FAST, n_splits=3, early_stopping_rounds=20)
    second = train(applicants, "repeat_model", FAST, n_splits=3, early_stopping_rounds=20)

    assert first["test"]["auc"] == second["test"]["auc"]
    # An equal score does not beat the champion, so the first version keeps the alias.
    assert (first["promoted"], second["promoted"]) == (True, False)


def test_weak_model_is_not_registered(applicants, tracking):
    shuffled = applicants.assign(
        TARGET=np.random.default_rng(1).permutation(applicants["TARGET"].to_numpy())
    )
    with pytest.raises(ModelQualityError, match="under the floor"):
        train(shuffled, "weak_model", FAST, n_splits=3, early_stopping_rounds=20)

    assert not mlflow.MlflowClient().search_registered_models("name = 'weak_model'")
