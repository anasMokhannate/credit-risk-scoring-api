"""Probability-of-default model: LightGBM, cross-validated, tracked in MLflow.

The training population is split once, deterministically, by a hash of the
applicant id:

* ``dev`` (about 80%): 5-fold stratified cross-validation, then the final fit.
* ``holdout`` (about 20%): never used for fitting or early stopping. It gives the
  test metrics, and later feeds the simulated scoring batches.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.config import Config, load_config
from credit_risk.metrics import decile_table, evaluate, ks_statistic
from credit_risk.plots import evaluation_figures

TARGET = "TARGET"
CHAMPION = "champion"
HOLDOUT_BUCKETS = 5  # 1 hash bucket in 5 is held out

# Not features: identifiers, the label, pipeline metadata, and gender, which must
# not drive a credit decision (it is kept in gold to measure fairness afterwards).
EXCLUDED = ("SK_ID_CURR", TARGET, "dataset", "split", "_processed_at", "CODE_GENDER")

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 3000,  # upper bound: early stopping picks the real number
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

BASELINE_FEATURES = (
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AGE_YEARS",
    "DAYS_EMPLOYED",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
)


class ModelQualityError(RuntimeError):
    """Raised when a trained model is too weak to be registered."""


# --------------------------------------------------------------------------- data


def load_labelled(spark, config: Config) -> pd.DataFrame:
    """Labelled applicants as pandas, with their ``split`` (dev or holdout)."""
    from pyspark.sql import functions as F

    bucket = F.pmod(F.xxhash64("SK_ID_CURR"), F.lit(HOLDOUT_BUCKETS))
    return (
        spark.table(config.table("gold", "features"))
        .where("dataset = 'train'")
        .withColumn("split", F.when(bucket == 0, "holdout").otherwise("dev"))
        .drop("_processed_at")
        .toPandas()
    )


def select_features(df: pd.DataFrame) -> dict:
    """Choose the model's inputs from ``df`` and record how to rebuild them.

    Columns that are empty or constant carry no information and are dropped. The
    returned spec is stored with the model so that scoring prepares its input in
    exactly the same way.
    """
    candidates = [c for c in df.columns if c not in EXCLUDED]
    dropped = [c for c in candidates if df[c].nunique(dropna=True) <= 1]
    features = [c for c in candidates if c not in dropped]
    categorical = {
        c: sorted(df[c].dropna().unique().tolist())
        for c in features
        if not pd.api.types.is_numeric_dtype(df[c])
    }
    return {"features": features, "categorical": categorical, "dropped": dropped}


def prepare(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Build the model input: the spec's columns, in order, with fixed categories.

    A category unseen at training time becomes missing, which LightGBM handles.
    """
    X = df[spec["features"]].copy()
    for column, categories in spec["categorical"].items():
        X[column] = pd.Categorical(X[column], categories=categories)
    return X


# ----------------------------------------------------------------------- training


def cross_validate(X, y, params: dict, n_splits: int = 5, early_stopping_rounds: int = 100):
    """Stratified K-fold with early stopping.

    Returns out-of-fold predictions (each row predicted by the model that did not
    see it), the AUC of each fold, and the number of trees each fold settled on.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=params["random_state"])
    oof = np.zeros(len(y))
    aucs, iterations = [], []
    for train_idx, valid_idx in folds.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        aucs.append(float(roc_auc_score(y.iloc[valid_idx], oof[valid_idx])))
        iterations.append(int(model.best_iteration_))
    return oof, aucs, iterations


def baseline_cv_auc(df: pd.DataFrame, y, n_splits: int = 5, seed: int = 42) -> float:
    """Out-of-fold AUC of a small logistic regression: what the complexity must beat."""
    columns = [c for c in BASELINE_FEATURES if c in df.columns]
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipeline, df[columns], y, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, oof))


def promote_if_better(client, model_name: str, version: str, test_auc: float) -> bool:
    """Give ``version`` the champion alias unless the current champion tests better."""
    try:
        current = client.get_model_version_by_alias(model_name, CHAMPION)
    except MlflowException:
        current = None
    if current is not None:
        champion_auc = client.get_run(current.run_id).data.metrics.get("test_auc", -math.inf)
        if test_auc <= champion_auc:
            return False
    client.set_registered_model_alias(model_name, CHAMPION, version)
    return True


def train(
    df: pd.DataFrame,
    model_name: str,
    params: dict | None = None,
    n_splits: int = 5,
    early_stopping_rounds: int = 100,
    min_cv_auc: float = 0.70,
) -> dict:
    """Cross-validate, fit the final model, evaluate on the holdout, log and register.

    ``df`` holds the labelled applicants with a ``split`` column. An MLflow
    experiment must already be active (see ``setup_mlflow``).
    """
    params = {**PARAMS, **(params or {})}
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    holdout = df[df["split"] == "holdout"].reset_index(drop=True)
    spec = select_features(dev)
    X_dev, y_dev = prepare(dev, spec), dev[TARGET].astype(int)
    X_test, y_test = prepare(holdout, spec), holdout[TARGET].astype(int)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "n_splits": n_splits,
                "early_stopping_rounds": early_stopping_rounds,
                "n_features": len(spec["features"]),
                "n_categorical": len(spec["categorical"]),
                "rows_dev": len(dev),
                "rows_holdout": len(holdout),
            }
        )
        mlflow.log_dict(spec, "feature_spec.json")

        # 1. Honest estimate of performance, and the number of trees to use.
        oof, fold_aucs, iterations = cross_validate(
            X_dev, y_dev, params, n_splits, early_stopping_rounds
        )
        cv = evaluate(y_dev, oof)
        for fold, auc in enumerate(fold_aucs):
            mlflow.log_metric("fold_auc", auc, step=fold)
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv.items()})
        mlflow.log_metrics(
            {
                "cv_auc_mean": float(np.mean(fold_aucs)),
                "cv_auc_std": float(np.std(fold_aucs)),
                "baseline_cv_auc": baseline_cv_auc(dev, y_dev, n_splits, params["random_state"]),
            }
        )
        if cv["auc"] < min_cv_auc:
            raise ModelQualityError(
                f"Cross-validated AUC {cv['auc']:.3f} is under the floor of {min_cv_auc}: "
                "the model is not registered. Check the feature table."
            )

        # 2. Final model on all dev rows, with the tree count found by cross-validation.
        n_estimators = round(float(np.mean(iterations)))
        model = lgb.LGBMClassifier(**{**params, "n_estimators": n_estimators})
        model.fit(X_dev, y_dev)
        mlflow.log_metric("final_n_estimators", n_estimators)
        mlflow.log_metric("train_auc", roc_auc_score(y_dev, model.predict_proba(X_dev)[:, 1]))

        # 3. The holdout is scored once, by the final model.
        test_scores = model.predict_proba(X_test)[:, 1]
        test = evaluate(y_test, test_scores)
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        # A neutral cut-off: the score that best separates the two populations,
        # chosen on out-of-fold predictions, never on the holdout.
        ks_threshold = ks_statistic(y_dev, oof)[1]
        mlflow.log_metric("ks_threshold", ks_threshold)
        mlflow.log_table(decile_table(y_test, test_scores), "test_default_rate_by_band.json")
        for name, figure in evaluation_figures(y_test, test_scores).items():
            mlflow.log_figure(figure, f"plots/test_{name}.png")
            plt.close(figure)

        # 4. Package and register.
        sample = X_dev.head(5)
        info = mlflow.lightgbm.log_model(
            model,
            name="model",
            signature=infer_signature(sample, model.predict_proba(sample)[:, 1]),
        )
        version = mlflow.register_model(info.model_uri, model_name).version
        promoted = promote_if_better(mlflow.MlflowClient(), model_name, version, test["auc"])
        mlflow.set_tag("promoted_to_champion", str(promoted))

    return {
        "run_id": run.info.run_id,
        "model_name": model_name,
        "model_version": str(version),
        "promoted": promoted,
        "n_features": len(spec["features"]),
        "dropped": spec["dropped"],
        "n_estimators": n_estimators,
        "fold_aucs": fold_aucs,
        "cv": cv,
        "test": test,
        "train_auc": float(mlflow.get_run(run.info.run_id).data.metrics["train_auc"]),
        "baseline_cv_auc": float(mlflow.get_run(run.info.run_id).data.metrics["baseline_cv_auc"]),
        "ks_threshold": ks_threshold,
    }


# ------------------------------------------------------------------------- MLflow


def setup_mlflow(config: Config, spark=None, local_dir: str | Path = ".local") -> None:
    """Point MLflow at Databricks (Unity Catalog registry) or at a local store."""
    if config.catalog:
        mlflow.set_registry_uri("databricks-uc")
        user = spark.sql("SELECT current_user()").first()[0]
        mlflow.set_experiment(f"/Users/{user}/credit_risk")
        return
    root = Path(local_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("credit_risk") is None:
        mlflow.create_experiment("credit_risk", artifact_location=(root / "mlartifacts").as_uri())
    mlflow.set_experiment("credit_risk")


def load_champion(config: Config):
    """Load the champion model with the feature spec it was trained with."""
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(config.model_name, CHAMPION)
    model = mlflow.lightgbm.load_model(f"models:/{config.model_name}@{CHAMPION}")
    spec = json.loads(mlflow.artifacts.load_text(f"runs:/{version.run_id}/feature_spec.json"))
    return model, spec, version


def run(spark, config: Config) -> dict:
    setup_mlflow(config, spark)
    result = train(load_labelled(spark, config), config.model_name)
    print(
        f"model   {result['model_name']} v{result['model_version']}"
        f"  cv AUC {result['cv']['auc']:.4f}  test AUC {result['test']['auc']:.4f}"
        f"  champion: {result['promoted']}"
    )
    return result


if __name__ == "__main__":
    from credit_risk.spark import get_spark

    summary = run(get_spark(), load_config())
    print(json.dumps(summary, indent=1))
