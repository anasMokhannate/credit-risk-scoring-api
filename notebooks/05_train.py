# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Train the probability-of-default model
# MAGIC Cross-validates a LightGBM model on `gold.features`, evaluates it on the holdout,
# MAGIC logs everything to MLflow and registers the model in Unity Catalog. The new version
# MAGIC gets the `champion` alias only if it beats the current champion on the holdout.

# COMMAND ----------

# MAGIC %pip install -q "lightgbm>=4" "mlflow>=2.15"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from credit_risk.config import load_config
from credit_risk.model import run

# COMMAND ----------

config = load_config()
result = run(spark, config)  # noqa: F821 (spark is provided by Databricks)

# COMMAND ----------

import pandas as pd

display(  # noqa: F821
    pd.DataFrame({"cross-validation": result["cv"], "holdout": result["test"]}).reset_index(
        names="metric"
    )
)

# COMMAND ----------

# Pass the registered version to the downstream tasks of the Workflow.
dbutils.jobs.taskValues.set(key="model_version", value=result["model_version"])  # noqa: F821
