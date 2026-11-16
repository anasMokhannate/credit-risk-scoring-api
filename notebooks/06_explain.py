# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Explain the champion model with SHAP
# MAGIC Computes SHAP contributions on a sample of the holdout, checks that they add up to
# MAGIC the model's score, adds the global importance table and plots to the champion's MLflow
# MAGIC run, and writes `gold.shap_global_importance`.

# COMMAND ----------

# MAGIC %pip install -q "lightgbm>=4" "mlflow>=2.15" "shap>=0.46"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from credit_risk.config import load_config
from credit_risk.explain import run

# COMMAND ----------

config = load_config()
importance = run(spark, config)  # noqa: F821 (spark is provided by Databricks)

# COMMAND ----------

display(importance.head(30))  # noqa: F821
