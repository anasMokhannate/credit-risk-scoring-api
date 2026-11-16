# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Gold features and quality gate 2
# MAGIC Aggregates the silver tables to one row per applicant in `gold.features`, then
# MAGIC validates the result. A critical quality failure raises and stops the pipeline.

# COMMAND ----------

# MAGIC %pip install -q "great_expectations>=1.3,<2"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from credit_risk.config import load_config
from credit_risk.features import build_features
from credit_risk.quality import run_gate

# COMMAND ----------

config = load_config()
stats = build_features(spark, config)  # noqa: F821 (spark is provided by Databricks)

# COMMAND ----------

results = run_gate(spark, config, "gold")  # noqa: F821

# COMMAND ----------

display(spark.table(config.table("gold", "features")).limit(20))  # noqa: F821
