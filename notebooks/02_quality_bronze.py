# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Quality gate 1: bronze
# MAGIC Validates the bronze tables with Great Expectations. Results are appended to
# MAGIC `gold.quality_results`; a critical failure raises and stops the pipeline.

# COMMAND ----------

# MAGIC %pip install -q "great_expectations>=1.3,<2"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from credit_risk.config import load_config
from credit_risk.quality import run_gate

# COMMAND ----------

config = load_config()
results = run_gate(spark, config, "bronze")  # noqa: F821 (spark is provided by Databricks)

# COMMAND ----------

display(  # noqa: F821
    spark.table(config.table("gold", "quality_results"))  # noqa: F821
    .where("layer = 'bronze'")
    .orderBy("validated_at", "table_name", ascending=[False, True])
)
