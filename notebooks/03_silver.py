# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Silver
# MAGIC Builds the typed, cleaned silver tables from bronze, then validates them.
# MAGIC A critical quality failure raises and stops the pipeline.

# COMMAND ----------

# MAGIC %pip install -q "great_expectations>=1.3,<2"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from credit_risk.config import load_config
from credit_risk.quality import run_gate
from credit_risk.silver import build_silver

# COMMAND ----------

config = load_config()
stats = build_silver(spark, config)  # noqa: F821 (spark is provided by Databricks)

# COMMAND ----------

results = run_gate(spark, config, "silver")  # noqa: F821

# COMMAND ----------

display(  # noqa: F821
    spark.table(config.table("gold", "quality_results"))  # noqa: F821
    .where("layer = 'silver' AND NOT success")
    .orderBy("validated_at", ascending=False)
)
