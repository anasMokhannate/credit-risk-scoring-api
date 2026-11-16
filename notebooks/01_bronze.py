# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Bronze ingestion
# MAGIC Lands the raw Home Credit CSVs from the Volume into `bronze` Delta tables, unchanged.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from credit_risk.config import load_config
from credit_risk.ingestion import ingest_all

# COMMAND ----------

config = load_config()
counts = ingest_all(spark, config)  # noqa: F821 (spark is provided by Databricks)

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {config.schema('bronze')}"))  # noqa: F821
