# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Batch scoring
# MAGIC Simulates six monthly batches from the holdout (the last three with a deliberate
# MAGIC change), scores each with the champion model, and stores the scores in
# MAGIC `gold.scores_history` and the reasons behind them in `gold.score_explanations`.

# COMMAND ----------

# MAGIC %pip install -q "lightgbm>=4" "mlflow>=2.15" "shap>=0.46"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from credit_risk.config import load_config
from credit_risk.scoring import run

# COMMAND ----------

config = load_config()
stats = run(spark, config)  # noqa: F821 (spark is provided by Databricks)

# COMMAND ----------

import pandas as pd

display(pd.DataFrame(stats))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: the reasons behind the five highest scores of the last batch

# COMMAND ----------

display(  # noqa: F821
    spark.sql(  # noqa: F821
        f"""
        SELECT s.SK_ID_CURR, round(s.pd_score, 3) AS pd_score, s.decision,
               e.rank, e.reason, e.feature_value, round(e.shap_value, 3) AS shap_value
        FROM {config.table("gold", "scores_history")} s
        JOIN {config.table("gold", "score_explanations")} e USING (SK_ID_CURR, batch_id)
        WHERE s.batch_id = 6
        QUALIFY dense_rank() OVER (ORDER BY s.pd_score DESC) <= 5
        ORDER BY s.pd_score DESC, e.rank
        """
    )
)
