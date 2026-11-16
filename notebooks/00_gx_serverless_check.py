# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Great Expectations on serverless: compatibility check
# MAGIC Serverless compute runs over Spark Connect, which Great Expectations supports only
# MAGIC partially. This notebook runs one expectation of each kind used by the pipeline on a
# MAGIC tiny DataFrame. Run it once before the quality gates; every row should show `ran = True`.

# COMMAND ----------

# MAGIC %pip install -q "great_expectations>=1.3,<2"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import great_expectations as gx
import great_expectations.expectations as gxe

df = spark.createDataFrame(  # noqa: F821 (spark is provided by Databricks)
    [(1, "a", 10.0), (2, "b", None), (2, "c", -5.0)], "id int, category string, amount double"
)
checks = {
    "row count": gxe.ExpectTableRowCountToBeBetween(min_value=1),
    "columns match set": gxe.ExpectTableColumnsToMatchSet(column_set=["id", "category", "amount"]),
    "not null": gxe.ExpectColumnValuesToNotBeNull(column="amount", mostly=0.5),
    "unique": gxe.ExpectColumnValuesToBeUnique(column="id"),
    "in set": gxe.ExpectColumnValuesToBeInSet(column="category", value_set=["a", "b"]),
    "between": gxe.ExpectColumnValuesToBeBetween(column="amount", min_value=0),
}
# Expected outcome of each check on the DataFrame above.
expected = {
    "row count": True,
    "columns match set": True,
    "not null": True,
    "unique": False,
    "in set": False,
    "between": False,
}

context = gx.get_context(mode="ephemeral")
source = context.data_sources.add_spark(name="spark", persist=False)

report = []
for i, (name, expectation) in enumerate(checks.items()):
    try:
        batch = source.add_dataframe_asset(name=f"a{i}").add_batch_definition_whole_dataframe("b")
        result = batch.get_batch(batch_parameters={"dataframe": df}).validate(expectation)
        report.append((name, True, result.success == expected[name], ""))
    except Exception as error:  # noqa: BLE001 (any failure is what this notebook reports)
        report.append((name, False, False, f"{type(error).__name__}: {error}"[:300]))

display(  # noqa: F821
    spark.createDataFrame(report, "check string, ran boolean, correct boolean, error string")  # noqa: F821
)

# COMMAND ----------

failed = [name for name, ran, correct, _ in report if not (ran and correct)]
assert not failed, f"Great Expectations does not work here for: {failed}"
print("Great Expectations works on this compute for every kind of check the pipeline uses.")
