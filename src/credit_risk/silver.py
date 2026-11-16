"""Silver layer: typed, cleaned tables at the same grain as the source.

Silver fixes what is objectively wrong (text types, placeholder values, duplicate
rows, rows that cannot be linked to an applicant) and nothing that is a modelling
choice: no imputation, no encoding, no aggregation.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from credit_risk.config import Config, load_config
from credit_risk.schemas import SCHEMAS
from credit_risk.spark import ensure_schemas, get_spark

# Silver table -> bronze tables it is built from.
SILVER_SOURCES = {
    "application": ("application_train", "application_test"),
    "bureau": ("bureau",),
    "bureau_balance": ("bureau_balance",),
    "previous_application": ("previous_application",),
    "pos_cash_balance": ("POS_CASH_balance",),
    "installments_payments": ("installments_payments",),
    "credit_card_balance": ("credit_card_balance",),
}

# Text placeholders meaning "not available" / "not applicable".
PLACEHOLDER_STRINGS = ("XNA", "XAP")

# 365243 days (about 1,000 years) stands for "no date" in these columns.
DAYS_PLACEHOLDER = 365243
DAYS_PLACEHOLDER_COLUMNS = {
    "application": ("DAYS_EMPLOYED",),
    "previous_application": (
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ),
}


class CastError(ValueError):
    """Raised when bronze values cannot be cast to the declared silver type."""


def _try_cast(column: str, dtype: str):
    # try_cast returns null instead of failing, in both ANSI and non-ANSI mode.
    return F.expr(f"try_cast(`{column}` AS {dtype})")


def count_cast_failures(df: DataFrame, schema: dict[str, str]) -> int:
    """Count non-null bronze values that do not fit their declared type."""
    columns = [c for c, t in schema.items() if t != "string" and c in df.columns]
    if not columns:
        return 0
    failures = [
        F.sum((F.col(c).isNotNull() & _try_cast(c, t).isNull()).cast("long"))
        for c, t in ((c, schema[c]) for c in columns)
    ]
    total = sum(failures[1:], failures[0])
    return df.agg(F.coalesce(total, F.lit(0)).alias("n")).first()["n"]


def cast_columns(df: DataFrame, schema: dict[str, str], keep: tuple[str, ...] = ()) -> DataFrame:
    """Cast to the declared schema. A declared column absent from ``df`` becomes null."""
    typed = [
        (_try_cast(c, t) if c in df.columns else F.lit(None).cast(t)).alias(c)
        for c, t in schema.items()
    ]
    return df.select(*typed, *keep)


def null_placeholders(df: DataFrame, table: str) -> DataFrame:
    """Turn XNA/XAP and the 365243-day placeholder into real nulls."""
    schema = SCHEMAS[table]
    if "DAYS_EMPLOYED" in DAYS_PLACEHOLDER_COLUMNS.get(table, ()):
        # Keep the information that the applicant is not employed.
        df = df.withColumn(
            "DAYS_EMPLOYED_ANOM",
            F.coalesce(F.col("DAYS_EMPLOYED") == DAYS_PLACEHOLDER, F.lit(False)).cast("int"),
        )
    for column in DAYS_PLACEHOLDER_COLUMNS.get(table, ()):
        df = df.withColumn(column, F.when(F.col(column) != DAYS_PLACEHOLDER, F.col(column)))
    strings = [c for c, t in schema.items() if t == "string"]
    if not strings:
        # installments_payments has no text column, and Spark Connect (serverless)
        # rejects withColumns() with nothing to change.
        return df
    return df.withColumns(
        {c: F.when(~F.col(c).isin(*PLACEHOLDER_STRINGS), F.col(c)) for c in strings}
    )


def clean(raw: DataFrame, table: str, keep: tuple[str, ...] = ()) -> DataFrame:
    """Apply every silver rule that needs only the table itself."""
    schema = SCHEMAS[table]
    failures = count_cast_failures(raw, schema)
    if failures:
        raise CastError(f"{table}: {failures} values do not fit the declared schema")
    return null_placeholders(cast_columns(raw, schema, keep), table).dropDuplicates()


def read_bronze(spark: SparkSession, config: Config, table: str) -> DataFrame:
    """Read the bronze source(s) of a silver table as one DataFrame."""
    sources = SILVER_SOURCES[table]
    if table == "application":
        train, test = (spark.table(config.table("bronze", s)) for s in sources)
        return train.withColumn("dataset", F.lit("train")).unionByName(
            test.withColumn("dataset", F.lit("test")), allowMissingColumns=True
        )
    return spark.table(config.table("bronze", sources[0]))


def build_table(spark: SparkSession, config: Config, table: str) -> dict[str, int]:
    raw = read_bronze(spark, config, table)
    df = clean(raw, table, keep=("dataset",) if table == "application" else ())
    if table == "bureau_balance":
        # A monthly status whose bureau credit is unknown cannot be tied to an applicant.
        bureau = spark.table(config.table("silver", "bureau")).select("SK_ID_BUREAU")
        df = df.join(bureau, "SK_ID_BUREAU", "left_semi")
    target = config.table("silver", table)
    (
        df.withColumn("_processed_at", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(target)
    )
    return {"rows_in": raw.count(), "rows_out": spark.table(target).count()}


def build_silver(spark: SparkSession, config: Config) -> dict[str, dict[str, int]]:
    ensure_schemas(spark, config)
    stats = {}
    for table in SILVER_SOURCES:  # bureau is built before bureau_balance
        stats[table] = build_table(spark, config, table)
        s = stats[table]
        print(
            f"silver  {config.table('silver', table):<40} {s['rows_out']:>12,} rows"
            f"  ({s['rows_in'] - s['rows_out']:,} removed)"
        )
    return stats


if __name__ == "__main__":
    build_silver(get_spark(), load_config())
