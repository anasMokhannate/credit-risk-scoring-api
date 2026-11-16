"""Bronze layer: land the raw Home Credit CSVs in Delta tables, unchanged.

Every column is kept as a string so that nothing is lost or silently coerced at
this stage; typing and cleaning happen in the silver layer. Two metadata columns
record where and when each row came from.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from credit_risk.config import SOURCE_TABLES, Config, load_config
from credit_risk.spark import ensure_schemas, get_spark

METADATA_COLUMNS = ("_source_file", "_ingested_at")


def read_raw(spark: SparkSession, config: Config, name: str) -> DataFrame:
    """Read one raw CSV with all columns as strings, plus ingestion metadata."""
    return (
        spark.read.option("header", True)
        .option("inferSchema", False)
        # The trailing * also accepts a gzip-compressed upload (name.csv.gz).
        .csv(config.raw_file(name) + "*")
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )


def ingest_table(spark: SparkSession, config: Config, name: str) -> int:
    """Load one source file into its bronze table and return the row count.

    The table is overwritten, so re-running the ingestion never duplicates rows.
    """
    table = config.table("bronze", name)
    (
        read_raw(spark, config, name)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(table)
    )
    return spark.table(table).count()


def ingest_all(
    spark: SparkSession, config: Config, names: tuple[str, ...] = SOURCE_TABLES
) -> dict[str, int]:
    ensure_schemas(spark, config)
    counts = {}
    for name in names:
        counts[name] = ingest_table(spark, config, name)
        print(f"bronze  {config.table('bronze', name):<45} {counts[name]:>12,} rows")
    return counts


if __name__ == "__main__":
    ingest_all(get_spark(), load_config())
