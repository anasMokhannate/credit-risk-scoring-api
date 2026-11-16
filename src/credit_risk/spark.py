"""Spark session and schema helpers shared by every pipeline step."""

from __future__ import annotations

import os
from pathlib import Path

from pyspark.sql import SparkSession

from credit_risk.config import LAYERS, Config, on_databricks


def get_spark(local_dir: str | Path = ".local") -> SparkSession:
    """Return the Databricks session, or a local Delta-enabled one.

    Locally, tables and the metastore are persisted under ``local_dir`` so that
    each pipeline step can be run as a separate process.
    """
    if on_databricks():
        return SparkSession.builder.getOrCreate()

    import pyspark
    from delta import configure_spark_with_delta_pip

    # Use the Spark bundled with the pip package: a system-wide SPARK_HOME pointing
    # at another Spark version would otherwise be launched and fail on a mismatch.
    os.environ["SPARK_HOME"] = str(Path(pyspark.__file__).parent)

    root = Path(local_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    builder = (
        SparkSession.builder.master("local[*]")
        .appName("credit-risk")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.warehouse.dir", str(root / "warehouse"))
        .config(
            "spark.hadoop.javax.jdo.option.ConnectionURL",
            f"jdbc:derby:;databaseName={root / 'metastore_db'};create=true",
        )
        .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={root}")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .enableHiveSupport()
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def ensure_schemas(spark: SparkSession, config: Config) -> None:
    """Create the bronze/silver/gold schemas if they do not exist."""
    for layer in LAYERS:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.schema(layer)}")
