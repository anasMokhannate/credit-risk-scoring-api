import great_expectations.expectations as gxe
import pytest

from credit_risk.config import SOURCE_TABLES, Config
from credit_risk.quality import (
    CRITICAL,
    WARNING,
    QualityGateError,
    bronze_suites,
    run_gate,
    silver_suites,
    validate,
)
from credit_risk.silver import SILVER_SOURCES
from credit_risk.spark import ensure_schemas


@pytest.fixture(scope="module")
def cfg(spark):
    cfg = Config(raw_path="unused", schema_prefix="qualitytest_")
    ensure_schemas(spark, cfg)
    return cfg


@pytest.fixture(scope="module")
def clients(spark, cfg):
    # Client 2 appears twice and one amount is missing.
    spark.createDataFrame(
        [(1, 100.0), (2, None), (2, 300.0)], "id int, amount double"
    ).write.format("delta").mode("overwrite").saveAsTable(cfg.table("silver", "clients"))
    return spark.table(cfg.table("silver", "clients"))


def test_validate_reports_each_expectation(clients):
    results = validate(
        clients,
        [
            gxe.ExpectColumnValuesToBeUnique(column="id", meta=CRITICAL),
            gxe.ExpectColumnValuesToNotBeNull(column="amount", mostly=0.5, meta=WARNING),
        ],
    )
    by_type = {r["expectation"]: r for r in results}

    unique = by_type["expect_column_values_to_be_unique"]
    assert (unique["success"], unique["severity"], unique["unexpected_count"]) == (
        False,
        "critical",
        2,
    )
    not_null = by_type["expect_column_values_to_not_be_null"]
    assert (not_null["success"], not_null["severity"], not_null["column_name"]) == (
        True,
        "warning",
        "amount",
    )


def test_critical_failure_stops_the_pipeline_after_recording_it(spark, cfg, clients):
    suites = {"clients": [gxe.ExpectColumnValuesToBeUnique(column="id", meta=CRITICAL)]}

    with pytest.raises(QualityGateError, match="clients.id"):
        run_gate(spark, cfg, "silver", suites, run_id="critical-run")

    stored = spark.table(cfg.table("gold", "quality_results")).where("run_id = 'critical-run'")
    assert [(r["table_name"], r["success"]) for r in stored.collect()] == [("clients", False)]


def test_warning_failure_is_recorded_without_stopping(spark, cfg, clients):
    suites = {"clients": [gxe.ExpectColumnValuesToNotBeNull(column="amount", meta=WARNING)]}

    rows = run_gate(spark, cfg, "silver", suites, run_id="warning-run")

    assert [(r["success"], r["severity"], r["unexpected_count"]) for r in rows] == [
        (False, "warning", 1)
    ]
    stored = spark.table(cfg.table("gold", "quality_results")).where("run_id = 'warning-run'")
    assert stored.count() == 1


def test_every_table_has_a_suite():
    assert set(bronze_suites()) == set(SOURCE_TABLES)
    assert set(silver_suites()) == set(SILVER_SOURCES)
