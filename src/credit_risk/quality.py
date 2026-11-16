"""Data quality gates built on Great Expectations.

A gate validates every table of a layer against its expectation suite, stores one
row per expectation in a Delta table, and raises if a critical expectation failed.
Warning-level failures are recorded but do not stop the pipeline.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")

import great_expectations as gx
import great_expectations.expectations as gxe
from pyspark.sql import DataFrame, SparkSession

from credit_risk.config import SOURCE_TABLES, Config, load_config
from credit_risk.ingestion import METADATA_COLUMNS
from credit_risk.schemas import SCHEMAS
from credit_risk.silver import DAYS_PLACEHOLDER, DAYS_PLACEHOLDER_COLUMNS
from credit_risk.spark import ensure_schemas, get_spark

CRITICAL = {"severity": "critical"}
WARNING = {"severity": "warning"}

RESULTS_SCHEMA = (
    "run_id string, validated_at timestamp, layer string, table_name string, "
    "expectation string, column_name string, severity string, success boolean, "
    "observed_value string, element_count long, unexpected_count long, unexpected_percent double"
)

# Bronze table -> (columns that must never be null, column that must be unique or None).
BRONZE_KEYS = {
    "application_train": (("SK_ID_CURR", "TARGET"), "SK_ID_CURR"),
    "application_test": (("SK_ID_CURR",), "SK_ID_CURR"),
    "bureau": (("SK_ID_CURR", "SK_ID_BUREAU"), "SK_ID_BUREAU"),
    "bureau_balance": (("SK_ID_BUREAU", "MONTHS_BALANCE"), None),
    "previous_application": (("SK_ID_CURR", "SK_ID_PREV"), "SK_ID_PREV"),
    "POS_CASH_balance": (("SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE"), None),
    "installments_payments": (("SK_ID_CURR", "SK_ID_PREV"), None),
    "credit_card_balance": (("SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE"), None),
}
_BRONZE_SCHEMA = {
    "application_train": "application",
    "application_test": "application",
    "POS_CASH_balance": "pos_cash_balance",
}


class QualityGateError(RuntimeError):
    """Raised when at least one critical expectation fails."""


def bronze_suites() -> dict[str, list]:
    """Gate 1: is the raw data structurally sound enough to be cleaned?"""
    suites = {}
    for table in SOURCE_TABLES:
        not_null, unique = BRONZE_KEYS[table]
        columns = list(SCHEMAS[_BRONZE_SCHEMA.get(table, table)])
        if table == "application_test":
            columns.remove("TARGET")
        suite = [
            gxe.ExpectTableRowCountToBeBetween(min_value=1, meta=CRITICAL),
            gxe.ExpectTableColumnsToMatchSet(
                column_set=[*columns, *METADATA_COLUMNS], exact_match=True, meta=CRITICAL
            ),
            *[gxe.ExpectColumnValuesToNotBeNull(column=c, meta=CRITICAL) for c in not_null],
        ]
        if unique:
            suite.append(gxe.ExpectColumnValuesToBeUnique(column=unique, meta=CRITICAL))
        if table == "application_train":
            suite.append(
                gxe.ExpectColumnValuesToBeInSet(
                    column="TARGET", value_set=["0", "1"], meta=CRITICAL
                )
            )
        suites[table] = suite
    return suites


def _between(column, low=None, high=None, meta=CRITICAL, **kwargs):
    return gxe.ExpectColumnValuesToBeBetween(
        column=column, min_value=low, max_value=high, meta=meta, **kwargs
    )


def _in_set(column, values, meta=CRITICAL, **kwargs):
    return gxe.ExpectColumnValuesToBeInSet(column=column, value_set=values, meta=meta, **kwargs)


def _not_null(column, meta=CRITICAL, **kwargs):
    return gxe.ExpectColumnValuesToNotBeNull(column=column, meta=meta, **kwargs)


def _unique(column):
    return gxe.ExpectColumnValuesToBeUnique(column=column, meta=CRITICAL)


def silver_suites() -> dict[str, list]:
    """Checks on typed data: keys, value ranges, known categories, placeholder removal."""
    no_placeholder = {
        table: [_between(c, high=DAYS_PLACEHOLDER - 1) for c in columns]
        for table, columns in DAYS_PLACEHOLDER_COLUMNS.items()
    }
    return {
        "application": [
            _not_null("SK_ID_CURR"),
            _unique("SK_ID_CURR"),
            _in_set("dataset", ["train", "test"]),
            _in_set("TARGET", [0, 1]),
            _in_set("CODE_GENDER", ["F", "M"]),
            _between("AMT_CREDIT", low=0, strict_min=True),
            _between("AMT_INCOME_TOTAL", low=0, strict_min=True),
            # Applicants are between 18 and 100 years old.
            _between("DAYS_BIRTH", low=-100 * 365, high=-18 * 365),
            _between("DAYS_EMPLOYED", high=0),
            *[_between(f"EXT_SOURCE_{i}", low=0, high=1) for i in (1, 2, 3)],
            _not_null("AMT_ANNUITY", meta=WARNING, mostly=0.99),
            _not_null("EXT_SOURCE_2", meta=WARNING, mostly=0.99),
            _not_null("EXT_SOURCE_3", meta=WARNING, mostly=0.75),
            *no_placeholder["application"],
        ],
        "bureau": [
            _not_null("SK_ID_CURR"),
            _not_null("SK_ID_BUREAU"),
            _unique("SK_ID_BUREAU"),
            _between("DAYS_CREDIT", high=0),
            _in_set("CREDIT_ACTIVE", ["Active", "Closed", "Sold", "Bad debt"]),
            _between("AMT_CREDIT_SUM", low=0, meta=WARNING),
        ],
        "bureau_balance": [
            _not_null("SK_ID_BUREAU"),
            _between("MONTHS_BALANCE", high=0),
            _in_set("STATUS", ["C", "X", "0", "1", "2", "3", "4", "5"]),
        ],
        "previous_application": [
            _not_null("SK_ID_CURR"),
            _not_null("SK_ID_PREV"),
            _unique("SK_ID_PREV"),
            _between("DAYS_DECISION", high=0),
            _in_set("NAME_CONTRACT_STATUS", ["Approved", "Refused", "Canceled", "Unused offer"]),
            *no_placeholder["previous_application"],
        ],
        "pos_cash_balance": [
            _not_null("SK_ID_CURR"),
            _not_null("SK_ID_PREV"),
            _between("MONTHS_BALANCE", high=0),
            _between("SK_DPD", low=0),
        ],
        "installments_payments": [
            _not_null("SK_ID_CURR"),
            _not_null("SK_ID_PREV"),
            _between("DAYS_INSTALMENT", high=0),
            _between("AMT_INSTALMENT", low=0),
            _not_null("AMT_PAYMENT", meta=WARNING, mostly=0.99),
        ],
        "credit_card_balance": [
            _not_null("SK_ID_CURR"),
            _not_null("SK_ID_PREV"),
            _between("MONTHS_BALANCE", high=0),
            _between("AMT_CREDIT_LIMIT_ACTUAL", low=0),
        ],
    }


def gold_suites() -> dict[str, list]:
    """Gate 2: is the feature table fit for modelling?"""
    shares = (
        "BUREAU_ACTIVE_SHARE",
        "BUREAU_BB_LATE_SHARE_MEAN",
        "PREV_APPROVED_SHARE",
        "PREV_REFUSED_SHARE",
        "INST_LATE_SHARE",
        "INST_LATE_SHARE_12M",
        "POS_DPD_SHARE",
        "CC_MIN_PAYMENT_MISSED_SHARE",
        "EXT_SOURCE_MEAN",
    )
    counts = ("BUREAU_COUNT", "PREV_COUNT", "INST_COUNT", "POS_LOAN_COUNT", "CC_CARD_COUNT")
    return {
        "features": [
            gxe.ExpectTableRowCountToBeBetween(min_value=1, meta=CRITICAL),
            _not_null("SK_ID_CURR"),
            _unique("SK_ID_CURR"),
            _in_set("dataset", ["train", "test"]),
            _in_set("TARGET", [0, 1]),
            _between("AGE_YEARS", low=18, high=100),
            _between("CREDIT_INCOME_RATIO", low=0, strict_min=True),
            _between("ANNUITY_INCOME_RATIO", low=0, strict_min=True),
            _between("EMPLOYED_AGE_RATIO", low=0, high=1),
            *[_between(c, low=0, high=1) for c in shares],
            *[_between(c, low=1) for c in counts],
            _between("INST_DAYS_LATE_MEAN", low=0),
            _between("INST_UNDERPAID_SUM", low=0),
            # Most applicants have a history; a collapse means a join or an upstream load broke.
            _not_null("BUREAU_COUNT", meta=WARNING, mostly=0.80),
            _not_null("PREV_COUNT", meta=WARNING, mostly=0.90),
            _not_null("INST_COUNT", meta=WARNING, mostly=0.90),
            # The default rate of the labelled population is about 8%.
            gxe.ExpectColumnMeanToBeBetween(
                column="TARGET", min_value=0.05, max_value=0.12, meta=WARNING
            ),
        ]
    }


SUITES = {"bronze": bronze_suites, "silver": silver_suites, "gold": gold_suites}


def _context():
    context = gx.get_context(mode="ephemeral")
    # persist=False: caching the DataFrame is not available on serverless compute.
    return context, context.data_sources.add_spark(name="spark", persist=False)


def validate(df: DataFrame, expectations: list, name: str = "table", context=None) -> list[dict]:
    """Run expectations against a DataFrame and return one dict per expectation."""
    context, source = context or _context()
    uid = f"{name}_{uuid.uuid4().hex[:8]}"
    batch = source.add_dataframe_asset(name=uid).add_batch_definition_whole_dataframe(uid)
    suite = context.suites.add(gx.ExpectationSuite(name=uid, expectations=expectations))
    definition = context.validation_definitions.add(
        gx.ValidationDefinition(name=uid, data=batch, suite=suite)
    )
    outcome = definition.run(batch_parameters={"dataframe": df}, result_format="BASIC")
    rows = []
    for item in outcome.results:
        item = item.to_json_dict()
        expectation, result = item["expectation_config"], item.get("result") or {}
        observed = result.get("observed_value")
        rows.append(
            {
                "expectation": expectation["type"],
                "column_name": expectation["kwargs"].get("column"),
                "severity": (expectation.get("meta") or {}).get("severity", "critical"),
                "success": bool(item["success"]),
                "observed_value": None if observed is None else str(observed)[:500],
                "element_count": result.get("element_count"),
                "unexpected_count": result.get("unexpected_count"),
                "unexpected_percent": result.get("unexpected_percent"),
            }
        )
    return rows


def run_gate(
    spark: SparkSession,
    config: Config,
    layer: str,
    suites: dict[str, list] | None = None,
    run_id: str | None = None,
) -> list[dict]:
    """Validate every table of ``layer``, store the results, raise on a critical failure."""
    suites = SUITES[layer]() if suites is None else suites
    run_id = run_id or uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    context = _context()
    rows = []
    for table, expectations in suites.items():
        results = validate(spark.table(config.table(layer, table)), expectations, table, context)
        failed = [r for r in results if not r["success"]]
        print(
            f"quality {config.table(layer, table):<40} "
            f"{len(results) - len(failed)}/{len(results)} expectations passed"
        )
        rows += [
            {"run_id": run_id, "validated_at": now, "layer": layer, "table_name": table, **r}
            for r in results
        ]

    ensure_schemas(spark, config)
    columns = [field.split()[0] for field in RESULTS_SCHEMA.split(", ")]
    (
        spark.createDataFrame([tuple(r[c] for c in columns) for r in rows], RESULTS_SCHEMA)
        .write.format("delta")
        .mode("append")
        .saveAsTable(config.table("gold", "quality_results"))
    )

    critical = [r for r in rows if not r["success"] and r["severity"] == "critical"]
    if critical:
        details = "\n".join(
            f"  {r['table_name']}.{r['column_name'] or '*'}: {r['expectation']} "
            f"({r['unexpected_count']} unexpected rows, observed {r['observed_value']})"
            for r in critical
        )
        raise QualityGateError(
            f"{len(critical)} critical expectation(s) failed on {layer}:\n{details}"
        )
    return rows


if __name__ == "__main__":
    import sys

    run_gate(get_spark(), load_config(), sys.argv[1] if len(sys.argv) > 1 else "bronze")
