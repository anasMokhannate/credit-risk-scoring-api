import pytest

from credit_risk.config import Config
from credit_risk.silver import CastError, build_table, clean
from credit_risk.spark import ensure_schemas


@pytest.fixture(scope="module")
def cfg(spark):
    cfg = Config(raw_path="unused", schema_prefix="silvertest_")
    ensure_schemas(spark, cfg)
    return cfg


def test_application_is_typed_and_placeholders_become_null(spark):
    raw = spark.createDataFrame(
        [
            ("100001", "1", "M", "-12000", "-2000", "School", "0.5"),
            ("100002", "0", "XNA", "-20000", "365243", "XNA", None),
        ],
        "SK_ID_CURR string, TARGET string, CODE_GENDER string, DAYS_BIRTH string, "
        "DAYS_EMPLOYED string, ORGANIZATION_TYPE string, EXT_SOURCE_1 string",
    )
    rows = {r["SK_ID_CURR"]: r for r in clean(raw, "application").collect()}

    employed, pensioner = rows[100001], rows[100002]
    assert (employed["TARGET"], employed["DAYS_EMPLOYED"], employed["EXT_SOURCE_1"]) == (
        1,
        -2000,
        0.5,
    )
    assert employed["DAYS_EMPLOYED_ANOM"] == 0
    # The 365243 placeholder becomes null but its meaning is kept in a flag.
    assert pensioner["DAYS_EMPLOYED"] is None
    assert pensioner["DAYS_EMPLOYED_ANOM"] == 1
    assert pensioner["CODE_GENDER"] is None
    assert pensioner["ORGANIZATION_TYPE"] is None
    # Declared columns missing from the source exist as nulls.
    assert employed["AMT_CREDIT"] is None


def test_exact_duplicates_are_removed(spark):
    raw = spark.createDataFrame(
        [("1", "-1", "C"), ("1", "-1", "C"), ("1", "-2", "0")],
        "SK_ID_BUREAU string, MONTHS_BALANCE string, STATUS string",
    )
    assert clean(raw, "bureau_balance").count() == 2


def test_uncastable_value_is_an_error(spark):
    raw = spark.createDataFrame(
        [("1", "abc", "C")], "SK_ID_BUREAU string, MONTHS_BALANCE string, STATUS string"
    )
    with pytest.raises(CastError, match="1 values"):
        clean(raw, "bureau_balance")


def _save(spark, cfg, name, rows, schema):
    spark.createDataFrame(rows, schema).write.format("delta").mode("overwrite").saveAsTable(
        cfg.table("bronze", name)
    )


def test_train_and_test_are_unioned(spark, cfg):
    _save(spark, cfg, "application_train", [("1", "1")], "SK_ID_CURR string, TARGET string")
    _save(spark, cfg, "application_test", [("2",)], "SK_ID_CURR string")

    assert build_table(spark, cfg, "application") == {"rows_in": 2, "rows_out": 2}
    rows = {r["SK_ID_CURR"]: r for r in spark.table(cfg.table("silver", "application")).collect()}
    assert (rows[1]["dataset"], rows[1]["TARGET"]) == ("train", 1)
    assert (rows[2]["dataset"], rows[2]["TARGET"]) == ("test", None)
    assert rows[1]["_processed_at"] is not None


def test_bureau_balance_without_bureau_is_dropped(spark, cfg):
    _save(spark, cfg, "bureau", [("1", "10")], "SK_ID_CURR string, SK_ID_BUREAU string")
    _save(
        spark,
        cfg,
        "bureau_balance",
        [("10", "-1", "C"), ("99", "-1", "C")],
        "SK_ID_BUREAU string, MONTHS_BALANCE string, STATUS string",
    )
    build_table(spark, cfg, "bureau")

    assert build_table(spark, cfg, "bureau_balance") == {"rows_in": 2, "rows_out": 1}


def test_table_without_text_columns_is_cleaned(spark):
    # installments_payments has no text column: nothing to replace, but it must still be typed.
    raw = spark.createDataFrame(
        [("1", "2", "-10.0", "100.5")],
        "SK_ID_PREV string, SK_ID_CURR string, DAYS_INSTALMENT string, AMT_PAYMENT string",
    )
    row = clean(raw, "installments_payments").first()
    assert (row["SK_ID_PREV"], row["DAYS_INSTALMENT"], row["AMT_PAYMENT"]) == (2 - 1, -10.0, 100.5)
