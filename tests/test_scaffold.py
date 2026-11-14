import pytest

from credit_risk.config import Config


def test_table_names_local():
    cfg = Config(raw_path="data/raw")
    assert cfg.table("bronze", "POS_CASH_balance") == "bronze.pos_cash_balance"
    assert cfg.raw_file("bureau") == "data/raw/bureau.csv"


def test_table_names_unity_catalog():
    cfg = Config(raw_path="/Volumes/credit_risk/bronze/raw/", catalog="credit_risk")
    assert cfg.table("gold", "features") == "credit_risk.gold.features"
    assert cfg.raw_file("bureau") == "/Volumes/credit_risk/bronze/raw/bureau.csv"


def test_unknown_layer_rejected():
    with pytest.raises(ValueError):
        Config(raw_path="x").schema("platinum")


def test_delta_round_trip(spark, config):
    table = config.table("bronze", "smoke")
    spark.createDataFrame([(1, "a"), (2, "b")], ["id", "v"]).write.format("delta").mode(
        "overwrite"
    ).saveAsTable(table)
    assert spark.table(table).count() == 2
    assert spark.sql(f"DESCRIBE DETAIL {table}").first()["format"] == "delta"
