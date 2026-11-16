from pathlib import Path

from credit_risk.ingestion import METADATA_COLUMNS, ingest_all

CSV = """SK_ID_CURR,AMT_CREDIT,NAME_CONTRACT_TYPE
100001,406597.5,Cash loans
100002,,Revolving loans
100003,1293502.5,"Cash, loans"
"""


def test_bronze_ingestion(spark, config):
    Path(config.raw_file("bureau")).write_text(CSV)

    counts = ingest_all(spark, config, names=("bureau",))
    assert counts == {"bureau": 3}

    df = spark.table(config.table("bronze", "bureau"))
    assert df.columns == ["SK_ID_CURR", "AMT_CREDIT", "NAME_CONTRACT_TYPE", *METADATA_COLUMNS]
    # Raw values are preserved as strings; empty fields become nulls.
    assert all(t == "string" for _, t in df.dtypes[:3])
    rows = {r["SK_ID_CURR"]: r for r in df.collect()}
    assert rows["100001"]["AMT_CREDIT"] == "406597.5"
    assert rows["100002"]["AMT_CREDIT"] is None
    assert rows["100003"]["NAME_CONTRACT_TYPE"] == "Cash, loans"
    assert rows["100001"]["_source_file"].endswith("bureau.csv")
    assert rows["100001"]["_ingested_at"] is not None

    # Re-running overwrites instead of appending.
    assert ingest_all(spark, config, names=("bureau",)) == {"bureau": 3}
