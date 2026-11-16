"""Each aggregate is checked on a tiny hand-built history whose answer is computed by hand."""

import pytest

from credit_risk.features import (
    application_features,
    assemble,
    bureau_features,
    credit_card_features,
    installments_features,
    pos_cash_features,
    previous_application_features,
)


def one(df, key=1):
    return df.where(f"SK_ID_CURR = {key}").first()


def test_application_ratios(spark):
    app = spark.createDataFrame(
        [
            (1, 200000.0, 100000.0, 10000.0, 180000.0, -1000, -10000, 2.0, 0.2, None, 0.6),
            (2, 200000.0, 0.0, None, None, None, -7305, None, None, None, None),
        ],
        "SK_ID_CURR int, AMT_CREDIT double, AMT_INCOME_TOTAL double, AMT_ANNUITY double, "
        "AMT_GOODS_PRICE double, DAYS_EMPLOYED int, DAYS_BIRTH int, CNT_FAM_MEMBERS double, "
        "EXT_SOURCE_1 double, EXT_SOURCE_2 double, EXT_SOURCE_3 double",
    )
    features = application_features(app)

    row = one(features)
    assert row["CREDIT_INCOME_RATIO"] == 2.0
    assert row["ANNUITY_INCOME_RATIO"] == 0.1
    assert row["CREDIT_TERM"] == 0.05
    assert row["GOODS_CREDIT_RATIO"] == 0.9
    assert row["EMPLOYED_AGE_RATIO"] == 0.1
    assert row["INCOME_PER_PERSON"] == 50000.0
    # Mean and minimum over the two known scores only.
    assert row["EXT_SOURCE_MEAN"] == pytest.approx(0.4)
    assert row["EXT_SOURCE_MIN"] == 0.2
    assert row["EXT_SOURCE_KNOWN_COUNT"] == 2
    assert row["APP_MISSING_COUNT"] == 1

    # Zero income and missing values give nulls, never infinity or an error.
    empty = one(features, 2)
    assert empty["CREDIT_INCOME_RATIO"] is None
    assert empty["EXT_SOURCE_MEAN"] is None
    assert empty["AGE_YEARS"] == pytest.approx(20.0)
    assert empty["APP_MISSING_COUNT"] == 7


def test_bureau_aggregates_two_levels(spark):
    bureau = spark.createDataFrame(
        [
            (1, 10, "Active", "Consumer credit", 1000.0, 400.0, 0.0, 0, 0, -100, 200.0),
            (1, 11, "Closed", "Credit card", 3000.0, 0.0, 50.0, 30, 1, -800, -400.0),
            (1, 12, "Active", "Consumer credit", None, 600.0, 0.0, 0, 0, -400, None),
        ],
        "SK_ID_CURR int, SK_ID_BUREAU int, CREDIT_ACTIVE string, CREDIT_TYPE string, "
        "AMT_CREDIT_SUM double, AMT_CREDIT_SUM_DEBT double, AMT_CREDIT_SUM_OVERDUE double, "
        "CREDIT_DAY_OVERDUE int, CNT_CREDIT_PROLONG int, DAYS_CREDIT int, "
        "DAYS_CREDIT_ENDDATE double",
    ).selectExpr("*", "CAST(NULL AS double) AS AMT_CREDIT_MAX_OVERDUE")
    # Credit 10: 4 months, 1 late (status 2). Credit 11: 2 months, both late (worst 5).
    # Credit 12 has no monthly history.
    balance = spark.createDataFrame(
        [(10, "0"), (10, "2"), (10, "C"), (10, "X"), (11, "1"), (11, "5")],
        "SK_ID_BUREAU int, STATUS string",
    )
    row = one(bureau_features(bureau, balance))

    assert row["BUREAU_COUNT"] == 3
    assert row["BUREAU_ACTIVE_COUNT"] == 2
    assert row["BUREAU_ACTIVE_SHARE"] == pytest.approx(2 / 3)
    assert row["BUREAU_CREDIT_TYPE_COUNT"] == 2
    assert row["BUREAU_CREDIT_SUM"] == 4000.0
    assert row["BUREAU_DEBT_SUM"] == 1000.0
    assert row["BUREAU_ACTIVE_DEBT_SUM"] == 1000.0
    assert row["BUREAU_DEBT_CREDIT_RATIO"] == 0.25
    assert row["BUREAU_OVERDUE_SUM"] == 50.0
    assert row["BUREAU_DAYS_OVERDUE_MAX"] == 30
    assert row["BUREAU_PROLONG_SUM"] == 1
    assert row["BUREAU_DAYS_CREDIT_MAX"] == -100
    assert row["BUREAU_COUNT_12M"] == 1
    assert row["BUREAU_MAX_OVERDUE_MAX"] is None
    # Averages over the two credits that have a monthly history.
    assert row["BUREAU_BB_MONTHS_MEAN"] == 3.0
    assert row["BUREAU_BB_LATE_SHARE_MEAN"] == pytest.approx((0.25 + 1.0) / 2)
    assert row["BUREAU_BB_WORST_STATUS_MAX"] == 5


def test_previous_application_aggregates(spark):
    previous = spark.createDataFrame(
        [
            (1, "Approved", 1000.0, 1000.0, 100.0, 0.1, 12.0, -100),
            (1, "Refused", 3000.0, 2000.0, 300.0, None, 24.0, -200),
            (1, "Refused", 500.0, 0.0, None, None, None, -900),
            (1, "Canceled", 0.0, 0.0, None, None, None, -1000),
        ],
        "SK_ID_CURR int, NAME_CONTRACT_STATUS string, AMT_APPLICATION double, AMT_CREDIT double, "
        "AMT_ANNUITY double, RATE_DOWN_PAYMENT double, CNT_PAYMENT double, DAYS_DECISION int",
    )
    row = one(previous_application_features(previous))

    assert row["PREV_COUNT"] == 4
    assert (row["PREV_APPROVED_COUNT"], row["PREV_REFUSED_COUNT"]) == (1, 2)
    assert (row["PREV_APPROVED_SHARE"], row["PREV_REFUSED_SHARE"]) == (0.25, 0.5)
    assert (row["PREV_COUNT_12M"], row["PREV_REFUSED_COUNT_12M"]) == (2, 1)
    assert row["PREV_AMT_CREDIT_MEAN"] == 750.0
    assert row["PREV_AMT_CREDIT_MAX"] == 2000.0
    # Only the two applications with a non-zero granted amount: 1000/1000 and 3000/2000.
    assert row["PREV_ASKED_GRANTED_RATIO_MEAN"] == 1.25
    assert row["PREV_RATE_DOWN_PAYMENT_MEAN"] == 0.1
    assert row["PREV_DAYS_DECISION_MAX"] == -100


def test_installments_aggregates(spark):
    installments = spark.createDataFrame(
        [
            (1, -100.0, -110.0, 100.0, 100.0),  # 10 days early, paid in full
            (1, -200.0, -190.0, 100.0, 50.0),  # 10 days late, half paid
            (1, -900.0, -870.0, 200.0, 200.0),  # 30 days late, more than a year ago
            (1, -950.0, None, 100.0, None),  # never paid
        ],
        "SK_ID_CURR int, DAYS_INSTALMENT double, DAYS_ENTRY_PAYMENT double, "
        "AMT_INSTALMENT double, AMT_PAYMENT double",
    )
    row = one(installments_features(installments))

    assert row["INST_COUNT"] == 4
    assert row["INST_DAYS_LATE_MAX"] == 30.0
    assert row["INST_DAYS_LATE_MEAN"] == pytest.approx((0 + 10 + 30) / 3)
    assert row["INST_LATE_SHARE"] == pytest.approx(2 / 3)
    assert row["INST_LATE_SHARE_12M"] == 0.5
    assert row["INST_PAYMENT_RATIO_MEAN"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert row["INST_UNDERPAID_SUM"] == 50.0
    assert row["INST_AMT_PAYMENT_SUM"] == 350.0


def test_pos_cash_aggregates(spark):
    pos_cash = spark.createDataFrame(
        [
            (1, 100, "Active", 0, 0, 10.0),
            (1, 100, "Completed", 0, 0, 0.0),
            (1, 200, "Active", 20, 5, 6.0),
            (1, 200, "Active", 40, 0, 5.0),
        ],
        "SK_ID_CURR int, SK_ID_PREV int, NAME_CONTRACT_STATUS string, SK_DPD int, "
        "SK_DPD_DEF int, CNT_INSTALMENT_FUTURE double",
    )
    row = one(pos_cash_features(pos_cash))

    assert (row["POS_LOAN_COUNT"], row["POS_MONTHS_COUNT"]) == (2, 4)
    assert (row["POS_DPD_MEAN"], row["POS_DPD_MAX"], row["POS_DPD_DEF_MAX"]) == (15.0, 40, 5)
    assert row["POS_DPD_SHARE"] == 0.5
    assert row["POS_COMPLETED_COUNT"] == 1
    assert row["POS_INSTALMENT_FUTURE_MEAN"] == 5.25


def test_credit_card_aggregates(spark):
    credit_card = spark.createDataFrame(
        [
            (1, 100, 500.0, 1000.0, 100.0, 50.0, 60.0, 0),
            (1, 100, 900.0, 1000.0, 300.0, 50.0, 20.0, 15),
            (1, 100, 0.0, 0.0, 0.0, 0.0, None, 0),  # card with no limit: no utilisation
        ],
        "SK_ID_CURR int, SK_ID_PREV int, AMT_BALANCE double, AMT_CREDIT_LIMIT_ACTUAL double, "
        "AMT_DRAWINGS_CURRENT double, AMT_INST_MIN_REGULARITY double, "
        "AMT_PAYMENT_CURRENT double, SK_DPD int",
    )
    row = one(credit_card_features(credit_card))

    assert (row["CC_CARD_COUNT"], row["CC_MONTHS_COUNT"]) == (1, 3)
    assert row["CC_BALANCE_MAX"] == 900.0
    assert row["CC_UTILISATION_MEAN"] == pytest.approx(0.7)
    assert row["CC_UTILISATION_MAX"] == 0.9
    assert row["CC_DPD_MAX"] == 15
    # The minimum was missed in 1 of the 2 months where a payment is recorded.
    assert row["CC_MIN_PAYMENT_MISSED_SHARE"] == 0.5


def test_applicant_without_history_keeps_their_row(spark):
    app = spark.createDataFrame(
        [(1, 1000.0), (2, 2000.0)], "SK_ID_CURR int, AMT_CREDIT double"
    ).selectExpr(
        "*",
        "100.0 AS AMT_INCOME_TOTAL",
        "10.0 AS AMT_ANNUITY",
        "1.0 AS AMT_GOODS_PRICE",
        "-1 AS DAYS_EMPLOYED",
        "-9000 AS DAYS_BIRTH",
        "1.0 AS CNT_FAM_MEMBERS",
        "0.5 AS EXT_SOURCE_1",
        "0.5 AS EXT_SOURCE_2",
        "0.5 AS EXT_SOURCE_3",
    )
    previous = spark.createDataFrame(
        [(1, "Approved", 1.0, 1.0, 1.0, 0.0, 1.0, -1)] * 3,
        "SK_ID_CURR int, NAME_CONTRACT_STATUS string, AMT_APPLICATION double, AMT_CREDIT double, "
        "AMT_ANNUITY double, RATE_DOWN_PAYMENT double, CNT_PAYMENT double, DAYS_DECISION int",
    )
    features = assemble(app, previous_application_features(previous))

    # One row per applicant, however many child rows there are.
    assert features.count() == 2
    assert one(features, 1)["PREV_COUNT"] == 3
    assert one(features, 2)["PREV_COUNT"] is None
    # The application's own AMT_CREDIT is not overwritten by the aggregate.
    assert one(features, 1)["AMT_CREDIT"] == 1000.0
