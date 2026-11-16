"""Gold layer: one row of features per applicant.

Every child table has many rows per applicant, so each is summarised to
``SK_ID_CURR`` first and the summaries are then left-joined to the application.
An applicant with no history in a table keeps their row, with nulls: the absence
of history is information the model can use.

Feature names are prefixed by their source (``BUREAU_``, ``PREV_``, ``INST_``,
``POS_``, ``CC_``) so that model explanations stay readable.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from credit_risk.config import Config, load_config
from credit_risk.spark import ensure_schemas, get_spark

KEY = "SK_ID_CURR"
RECENT_DAYS = 365  # window of the "last 12 months" features

_EXT_SOURCES = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")


def _ratio(numerator, denominator) -> Column:
    """Division that gives null, never an error or infinity, on a zero denominator."""
    return F.try_divide(F.col(numerator), F.col(denominator))


def _share(condition: Column) -> Column:
    """Share of rows where ``condition`` holds (rows where it is null are ignored)."""
    return F.avg(condition.cast("double"))


def _count(condition: Column) -> Column:
    return F.sum(F.coalesce(condition, F.lit(False)).cast("int"))


def application_features(app: DataFrame) -> DataFrame:
    """Ratios describing affordability, loan structure and job stability."""
    known_scores = sum(F.col(c).isNotNull().cast("int") for c in _EXT_SOURCES)
    score_total = sum(F.coalesce(F.col(c), F.lit(0.0)) for c in _EXT_SOURCES)
    missing = sum(F.col(c).isNull().cast("int") for c in app.columns)
    return app.withColumns(
        {
            "APP_MISSING_COUNT": missing,
            "AGE_YEARS": -F.col("DAYS_BIRTH") / 365.25,
            "CREDIT_INCOME_RATIO": _ratio("AMT_CREDIT", "AMT_INCOME_TOTAL"),
            "ANNUITY_INCOME_RATIO": _ratio("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
            # Share of the loan repaid each year: the inverse of its duration.
            "CREDIT_TERM": _ratio("AMT_ANNUITY", "AMT_CREDIT"),
            "GOODS_CREDIT_RATIO": _ratio("AMT_GOODS_PRICE", "AMT_CREDIT"),
            "EMPLOYED_AGE_RATIO": _ratio("DAYS_EMPLOYED", "DAYS_BIRTH"),
            "INCOME_PER_PERSON": _ratio("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
            "EXT_SOURCE_MEAN": F.try_divide(score_total, known_scores),
            "EXT_SOURCE_MIN": F.least(*_EXT_SOURCES),
            "EXT_SOURCE_KNOWN_COUNT": known_scores,
        }
    ).drop("_processed_at")


def bureau_features(bureau: DataFrame, bureau_balance: DataFrame) -> DataFrame:
    """Indebtedness and repayment behaviour at other lenders."""
    # Level 1: one row per bureau credit, from its monthly statuses.
    # Status 1 to 5 means 1 to 5 months past due; C is closed, X unknown, 0 on time.
    late = F.col("STATUS").isin("1", "2", "3", "4", "5")
    per_credit = bureau_balance.groupBy("SK_ID_BUREAU").agg(
        F.count("*").alias("BB_MONTHS"),
        _share(late).alias("BB_LATE_SHARE"),
        F.max(F.when(late, F.col("STATUS").cast("int")).otherwise(0)).alias("BB_WORST_STATUS"),
    )
    active = F.col("CREDIT_ACTIVE") == "Active"
    # Level 2: one row per applicant.
    return (
        bureau.join(per_credit, "SK_ID_BUREAU", "left")
        .groupBy(KEY)
        .agg(
            F.count("*").alias("BUREAU_COUNT"),
            _count(active).alias("BUREAU_ACTIVE_COUNT"),
            _share(active).alias("BUREAU_ACTIVE_SHARE"),
            F.countDistinct("CREDIT_TYPE").alias("BUREAU_CREDIT_TYPE_COUNT"),
            F.sum("AMT_CREDIT_SUM").alias("BUREAU_CREDIT_SUM"),
            F.sum("AMT_CREDIT_SUM_DEBT").alias("BUREAU_DEBT_SUM"),
            F.sum(F.when(active, F.col("AMT_CREDIT_SUM_DEBT"))).alias("BUREAU_ACTIVE_DEBT_SUM"),
            F.sum("AMT_CREDIT_SUM_OVERDUE").alias("BUREAU_OVERDUE_SUM"),
            F.max("AMT_CREDIT_MAX_OVERDUE").alias("BUREAU_MAX_OVERDUE_MAX"),
            F.max("CREDIT_DAY_OVERDUE").alias("BUREAU_DAYS_OVERDUE_MAX"),
            F.sum("CNT_CREDIT_PROLONG").alias("BUREAU_PROLONG_SUM"),
            F.max("DAYS_CREDIT").alias("BUREAU_DAYS_CREDIT_MAX"),  # most recent credit
            F.avg("DAYS_CREDIT").alias("BUREAU_DAYS_CREDIT_MEAN"),
            F.max("DAYS_CREDIT_ENDDATE").alias("BUREAU_DAYS_ENDDATE_MAX"),
            _count(F.col("DAYS_CREDIT") >= -RECENT_DAYS).alias("BUREAU_COUNT_12M"),
            F.avg("BB_MONTHS").alias("BUREAU_BB_MONTHS_MEAN"),
            F.avg("BB_LATE_SHARE").alias("BUREAU_BB_LATE_SHARE_MEAN"),
            F.max("BB_WORST_STATUS").alias("BUREAU_BB_WORST_STATUS_MAX"),
        )
        .withColumn("BUREAU_DEBT_CREDIT_RATIO", _ratio("BUREAU_DEBT_SUM", "BUREAU_CREDIT_SUM"))
    )


def previous_application_features(previous: DataFrame) -> DataFrame:
    """History with Home Credit itself: past requests, approvals and refusals."""
    approved = F.col("NAME_CONTRACT_STATUS") == "Approved"
    refused = F.col("NAME_CONTRACT_STATUS") == "Refused"
    recent = F.col("DAYS_DECISION") >= -RECENT_DAYS
    return previous.groupBy(KEY).agg(
        F.count("*").alias("PREV_COUNT"),
        _count(approved).alias("PREV_APPROVED_COUNT"),
        _count(refused).alias("PREV_REFUSED_COUNT"),
        _share(approved).alias("PREV_APPROVED_SHARE"),
        _share(refused).alias("PREV_REFUSED_SHARE"),
        _count(refused & recent).alias("PREV_REFUSED_COUNT_12M"),
        _count(recent).alias("PREV_COUNT_12M"),
        F.avg("AMT_APPLICATION").alias("PREV_AMT_APPLICATION_MEAN"),
        F.avg("AMT_CREDIT").alias("PREV_AMT_CREDIT_MEAN"),
        F.max("AMT_CREDIT").alias("PREV_AMT_CREDIT_MAX"),
        F.avg("AMT_ANNUITY").alias("PREV_AMT_ANNUITY_MEAN"),
        # Above 1: the client asked for more than was granted.
        F.avg(_ratio("AMT_APPLICATION", "AMT_CREDIT")).alias("PREV_ASKED_GRANTED_RATIO_MEAN"),
        F.avg("RATE_DOWN_PAYMENT").alias("PREV_RATE_DOWN_PAYMENT_MEAN"),
        F.avg("CNT_PAYMENT").alias("PREV_CNT_PAYMENT_MEAN"),
        F.max("DAYS_DECISION").alias("PREV_DAYS_DECISION_MAX"),  # most recent decision
        F.avg("DAYS_DECISION").alias("PREV_DAYS_DECISION_MEAN"),
    )


def installments_features(installments: DataFrame) -> DataFrame:
    """Actual payment discipline: paid late or not, paid in full or not."""
    # greatest() skips nulls, so an instalment with no recorded payment would count as
    # "0 days late" and "fully paid". Keep it null instead: its outcome is unknown.
    delay = F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT")
    shortfall = F.col("AMT_INSTALMENT") - F.col("AMT_PAYMENT")
    days_late = F.when(delay.isNotNull(), F.greatest(delay, F.lit(0.0)))
    late = F.col("DAYS_ENTRY_PAYMENT") > F.col("DAYS_INSTALMENT")
    underpaid = F.when(shortfall.isNotNull(), F.greatest(shortfall, F.lit(0.0)))
    recent = F.col("DAYS_INSTALMENT") >= -RECENT_DAYS
    return installments.groupBy(KEY).agg(
        F.count("*").alias("INST_COUNT"),
        F.avg(days_late).alias("INST_DAYS_LATE_MEAN"),
        F.max(days_late).alias("INST_DAYS_LATE_MAX"),
        _share(late).alias("INST_LATE_SHARE"),
        _share(F.when(recent, late)).alias("INST_LATE_SHARE_12M"),
        F.avg(_ratio("AMT_PAYMENT", "AMT_INSTALMENT")).alias("INST_PAYMENT_RATIO_MEAN"),
        F.sum(underpaid).alias("INST_UNDERPAID_SUM"),
        F.sum("AMT_PAYMENT").alias("INST_AMT_PAYMENT_SUM"),
    )


def pos_cash_features(pos_cash: DataFrame) -> DataFrame:
    """Behaviour on point-of-sale and cash loans, from monthly snapshots."""
    return pos_cash.groupBy(KEY).agg(
        F.countDistinct("SK_ID_PREV").alias("POS_LOAN_COUNT"),
        F.count("*").alias("POS_MONTHS_COUNT"),
        F.avg("SK_DPD").alias("POS_DPD_MEAN"),
        F.max("SK_DPD").alias("POS_DPD_MAX"),
        F.max("SK_DPD_DEF").alias("POS_DPD_DEF_MAX"),
        _share(F.col("SK_DPD") > 0).alias("POS_DPD_SHARE"),
        _count(F.col("NAME_CONTRACT_STATUS") == "Completed").alias("POS_COMPLETED_COUNT"),
        F.avg("CNT_INSTALMENT_FUTURE").alias("POS_INSTALMENT_FUTURE_MEAN"),
    )


def credit_card_features(credit_card: DataFrame) -> DataFrame:
    """Revolving credit stress: how much of the limit is used, and missed minimums."""
    utilisation = _ratio("AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL")
    missed_minimum = F.col("AMT_PAYMENT_CURRENT") < F.col("AMT_INST_MIN_REGULARITY")
    return credit_card.groupBy(KEY).agg(
        F.countDistinct("SK_ID_PREV").alias("CC_CARD_COUNT"),
        F.count("*").alias("CC_MONTHS_COUNT"),
        F.avg("AMT_BALANCE").alias("CC_BALANCE_MEAN"),
        F.max("AMT_BALANCE").alias("CC_BALANCE_MAX"),
        F.avg(utilisation).alias("CC_UTILISATION_MEAN"),
        F.max(utilisation).alias("CC_UTILISATION_MAX"),
        F.avg("AMT_DRAWINGS_CURRENT").alias("CC_DRAWINGS_MEAN"),
        F.max("SK_DPD").alias("CC_DPD_MAX"),
        _share(missed_minimum).alias("CC_MIN_PAYMENT_MISSED_SHARE"),
    )


def assemble(application: DataFrame, *aggregates: DataFrame) -> DataFrame:
    """Left-join each per-applicant aggregate to the application features."""
    features = application_features(application)
    for aggregate in aggregates:
        features = features.join(aggregate, KEY, "left")
    return features


def build_features(spark: SparkSession, config: Config) -> dict[str, int]:
    def silver(name):
        return spark.table(config.table("silver", name))

    application = silver("application")
    features = assemble(
        application,
        bureau_features(silver("bureau"), silver("bureau_balance")),
        previous_application_features(silver("previous_application")),
        installments_features(silver("installments_payments")),
        pos_cash_features(silver("pos_cash_balance")),
        credit_card_features(silver("credit_card_balance")),
    )
    ensure_schemas(spark, config)
    target = config.table("gold", "features")
    (
        features.withColumn("_processed_at", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(target)
    )
    result = spark.table(target)
    stats = {
        "applicants": application.count(),
        "rows": result.count(),
        "columns": len(result.columns),
    }
    if stats["rows"] != stats["applicants"]:
        raise ValueError(
            f"{target} has {stats['rows']} rows for {stats['applicants']} applicants: "
            "an aggregate is not unique per applicant"
        )
    print(f"gold    {target:<40} {stats['rows']:>12,} rows, {stats['columns']} columns")
    return stats


if __name__ == "__main__":
    build_features(get_spark(), load_config())
