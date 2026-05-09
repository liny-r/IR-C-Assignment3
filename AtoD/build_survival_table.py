"""
build_survival_table.py
========================
Step 1 of the pipeline: construct the loan-level survival table.

For every 30-year FRM loan in the Freddie Mac Standard Dataset between
1999Q1 and 2022Q4:
  - origination covariates (FICO, LTV, DTI, ...)
  - duration (months) and event indicator (Event_Prepay = ZBC == 1)
  - vintage info for stratification

Reads yearly origination and performance parquets and emits a single
``survival_table.parquet`` that all downstream parts (A, B, C, D)
consume.

Memory: peak ~2-3 GB
Runtime: ~5 minutes for 1999-2022 vintages

Inputs:
  - Origination:  historical_data_{year}.parquet      (in ORIG_DIR)
  - Performance:  historical_data_time_{year}.parquet  (in PERF_DIR)

Output:
  - utilities.SURVIVAL_TABLE  (one row per loan)

Notes on changes from previous version:
  - Concern 9 fix: ``LastInterestRate`` now uses sort_by(MonthlyReporting
    Period).last() instead of bare last(). Same for any other "last
    value" aggregation. RateChange is now reliable.
  - Column rename: ``Defaulted`` → ``CreditTerminated`` (ZBC ∈ {2,3,9}
    is a credit-related termination, not a "default" in the strict
    sense). Downstream parts (A in particular) use the new name.
  - Logging: uses utilities.make_logger / log_step / log_section instead
    of bare print(). Every time/memory-sensitive operation is wrapped in
    a log_step context manager so a crashing script always leaves a
    visible STARTING line at the death point.

Usage:
  python build_survival_table.py
"""

from __future__ import annotations

import gc
import os

import polars as pl

from utilities import (
    SURVIVAL_TABLE,
    SENTINEL_RULES,
    FICO_BUCKETS,
    LTV_BUCKETS,
    VINTAGE_BUCKETS,
    PROJECT_ROOT,
    make_logger,
    log_step,
    log_section,
)


# ============================================================
# CONFIGURATION — adjust paths to match your layout
# ============================================================
ORIG_DIR = "../Origination_Historical_Data"
PERF_DIR = "../Monthly_Performance_historical_data_time"

VINTAGES = list(range(1999, 2023))


# String-valued sentinels (numeric ones live in utilities.SENTINEL_RULES).
# Kept inline here because they apply only at the Polars level for the
# raw Freddie Mac data; downstream pandas code only ever sees strings or
# nulls in these columns.
STRING_SENTINELS = {
    "Channel": "9",
    "FirstTimeHomebuyerFlag": "9",
    "OccupancyStatus": "9",
    "LoanPurpose": "9",
    "PropertyType": "99",
}


ORIG_COLS = [
    "LoanSequenceNumber",
    "CreditScore",
    "FirstPaymentDate",
    "FirstTimeHomebuyerFlag",
    "MaturityDate",
    "MortgageInsurancePercentage",
    "NumberofUnits",
    "OccupancyStatus",
    "OriginalCombinedLoantoValueCLTV",
    "OriginalDebttoIncomeRatio",
    "OriginalUPB",
    "OriginalLoantoValueLTV",
    "OriginalInterestRate",
    "Channel",
    "PropertyState",
    "PropertyType",
    "LoanPurpose",
    "OriginalLoanTerm",
    "NumberofBorrowers",
    "PrepaymentPenaltyMortgageFlag",
    "SuperConformingFlag",
    "HARPIndicator",
    "ProgramIndicator",
]

PERF_COLS = [
    "LoanSequenceNumber",
    "MonthlyReportingPeriod",
    "LoanAge",
    "ZeroBalanceCode",
    "CurrentInterestRate",
    "CurrentLoanDelinquencyStatus",
    "CurrentActualUPB",
]


log = make_logger(
    "build_survival_table",
    log_file=PROJECT_ROOT / "build_survival_table.log",
)


# ============================================================
# Helpers (Polars-level sentinel cleaning)
# ============================================================
def _clean_numeric_sentinels(df: pl.DataFrame) -> pl.DataFrame:
    """Polars equivalent of utilities.clean_sentinels — same dictionary."""
    for col, sentinel in SENTINEL_RULES.items():
        if col not in df.columns:
            continue
        df = df.with_columns(
            pl.when(pl.col(col) >= sentinel)
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    return df


def _clean_string_sentinels(df: pl.DataFrame) -> pl.DataFrame:
    for col, sentinel in STRING_SENTINELS.items():
        if col not in df.columns:
            continue
        df = df.with_columns(
            pl.when(pl.col(col) == sentinel)
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    return df


# ============================================================
# Step 1: Load origination data
# ============================================================
def load_origination(vintages: list[int]) -> pl.DataFrame:
    log_section(log, "STEP 1: Loading origination data")

    frames: list[pl.DataFrame] = []
    for year in vintages:
        path = os.path.join(ORIG_DIR, f"historical_data_{year}.parquet")
        if not os.path.exists(path):
            log.warning(f"  SKIP {year} (file not found: {path})")
            continue
        with log_step(log, f"  Loading vintage {year}"):
            df = (
                pl.scan_parquet(path)
                .select(ORIG_COLS)
                .filter(pl.col("OriginalLoanTerm") == 360)  # 30yr FRM only
                .collect()
            )
            log.info(f"    {df.height:,} loans (30yr FRM)")
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No origination files found under {ORIG_DIR} for vintages "
            f"{min(vintages)}-{max(vintages)}"
        )

    with log_step(log, "  Concatenating vintages"):
        orig = pl.concat(frames)
        del frames
        gc.collect()
    log.info(f"  Total 30yr FRM loans: {orig.height:,}")
    log.info(f"  Memory: {orig.estimated_size('mb'):.1f} MB")

    with log_step(log, "  Cleaning numeric sentinels"):
        orig = _clean_numeric_sentinels(orig)

    with log_step(log, "  Cleaning string sentinels"):
        orig = _clean_string_sentinels(orig)

    with log_step(log, "  Computing derived fields"):
        orig = _add_derived_fields(orig)

    return orig


def _add_derived_fields(orig: pl.DataFrame) -> pl.DataFrame:
    """Vintage year/quarter/bucket and FICO/LTV buckets."""
    # Loan Sequence Number format: F YY Qn XXXXXXX
    #   F = product (always F here, since we filter to FRM)
    #   YY = 2-digit year, Qn = quarter
    orig = orig.with_columns(
        pl.col("LoanSequenceNumber").str.slice(1, 2).alias("_yy"),
        pl.col("LoanSequenceNumber").str.slice(4, 1).alias("_q"),
    )
    orig = orig.with_columns(
        # YY → full year (Freddie Mac SFLLD starts 1999, so YY < 90 → 20YY)
        pl.when(pl.col("_yy").cast(pl.Int16) >= 90)
        .then(pl.col("_yy").cast(pl.Int16) + 1900)
        .otherwise(pl.col("_yy").cast(pl.Int16) + 2000)
        .alias("VintageYear"),
        pl.col("_q").cast(pl.Int16).alias("VintageQuarter"),
    ).drop(["_yy", "_q"])

    # FICO bucket
    orig = orig.with_columns(
        pl.when(pl.col("CreditScore").is_null()).then(pl.lit("Missing"))
        .when(pl.col("CreditScore") < 620).then(pl.lit(FICO_BUCKETS[0]))
        .when(pl.col("CreditScore") < 660).then(pl.lit(FICO_BUCKETS[1]))
        .when(pl.col("CreditScore") < 700).then(pl.lit(FICO_BUCKETS[2]))
        .when(pl.col("CreditScore") < 740).then(pl.lit(FICO_BUCKETS[3]))
        .when(pl.col("CreditScore") < 780).then(pl.lit(FICO_BUCKETS[4]))
        .otherwise(pl.lit(FICO_BUCKETS[5]))
        .alias("FICO_bucket"),
    )

    # LTV bucket
    orig = orig.with_columns(
        pl.when(pl.col("OriginalLoantoValueLTV").is_null()).then(pl.lit("Missing"))
        .when(pl.col("OriginalLoantoValueLTV") <= 60).then(pl.lit(LTV_BUCKETS[0]))
        .when(pl.col("OriginalLoantoValueLTV") <= 70).then(pl.lit(LTV_BUCKETS[1]))
        .when(pl.col("OriginalLoantoValueLTV") <= 80).then(pl.lit(LTV_BUCKETS[2]))
        .when(pl.col("OriginalLoantoValueLTV") <= 90).then(pl.lit(LTV_BUCKETS[3]))
        .when(pl.col("OriginalLoantoValueLTV") <= 95).then(pl.lit(LTV_BUCKETS[4]))
        .otherwise(pl.lit(LTV_BUCKETS[5]))
        .alias("LTV_bucket"),
    )

    # Vintage bucket (matches utilities.VINTAGE_BUCKETS)
    orig = orig.with_columns(
        pl.when(pl.col("VintageYear") <= 2003).then(pl.lit(VINTAGE_BUCKETS[0]))
        .when(pl.col("VintageYear") <= 2007).then(pl.lit(VINTAGE_BUCKETS[1]))
        .when(pl.col("VintageYear") <= 2011).then(pl.lit(VINTAGE_BUCKETS[2]))
        .when(pl.col("VintageYear") <= 2015).then(pl.lit(VINTAGE_BUCKETS[3]))
        .when(pl.col("VintageYear") <= 2019).then(pl.lit(VINTAGE_BUCKETS[4]))
        .otherwise(pl.lit(VINTAGE_BUCKETS[5]))
        .alias("VintageBucket"),
    )

    return orig


# ============================================================
# Step 2: Extract terminal records from performance data
# ============================================================
def extract_terminal_records(vintages: list[int]) -> pl.DataFrame:
    """
    For each loan, extract the terminal record from monthly performance:

        Duration             = max LoanAge ever observed
        LastReportingPeriod  = max MonthlyReportingPeriod
        ZeroBalanceCode      = the (single) non-null ZBC value
        LastInterestRate     = CurrentInterestRate at the latest reporting
                               period — sort-aware (concern 9 fix)
        MaxDelinquency       = max delinquency status ever observed
        EverDelinquent90Plus = ever 90+ days delinquent

    Concern 9: bare ``.last()`` inside a group_by/agg returns the row in
    physical scan order, not in MonthlyReportingPeriod order. The yearly
    parquets are usually already sorted by reporting period so the
    pre-existing code happened to be right, but it's not guaranteed. We
    use ``.sort_by(MonthlyReportingPeriod).last()`` to make the
    invariant explicit. Same logic applies wherever we want a
    "value at the last reporting period".
    """
    log_section(log, "STEP 2: Extracting terminal records from performance data")

    frames: list[pl.DataFrame] = []
    for year in vintages:
        path = os.path.join(PERF_DIR, f"historical_data_time_{year}.parquet")
        if not os.path.exists(path):
            log.warning(f"  SKIP {year} (file not found: {path})")
            continue

        with log_step(log, f"  Aggregating performance for {year}"):
            lf = pl.scan_parquet(path).select(PERF_COLS)

            # Cast ZBC to Int16; handles potential dtype mismatches
            # across years. ``strict=False`` means non-numeric strings
            # become null (e.g. "RA").
            lf = lf.with_columns(
                pl.col("ZeroBalanceCode").cast(pl.Int16, strict=False),
            )

            terminal = (
                lf.group_by("LoanSequenceNumber")
                .agg(
                    # Duration = max LoanAge (order-invariant aggregate)
                    pl.col("LoanAge").max().alias("Duration"),

                    # Last reporting period (max — order-invariant)
                    pl.col("MonthlyReportingPeriod").max()
                        .alias("LastReportingPeriod"),

                    # ZBC: at most one non-null value per loan
                    pl.col("ZeroBalanceCode").drop_nulls().first()
                        .alias("ZeroBalanceCode"),

                    # ── CONCERN 9 FIX ──
                    # Use sort_by(MonthlyReportingPeriod).last() so we
                    # genuinely get the value at the latest reporting
                    # period regardless of underlying scan order.
                    pl.col("CurrentInterestRate")
                        .sort_by("MonthlyReportingPeriod")
                        .last()
                        .alias("LastInterestRate"),

                    # Max delinquency (order-invariant aggregate)
                    pl.col("CurrentLoanDelinquencyStatus")
                        .cast(pl.Int16, strict=False)
                        .max()
                        .alias("MaxDelinquency"),

                    # Ever 90+ delinquent (order-invariant aggregate)
                    (
                        pl.col("CurrentLoanDelinquencyStatus")
                        .cast(pl.Int16, strict=False)
                        >= 3
                    ).any().alias("EverDelinquent90Plus"),
                )
                .collect()
            )
            log.info(f"    {terminal.height:,} loans")
            frames.append(terminal)
            gc.collect()

    if not frames:
        raise FileNotFoundError(
            f"No performance files found under {PERF_DIR} for vintages "
            f"{min(vintages)}-{max(vintages)}"
        )

    with log_step(log, "  Concatenating yearly terminal records"):
        perf = pl.concat(frames)
        del frames
        gc.collect()

    log.info(f"  Total terminal records: {perf.height:,}")
    log.info(f"  Memory: {perf.estimated_size('mb'):.1f} MB")
    return perf


# ============================================================
# Step 3: Build survival table
# ============================================================
def build_survival_table_df(
    orig: pl.DataFrame, perf: pl.DataFrame
) -> pl.DataFrame:
    """Inner-join origination and performance and build event/duration."""
    log_section(log, "STEP 3: Joining origination and performance data")

    with log_step(log, "  Inner join on LoanSequenceNumber"):
        surv = orig.join(perf, on="LoanSequenceNumber", how="inner")
    log.info(f"  Joined rows: {surv.height:,}")
    log.info(
        f"  (Origination had {orig.height:,}, "
        f"performance had {perf.height:,})"
    )

    with log_step(log, "  Building event indicators"):
        surv = surv.with_columns(
            # Voluntary payoff (prepayment)
            (pl.col("ZeroBalanceCode") == 1)
                .fill_null(False).alias("Prepaid"),

            # Credit-related termination — was "Defaulted" in the old
            # code; renamed because ZBC ∈ {2,3,9} is a credit-related
            # *termination* (third-party sale, short sale, REO), not a
            # strict default. Downstream parts use the new name.
            pl.col("ZeroBalanceCode").is_in([2, 3, 9])
                .fill_null(False).alias("CreditTerminated"),

            # Any termination
            pl.col("ZeroBalanceCode").is_not_null().alias("Terminated"),

            # Event indicator for prepayment-only survival analysis
            (pl.col("ZeroBalanceCode") == 1)
                .fill_null(False).cast(pl.Int8).alias("Event_Prepay"),
        )

    with log_step(log, "  Cleaning duration"):
        # Floor at 1 month (handle null/0 from very-recent originations)
        surv = surv.with_columns(
            pl.when(pl.col("Duration").is_null() | (pl.col("Duration") <= 0))
            .then(pl.lit(1))
            .otherwise(pl.col("Duration"))
            .cast(pl.Int16)
            .alias("Duration"),
        )

    # Rough refinance-incentive proxy.
    # Now reliable thanks to the concern-9 fix above.
    surv = surv.with_columns(
        (pl.col("OriginalInterestRate") - pl.col("LastInterestRate"))
            .alias("RateChange"),
    )

    # ----- Summary (concern A1: properly broken-down censoring) -----
    n = surv.height
    n_prepaid = surv["Prepaid"].sum()
    n_credit = surv["CreditTerminated"].sum()
    n_terminated = surv["Terminated"].sum()
    n_other_term = n_terminated - n_prepaid - n_credit
    n_active = n - n_terminated

    log.info("")
    log.info(f"  Final survival table: {n:,} loans")
    log.info(f"    Prepaid (event):              "
             f"{n_prepaid:>10,} ({100 * n_prepaid / n:5.1f}%)")
    log.info(f"    Credit-related termination:   "
             f"{n_credit:>10,} ({100 * n_credit / n:5.1f}%)")
    log.info(f"    Other termination (whole-loan / RPL etc.): "
             f"{n_other_term:>10,} ({100 * n_other_term / n:5.1f}%)")
    log.info(f"    Still active (right-censored):"
             f"{n_active:>10,} ({100 * n_active / n:5.1f}%)")
    log.info(f"    --- For prepayment KM, censored = "
             f"{n - n_prepaid:,} ({100 * (n - n_prepaid) / n:.1f}%)")

    return surv


# ============================================================
# Main
# ============================================================
def main():
    log_section(log, "BUILD SURVIVAL TABLE")
    log.info(f"  Vintages: {min(VINTAGES)}-{max(VINTAGES)} "
             f"({len(VINTAGES)} years)")
    log.info(f"  Origination dir:  {ORIG_DIR}")
    log.info(f"  Performance dir:  {PERF_DIR}")
    log.info(f"  Output:           {SURVIVAL_TABLE}")

    with log_step(log, "Loading origination data"):
        orig = load_origination(VINTAGES)
    gc.collect()

    with log_step(log, "Extracting terminal performance records"):
        perf = extract_terminal_records(VINTAGES)
    gc.collect()

    with log_step(log, "Building survival table"):
        surv = build_survival_table_df(orig, perf)
    del orig, perf
    gc.collect()

    with log_step(log, f"Writing {SURVIVAL_TABLE.name}"):
        surv.write_parquet(
            SURVIVAL_TABLE, compression="zstd", compression_level=8
        )

    size_mb = os.path.getsize(SURVIVAL_TABLE) / (1024 * 1024)
    log_section(log, "SUMMARY")
    log.info(f"  Loans:        {surv.height:,}")
    log.info(f"  Columns:      {surv.width}")
    log.info(
        f"  Duration:     median={surv['Duration'].median()}, "
        f"mean={surv['Duration'].mean():.1f}, "
        f"max={surv['Duration'].max()}"
    )
    log.info(f"  Prepay rate:  {surv['Prepaid'].mean():.4f}")
    log.info(f"  File:         {SURVIVAL_TABLE} ({size_mb:.1f} MB)")
    log.info(f"  Columns:      {surv.columns}")


if __name__ == "__main__":
    main()
