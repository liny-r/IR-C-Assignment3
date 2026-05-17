"""
part_a_survival_analysis.py
============================
Part A — Exploratory Survival Analysis (compute only)

A(i)   Overall Kaplan-Meier survival curve for prepayment
A(ii)  Implied hazard rates (and Nelson-Aalen cumulative hazard)
A(iii) Stratified survival curves by FICO, LTV, vintage, loan purpose

This script does NO plotting. It writes parquet files into
``results_a/`` which the notebook ``results_a/part_a_plot.ipynb`` reads
to render figures.

Outputs (all parquet under results_a/):
  - summary.parquet           — overall counts (concern A1)
  - vintage_summary.parquet   — counts and median duration by vintage
  - km_overall.parquet        — KM(t) and CI, t = 1..360
  - hazard_overall.parquet    — discrete hazard, NA cumulative hazard
  - km_by_fico.parquet        — KM(t) per FICO bucket
  - km_by_ltv.parquet         — KM(t) per LTV bucket
  - km_by_vintage.parquet     — KM(t) per vintage bucket
  - km_by_purpose.parquet     — KM(t) per loan purpose
  - stratum_medians.parquet   — median survival per stratum

Notes on changes from the previous version:
  - Plotting moved to the notebook.
  - Outputs go to ``results_a/`` (concern 4).
  - Summary uses CreditTerminated (renamed from Defaulted, A2).
  - Censoring breakdown is reported correctly (concern A1).
  - Cause-specific framing of KM is documented; competing risks treated
    as non-informative censoring (concern 10 — documentation only here,
    notebook surfaces it).
  - Cumulative-hazard column is labeled "Cumulative hazard implied by
    KM" (Λ̂ = -ln Ŝ_KM); a separate Nelson-Aalen estimate is also
    computed and stored so the notebook can show both (concern 11).
  - Logging via utilities helpers, with STARTING-before / DONE-after
    every time-sensitive block.

Usage:
  python part_a_survival_analysis.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
from lifelines import KaplanMeierFitter, NelsonAalenFitter

from utilities import (
    SURVIVAL_TABLE,
    RESULTS_A,
    PROJECT_ROOT,
    FICO_BUCKETS,
    LTV_BUCKETS,
    VINTAGE_BUCKETS,
    PURPOSE_LABELS,
    ensure_results_dir,
    make_logger,
    log_step,
    log_section,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIGURATION
# ============================================================
TIME_GRID = np.arange(1, 361)              # months 1..360
KEY_TIMEPOINTS = [12, 24, 36, 60, 84, 120, 180, 240]
HAZARD_SMOOTHING_WINDOW = 6                # for the notebook's smoothed hazard line

ensure_results_dir(RESULTS_A)
log = make_logger(
    "part_a_survival_analysis",
    log_file=PROJECT_ROOT / "part_a.log",
)


# ============================================================
# Data loading
# ============================================================
def load_data() -> pd.DataFrame:
    """Load the survival table (Polars → pandas)."""
    with log_step(log, "Loading survival table"):
        df = pl.read_parquet(SURVIVAL_TABLE)
        log.info(f"  {df.height:,} loans, {df.width} columns")
    with log_step(log, "Converting to pandas"):
        pdf = df.to_pandas()
    log.info(
        f"  Duration: median={pdf['Duration'].median():.0f}, "
        f"mean={pdf['Duration'].mean():.1f}, max={pdf['Duration'].max()}"
    )
    log.info(f"  Prepayment rate: {pdf['Event_Prepay'].mean():.4f}")
    return pdf


# ============================================================
# Summary  (concern A1: clean censoring breakdown)
# ============================================================
def write_summary(pdf: pd.DataFrame) -> None:
    """
    Write summary.parquet and vintage_summary.parquet.

    The KM estimator for prepayment treats every non-prepayment as
    censored, so the summary reports the censoring components clearly.
    """
    log_section(log, "DATASET SUMMARY")

    n = len(pdf)
    n_prepaid = int(pdf["Event_Prepay"].sum())
    n_credit = int(pdf["CreditTerminated"].sum())
    n_terminated = int(pdf["Terminated"].sum())
    n_other_term = n_terminated - n_prepaid - n_credit
    n_active = n - n_terminated
    n_censored_for_prepay_km = n - n_prepaid  # everything that isn't a prepay

    summary = pd.DataFrame(
        [
            ("total", n, 100.0),
            ("prepaid_event", n_prepaid, 100 * n_prepaid / n),
            ("credit_terminated", n_credit, 100 * n_credit / n),
            ("other_terminated", n_other_term, 100 * n_other_term / n),
            ("still_active", n_active, 100 * n_active / n),
            ("censored_for_prepay_km",
             n_censored_for_prepay_km,
             100 * n_censored_for_prepay_km / n),
        ],
        columns=["bucket", "n", "pct"],
    )
    summary.to_parquet(RESULTS_A / "summary.parquet", index=False)

    log.info(f"  Total 30yr FRM loans:           {n:>12,}")
    log.info(f"  Prepaid (event):                {n_prepaid:>12,}  "
             f"({100 * n_prepaid / n:5.1f}%)")
    log.info(f"  Credit-related termination:     {n_credit:>12,}  "
             f"({100 * n_credit / n:5.1f}%)")
    log.info(f"  Other termination:              {n_other_term:>12,}  "
             f"({100 * n_other_term / n:5.1f}%)")
    log.info(f"  Still active (right-censored):  {n_active:>12,}  "
             f"({100 * n_active / n:5.1f}%)")
    log.info(f"  ------ For prepayment KM, censored = "
             f"{n_censored_for_prepay_km:,} "
             f"({100 * n_censored_for_prepay_km / n:.1f}%)")

    # Vintage breakdown
    vintage_rows = []
    for bucket in VINTAGE_BUCKETS:
        mask = pdf["VintageBucket"] == bucket
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        prepay_rate = float(pdf.loc[mask, "Event_Prepay"].mean())
        median_dur = float(pdf.loc[mask, "Duration"].median())
        n_prepay_b = int(pdf.loc[mask, "Event_Prepay"].sum())
        vintage_rows.append({
            "VintageBucket": bucket, "n": n_b, "n_prepay": n_prepay_b,
            "prepay_rate": prepay_rate, "median_duration": median_dur,
        })
    vintage_df = pd.DataFrame(vintage_rows)
    vintage_df.to_parquet(RESULTS_A / "vintage_summary.parquet", index=False)

    log.info("  By vintage:")
    for r in vintage_rows:
        log.info(
            f"    {r['VintageBucket']}: {r['n']:>10,} loans, "
            f"prepay rate={r['prepay_rate']:.3f}, "
            f"median duration={r['median_duration']:.0f} mo"
        )


# ============================================================
# A(i) Overall KM
# ============================================================
def fit_overall_km(pdf: pd.DataFrame) -> KaplanMeierFitter:
    """Fit overall KM and write km_overall.parquet."""
    log_section(log, "A(i): Kaplan-Meier Survival Curve for Prepayment")

    with log_step(log, "Fitting overall KM"):
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=pdf["Duration"],
            event_observed=pdf["Event_Prepay"],
            label="Prepayment",
        )

    median_surv = kmf.median_survival_time_
    n = len(pdf)
    n_events = int(pdf["Event_Prepay"].sum())
    log.info(f"  Median survival time: {median_surv:.1f} months")
    log.info(f"  Number at risk at t=0: {n:,}")
    log.info(f"  Number of events: {n_events:,}")
    log.info(f"  Censored (everything that isn't a prepay): "
             f"{n - n_events:,}")

    # Predict KM(t) and 95% CI on the canonical time grid.
    # Both the survival function and its CI are step functions; using
    # linear interpolation for the CI but step evaluation for S would
    # let `S > S_upper` for early t values where S=1 exactly. Use
    # 'previous-value' lookup (right=False, left=1.0) to keep S, S_lower,
    # S_upper consistent as step functions.
    with log_step(log, "Computing KM(t) on time grid"):
        s = kmf.survival_function_at_times(TIME_GRID).values

        ci = kmf.confidence_interval_survival_function_
        # searchsorted-based step lookup: at time t, take the value at
        # the largest event time <= t. Before any event, both bounds = 1.
        ci_idx = ci.index.values
        idx_for_t = np.searchsorted(ci_idx, TIME_GRID, side="right") - 1
        ci_lower = np.where(idx_for_t < 0, 1.0,
                            ci.iloc[:, 0].values[np.clip(idx_for_t, 0, None)])
        ci_upper = np.where(idx_for_t < 0, 1.0,
                            ci.iloc[:, 1].values[np.clip(idx_for_t, 0, None)])
        km_df = pd.DataFrame({
            "t": TIME_GRID,
            "S": s,
            "S_lower": ci_lower,
            "S_upper": ci_upper,
        })
        # Median: NaN if KM never crosses 0.5 (censored too early)
        if np.isfinite(median_surv):
            km_df.attrs["median_survival_time"] = float(median_surv)
        km_df.to_parquet(RESULTS_A / "km_overall.parquet", index=False)
        log.info(f"  Saved {len(km_df)} rows to km_overall.parquet")

    log.info("  Survival probability at key timepoints:")
    for t in KEY_TIMEPOINTS:
        if t <= len(s):
            s_t = s[t - 1]
            log.info(
                f"    S({t:3d} months) = {s_t:.4f}  "
                f"({100 * (1 - s_t):.1f}% prepaid by month {t})"
            )

    # Stash the KM object for A(ii)
    return kmf


# ============================================================
# A(ii) Hazard rates
# ============================================================
def write_hazard(pdf: pd.DataFrame, kmf: KaplanMeierFitter) -> None:
    """
    Write hazard_overall.parquet with three columns:
      - h_discrete       : 1 - S(t)/S(t-1), the discrete monthly hazard
                            implied by KM
      - cum_hazard_km    : -ln S_KM(t), cumulative hazard implied by KM
                            (concern 11: NOT Nelson-Aalen, even if the
                            previous code labelled it that way)
      - cum_hazard_na    : Nelson-Aalen Λ̂(t) = Σ_{t_i ≤ t} d_i/n_i,
                            the actual NA estimator
      - h_smoothed       : centered moving average of h_discrete (window
                            HAZARD_SMOOTHING_WINDOW), a hint for the
                            notebook
    """
    log_section(log, "A(ii): Hazard Rates")

    with log_step(log, "Computing discrete hazard from KM"):
        s_grid = kmf.survival_function_at_times(TIME_GRID).values
        s_full = np.concatenate([[1.0], s_grid])  # prepend S(0) = 1
        h_discrete = 1.0 - s_full[1:] / np.maximum(s_full[:-1], 1e-10)
        cum_hazard_km = -np.log(np.maximum(s_grid, 1e-10))
        h_smoothed = (
            pd.Series(h_discrete)
            .rolling(HAZARD_SMOOTHING_WINDOW, center=True, min_periods=1)
            .mean().values
        )

    with log_step(log, "Fitting Nelson-Aalen cumulative hazard"):
        naf = NelsonAalenFitter()
        naf.fit(durations=pdf["Duration"], event_observed=pdf["Event_Prepay"])
        # NA gives a step function in `cumulative_hazard_`; interpolate
        # onto our canonical grid.
        na_idx = naf.cumulative_hazard_.index.values
        na_val = naf.cumulative_hazard_.iloc[:, 0].values
        cum_hazard_na = np.interp(TIME_GRID, na_idx, na_val)

    haz_df = pd.DataFrame({
        "t": TIME_GRID,
        "h_discrete": h_discrete,
        "h_smoothed": h_smoothed,
        "cum_hazard_km": cum_hazard_km,
        "cum_hazard_na": cum_hazard_na,
    })
    haz_df.to_parquet(RESULTS_A / "hazard_overall.parquet", index=False)
    log.info(f"  Saved {len(haz_df)} rows to hazard_overall.parquet")

    log.info("  Hazard at key points:")
    for t in [6, 12, 24, 36, 60, 84, 120]:
        if t <= len(h_discrete):
            ann = 100 * (1 - (1 - h_discrete[t - 1]) ** 12)
            log.info(
                f"    h({t:3d}) = {h_discrete[t - 1]:.6f}  "
                f"(annualized: {ann:.2f}%)"
            )


# ============================================================
# A(iii) Stratified KM
# ============================================================
def _km_by_stratum(
    pdf: pd.DataFrame,
    column: str,
    bucket_order: list[str],
    label: str,
    output_file: str,
    min_n: int = 100,
) -> list[dict]:
    """
    Fit KM separately within each stratum and write a long-format parquet:
       columns = [stratum, t, S, S_lower, S_upper, n_in_stratum, n_events]
    Returns a list of summary dicts (median per stratum) for the caller.
    """
    log_section(log, f"A(iii): KM stratified by {label}")
    rows = []
    medians = []
    kmf = KaplanMeierFitter()

    for bucket in bucket_order:
        mask = pdf[column] == bucket
        n_in = int(mask.sum())
        if n_in < min_n:
            log.info(f"  {bucket}: {n_in} loans — skipping (too few)")
            continue

        sub = pdf.loc[mask]
        n_events = int(sub["Event_Prepay"].sum())
        with log_step(log, f"  Fitting KM for {label}={bucket} (n={n_in:,})"):
            kmf.fit(sub["Duration"], sub["Event_Prepay"], label=str(bucket))
            s = kmf.survival_function_at_times(TIME_GRID).values
            ci = kmf.confidence_interval_survival_function_
            ci_idx = ci.index.values
            idx_for_t = np.searchsorted(ci_idx, TIME_GRID, side="right") - 1
            ci_lower = np.where(
                idx_for_t < 0, 1.0,
                ci.iloc[:, 0].values[np.clip(idx_for_t, 0, None)],
            )
            ci_upper = np.where(
                idx_for_t < 0, 1.0,
                ci.iloc[:, 1].values[np.clip(idx_for_t, 0, None)],
            )
        median_t = kmf.median_survival_time_
        median_t_safe = float(median_t) if np.isfinite(median_t) else np.nan

        for i, t in enumerate(TIME_GRID):
            rows.append({
                "stratum": str(bucket),
                "t": int(t),
                "S": float(s[i]),
                "S_lower": float(ci_lower[i]),
                "S_upper": float(ci_upper[i]),
                "n_in_stratum": n_in,
                "n_events": n_events,
            })
        medians.append({
            "variable": label,
            "stratum": str(bucket),
            "n": n_in,
            "n_events": n_events,
            "median_survival_time": median_t_safe,
            "prepay_rate": float(sub["Event_Prepay"].mean()),
        })
        log.info(
            f"  {bucket}: n={n_in:,}, "
            f"median={median_t_safe if np.isnan(median_t_safe) else f'{median_t_safe:.0f}'} "
            f"months, prepay rate={sub['Event_Prepay'].mean():.3f}"
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(RESULTS_A / output_file, index=False)
    log.info(f"  Saved {len(out_df)} rows to {output_file}")
    return medians


def write_stratified_km(pdf: pd.DataFrame) -> None:
    """Run all four stratifications and write a unified medians table."""
    all_medians: list[dict] = []

    all_medians.extend(_km_by_stratum(
        pdf, "FICO_bucket", FICO_BUCKETS, "FICO", "km_by_fico.parquet",
    ))
    all_medians.extend(_km_by_stratum(
        pdf, "LTV_bucket", LTV_BUCKETS, "LTV", "km_by_ltv.parquet",
    ))
    all_medians.extend(_km_by_stratum(
        pdf, "VintageBucket", VINTAGE_BUCKETS, "Vintage",
        "km_by_vintage.parquet",
    ))

    # Loan purpose: maps single-letter codes to readable labels
    purpose_codes = list(PURPOSE_LABELS.keys())
    all_medians.extend(_km_by_stratum(
        pdf, "LoanPurpose", purpose_codes, "LoanPurpose",
        "km_by_purpose.parquet",
    ))

    pd.DataFrame(all_medians).to_parquet(
        RESULTS_A / "stratum_medians.parquet", index=False
    )
    log.info(
        f"  Saved {len(all_medians)} rows to stratum_medians.parquet"
    )


# ============================================================
# Main
# ============================================================
def main():
    log_section(log, "PART A — EXPLORATORY SURVIVAL ANALYSIS")
    log.info(f"  Input:  {SURVIVAL_TABLE}")
    log.info(f"  Output: {RESULTS_A}/")
    log.info("")
    log.info("  Note: this script computes only. Run the notebook")
    log.info("        results_a/part_a_plot.ipynb to render figures.")
    log.info("")
    log.info("  Estimand: cause-specific KM survival function for")
    log.info("  prepayment. Non-prepayment terminations are treated as")
    log.info("  non-informative censoring. See notebook for discussion.")
    log.info("")

    pdf = load_data()
    write_summary(pdf)
    kmf = fit_overall_km(pdf)
    write_hazard(pdf, kmf)
    write_stratified_km(pdf)

    log_section(log, "DONE")
    log.info(f"  All outputs written to {RESULTS_A}/")
    log.info("  Files produced:")
    for f in sorted(RESULTS_A.glob("*.parquet")):
        size_kb = f.stat().st_size / 1024
        log.info(f"    {f.name:40s} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
