"""
part_b_cox_model.py
====================
Part B — Classical Cox Proportional Hazards Modelling (compute only)

B(i)   Fit Cox PH on origination covariates
B(ii)  Hazard ratio table for B(i) (and for B(iv) macro model)
B(iii) Test the proportional hazards assumption
       — Schoenfeld residual test (formal)
       — Schoenfeld residual scatter (visual, for the notebook)
       — Log-log survival plot inputs (visual, for the notebook)
B(iv)  Cox PH with macroeconomic covariates joined at origination

This script does NO plotting. It writes parquet files into
``results_b/`` which the notebook ``results_b/part_b_plot.ipynb``
reads to render figures.

Outputs (under results_b/):
  - cox_base_summary.parquet         — coefficient table for B(i)
  - cox_base_metadata.parquet        — concordance, AIC, n, n_events
  - hazard_ratios_base.parquet       — sorted HRs with 95% CIs
  - ph_test.parquet                  — Schoenfeld correlation test results
  - schoenfeld_residuals.parquet     — subsampled residuals for plotting
  - loglog_curves.parquet            — log-log step functions per stratum
  - macro_panel.parquet              — monthly FRED macro panel
  - cox_macro_summary.parquet        — coefficient table for B(iv)
  - cox_macro_metadata.parquet       — same metadata, with macro
  - hazard_ratios_macro.parquet      — sorted HRs with 95% CIs

Notes on changes from the previous version:
  - Plotting moved to the notebook (concern 4).
  - Outputs go to ``results_b/`` (concern 4).
  - ``clean_sentinels`` is called early so MI=999 is NaN before
    ``has_MI`` is computed (concern B2).
  - B(iv) is documented as a vintage-macro model: macro covariates are
    joined at FirstPaymentDate, capturing the macro environment at the
    loan's *origination*, NOT dynamic refinancing incentive during
    its life. A true time-varying Cox model needs the loan-month panel
    (memory-budget out of reach for the assignment). Concern 8.
  - Logging via utilities helpers, with STARTING-before / DONE-after
    every time-sensitive block.

Usage:
  python part_b_cox_model.py
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

from utilities import (
    SURVIVAL_TABLE,
    RESULTS_B,
    PROJECT_ROOT,
    SAMPLE_FRAC_B,
    RANDOM_SEED,
    LABEL_MAP,
    clean_sentinels,
    ensure_results_dir,
    make_logger,
    log_step,
    log_section,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIGURATION
# ============================================================
FRED_DIR = PROJECT_ROOT / ".." / "FRED data"

PENALIZER = 0.001  # ridge penalty for Cox; tiny — just for stability
SCHOENFELD_SUBSAMPLE = 0.1  # fraction of rows used for B(iii) diagnostics

ensure_results_dir(RESULTS_B)
log = make_logger(
    "part_b_cox_model",
    log_file=PROJECT_ROOT / "part_b.log",
)


# ============================================================
# 1. FRED data loading
# ============================================================
def _load_fred_csv(filename: str, value_col: str) -> pd.DataFrame | None:
    """
    Load a single FRED CSV, return DataFrame with [date, <value_col>].
    Returns None if file missing.
    """
    p = FRED_DIR / filename
    if not p.exists():
        log.warning(f"  {filename} not found at {p} — skipping")
        return None
    raw = pd.read_csv(p)
    # Modern FRED uses 'observation_date'; older dumps used 'DATE'
    date_col = (
        "observation_date"
        if "observation_date" in raw.columns
        else "DATE" if "DATE" in raw.columns else None
    )
    if date_col is None:
        log.warning(
            f"  {filename}: no observation_date or DATE column "
            f"(got {list(raw.columns)}) — skipping"
        )
        return None
    raw[date_col] = pd.to_datetime(raw[date_col])
    series_col = [c for c in raw.columns if c != date_col][0]
    raw = raw.rename(columns={date_col: "date", series_col: value_col})
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    return raw[["date", value_col]].dropna()


def build_macro_panel() -> pd.DataFrame:
    """
    Build a monthly macro panel from the five FRED CSVs.

    Frequencies in source files vary (daily DGS10, weekly MORTGAGE30US,
    monthly UNRATE/CSUSHPINSA, annual FPCPITOTLZGUSA). We resample
    everything to month-start, then forward-fill annual CPI within the
    year. HPI year-over-year growth is derived from CSUSHPINSA.

    Returns a DataFrame with columns:
        YYYYMM, MortgageRate, Treasury10Y, Unemployment, HPI, HPI_YoY,
        CPI_YoY (if available)

    YYYYMM is an integer like 202503 (March 2025) so it joins directly
    against the survival table's FirstPaymentDate.
    """
    log_section(log, "Loading FRED macro panel")
    log.info(f"  FRED dir: {FRED_DIR.resolve()}")

    # Build a monthly index spanning the union of all sources
    series = {}
    for fname, vcol in [
        ("MORTGAGE30US.csv", "MortgageRate"),
        ("DGS10.csv", "Treasury10Y"),
        ("UNRATE.csv", "Unemployment"),
        ("CSUSHPINSA.csv", "HPI"),
        ("FPCPITOTLZGUSA.csv", "CPI_YoY"),  # annual %, already YoY
    ]:
        with log_step(log, f"  Loading {fname}"):
            df = _load_fred_csv(fname, vcol)
            if df is not None:
                series[vcol] = df

    if not series:
        raise FileNotFoundError(
            f"No FRED CSV files loaded from {FRED_DIR}. Cannot build " "macro panel."
        )

    # Compute global month range
    earliest = min(df["date"].min() for df in series.values())
    latest = max(df["date"].max() for df in series.values())
    month_range = pd.date_range(
        earliest.normalize().replace(day=1),
        latest.normalize().replace(day=1),
        freq="MS",
    )
    panel = pd.DataFrame({"date": month_range})

    with log_step(log, "  Resampling each series to monthly"):
        for vcol, df in series.items():
            df_monthly = (
                df.set_index("date")[vcol]
                .resample("MS")
                .mean()  # daily → monthly mean (weekly/monthly are identity)
                .ffill()
                .reset_index()
            )
            panel = panel.merge(df_monthly, on="date", how="left")

    # Annual CPI series only has Jan 1 observations — forward-fill
    if "CPI_YoY" in panel.columns:
        panel["CPI_YoY"] = panel["CPI_YoY"].ffill()

    # HPI year-over-year growth (computed from level)
    if "HPI" in panel.columns:
        panel["HPI_YoY"] = panel["HPI"].pct_change(periods=12) * 100.0

    # YYYYMM integer key
    panel["YYYYMM"] = panel["date"].dt.year * 100 + panel["date"].dt.month
    panel = panel.drop(columns=["date"])

    # Reorder
    front_cols = ["YYYYMM"]
    other_cols = [c for c in panel.columns if c not in front_cols]
    panel = panel[front_cols + other_cols]

    log.info(
        f"  Macro panel: {len(panel)} months, "
        f"{panel['YYYYMM'].min()}..{panel['YYYYMM'].max()}"
    )
    log.info(f"  Columns: {list(panel.columns)}")

    panel.to_parquet(RESULTS_B / "macro_panel.parquet", index=False)
    log.info(f"  Saved {RESULTS_B / 'macro_panel.parquet'}")
    return panel


# ============================================================
# 2. Survival table loading and feature prep
# ============================================================
BASE_COVARIATES = [
    "FICO_z",
    "LTV_z",
    "Rate_z",
    "UPB_z",
    "DTI_z",
    "DTI_missing",
    "is_Purchase",
    "is_CashOutRefi",
    "is_Investment",
    "is_SecondHome",
    "is_FirstTimeBuyer",
    "is_Condo",
    "is_PUD",
    "has_MI",
    "is_MultiBorrower",
]

COX_COLS = ["Duration", "Event_Prepay"]


def load_and_prepare() -> pd.DataFrame:
    """
    Load survival table, sample, clean sentinels, build covariates.

    Sentinel cleaning happens BEFORE has_MI is built. In the previous
    version this was a defensive concern (B2): MI=999 is a sentinel for
    "not available", but the old has_MI line did
    ``has_MI = (MI.fillna(0) > 0)`` which would treat 999 as "yes there
    is MI" because 999 > 0. Calling clean_sentinels first converts 999
    to NaN, after which fillna(0) treats it correctly as "missing → no
    MI assumed".
    """
    log_section(log, "Loading and preparing survival table")
    with log_step(log, "Reading parquet"):
        df = pl.read_parquet(SURVIVAL_TABLE)
        log.info(f"  Full dataset: {df.height:,} loans")

    with log_step(log, f"Sampling {SAMPLE_FRAC_B*100:.0f}% (seed={RANDOM_SEED})"):
        df = df.sample(fraction=SAMPLE_FRAC_B, seed=RANDOM_SEED)
        log.info(f"  Sampled: {df.height:,} loans")

    pdf = df.to_pandas()
    del df
    gc.collect()

    # ── Concern B2: defensive sentinel cleaning BEFORE deriving has_MI
    with log_step(log, "Cleaning sentinels (defensive)"):
        pdf = clean_sentinels(pdf)

    # Filter
    pdf = pdf[pdf["Duration"] > 0].copy()

    key_cols = ["CreditScore", "OriginalLoantoValueLTV", "OriginalInterestRate"]
    before = len(pdf)
    pdf = pdf.dropna(subset=key_cols)
    log.info(
        f"  After dropping missing key covariates: {len(pdf):,} "
        f"(dropped {before - len(pdf):,})"
    )

    # ── Continuous covariates: standardize
    with log_step(log, "Standardizing continuous covariates"):
        for src, dst in [
            ("CreditScore", "FICO_z"),
            ("OriginalLoantoValueLTV", "LTV_z"),
            ("OriginalInterestRate", "Rate_z"),
            ("OriginalUPB", "UPB_z"),
        ]:
            mean = pdf[src].mean()
            std = pdf[src].std()
            pdf[dst] = (pdf[src] - mean) / std

        # DTI: median-impute + missing indicator
        dti_median = pdf["OriginalDebttoIncomeRatio"].median()
        pdf["DTI_filled"] = pdf["OriginalDebttoIncomeRatio"].fillna(dti_median)
        pdf["DTI_z"] = (pdf["DTI_filled"] - pdf["DTI_filled"].mean()) / pdf[
            "DTI_filled"
        ].std()
        pdf["DTI_missing"] = pdf["OriginalDebttoIncomeRatio"].isna().astype(int)

    # ── Dummies (cleanly handled now that sentinels are NaN)
    with log_step(log, "Building categorical dummies"):
        pdf["is_Purchase"] = (pdf["LoanPurpose"] == "P").astype(int)
        pdf["is_CashOutRefi"] = (pdf["LoanPurpose"] == "C").astype(int)
        pdf["is_Investment"] = (pdf["OccupancyStatus"] == "I").astype(int)
        pdf["is_SecondHome"] = (pdf["OccupancyStatus"] == "S").astype(int)
        pdf["is_FirstTimeBuyer"] = (pdf["FirstTimeHomebuyerFlag"] == "Y").astype(int)
        pdf["is_Condo"] = (pdf["PropertyType"] == "CO").astype(int)
        pdf["is_PUD"] = (pdf["PropertyType"] == "PU").astype(int)
        # Concern B2: MI=999 has already been NaN'd above → fillna(0)
        # correctly treats it as 0%
        pdf["has_MI"] = (pdf["MortgageInsurancePercentage"].fillna(0) > 0).astype(int)
        pdf["is_MultiBorrower"] = (pdf["NumberofBorrowers"].fillna(1) > 1).astype(int)

    return pdf


def make_cox_df(pdf: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Extract just the columns lifelines needs — numeric, no nulls."""
    cox_df = pdf[COX_COLS + covariates].copy().dropna()
    for c in cox_df.columns:
        cox_df[c] = pd.to_numeric(cox_df[c], errors="coerce")
    return cox_df.dropna()


# ============================================================
# 3. B(i) — base Cox model
# ============================================================
def fit_cox(cox_df: pd.DataFrame, label: str) -> tuple[CoxPHFitter, dict]:
    """Fit a Cox model and return (fitter, metadata-dict)."""
    log.info(
        f"  Fitting {label}: {len(cox_df):,} loans, "
        f"{int(cox_df['Event_Prepay'].sum())} events, "
        f"{cox_df.shape[1] - 2} covariates"
    )
    cph = CoxPHFitter(penalizer=PENALIZER)
    with log_step(log, f"  cph.fit ({label})"):
        cph.fit(
            cox_df,
            duration_col="Duration",
            event_col="Event_Prepay",
            show_progress=False,
        )
    md = {
        "label": label,
        "n": int(len(cox_df)),
        "n_events": int(cox_df["Event_Prepay"].sum()),
        "n_covariates": int(cox_df.shape[1] - 2),
        "concordance": float(cph.concordance_index_),
        "partial_AIC": float(cph.AIC_partial_),
        "log_likelihood": float(cph.log_likelihood_),
        "penalizer": float(PENALIZER),
    }
    log.info(f"    Concordance: {md['concordance']:.4f}")
    log.info(f"    Partial AIC: {md['partial_AIC']:.1f}")
    return cph, md


def write_cox_outputs(
    cph: CoxPHFitter,
    md: dict,
    summary_filename: str,
    metadata_filename: str,
    hazard_ratios_filename: str,
) -> None:
    """Persist the three standard parquet files for one Cox fit."""
    # Coefficient summary — preserve covariate name as a column
    summary = cph.summary.copy().reset_index().rename(columns={"index": "covariate"})
    if "covariate" not in summary.columns:
        # In some lifelines versions index has another name
        summary.rename(columns={summary.columns[0]: "covariate"}, inplace=True)
    summary.to_parquet(RESULTS_B / summary_filename, index=False)
    log.info(f"  Saved {summary_filename}")

    # Metadata as a 1-row parquet
    pd.DataFrame([md]).to_parquet(RESULTS_B / metadata_filename, index=False)
    log.info(f"  Saved {metadata_filename}")

    # Hazard ratio table sorted by coef
    hr = (
        cph.summary[
            [
                "coef",
                "exp(coef)",
                "exp(coef) lower 95%",
                "exp(coef) upper 95%",
                "se(coef)",
                "z",
                "p",
            ]
        ]
        .copy()
        .reset_index()
        .rename(columns={"index": "covariate"})
        .sort_values("coef")
        .reset_index(drop=True)
    )
    if "covariate" not in hr.columns:
        hr.rename(columns={hr.columns[0]: "covariate"}, inplace=True)
    hr["label"] = hr["covariate"].map(lambda c: LABEL_MAP.get(c, c))
    hr.to_parquet(RESULTS_B / hazard_ratios_filename, index=False)
    log.info(f"  Saved {hazard_ratios_filename}")


# ============================================================
# 4. B(iii) — proportional hazards diagnostics
# ============================================================
def make_schoenfeld_sample(cox_df: pd.DataFrame) -> pd.DataFrame:
    """Return the deterministic row-fraction sample used for B(iii)."""
    if not 0 < SCHOENFELD_SUBSAMPLE <= 1:
        raise ValueError(
            "SCHOENFELD_SUBSAMPLE must be a proportion in the interval (0, 1]."
        )

    if SCHOENFELD_SUBSAMPLE == 1:
        return cox_df

    n_rows = len(cox_df)
    n_sample = max(1, int(round(n_rows * SCHOENFELD_SUBSAMPLE)))
    sch_df = cox_df.sample(n=n_sample, random_state=RANDOM_SEED).sort_index()
    log.info(
        f"  Using {SCHOENFELD_SUBSAMPLE:.1%} Schoenfeld diagnostic sample: "
        f"{len(sch_df):,} of {n_rows:,} rows"
    )
    return sch_df


def fit_schoenfeld_cox(
    cph: CoxPHFitter, cox_df: pd.DataFrame, sch_df: pd.DataFrame
) -> CoxPHFitter:
    """
    Return a Cox fitter paired to the B(iii) diagnostic dataframe.

    Lifelines stores training-row state inside a fitted CoxPHFitter. Its
    Schoenfeld test and residual routines expect the dataframe passed later to
    have the same row count as the dataframe used in fit().
    """
    if len(sch_df) == len(cox_df) and sch_df.index.equals(cox_df.index):
        return cph

    cph_diag = CoxPHFitter(penalizer=PENALIZER)
    with log_step(log, "  cph.fit (Schoenfeld diagnostic sample)"):
        cph_diag.fit(
            sch_df,
            duration_col="Duration",
            event_col="Event_Prepay",
            show_progress=False,
        )
    return cph_diag


def write_ph_diagnostics(
    cph: CoxPHFitter, cox_df: pd.DataFrame, pdf: pd.DataFrame
) -> None:
    """
    Three diagnostic outputs for the notebook:
      1. ph_test.parquet         — Schoenfeld correlation test per covariate
      2. schoenfeld_residuals.parquet — subsampled residuals for plotting
      3. loglog_curves.parquet   — step functions of log(-log S(t)) per stratum
    """
    log_section(log, "B(iii): Proportional Hazards diagnostics")
    sch_df = make_schoenfeld_sample(cox_df)
    cph_diag = fit_schoenfeld_cox(cph, cox_df, sch_df)

    # 1) Formal test
    with log_step(log, "  Schoenfeld correlation test"):
        try:
            test_results = proportional_hazard_test(
                cph_diag, sch_df, time_transform="rank"
            )
            ph_df = (
                test_results.summary.copy()
                .reset_index()
                .rename(columns={"index": "covariate"})
            )
            if "covariate" not in ph_df.columns:
                ph_df.rename(columns={ph_df.columns[0]: "covariate"}, inplace=True)
            ph_df["label"] = ph_df["covariate"].map(lambda c: LABEL_MAP.get(c, c))
            ph_df.to_parquet(RESULTS_B / "ph_test.parquet", index=False)
            log.info(f"    Saved ph_test.parquet " f"({len(ph_df)} covariates)")
        except Exception as e:
            log.warning(f"    proportional_hazard_test failed: {e}")
            pd.DataFrame(
                columns=["covariate", "test_statistic", "p", "-log2(p)", "label"]
            ).to_parquet(RESULTS_B / "ph_test.parquet", index=False)

    # 2) Schoenfeld residuals (same subsample, for plotting)
    with log_step(log, "  Schoenfeld residuals"):
        try:
            sch = cph_diag.compute_residuals(sch_df, kind="schoenfeld")
            # Long format: time, covariate, residual
            long = sch.reset_index().rename(columns={"index": "Duration"})
            long = long.melt(
                id_vars="Duration", var_name="covariate", value_name="residual"
            )
            long["label"] = long["covariate"].map(lambda c: LABEL_MAP.get(c, c))
            long.to_parquet(
                RESULTS_B / "schoenfeld_residuals.parquet",
                index=False,
            )
            log.info(f"    Saved schoenfeld_residuals.parquet " f"({len(long):,} rows)")
        except Exception as e:
            log.warning(f"    Schoenfeld residual computation failed: {e}")
            pd.DataFrame(
                columns=["Duration", "covariate", "residual", "label"]
            ).to_parquet(RESULTS_B / "schoenfeld_residuals.parquet", index=False)

    # 3) Log-log survival curves for visual PH check
    with log_step(log, "  Log-log survival curves"):
        rows = []
        kmf = KaplanMeierFitter()
        strat_configs = [
            ("FICO_bucket", ["<620", "660-699", "740-779", "780+"], "FICO Score"),
            ("LTV_bucket", ["<=60", "71-80", "81-90", "96+"], "Original LTV"),
            ("LoanPurpose", ["P", "N", "C"], "Loan Purpose"),
        ]
        for col, values, title in strat_configs:
            for val in values:
                mask = pdf[col] == val
                n = int(mask.sum())
                if n < 100:
                    continue
                sub = pdf[mask]
                kmf.fit(sub["Duration"], sub["Event_Prepay"])
                sf = kmf.survival_function_.copy()
                # Trim degenerate ends so log(-log) is well-defined
                sf = sf[(sf.iloc[:, 0] > 0) & (sf.iloc[:, 0] < 1)]
                t = sf.index.values
                s = sf.iloc[:, 0].values
                if len(t) == 0:
                    continue
                # The log-log values are what plotting code overlays;
                # store them so the notebook is purely cosmetic
                log_t = np.log(t + 1)
                log_log_s = np.log(-np.log(s))
                for i in range(len(t)):
                    rows.append(
                        {
                            "stratum_var": col,
                            "stratum_label": title,
                            "stratum": str(val),
                            "n_in_stratum": n,
                            "t": int(t[i]),
                            "log_t": float(log_t[i]),
                            "S": float(s[i]),
                            "log_log_S": float(log_log_s[i]),
                        }
                    )

        loglog_df = pd.DataFrame(rows)
        loglog_df.to_parquet(RESULTS_B / "loglog_curves.parquet", index=False)
        log.info(
            f"    Saved loglog_curves.parquet "
            f"({len(loglog_df)} rows, "
            f"{loglog_df['stratum_var'].nunique() if len(loglog_df) else 0} variables)"
        )


# ============================================================
# 5. B(iv) — Cox model with vintage-macro covariates
# ============================================================
def fit_macro_cox(pdf: pd.DataFrame, macro: pd.DataFrame) -> CoxPHFitter | None:
    """
    Fit Cox PH with macro covariates joined at the loan's
    FirstPaymentDate. NOTE: this is a vintage-macro model — macro
    covariates are static at origination, NOT time-varying. They
    capture the macro environment at origination, which is itself
    informative (vintages originated when rates were high have
    different prepayment patterns) but is not the same thing as a
    refinancing-incentive model. See concern 8 in the response doc.
    """
    log_section(log, "B(iv): Cox PH with vintage-macro covariates")
    if macro is None or len(macro) == 0:
        log.warning("  Macro panel empty — skipping B(iv)")
        return None

    with log_step(log, "  Joining macro at FirstPaymentDate"):
        pdf_m = pdf.merge(
            macro, left_on="FirstPaymentDate", right_on="YYYYMM", how="left"
        )

    macro_cols_to_use = ["MortgageRate", "Treasury10Y", "Unemployment", "HPI_YoY"]
    macro_covs_z = []
    with log_step(log, "  Standardizing macro covariates"):
        for col in macro_cols_to_use:
            if col not in pdf_m.columns:
                continue
            mean = pdf_m[col].mean()
            std = pdf_m[col].std()
            zcol = f"{col}_z"
            if std > 0:
                pdf_m[zcol] = (pdf_m[col] - mean) / std
            else:
                pdf_m[zcol] = 0.0
            n_miss = int(pdf_m[col].isna().sum())
            pdf_m[zcol] = pdf_m[zcol].fillna(0)
            macro_covs_z.append(zcol)
            log.info(f"    {col}: mean={mean:.3f}, std={std:.3f}, " f"missing={n_miss}")

    all_covs = BASE_COVARIATES + macro_covs_z
    cox_df = make_cox_df(pdf_m, all_covs)
    log.info(f"  Covariates: {all_covs}")

    cph, md = fit_cox(cox_df, label="macro")
    write_cox_outputs(
        cph,
        md,
        summary_filename="cox_macro_summary.parquet",
        metadata_filename="cox_macro_metadata.parquet",
        hazard_ratios_filename="hazard_ratios_macro.parquet",
    )
    return cph


# ============================================================
# Main
# ============================================================
def main():
    log_section(log, "PART B — CLASSICAL COX MODELLING")
    log.info(f"  Input:  {SURVIVAL_TABLE}")
    log.info(f"  Output: {RESULTS_B}/")
    log.info("")
    log.info("  Note: this script computes only. Run the notebook")
    log.info("        results_b/part_b_plot.ipynb to render figures.")
    log.info("")
    log.info("  B(iv) is a vintage-macro model: macro covariates are")
    log.info("  joined at FirstPaymentDate (origination), not time-")
    log.info("  varying during the loan's life. See concern 8.")
    log.info("")

    # Macro panel first — needed for B(iv); also sanity-checks FRED inputs
    macro = build_macro_panel()

    # Load and prepare loan data
    pdf = load_and_prepare()

    # B(i): fit base Cox model
    log_section(log, "B(i): Cox PH with origination covariates")
    cox_df = make_cox_df(pdf, BASE_COVARIATES)
    cph_base, md_base = fit_cox(cox_df, label="base")
    write_cox_outputs(
        cph_base,
        md_base,
        summary_filename="cox_base_summary.parquet",
        metadata_filename="cox_base_metadata.parquet",
        hazard_ratios_filename="hazard_ratios_base.parquet",
    )

    # B(iii): PH diagnostics
    write_ph_diagnostics(cph_base, cox_df, pdf)

    # B(iv): macro extension
    fit_macro_cox(pdf, macro)

    log_section(log, "DONE")
    log.info(f"  All outputs written to {RESULTS_B}/")
    for f in sorted(RESULTS_B.glob("*.parquet")):
        size_kb = f.stat().st_size / 1024
        log.info(f"    {f.name:40s} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
