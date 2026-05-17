"""
utilities.py — Shared helpers and constants for IR&C Assignment 3
==================================================================
Used by:
  - build_survival_table.py
  - part_a_survival_analysis.py
  - part_b_cox_model.py
  - part_c_ml_models.py
  - part_d_deep_cox.py
  - results_*/part_*_plot.ipynb

Sections:
  1. Logging (make_logger, log_step, log_section)
  2. Path constants
  3. Project constants (sample fractions, seeds, horizons)
  4. Sentinel value cleaning
  5. Split helpers (vintage_split, carve_validation)
  6. Canonical split file (concern 5)
  7. Feature engineering for Parts C/D (build_feature_matrix)
  8. Standardization for Part D (standardize_three)
  9. Target construction (concern 6: TARGET_MODE flag)
 10. Metrics and calibration (compute_metrics, compute_calibration_bins)
 11. Plot styling and labels

This module is meant to be boring and stable. Modelling decisions
(hyperparameters, architectures) live in the part scripts that own them.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ============================================================
# 1. LOGGING
# ============================================================


def make_logger(
    name: str,
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a configured logger that writes to stdout, optionally also to a
    file. Format: ``[HH:MM:SS] LEVEL message``.

    Idempotent: calling twice with the same name returns the same logger
    without duplicating handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def _process_mem_mb() -> float:
    """Resident memory of the current process in MB. NaN if psutil missing."""
    if not _HAS_PSUTIL:
        return float("nan")
    return psutil.Process().memory_info().rss / (1024**2)


@contextmanager
def log_step(logger: logging.Logger, msg: str):
    """
    Context manager that prints "STARTING: <msg>" before the block and
    "DONE: <msg> (Xs, ...)" after. If psutil is available, also reports
    current memory and the delta.

    The point of the STARTING line is concern-9 / "output before
    time-sensitive parts": if the script crashes inside the block, the
    last visible log line will be STARTING, so it's obvious where it died.
    """
    logger.info(f"STARTING: {msg}")
    t0 = time.time()
    m0 = _process_mem_mb()
    try:
        yield
    finally:
        dt = time.time() - t0
        m1 = _process_mem_mb()
        if _HAS_PSUTIL and not (np.isnan(m0) or np.isnan(m1)):
            logger.info(
                f"DONE: {msg}  ({dt:.1f}s, mem {m1:.0f}MB diff{m1 - m0:+.0f}MB)"
            )
        else:
            logger.info(f"DONE: {msg}  ({dt:.1f}s)")


def log_section(logger: logging.Logger, title: str) -> None:
    """Visible divider for major phase boundaries."""
    bar = "=" * 60
    logger.info(bar)
    logger.info(title)
    logger.info(bar)


# ============================================================
# 2. PATH CONSTANTS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
SURVIVAL_TABLE = PROJECT_ROOT / "survival_table.parquet"

RESULTS_A = PROJECT_ROOT / "results_a"
RESULTS_B = PROJECT_ROOT / "results_b"
RESULTS_CD = PROJECT_ROOT / "results_cd"

CANONICAL_SPLIT_FILE = RESULTS_CD / "canonical_split.parquet"
ENCODERS_FILE = RESULTS_CD / "encoders.pkl"
TARGET_MODE_FILE = RESULTS_CD / "target_mode.txt"


def ensure_results_dir(path: Path) -> Path:
    """mkdir -p; returns path unchanged."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# 3. PROJECT CONSTANTS
# ============================================================
RANDOM_SEED = 42

# Parts C and D MUST share this (canonical-split alignment depends on it)
SAMPLE_FRAC_CD = 0.10

# Part B can be smaller — Cox PH is slow on big n
SAMPLE_FRAC_B = 0.05

# Last performance period observed in the data (YYYYMM)
DATA_CUTOFF_YYYYMM = 202509

# Vintage-based train/test split
TRAIN_END_YEAR = 2014
TEST_START_YEAR = 2015
TEST_END_YEAR = 2019

# Validation carve fraction within training set (for Part D early stopping)
VAL_FRAC = 0.10

# Evaluation horizons in months
HORIZONS = [12, 24, 36, 60]


# ============================================================
# 4. SENTINEL VALUE CLEANING
# ============================================================
# Numeric sentinels per the Freddie Mac data dictionary.
# Used in pandas (here) AND in build_survival_table.py at the Polars level.
# Same dictionary, two implementations because Polars and pandas don't
# share an API.
SENTINEL_RULES = {
    "CreditScore": 9999,
    "OriginalLoantoValueLTV": 999,
    "OriginalCombinedLoantoValueCLTV": 999,
    "OriginalDebttoIncomeRatio": 999,
    "MortgageInsurancePercentage": 999,
    "NumberofBorrowers": 99,
    "NumberofUnits": 99,
}


def clean_sentinels(
    pdf: pd.DataFrame,
    rules: dict | None = None,
) -> pd.DataFrame:
    """
    Convert sentinel values to NaN, in place. Returns the same DataFrame
    for chaining.

    Defensive: build_survival_table.py already cleans sentinels at the
    Polars level. This helper guarantees the property after any
    Polars→pandas round-trip and is safe to call multiple times.
    """
    if rules is None:
        rules = SENTINEL_RULES
    for col, sentinel in rules.items():
        if col in pdf.columns:
            pdf.loc[pdf[col] >= sentinel, col] = np.nan
    return pdf


# ============================================================
# 5. SPLIT HELPERS
# ============================================================
def vintage_split(
    pdf: pd.DataFrame,
    train_end_year: int = TRAIN_END_YEAR,
    test_start_year: int = TEST_START_YEAR,
    test_end_year: int = TEST_END_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vintage-based train/test split on `VintageYear`.

    Train: VintageYear <= train_end_year
    Test:  test_start_year <= VintageYear <= test_end_year

    Returns (pdf_train, pdf_test). Both are independent copies.
    """
    train_mask = pdf["VintageYear"] <= train_end_year
    test_mask = (pdf["VintageYear"] >= test_start_year) & (
        pdf["VintageYear"] <= test_end_year
    )
    return pdf.loc[train_mask].copy(), pdf.loc[test_mask].copy()


def carve_validation(
    pdf_train: pd.DataFrame,
    frac: float = VAL_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random row split of an already-built training set, for early-stopping
    validation in Part D.

    Returns (pdf_train_inner, pdf_val). Both copies, no shared index.

    Concern 3: this should be called BEFORE build_feature_matrix /
    standardize_three so the validation set never influences the fitted
    medians, level vocabularies, or scaling stats.
    """
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    rng = np.random.RandomState(seed)
    n = len(pdf_train)
    n_val = int(round(n * frac))
    val_idx = rng.choice(n, size=n_val, replace=False)
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    return (
        pdf_train.iloc[~val_mask].copy().reset_index(drop=True),
        pdf_train.iloc[val_mask].copy().reset_index(drop=True),
    )


# ============================================================
# 6. CANONICAL SPLIT FILE  (concern 5)
# ============================================================
def write_canonical_split(
    pdf_train: pd.DataFrame,
    pdf_test: pd.DataFrame,
    pdf_val: pd.DataFrame | None = None,
    path: Path = CANONICAL_SPLIT_FILE,
) -> None:
    """
    Write a long parquet file with columns
    ``[LoanSequenceNumber, split, VintageYear]`` where
    split ∈ {"train", "val", "test"}. Order within each split matches
    the order of the input DataFrames.

    This file is the single source of truth for the train/val/test
    partition. Part C is its only writer; Part D loads it. Concern 5.
    """
    parts = []
    blocks = [("train", pdf_train), ("val", pdf_val), ("test", pdf_test)]
    for split_name, df in blocks:
        if df is None:
            continue
        if "LoanSequenceNumber" not in df.columns or "VintageYear" not in df.columns:
            raise KeyError(
                f"DataFrame for split={split_name!r} must contain "
                "'LoanSequenceNumber' and 'VintageYear'"
            )
        sub = df[["LoanSequenceNumber", "VintageYear"]].copy()
        sub["split"] = split_name
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)[
        ["LoanSequenceNumber", "split", "VintageYear"]
    ]
    ensure_results_dir(path.parent)
    out.to_parquet(path, index=False)


def load_canonical_split(path: Path = CANONICAL_SPLIT_FILE) -> pd.DataFrame:
    """Load the canonical split file. Errors clearly if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical split file not found at {path}. "
            "Run part_c_ml_models.py first to produce it."
        )
    return pd.read_parquet(path)


def restrict_to_canonical_split(
    pdf: pd.DataFrame,
    split_value: str,
    canonical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Filter `pdf` to the loans the canonical split tags as `split_value`,
    AND reorder so row order matches the canonical file. Returns a new
    DataFrame with reset index.

    Used by Part D to align with Part C's test set.
    """
    if canonical is None:
        canonical = load_canonical_split()
    if split_value not in {"train", "val", "test"}:
        raise ValueError(f"split_value must be train/val/test, got {split_value!r}")

    target_ids = canonical.loc[
        canonical["split"] == split_value, "LoanSequenceNumber"
    ].tolist()
    if not target_ids:
        raise ValueError(f"Canonical split has no rows for split={split_value!r}")

    pdf_r = pdf[pdf["LoanSequenceNumber"].isin(target_ids)].copy()
    pdf_r["_order"] = pd.Categorical(
        pdf_r["LoanSequenceNumber"], categories=target_ids, ordered=True
    )
    pdf_r = pdf_r.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return pdf_r


# ============================================================
# 7. FEATURE ENGINEERING (Parts C and D)
# ============================================================
NUMERIC_FEATURES = [
    "CreditScore",
    "OriginalLoantoValueLTV",
    "OriginalCombinedLoantoValueCLTV",
    "OriginalDebttoIncomeRatio",
    "OriginalUPB",
    "OriginalInterestRate",
    "MortgageInsurancePercentage",
    "NumberofBorrowers",
    "OriginalLoanTerm",
]

CATEGORICAL_FEATURES = [
    "LoanPurpose",
    "OccupancyStatus",
    "FirstTimeHomebuyerFlag",
    "PropertyType",
    "Channel",
    "PropertyState",
]

CATEGORICAL_TOP_K = 20


def build_feature_matrix(
    pdf: pd.DataFrame,
    fit_encoders: bool = True,
    encoders: dict | None = None,
    drop_vintage_year: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Construct the design matrix X for ML / Deep Cox.

    Numeric columns:
      - median impute (median fitted on this pdf when fit_encoders=True,
        else taken from `encoders`)
      - emits a ``<col>__missing`` 0/1 indicator

    Categorical columns:
      - keep top-CATEGORICAL_TOP_K levels (fitted/loaded similarly)
      - lump rest into 'OTHER', map NaN → 'MISSING'
      - one-hot encoded

    VintageYear is emitted as a single int column unless
    ``drop_vintage_year=True`` (concern 12 sensitivity).

    Returns ``(X, encoders)``. ``encoders`` has keys ``median__<col>``
    and ``levels__<col>``.

    CRITICAL — concerns 1, 2: callers MUST pass the train DataFrame here
    with fit_encoders=True, then transform val/test with fit_encoders=False
    and the same encoders dict. Never call with the full unsplit pdf.
    """
    if encoders is None:
        encoders = {}
    X = pd.DataFrame(index=pdf.index)

    # Numeric: median impute + missing indicator
    for col in NUMERIC_FEATURES:
        if col not in pdf.columns:
            continue
        if fit_encoders:
            median = pdf[col].median()
            encoders[f"median__{col}"] = median
        else:
            try:
                median = encoders[f"median__{col}"]
            except KeyError as e:
                raise KeyError(
                    f"Encoder for {col!r} is missing — did you call "
                    f"build_feature_matrix(fit_encoders=True) on training "
                    f"data first?"
                ) from e
        X[col] = pdf[col].fillna(median).astype(np.float32)
        X[f"{col}__missing"] = pdf[col].isna().astype(np.int8)

    # Categorical: top-K + OTHER + MISSING, one-hot
    for col in CATEGORICAL_FEATURES:
        if col not in pdf.columns:
            continue
        if fit_encoders:
            top_levels = pdf[col].value_counts().head(CATEGORICAL_TOP_K).index.tolist()
            encoders[f"levels__{col}"] = top_levels
        else:
            try:
                top_levels = encoders[f"levels__{col}"]
            except KeyError as e:
                raise KeyError(
                    f"Encoder levels for {col!r} are missing — did you "
                    f"call build_feature_matrix(fit_encoders=True) on "
                    f"training data first?"
                ) from e
        s = pdf[col].where(pdf[col].isin(top_levels), other="OTHER").fillna("MISSING")
        for level in list(top_levels) + ["OTHER", "MISSING"]:
            X[f"{col}__{level}"] = (s == level).astype(np.int8)

    # VintageYear (concern 12 sensitivity)
    if not drop_vintage_year and "VintageYear" in pdf.columns:
        X["VintageYear"] = pdf["VintageYear"].astype(np.int16)

    return X, encoders


# ============================================================
# 8. STANDARDIZATION (Part D)
# ============================================================
def standardize_three(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Z-score using mean/std fitted on X_train ONLY (concern 3 fix).

    Returns ``(X_train_z, X_val_z, X_test_z, scaler)`` where the three
    arrays are float32 numpy and ``scaler`` carries the fitted means,
    stds, and column order so the transform can be replayed later.
    """
    means = X_train.mean()
    stds = X_train.std().replace(0, 1)

    def _z(df: pd.DataFrame) -> np.ndarray:
        return ((df - means) / stds).astype(np.float32).values

    return (
        _z(X_train),
        _z(X_val),
        _z(X_test),
        {
            "means": means,
            "stds": stds,
            "columns": X_train.columns.tolist(),
        },
    )


# ============================================================
# 9. TARGET CONSTRUCTION  (concern 6)
# ============================================================
def build_horizon_targets(
    pdf: pd.DataFrame,
    horizons: list[int] = HORIZONS,
    mode: str = "cause_specific",
) -> pd.DataFrame:
    """
    Add columns ``Target_T<h>`` for each horizon in `horizons`. Modifies
    pdf in place AND returns it.

    Requires columns: Event_Prepay, Duration, Terminated, FirstPaymentDate.

    Concern 6 boundary fix:
      observed_through_T   : Duration >= T   (was Duration > T)
      prepaid_by_T         : Event_Prepay==1 AND Duration <= T

    mode='cause_specific':
      prepaid_by_T                                → 1
      observed_through_T & ~prepaid_by_T          → 0
      non-prepay termination at Duration <= T     → NaN (censored)

    mode='binary':
      prepaid_by_T                                → 1
      observed_through_T & ~prepaid_by_T          → 0
      non-prepay termination at Duration <= T     → 0

    Also computes ``MaxObsMonths`` (the calendar-time observation window
    available for each loan) so downstream code can enforce
    "sufficient observation window for horizon T".
    """
    if mode not in ("cause_specific", "binary"):
        raise ValueError(f"Unknown mode: {mode!r}")

    # MaxObsMonths = months between FirstPaymentDate and DATA_CUTOFF
    fp = pdf["FirstPaymentDate"].astype(int)
    fp_year = fp // 100
    fp_month = fp % 100
    cutoff_year = DATA_CUTOFF_YYYYMM // 100
    cutoff_month = DATA_CUTOFF_YYYYMM % 100
    pdf["MaxObsMonths"] = (cutoff_year * 12 + cutoff_month) - (fp_year * 12 + fp_month)

    for T in horizons:
        target = pd.Series(np.nan, index=pdf.index, dtype=float)

        prepaid_by_T = (pdf["Event_Prepay"] == 1) & (pdf["Duration"] <= T)
        observed_through_T = pdf["Duration"] >= T  # boundary fix
        non_prepay_term_pre_T = (
            (pdf["Terminated"].astype(bool))
            & (pdf["Event_Prepay"] == 0)
            & (pdf["Duration"] <= T)
        )

        target[prepaid_by_T] = 1.0
        if mode == "cause_specific":
            target[observed_through_T & ~prepaid_by_T & ~non_prepay_term_pre_T] = 0.0
        elif mode == "binary":
            target[observed_through_T & ~prepaid_by_T] = 0.0
            target[non_prepay_term_pre_T] = 0.0
        # else: cause_specific → those rows stay NaN (censored)

        pdf[f"Target_T{T}"] = target

    return pdf


def write_target_mode(mode: str, path: Path = TARGET_MODE_FILE) -> None:
    """Save the target mode so plotting notebooks can label charts."""
    ensure_results_dir(path.parent)
    path.write_text(mode + "\n")


def read_target_mode(
    path: Path = TARGET_MODE_FILE, default: str = "cause_specific"
) -> str:
    """Read the target mode written by Part C; default if missing."""
    if not path.exists():
        return default
    return path.read_text().strip()


# ============================================================
# 10. METRICS AND CALIBRATION
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Standard binary classification metrics: AUC, Brier, log-loss,
    accuracy@0.5. Robust to empty or single-class y_true (returns NaN
    metrics with n, n_events still real).
    """
    from sklearn.metrics import (  # local import to keep utilities lazy
        roc_auc_score,
        brier_score_loss,
        log_loss,
        accuracy_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = int(len(y_true))
    n_events = int(np.sum(y_true)) if n > 0 else 0

    if n == 0 or len(np.unique(y_true)) < 2:
        return {
            "auc": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "accuracy": np.nan,
            "n": n,
            "n_events": n_events,
        }

    y_pred_c = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred_c)),
        "log_loss": float(log_loss(y_true, y_pred_c)),
        "accuracy": float(accuracy_score(y_true, y_pred_c >= 0.5)),
        "n": n,
        "n_events": n_events,
    }


def compute_calibration_bins(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Decile calibration: predicted-mean vs observed-rate per score bin.

    Columns: ``bin, pred_mean, observed_rate, pred_lower, pred_upper,
    n_in_bin``. Robust to ties via ``duplicates='drop'``.
    """
    if len(y_true) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred)})
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        df["bin"] = 0

    return (
        df.groupby("bin")
        .agg(
            pred_mean=("p", "mean"),
            observed_rate=("y", "mean"),
            pred_lower=("p", "min"),
            pred_upper=("p", "max"),
            n_in_bin=("y", "size"),
        )
        .reset_index()
    )


# ============================================================
# 11. PLOT STYLING AND LABELS
# ============================================================
DPI = 150

# Buckets used for stratified analysis (Part A KM, Parts C/D plotting)
FICO_BUCKETS = ["<620", "620-659", "660-699", "700-739", "740-779", "780+"]
LTV_BUCKETS = ["<=60", "61-70", "71-80", "81-90", "91-95", "96+"]
VINTAGE_BUCKETS = [
    "1999-2003",
    "2004-2007",
    "2008-2011",
    "2012-2015",
    "2016-2019",
    "2020+",
]
PURPOSE_LABELS = {
    "P": "Purchase",
    "N": "Refi (No Cash-Out)",
    "C": "Refi (Cash-Out)",
}

# Colours for stratified plots
FICO_COLORS = ["#DC2626", "#EA580C", "#CA8A04", "#16A34A", "#2563EB", "#7C3AED"]
LTV_COLORS = ["#7C3AED", "#2563EB", "#16A34A", "#CA8A04", "#EA580C", "#DC2626"]
VINTAGE_COLORS = ["#7C3AED", "#2563EB", "#DC2626", "#16A34A", "#EA580C", "#CA8A04"]
PURPOSE_COLORS = {"P": "#2563EB", "N": "#16A34A", "C": "#DC2626"}

# Model display order and colours (Parts C/D)
MODEL_ORDER = ["LogReg", "RF", "LGBM", "Cox", "LinearCox", "DeepCox"]
MODEL_COLORS = {
    "LogReg": "#9CA3AF",
    "RF": "#16A34A",
    "LGBM": "#EA580C",
    "Cox": "#2563EB",
    "LinearCox": "#7C3AED",
    "DeepCox": "#DC2626",
}

# Friendly labels for Cox covariates (Part B)
LABEL_MAP = {
    "FICO_z": "FICO (standardized)",
    "LTV_z": "LTV (standardized)",
    "Rate_z": "Orig. Interest Rate (std)",
    "UPB_z": "Loan Balance (standardized)",
    "DTI_z": "DTI (standardized)",
    "DTI_missing": "DTI Missing",
    "is_Purchase": "Purchase (vs Refi)",
    "is_CashOutRefi": "Cash-Out Refi",
    "is_Investment": "Investment Property",
    "is_SecondHome": "Second Home",
    "is_FirstTimeBuyer": "First-Time Buyer",
    "is_Condo": "Condo",
    "is_PUD": "PUD",
    "has_MI": "Has Mortgage Insurance",
    "is_MultiBorrower": "Multiple Borrowers",
    "MortgageRate_z": "30yr Mortgage Rate (std)",
    "Treasury10Y_z": "10yr Treasury Rate (std)",
    "Unemployment_z": "Unemployment Rate (std)",
    "HPI_YoY_z": "HPI Growth YoY (std)",
}


def apply_plot_style(dpi: int = DPI) -> None:
    """Set project-wide matplotlib rcParams. Call at top of plotting nb."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": dpi,
        }
    )
