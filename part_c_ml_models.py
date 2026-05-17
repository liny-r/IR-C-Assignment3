"""
part_c_ml_models.py
====================
Part C — ML Models for Mortgage Prepayment (compute only)

C(i)   Predictive models: logistic regression, random forest, LightGBM
C(ii)  Compare against Cox PH baseline
C(iii) Out-of-sample backtesting (vintage-based train/test split)

This script does NO plotting. It writes parquet files into
``results_cd/`` which the notebook ``results_cd/part_c_plot.ipynb``
reads to render figures.

Outputs (under results_cd/):
  - target_mode.txt              — chosen target mode (cause_specific|binary)
  - canonical_split.parquet      — split assignments (concern 5)
  - encoders.pkl                 — fitted encoders (re-used by Part D)
  - cox_encoders.pkl             — fitted Cox-baseline encoders
  - C_predictions.parquet        — per-loan predictions for all models / horizons
  - C_metrics.parquet            — AUC, Brier, log-loss, accuracy
  - C_calibration.parquet        — decile calibration bins per model/horizon
  - C_feature_importance.parquet — per-model feature importance

Notes on changes from the previous version:
  - Plotting moved to the notebook (concern 4).
  - **Concern 1 fix:** vintage_split is called BEFORE build_feature_matrix.
    Encoders (medians, top-K levels) are fit on training data only.
  - **Concern 5 fix:** canonical_split.parquet is written so Part D can
    align without re-sampling.
  - **Concern 6 fix:** targets built via utilities.build_horizon_targets,
    which fixes the Duration >= T boundary and exposes a TARGET_MODE
    flag. Default 'cause_specific' matches Cox naturally.
  - **Concern 12 fix:** DROP_VINTAGE_YEAR_FEATURE flag (False by default)
    enables a sensitivity comparison.
  - Logging via utilities helpers, with STARTING-before / DONE-after
    every time-sensitive block.

Usage:
  python part_c_ml_models.py
  python part_c_ml_models.py --drop-vintage-year      # sensitivity run
  python part_c_ml_models.py --target-mode binary     # binary framing
"""

from __future__ import annotations

import argparse
import gc
import warnings

import joblib
import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb

    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False

from utilities import (
    SURVIVAL_TABLE,
    RESULTS_CD,
    PROJECT_ROOT,
    SAMPLE_FRAC_CD,
    RANDOM_SEED,
    TRAIN_END_YEAR,
    TEST_START_YEAR,
    TEST_END_YEAR,
    HORIZONS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ensure_results_dir,
    make_logger,
    log_step,
    log_section,
    clean_sentinels,
    vintage_split,
    build_feature_matrix,
    build_horizon_targets,
    write_canonical_split,
    write_target_mode,
    compute_metrics,
    compute_calibration_bins,
    FICO_BUCKETS,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Hyperparameters (Part-C specific — kept here, not in utilities)
# ============================================================
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 200
RF_N_JOBS = -1

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

LOGREG_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 200,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

COX_PENALIZER = 0.001


# ============================================================
# Cox-baseline features (separate from ML feature matrix because Cox
# wants the same standardized covariates Part B uses, not one-hot
# expansions)
# ============================================================
COX_COVARIATES = [
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


# ============================================================
# Logger setup
# ============================================================
ensure_results_dir(RESULTS_CD)
log = make_logger(
    "part_c_ml_models",
    log_file=PROJECT_ROOT / "part_c.log",
)


# ============================================================
# Data loading & target construction
# ============================================================
def load_and_prepare(target_mode: str) -> pd.DataFrame:
    """
    Load the survival table, sample, clean sentinels, build all horizon
    targets via utilities. Returns the prepared pandas DataFrame.

    Targets use the boundary fix (Duration >= T for "observed through T")
    and the chosen TARGET_MODE (cause_specific | binary). Concern 6.
    """
    log_section(log, "Loading and preparing survival table")
    with log_step(log, "Reading parquet"):
        df = pl.read_parquet(SURVIVAL_TABLE)
        log.info(f"  Full dataset: {df.height:,} loans")

    with log_step(log, f"Sampling {SAMPLE_FRAC_CD*100:.0f}% (seed={RANDOM_SEED})"):
        df = df.sample(fraction=SAMPLE_FRAC_CD, seed=RANDOM_SEED)
        log.info(f"  Sampled: {df.height:,} loans")

    pdf = df.to_pandas()
    del df
    gc.collect()

    with log_step(log, "Cleaning sentinels (defensive)"):
        pdf = clean_sentinels(pdf)

    pdf = pdf[pdf["Duration"] > 0].copy().reset_index(drop=True)
    log.info(f"  After Duration > 0 filter: {len(pdf):,} loans")

    with log_step(log, f"Building horizon targets (mode={target_mode!r})"):
        pdf = build_horizon_targets(pdf, horizons=HORIZONS, mode=target_mode)
    log.info("  Target counts (events / observed / censored at NaN):")
    for T in HORIZONS:
        t = pdf[f"Target_T{T}"]
        log.info(
            f"    T={T:3d}  positives={int((t == 1).sum()):>7,}  "
            f"negatives={int((t == 0).sum()):>7,}  "
            f"NaN={int(t.isna().sum()):>7,}"
        )

    return pdf


# ============================================================
# Cox baseline features (separate from ML features)
# ============================================================
def build_cox_features(
    pdf: pd.DataFrame, fit: bool = True, cox_encoders: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Same feature shape as Part B's BASE_COVARIATES. Standardizes
    continuous covariates and constructs binary dummies. The
    `fit`/`cox_encoders` parameter pair is the leakage-prevention
    handle: pass fit=True only for the training DataFrame, then reuse
    the returned encoders dict for the test DataFrame with fit=False.
    """
    if cox_encoders is None:
        cox_encoders = {}
    out = pd.DataFrame(index=pdf.index)

    for raw, std_col in [
        ("CreditScore", "FICO_z"),
        ("OriginalLoantoValueLTV", "LTV_z"),
        ("OriginalInterestRate", "Rate_z"),
        ("OriginalUPB", "UPB_z"),
    ]:
        if fit:
            mean = pdf[raw].mean()
            std = pdf[raw].std()
            cox_encoders[f"{raw}__mean"] = mean
            cox_encoders[f"{raw}__std"] = std
        else:
            mean = cox_encoders[f"{raw}__mean"]
            std = cox_encoders[f"{raw}__std"]
        out[std_col] = (pdf[raw] - mean) / std

    if fit:
        cox_encoders["DTI__median"] = pdf["OriginalDebttoIncomeRatio"].median()
    dti_filled = pdf["OriginalDebttoIncomeRatio"].fillna(cox_encoders["DTI__median"])
    if fit:
        cox_encoders["DTI__mean"] = dti_filled.mean()
        cox_encoders["DTI__std"] = dti_filled.std()
    out["DTI_z"] = (dti_filled - cox_encoders["DTI__mean"]) / cox_encoders["DTI__std"]
    out["DTI_missing"] = pdf["OriginalDebttoIncomeRatio"].isna().astype(int)

    out["is_Purchase"] = (pdf["LoanPurpose"] == "P").astype(int)
    out["is_CashOutRefi"] = (pdf["LoanPurpose"] == "C").astype(int)
    out["is_Investment"] = (pdf["OccupancyStatus"] == "I").astype(int)
    out["is_SecondHome"] = (pdf["OccupancyStatus"] == "S").astype(int)
    out["is_FirstTimeBuyer"] = (pdf["FirstTimeHomebuyerFlag"] == "Y").astype(int)
    out["is_Condo"] = (pdf["PropertyType"] == "CO").astype(int)
    out["is_PUD"] = (pdf["PropertyType"] == "PU").astype(int)
    # Concern B2: clean_sentinels already converted MI=999 to NaN
    out["has_MI"] = (pdf["MortgageInsurancePercentage"].fillna(0) > 0).astype(int)
    out["is_MultiBorrower"] = (pdf["NumberofBorrowers"].fillna(1) > 1).astype(int)

    return out, cox_encoders


# ============================================================
# Model fitting
# ============================================================
def fit_logreg(X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    """L2-regularized logistic regression. Standardize internally so the
    bundle is self-contained."""
    means = X_train.mean()
    stds = X_train.std().replace(0, 1)
    X_scaled = ((X_train - means) / stds).values
    model = LogisticRegression(**LOGREG_PARAMS)
    model.fit(X_scaled, y_train)
    return {"model": model, "means": means, "stds": stds}


def predict_logreg(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    X_scaled = ((X - bundle["means"]) / bundle["stds"]).values
    return bundle["model"].predict_proba(X_scaled)[:, 1]


def fit_random_forest(
    X_train: pd.DataFrame, y_train: np.ndarray
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train.values, y_train)
    return model


def predict_rf(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X.values)[:, 1]


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    if X_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def predict_lgbm(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def fit_cox(pdf_train: pd.DataFrame) -> tuple[CoxPHFitter, dict]:
    """Fit Cox PH on training set. Returns (fitter, encoders)."""
    cox_feat, cox_encoders = build_cox_features(pdf_train, fit=True)
    cox_df = cox_feat.copy()
    cox_df["Duration"] = pdf_train["Duration"].values
    cox_df["Event_Prepay"] = pdf_train["Event_Prepay"].values
    cox_df = cox_df.dropna()

    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(
        cox_df,
        duration_col="Duration",
        event_col="Event_Prepay",
        show_progress=False,
    )
    log.info(f"    Cox concordance: {cph.concordance_index_:.4f}")
    return cph, cox_encoders


def predict_cox_at_horizons(
    cph: CoxPHFitter, cox_features: pd.DataFrame, horizons: list[int]
) -> dict[int, np.ndarray]:
    """P(prepaid by T | X) = 1 - S(T | X)."""
    sf = cph.predict_survival_function(cox_features, times=horizons)
    preds = {}
    for T in horizons:
        if T in sf.index:
            preds[T] = (1.0 - sf.loc[T].values).astype(np.float32)
        else:
            idx = sf.index[np.argmin(np.abs(sf.index - T))]
            preds[T] = (1.0 - sf.loc[idx].values).astype(np.float32)
    return preds


# ============================================================
# Feature importance helper
# ============================================================
def extract_feature_importance(
    model_name: str, model, feature_names: list[str], horizon: int
) -> pd.DataFrame | None:
    rows = []
    if model_name == "LogReg":
        coefs = model["model"].coef_[0]
        for name, c in zip(feature_names, coefs):
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "feature": name,
                    "importance": float(abs(c)),
                    "raw_value": float(c),
                }
            )
    elif model_name == "RF":
        for name, imp in zip(feature_names, model.feature_importances_):
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "feature": name,
                    "importance": float(imp),
                    "raw_value": float(imp),
                }
            )
    elif model_name == "LGBM":
        for name, imp in zip(feature_names, model.feature_importances_):
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "feature": name,
                    "importance": float(imp),
                    "raw_value": float(imp),
                }
            )
    if not rows:
        return None
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-mode",
        choices=["cause_specific", "binary"],
        default="cause_specific",
        help="Target construction mode (concern 6). Default cause_specific.",
    )
    parser.add_argument(
        "--drop-vintage-year",
        action="store_true",
        help="Drop VintageYear from feature matrix (concern 12 sensitivity).",
    )
    args = parser.parse_args()

    log_section(log, "PART C — ML MODELS FOR PREPAYMENT")
    log.info(f"  Input:                 {SURVIVAL_TABLE}")
    log.info(f"  Output:                {RESULTS_CD}/")
    log.info(f"  TARGET_MODE:           {args.target_mode}")
    log.info(f"  DROP_VINTAGE_YEAR:     {args.drop_vintage_year}")
    log.info(f"  HORIZONS:              {HORIZONS}")
    log.info(f"  Train years:           ...{TRAIN_END_YEAR}")
    log.info(f"  Test years:            {TEST_START_YEAR}..{TEST_END_YEAR}")
    log.info("")

    # Persist target mode so notebooks know how to label charts
    write_target_mode(args.target_mode)

    # ── 1. Load & prepare ─────────────────────────────────────
    pdf = load_and_prepare(args.target_mode)

    # ── 2. SPLIT FIRST (concern 1 fix) ────────────────────────
    log_section(log, "Splitting train/test (vintage-based)")
    with log_step(
        log,
        f"vintage_split: train =<{TRAIN_END_YEAR}, "
        f"test {TEST_START_YEAR}..{TEST_END_YEAR}",
    ):
        pdf_train, pdf_test = vintage_split(
            pdf,
            train_end_year=TRAIN_END_YEAR,
            test_start_year=TEST_START_YEAR,
            test_end_year=TEST_END_YEAR,
        )
    log.info(f"  Train: {len(pdf_train):,} loans")
    log.info(f"  Test:  {len(pdf_test):,} loans")

    # ── 3. Write canonical split file (concern 5) ─────────────
    with log_step(log, "Writing canonical_split.parquet"):
        write_canonical_split(pdf_train=pdf_train, pdf_test=pdf_test)

    # ── 4. Build feature matrix ON TRAINING ONLY (concern 1) ──
    log_section(log, "Building feature matrix (encoders fit on train only)")
    with log_step(log, "build_feature_matrix(train, fit_encoders=True)"):
        X_train, encoders = build_feature_matrix(
            pdf_train,
            fit_encoders=True,
            drop_vintage_year=args.drop_vintage_year,
        )
        feature_names = X_train.columns.tolist()
    log.info(f"  Train feature matrix: {X_train.shape}")
    log.info(f"  Feature columns:      {len(feature_names)}")

    with log_step(log, "build_feature_matrix(test, fit_encoders=False)"):
        X_test, _ = build_feature_matrix(
            pdf_test,
            fit_encoders=False,
            encoders=encoders,
            drop_vintage_year=args.drop_vintage_year,
        )
    log.info(f"  Test feature matrix: {X_test.shape}")

    # Persist encoders so Part D can reuse them
    with log_step(log, "Saving encoders.pkl"):
        joblib.dump(encoders, RESULTS_CD / "encoders.pkl", compress=3)

    # ── 5. Fit models per horizon ─────────────────────────────
    log_section(log, "Fitting horizon-specific models")
    fitted_models: dict = {}
    feat_imp_rows: list[pd.DataFrame] = []

    for T in HORIZONS:
        target_col = f"Target_T{T}"
        train_mask = pdf_train[target_col].notna()
        n_train = int(train_mask.sum())
        if n_train < 1000:
            log.warning(f"  T={T}: only {n_train} training rows — skipping")
            continue

        Xt = X_train.loc[train_mask]
        yt = pdf_train.loc[train_mask, target_col].values.astype(np.int8)
        log.info(
            f"  T={T:3d}: training on {len(yt):,} rows, "
            f"{int(yt.sum()):,} positives "
            f"({yt.mean():.4f})"
        )

        with log_step(log, f"  LogReg(T={T})"):
            bundle = fit_logreg(Xt, yt)
            fitted_models[("LogReg", T)] = bundle
            fi = extract_feature_importance("LogReg", bundle, feature_names, T)
            if fi is not None:
                feat_imp_rows.append(fi)

        with log_step(log, f"  RandomForest(T={T})"):
            rf_model = fit_random_forest(Xt, yt)
            fitted_models[("RF", T)] = rf_model
            fi = extract_feature_importance("RF", rf_model, feature_names, T)
            if fi is not None:
                feat_imp_rows.append(fi)

        if HAVE_LGBM:
            with log_step(log, f"  LightGBM(T={T}) with early stopping"):
                X_tr2, X_val, y_tr2, y_val = train_test_split(
                    Xt,
                    yt,
                    test_size=0.1,
                    random_state=RANDOM_SEED,
                )
                lgbm_model = fit_lgbm(X_tr2, y_tr2, X_val, y_val)
                fitted_models[("LGBM", T)] = lgbm_model
                fi = extract_feature_importance("LGBM", lgbm_model, feature_names, T)
                if fi is not None:
                    feat_imp_rows.append(fi)

        gc.collect()

    # ── 6. Cox baseline ───────────────────────────────────────
    log_section(log, "Fitting Cox PH baseline")
    cph = None
    cox_encoders = {}
    try:
        with log_step(log, "fit_cox(train)"):
            cph, cox_encoders = fit_cox(pdf_train)
            fitted_models[("Cox", "all")] = (cph, cox_encoders)
        with log_step(log, "Saving cox_encoders.pkl"):
            joblib.dump(cox_encoders, RESULTS_CD / "cox_encoders.pkl", compress=3)
    except Exception as e:
        log.warning(f"  Cox fit failed: {type(e).__name__}: {e}")
        log.warning("  Continuing without Cox baseline. Part D requires")
        log.warning("  cox_encoders.pkl, so it will need to refit.")
        # Persist an empty encoders file so Part D's "load it" check passes
        joblib.dump({}, RESULTS_CD / "cox_encoders.pkl", compress=3)

    # ── 7. Predict on test set ────────────────────────────────
    log_section(log, "Predicting on test set")
    pred_columns: dict[str, np.ndarray] = {}

    for (model_name, T), model in fitted_models.items():
        if model_name == "Cox":
            continue
        with log_step(log, f"  Predicting {model_name} T={T}"):
            if model_name == "LogReg":
                preds = predict_logreg(model, X_test)
            elif model_name == "RF":
                preds = predict_rf(model, X_test)
            elif model_name == "LGBM":
                preds = predict_lgbm(model, X_test)
            else:
                continue
            pred_columns[f"pred_{model_name}_T{T}"] = preds.astype(np.float32)

    # Cox predictions: build features with same encoders, predict at all horizons
    if cph is not None:
        with log_step(log, "  Cox: build features and predict at horizons"):
            cox_feat_test, _ = build_cox_features(
                pdf_test, fit=False, cox_encoders=cox_encoders
            )
            cox_valid_mask = cox_feat_test.notna().all(axis=1)
            log.info(
                f"    Valid Cox feature rows: "
                f"{int(cox_valid_mask.sum()):,} / {len(cox_feat_test):,}"
            )
            cox_preds_at_T = predict_cox_at_horizons(
                cph,
                cox_feat_test[cox_valid_mask],
                HORIZONS,
            )
            for T in HORIZONS:
                pred_arr = np.full(len(pdf_test), np.nan, dtype=np.float32)
                pred_arr[cox_valid_mask.values] = cox_preds_at_T[T]
                pred_columns[f"pred_Cox_T{T}"] = pred_arr
    else:
        log.info("  Skipping Cox predictions (Cox fit was not successful).")
        # Emit all-NaN columns so downstream metrics code can iterate uniformly
        for T in HORIZONS:
            pred_columns[f"pred_Cox_T{T}"] = np.full(
                len(pdf_test),
                np.nan,
                dtype=np.float32,
            )

    # ── 8. Build & save predictions parquet ───────────────────
    log_section(log, "Building predictions output")
    out_predictions = pd.DataFrame(
        {
            "LoanSequenceNumber": pdf_test["LoanSequenceNumber"].values,
            "VintageYear": pdf_test["VintageYear"].values.astype(np.int16),
            "VintageBucket": pdf_test["VintageBucket"].values,
            "FICO_bucket": pdf_test["FICO_bucket"].values,
            "LTV_bucket": pdf_test["LTV_bucket"].values,
            "LoanPurpose": pdf_test["LoanPurpose"].values,
            "Duration": pdf_test["Duration"].values.astype(np.int16),
            "Event_Prepay": pdf_test["Event_Prepay"].values.astype(np.int8),
            "MaxObsMonths": pdf_test["MaxObsMonths"].values.astype(np.int16),
        }
    )
    for T in HORIZONS:
        out_predictions[f"Target_T{T}"] = pdf_test[f"Target_T{T}"].values
    for col, arr in pred_columns.items():
        out_predictions[col] = arr

    with log_step(log, "Writing C_predictions.parquet"):
        pl.from_pandas(out_predictions).write_parquet(
            RESULTS_CD / "C_predictions.parquet",
            compression="zstd",
            compression_level=8,
        )
    log.info(
        f"  C_predictions.parquet: {len(out_predictions):,} rows, "
        f"{out_predictions.shape[1]} cols"
    )

    # ── 9. Compute metrics ────────────────────────────────────
    log_section(log, "Computing metrics")
    model_names_active = (
        ["LogReg", "RF", "LGBM", "Cox"] if HAVE_LGBM else ["LogReg", "RF", "Cox"]
    )
    metric_rows: list[dict] = []

    with log_step(log, "Computing per-model / per-horizon / per-stratum metrics"):
        for T in HORIZONS:
            target_col = f"Target_T{T}"
            valid_mask = out_predictions[target_col].notna().values
            if not valid_mask.any():
                continue
            y_all = out_predictions.loc[valid_mask, target_col].values

            for mname in model_names_active:
                pred_col = f"pred_{mname}_T{T}"
                if pred_col not in out_predictions.columns:
                    continue
                preds = out_predictions.loc[valid_mask, pred_col].values
                pred_valid = ~np.isnan(preds)

                # Overall
                metric_rows.append(
                    {
                        "model": mname,
                        "horizon": T,
                        "stratum_type": "all",
                        "stratum_value": "all",
                        **compute_metrics(y_all[pred_valid], preds[pred_valid]),
                    }
                )

                # By vintage year
                for v in sorted(
                    out_predictions.loc[valid_mask, "VintageYear"].unique()
                ):
                    sub_idx = (
                        out_predictions.loc[valid_mask, "VintageYear"] == v
                    ).values
                    if sub_idx.sum() < 100:
                        continue
                    metric_rows.append(
                        {
                            "model": mname,
                            "horizon": T,
                            "stratum_type": "vintage",
                            "stratum_value": str(int(v)),
                            **compute_metrics(
                                y_all[sub_idx & pred_valid],
                                preds[sub_idx & pred_valid],
                            ),
                        }
                    )

                # By FICO bucket
                for fb in FICO_BUCKETS:
                    sub_idx = (
                        out_predictions.loc[valid_mask, "FICO_bucket"] == fb
                    ).values
                    if sub_idx.sum() < 100:
                        continue
                    metric_rows.append(
                        {
                            "model": mname,
                            "horizon": T,
                            "stratum_type": "FICO",
                            "stratum_value": fb,
                            **compute_metrics(
                                y_all[sub_idx & pred_valid],
                                preds[sub_idx & pred_valid],
                            ),
                        }
                    )

    metrics_df = pd.DataFrame(metric_rows)
    pl.from_pandas(metrics_df).write_parquet(
        RESULTS_CD / "C_metrics.parquet",
        compression="zstd",
        compression_level=8,
    )
    log.info(f"  C_metrics.parquet: {len(metrics_df):,} rows")

    # ── 10. Calibration ───────────────────────────────────────
    log_section(log, "Computing calibration bins")
    calib_rows: list[pd.DataFrame] = []
    with log_step(log, "Computing calibration"):
        for T in HORIZONS:
            target_col = f"Target_T{T}"
            valid_mask = out_predictions[target_col].notna().values
            if not valid_mask.any():
                continue
            y_all = out_predictions.loc[valid_mask, target_col].values

            for mname in model_names_active:
                pred_col = f"pred_{mname}_T{T}"
                if pred_col not in out_predictions.columns:
                    continue
                preds = out_predictions.loc[valid_mask, pred_col].values
                pred_valid = ~np.isnan(preds)
                cb = compute_calibration_bins(
                    y_all[pred_valid],
                    preds[pred_valid],
                    n_bins=10,
                )
                if len(cb) == 0:
                    continue
                cb["model"] = mname
                cb["horizon"] = T
                calib_rows.append(cb)

    if calib_rows:
        calib_df = pd.concat(calib_rows, ignore_index=True)
        pl.from_pandas(calib_df).write_parquet(
            RESULTS_CD / "C_calibration.parquet",
            compression="zstd",
            compression_level=8,
        )
        log.info(f"  C_calibration.parquet: {len(calib_df):,} rows")

    # ── 11. Feature importance ────────────────────────────────
    log_section(log, "Saving feature importance")
    if feat_imp_rows:
        fi_df = pd.concat(feat_imp_rows, ignore_index=True)
        pl.from_pandas(fi_df).write_parquet(
            RESULTS_CD / "C_feature_importance.parquet",
            compression="zstd",
            compression_level=8,
        )
        log.info(f"  C_feature_importance.parquet: {len(fi_df):,} rows")

    # ── 12. Summary ───────────────────────────────────────────
    log_section(log, "METRICS SUMMARY (overall, by horizon)")
    overall = metrics_df[metrics_df["stratum_type"] == "all"].copy()
    if len(overall):
        log.info("AUC by model and horizon:")
        for line in (
            overall.pivot_table(
                index="model",
                columns="horizon",
                values="auc",
                aggfunc="first",
            )
            .round(4)
            .to_string()
            .splitlines()
        ):
            log.info(f"  {line}")
        log.info("Brier by model and horizon:")
        for line in (
            overall.pivot_table(
                index="model",
                columns="horizon",
                values="brier",
                aggfunc="first",
            )
            .round(4)
            .to_string()
            .splitlines()
        ):
            log.info(f"  {line}")

    log.info(f"  Output files in {RESULTS_CD}/:")
    for f in sorted(RESULTS_CD.glob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            log.info(f"    {f.name:40s} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
