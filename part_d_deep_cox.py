"""
part_d_deep_cox.py
===================
Part D — Deep Cox Model (DeepSurv) — compute only

  D(i)   Implement a Deep Cox model: lambda(t|X) = lambda_0(t) * exp(f_theta(X))
  D(ii)  f_theta is an MLP (replaces the linear predictor in classical Cox)
  D(iii) Trained by minimizing the Cox partial likelihood (mini-batch
         stochastic approximation — see note below)
  D(iv)  Out-of-sample comparison with classical Cox (Part B) and the
         ML models from Part C (LogReg, RF, LightGBM)
  D(v)   Linear baseline (depth=0) vs Deep ablation, to isolate the
         contribution of nonlinear effects in mortgage prepayment

This script does NO plotting. It writes parquet/state-dict files into
``results_cd/`` (alongside Part C). The notebook
``results_cd/part_d_plot.ipynb`` reads those files to render figures.

Inputs (must already exist; produced by Part C):
  - results_cd/canonical_split.parquet   single source of truth for split
  - results_cd/encoders.pkl              fitted on Part C training data
  - results_cd/target_mode.txt           'cause_specific' or 'binary'
  - results_cd/C_predictions.parquet     used for row-alignment sanity check

Outputs (under results_cd/):
  - canonical_split_with_val.parquet     train/val/test (val carved here)
  - D_predictions.parquet                Deep + Linear Cox preds, aligned
                                         row-by-row with C_predictions
  - D_metrics.parquet                    AUC/Brier/log-loss/accuracy by
                                         (model, horizon, stratum)
  - D_calibration.parquet                decile calibration bins
  - D_training_history.parquet           per-epoch train/val loss
  - D_linear_vs_deep.parquet             D(v) summary: Δ-AUC by horizon
  - D_model.pt                           Deep Cox PyTorch state dict
  - D_linear_model.pt                    Linear Cox PyTorch state dict
  - D_model_meta.json                    feature names, scaler stats,
                                         arch, training config

Notes / fixes from the previous version:
  - **Concern 2 fix (preprocessing leakage):** features are built only
    after the canonical train/val/test split, with encoders loaded from
    Part C (fit on training data only).
  - **Concern 3 fix (validation leakage):** validation is carved from
    train BEFORE feature transformation and standardization. The scaler
    is fit on the inner training set only; val and test are transformed.
  - **Concern 4 fix (folder layout):** outputs go to ``results_cd/``,
    alongside Part C, so the comparison plot reads from one folder.
  - **Concern 5 fix (alignment):** Part C's canonical_split.parquet is
    the single source of truth. Part D restricts the survival table to
    those exact loan IDs and preserves the canonical row order (no
    independent re-sampling). A sanity check verifies the test rows
    match C_predictions.parquet exactly.
  - **Concern 7 (mini-batch Cox):** the Cox partial likelihood is
    computed per mini-batch — risk sets are batch-local, not global.
    This is a stochastic approximation to the exact partial likelihood
    over the full risk set. Exact full-batch training would be more
    accurate but more expensive in memory and time; mini-batch is the
    standard pycox approach. With BATCH_SIZE = 512 the approximation
    is the standard one used in the DeepSurv literature.
  - **Concern 12 (VintageYear sensitivity):** ``--drop-vintage-year``
    flag is plumbed through. Use the same value as the Part C run
    you want to compare against (otherwise feature counts will mismatch).

Usage:
  python part_d_deep_cox.py
  python part_d_deep_cox.py --drop-vintage-year     # sensitivity run
  python part_d_deep_cox.py --target-mode binary    # binary framing

Memory: peak ~2-3 GB
Runtime: ~20-60 minutes on CPU; faster with CUDA
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from utilities import (
    SURVIVAL_TABLE,
    RESULTS_CD,
    PROJECT_ROOT,
    RANDOM_SEED,
    SAMPLE_FRAC_CD,
    VAL_FRAC,
    HORIZONS,
    FICO_BUCKETS,
    CANONICAL_SPLIT_FILE,
    ENCODERS_FILE,
    ensure_results_dir,
    make_logger,
    log_step,
    log_section,
    clean_sentinels,
    build_horizon_targets,
    build_feature_matrix,
    standardize_three,
    load_canonical_split,
    restrict_to_canonical_split,
    carve_validation,
    write_canonical_split,
    read_target_mode,
    compute_metrics,
    compute_calibration_bins,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Hyperparameters (Part-D specific — kept here, not in utilities)
# ============================================================
ARCH_DEEP = {
    "num_nodes": [64, 64, 64],   # 3 hidden layers
    "dropout": 0.1,
    "batch_norm": True,
    "out_features": 1,
    "output_bias": False,
    "activation": "relu",
}

# D(v) ablation: depth=0 — pure linear predictor inside the Cox NN.
# This is functionally a re-implementation of classical Cox via gradient
# descent + Breslow baseline; comparison with ARCH_DEEP isolates the
# contribution of nonlinear interactions.
ARCH_LINEAR = {
    "num_nodes": [],             # no hidden layers
    "dropout": 0.0,
    "batch_norm": False,
    "out_features": 1,
    "output_bias": False,
    "activation": "relu",
}

EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 8
PREDICT_CHUNK = 50_000  # rows per chunk in predict_at_horizons


# ============================================================
# Logger setup
# ============================================================
ensure_results_dir(RESULTS_CD)
log = make_logger(
    "part_d_deep_cox",
    log_file=PROJECT_ROOT / "part_d.log",
)


# ============================================================
# Deferred torch / pycox import
# ============================================================
def import_torch_stack() -> dict:
    """
    Import PyTorch and pycox once, log versions, return a dict of refs.
    Deferred so the failure mode 'package not installed' surfaces with a
    nice message before we touch the data.
    """
    log_section(log, "Importing PyTorch and pycox")
    try:
        import torch
        import torch.nn as nn  # noqa: F401  (re-exported via stack)
        import torchtuples as tt
        from pycox.models import CoxPH
    except ImportError as e:
        log.error(f"Required package missing: {e}")
        log.error("Install with: pip install torch pycox torchtuples")
        sys.exit(1)

    log.info(f"  torch version: {torch.__version__}")
    log.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        log.info("  Using CPU.")

    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    return {"torch": torch, "tt": tt, "CoxPH": CoxPH}


# ============================================================
# Data loading
# ============================================================
def load_survival_for_part_d(target_mode: str) -> pd.DataFrame:
    """
    Load the survival table, sample with the SAME fraction and seed as
    Part C, sentinel-clean, build horizon targets per `target_mode`.

    The Part-C-canonical loan-ID set is enforced separately via
    restrict_to_canonical_split() — which is the single source of truth
    for the actual train/test partition. Sampling here only matters for
    consistency with Part C's universe; restrict_to_canonical_split()
    will further intersect with the canonical IDs.
    """
    log_section(log, "Loading and preparing survival table")
    with log_step(log, "Reading parquet"):
        df = pl.read_parquet(SURVIVAL_TABLE)
        log.info(f"  Full dataset: {df.height:,} loans")

    with log_step(
        log, f"Sampling {SAMPLE_FRAC_CD * 100:.0f}% (seed={RANDOM_SEED})"
    ):
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
    log.info("  Target counts (positives / negatives / censored):")
    for T in HORIZONS:
        t = pdf[f"Target_T{T}"]
        log.info(
            f"    T={T:3d}  positives={int((t == 1).sum()):>7,}  "
            f"negatives={int((t == 0).sum()):>7,}  "
            f"NaN={int(t.isna().sum()):>7,}"
        )

    return pdf


# ============================================================
# Deep Cox model construction & training
# ============================================================
def build_mlp(stack: dict, input_dim: int, arch: dict):
    """Build an MLP via torchtuples.practical.MLPVanilla."""
    tt = stack["tt"]
    return tt.practical.MLPVanilla(
        in_features=input_dim,
        num_nodes=arch["num_nodes"],
        out_features=arch["out_features"],
        batch_norm=arch["batch_norm"],
        dropout=arch["dropout"],
        output_bias=arch["output_bias"],
    )


def train_deepsurv(
    stack: dict,
    X_train: np.ndarray,
    durations_train: np.ndarray,
    events_train: np.ndarray,
    X_val: np.ndarray,
    durations_val: np.ndarray,
    events_val: np.ndarray,
    arch: dict,
    model_name: str = "DeepCox",
) -> tuple:
    """
    Train a DeepSurv model. Returns (model, history_df, train_time_s).

    The Cox partial likelihood is computed per mini-batch — risk sets
    are batch-local. This is a stochastic approximation that pycox uses
    by default. With BATCH_SIZE=512 the approximation is reasonable and
    is what the DeepSurv literature uses.
    """
    torch = stack["torch"]
    tt = stack["tt"]
    CoxPH = stack["CoxPH"]

    log.info(
        f"  Building {model_name} network with arch={arch['num_nodes']} "
        f"(in_dim={X_train.shape[1]})"
    )
    net = build_mlp(stack, X_train.shape[1], arch)
    n_params = sum(p.numel() for p in net.parameters())
    log.info(f"    Network parameters: {n_params:,}")

    model = CoxPH(net, optimizer=tt.optim.Adam(lr=LEARNING_RATE))

    # pycox expects target as (durations, events) tuple of float32
    y_train = (
        durations_train.astype(np.float32),
        events_train.astype(np.float32),
    )
    y_val = (
        durations_val.astype(np.float32),
        events_val.astype(np.float32),
    )

    callbacks = [tt.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE)]

    log.info(
        f"  Training: up to {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
        f"early-stopping patience={EARLY_STOPPING_PATIENCE}"
    )
    log.info(
        f"    Inner train: {len(X_train):,} rows, "
        f"{int(events_train.sum()):,} events ({events_train.mean():.4f})"
    )
    log.info(
        f"    Val:         {len(X_val):,} rows, "
        f"{int(events_val.sum()):,} events ({events_val.mean():.4f})"
    )

    t0 = time.time()
    with log_step(log, f"  Training {model_name}"):
        log_obj = model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=False,
            val_data=(X_val, y_val),
            val_batch_size=BATCH_SIZE,
        )
    train_time = time.time() - t0
    log.info(
        f"  Training complete in {train_time:.1f}s, "
        f"{log_obj.epoch + 1} epochs"
    )

    # Extract per-epoch history.
    try:
        hist = (
            log_obj.to_pandas()
            .reset_index()
            .rename(columns={"index": "epoch"})
        )
    except Exception as e:
        log.warning(f"  Could not extract training history: {e}")
        hist = pd.DataFrame()

    if not hist.empty:
        log.info("  Per-epoch losses:")
        for _, r in hist.iterrows():
            ep = int(r.get("epoch", -1))
            tl = float(r.get("train_loss", np.nan))
            vl = float(r.get("val_loss", np.nan))
            log.info(f"    epoch {ep:3d}: train={tl:.4f}  val={vl:.4f}")

    return model, hist, train_time


def compute_baseline_hazards(
    model,
    X_train: np.ndarray,
    durations_train: np.ndarray,
    events_train: np.ndarray,
) -> None:
    """
    Compute Breslow baseline hazards on the training set.

    Note: per pycox convention this uses the *full* training data
    (inner-train ∪ val). The val set has already been used for early
    stopping, so re-using it for the non-trainable Breslow estimator
    (a weighted-KM-style step function) does not introduce additional
    leakage to the test set. Doing so simply gives a slightly more
    accurate baseline-hazard step function.
    """
    with log_step(log, "  Computing Breslow baseline hazards"):
        model.compute_baseline_hazards(
            X_train,
            (
                durations_train.astype(np.float32),
                events_train.astype(np.float32),
            ),
            batch_size=BATCH_SIZE,
        )


def predict_at_horizons(
    model, X_test: np.ndarray, horizons: list[int],
    chunk_size: int = PREDICT_CHUNK,
) -> dict[int, np.ndarray]:
    """
    Predict P(prepaid by T | X) = 1 - S(T | X) for each T in `horizons`.

    Chunked to control memory: pycox's predict_surv_df returns a wide
    DataFrame (rows = training event times, columns = subjects).
    """
    log.info(
        f"  Predicting on {len(X_test):,} test rows "
        f"in chunks of {chunk_size:,}"
    )

    predictions = {
        T: np.zeros(len(X_test), dtype=np.float32) for T in horizons
    }
    n_chunks = (len(X_test) + chunk_size - 1) // chunk_size
    t0 = time.time()
    for i, start in enumerate(range(0, len(X_test), chunk_size)):
        end = min(start + chunk_size, len(X_test))
        X_chunk = X_test[start:end]

        # surv: rows = event times, columns = subjects
        surv = model.predict_surv_df(X_chunk)

        for T in horizons:
            # Find largest training event time <= T
            valid_times = surv.index[surv.index <= T]
            if len(valid_times) == 0:
                # No training events before T → S(T|X) is still 1, so
                # P(prepaid by T) = 0
                predictions[T][start:end] = 0.0
            else:
                t_use = valid_times[-1]
                predictions[T][start:end] = (
                    1.0 - surv.loc[t_use].values
                ).astype(np.float32)

        if (i + 1) % 5 == 0 or (i + 1) == n_chunks:
            elapsed = time.time() - t0
            log.info(f"    chunk {i + 1}/{n_chunks}: {elapsed:.1f}s elapsed")

    log.info(f"  Total prediction time: {time.time() - t0:.1f}s")
    return predictions


# ============================================================
# Evaluation (mirrors Part C's per-stratum metric layout)
# ============================================================
def evaluate_model(
    predictions: dict[int, np.ndarray],
    pdf_test: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-stratum metrics & calibration, mirroring Part C."""
    metric_rows: list[dict] = []
    calib_rows: list[pd.DataFrame] = []

    for T in HORIZONS:
        target_col = f"Target_T{T}"
        if target_col not in pdf_test.columns:
            continue
        target = pdf_test[target_col]
        valid = target.notna()
        if not valid.any():
            continue
        y_all = target[valid].values.astype(int)
        p_all = predictions[T][valid.values]

        # Overall
        metric_rows.append({
            "model": model_name, "horizon": T,
            "stratum_type": "all", "stratum_value": "all",
            **compute_metrics(y_all, p_all),
        })

        # Calibration
        cb = compute_calibration_bins(y_all, p_all, n_bins=10)
        if len(cb):
            cb = cb.copy()
            cb["model"] = model_name
            cb["horizon"] = T
            calib_rows.append(cb)

        # By vintage year (only on the valid subset)
        vintage = pdf_test.loc[valid, "VintageYear"].values
        for v in np.unique(vintage):
            mask = vintage == v
            if mask.sum() < 100:
                continue
            metric_rows.append({
                "model": model_name, "horizon": T,
                "stratum_type": "vintage", "stratum_value": str(int(v)),
                **compute_metrics(y_all[mask], p_all[mask]),
            })

        # By FICO bucket
        fico = pdf_test.loc[valid, "FICO_bucket"].values
        for fb in FICO_BUCKETS:
            mask = fico == fb
            if mask.sum() < 100:
                continue
            metric_rows.append({
                "model": model_name, "horizon": T,
                "stratum_type": "FICO", "stratum_value": fb,
                **compute_metrics(y_all[mask], p_all[mask]),
            })

    metrics_df = pd.DataFrame(metric_rows)
    calib_df = (
        pd.concat(calib_rows, ignore_index=True)
        if calib_rows else pd.DataFrame()
    )
    return metrics_df, calib_df


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument(
        "--target-mode", choices=["cause_specific", "binary"],
        default=None,
        help="Override target mode. Default: read from results_cd/"
             "target_mode.txt (matches the Part C run).",
    )
    parser.add_argument(
        "--drop-vintage-year", action="store_true",
        help="Drop VintageYear from the feature matrix (concern 12 "
             "sensitivity). Should match the Part C run you compare to.",
    )
    return parser.parse_args()


def assert_alignment(pdf_test: pd.DataFrame) -> None:
    """
    Verify the test rows we're about to predict on match Part C's test
    rows row-by-row. Without this, a silent ordering drift would
    invalidate the side-by-side comparison.
    """
    c_path = RESULTS_CD / "C_predictions.parquet"
    if not c_path.exists():
        log.warning(
            "  C_predictions.parquet not found — skipping alignment check. "
            "Plotting comparisons may be misaligned if Part C wasn't run."
        )
        return
    c_ids = pl.read_parquet(c_path, columns=["LoanSequenceNumber"])\
              .to_pandas()["LoanSequenceNumber"].values
    d_ids = pdf_test["LoanSequenceNumber"].values
    if len(c_ids) != len(d_ids):
        raise RuntimeError(
            f"Test row count mismatch: C has {len(c_ids):,} rows, "
            f"D has {len(d_ids):,} rows. Re-run Part C (or check that "
            f"--target-mode and --drop-vintage-year match)."
        )
    if not np.array_equal(c_ids, d_ids):
        raise RuntimeError(
            "Test row order mismatch between C and D. "
            "restrict_to_canonical_split should have ensured equality. "
            "Did the canonical_split.parquet file change between C and D?"
        )
    log.info(
        f"  Alignment OK: {len(d_ids):,} test rows match "
        f"C_predictions.parquet exactly."
    )


def main() -> None:
    args = parse_args()
    t_start = time.time()

    log_section(log, "PART D — DEEP COX MODEL")
    log.info(f"  Input survival table: {SURVIVAL_TABLE}")
    log.info(f"  Input canonical split:{CANONICAL_SPLIT_FILE}")
    log.info(f"  Input encoders:       {ENCODERS_FILE}")
    log.info(f"  Output dir:           {RESULTS_CD}/")
    log.info("")

    # ── Resolve target_mode ──────────────────────────────────
    if args.target_mode is not None:
        target_mode = args.target_mode
        log.info(f"  Target mode (CLI override): {target_mode!r}")
    else:
        target_mode = read_target_mode()
        log.info(f"  Target mode (from target_mode.txt): {target_mode!r}")
    log.info(f"  Drop VintageYear:     {args.drop_vintage_year}")
    log.info("")

    # ── 1. Verify Part C ran ─────────────────────────────────
    log_section(log, "Step 1/9: Verify Part C outputs exist")
    if not CANONICAL_SPLIT_FILE.exists():
        log.error(
            f"  Canonical split file not found at {CANONICAL_SPLIT_FILE}. "
            f"Run part_c_ml_models.py first."
        )
        sys.exit(1)
    if not ENCODERS_FILE.exists():
        log.error(
            f"  Encoders file not found at {ENCODERS_FILE}. "
            f"Run part_c_ml_models.py first."
        )
        sys.exit(1)
    log.info(f"  Found: {CANONICAL_SPLIT_FILE}")
    log.info(f"  Found: {ENCODERS_FILE}")

    # ── 2. Import torch / pycox ──────────────────────────────
    log_section(log, "Step 2/9: Import torch and pycox")
    stack = import_torch_stack()

    # ── 3. Load encoders and canonical split ─────────────────
    log_section(log, "Step 3/9: Load encoders & canonical split from Part C")
    with log_step(log, "Loading encoders.pkl"):
        encoders = joblib.load(ENCODERS_FILE)
    log.info(f"  Encoders dict: {len(encoders)} keys")

    with log_step(log, "Loading canonical_split.parquet"):
        canonical = load_canonical_split()
    n_train = int((canonical["split"] == "train").sum())
    n_test = int((canonical["split"] == "test").sum())
    log.info(f"  Canonical: train={n_train:,} test={n_test:,}")

    # ── 4. Load survival table & build horizon targets ───────
    log_section(log, "Step 4/9: Load survival table")
    pdf = load_survival_for_part_d(target_mode)

    # ── 5. Restrict to canonical split, carve val ────────────
    log_section(log, "Step 5/9: Restrict to canonical split, carve val")
    with log_step(log, "Restrict to canonical train"):
        pdf_train_full = restrict_to_canonical_split(
            pdf, "train", canonical=canonical
        )
    log.info(f"  Train (full): {len(pdf_train_full):,} loans")

    with log_step(log, "Restrict to canonical test"):
        pdf_test = restrict_to_canonical_split(
            pdf, "test", canonical=canonical
        )
    log.info(f"  Test:         {len(pdf_test):,} loans")
    del pdf
    gc.collect()

    # Concern 3 fix: carve val BEFORE feature transformation / scaling.
    with log_step(
        log, f"Carve {VAL_FRAC*100:.0f}% val (seed={RANDOM_SEED})"
    ):
        pdf_train, pdf_val = carve_validation(
            pdf_train_full, frac=VAL_FRAC, seed=RANDOM_SEED
        )
    log.info(
        f"  Inner train: {len(pdf_train):,} loans, "
        f"Val: {len(pdf_val):,} loans"
    )

    # Persist train/val/test split file so the carve is reproducible
    with log_step(log, "Writing canonical_split_with_val.parquet"):
        write_canonical_split(
            pdf_train=pdf_train,
            pdf_val=pdf_val,
            pdf_test=pdf_test,
            path=RESULTS_CD / "canonical_split_with_val.parquet",
        )

    # ── 6. Build features (reuse C's encoders) ───────────────
    log_section(log, "Step 6/9: Build feature matrices (encoders from Part C)")
    with log_step(log, "build_feature_matrix(train, fit_encoders=False)"):
        X_train, _ = build_feature_matrix(
            pdf_train, fit_encoders=False, encoders=encoders,
            drop_vintage_year=args.drop_vintage_year,
        )
        feature_names = X_train.columns.tolist()
    log.info(f"  Inner train X: {X_train.shape}")
    log.info(f"  Feature columns: {len(feature_names)}")

    with log_step(log, "build_feature_matrix(val, fit_encoders=False)"):
        X_val, _ = build_feature_matrix(
            pdf_val, fit_encoders=False, encoders=encoders,
            drop_vintage_year=args.drop_vintage_year,
        )

    with log_step(log, "build_feature_matrix(test, fit_encoders=False)"):
        X_test, _ = build_feature_matrix(
            pdf_test, fit_encoders=False, encoders=encoders,
            drop_vintage_year=args.drop_vintage_year,
        )
    log.info(f"  Val   X: {X_val.shape}")
    log.info(f"  Test  X: {X_test.shape}")

    # Sanity: shapes must match
    if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise RuntimeError(
            f"Feature column count mismatch: train={X_train.shape[1]} "
            f"val={X_val.shape[1]} test={X_test.shape[1]}"
        )

    # Alignment sanity check (concern 5)
    assert_alignment(pdf_test)

    # ── 7. Standardize (fit on inner train only, concern 3) ──
    log_section(log, "Step 7/9: Standardize features (scaler fit on train)")
    with log_step(log, "standardize_three(train, val, test)"):
        X_train_arr, X_val_arr, X_test_arr, scaler = standardize_three(
            X_train, X_val, X_test
        )
    log.info(
        f"  X_train_arr mean={X_train_arr.mean():.4f}, "
        f"std={X_train_arr.std():.4f}  (should be ~0/1)"
    )
    log.info(
        f"  X_val_arr mean={X_val_arr.mean():.4f}, "
        f"std={X_val_arr.std():.4f}"
    )
    log.info(
        f"  X_test_arr mean={X_test_arr.mean():.4f}, "
        f"std={X_test_arr.std():.4f}"
    )
    # Free the pandas copies; we only need numpy arrays from here on
    del X_train, X_val, X_test
    gc.collect()

    # Pull duration/event arrays
    d_tr = pdf_train["Duration"].values.astype(np.float32)
    e_tr = pdf_train["Event_Prepay"].values.astype(np.float32)
    d_val = pdf_val["Duration"].values.astype(np.float32)
    e_val = pdf_val["Event_Prepay"].values.astype(np.float32)

    log.info(
        f"  Train events: {int(e_tr.sum()):,} / {len(e_tr):,} "
        f"({e_tr.mean():.4f})"
    )
    log.info(
        f"  Val events:   {int(e_val.sum()):,} / {len(e_val):,} "
        f"({e_val.mean():.4f})"
    )

    # Full-train arrays for baseline-hazards estimation (concatenate
    # inner train + val; see compute_baseline_hazards docstring)
    X_train_full_arr = np.concatenate([X_train_arr, X_val_arr], axis=0)
    d_train_full = np.concatenate([d_tr, d_val], axis=0)
    e_train_full = np.concatenate([e_tr, e_val], axis=0)

    # ── 8. Train Deep Cox ────────────────────────────────────
    log_section(log, "Step 8/9: Train Deep Cox model — D(i)–(iii)")
    deep_model, deep_history, deep_time = train_deepsurv(
        stack, X_train_arr, d_tr, e_tr, X_val_arr, d_val, e_val,
        arch=ARCH_DEEP, model_name="DeepCox",
    )
    deep_history["model"] = "DeepCox"

    compute_baseline_hazards(
        deep_model, X_train_full_arr, d_train_full, e_train_full
    )

    with log_step(log, "  DeepCox: predict on test"):
        deep_predictions = predict_at_horizons(
            deep_model, X_test_arr, HORIZONS
        )

    log.info("  Deep prediction sanity:")
    for T in HORIZONS:
        p = deep_predictions[T]
        log.info(
            f"    T={T:3d}: mean={p.mean():.4f} "
            f"min={p.min():.4f} max={p.max():.4f}"
        )

    deep_metrics, deep_calib = evaluate_model(
        deep_predictions, pdf_test, "DeepCox"
    )
    log.info("  DeepCox overall metrics:")
    for _, r in deep_metrics[deep_metrics["stratum_type"] == "all"].iterrows():
        log.info(
            f"    T={int(r['horizon']):3d}: "
            f"AUC={r['auc']:.4f}, Brier={r['brier']:.4f}, "
            f"LogLoss={r['log_loss']:.4f}, n={int(r['n']):,}"
        )

    # ── 9. Train Linear Cox baseline (D(v) ablation) ─────────
    log_section(log, "Step 9/9: Train Linear Cox baseline — D(v) ablation")
    linear_model, linear_history, linear_time = train_deepsurv(
        stack, X_train_arr, d_tr, e_tr, X_val_arr, d_val, e_val,
        arch=ARCH_LINEAR, model_name="LinearCox",
    )
    linear_history["model"] = "LinearCox"

    compute_baseline_hazards(
        linear_model, X_train_full_arr, d_train_full, e_train_full
    )

    with log_step(log, "  LinearCox: predict on test"):
        linear_predictions = predict_at_horizons(
            linear_model, X_test_arr, HORIZONS
        )

    linear_metrics, linear_calib = evaluate_model(
        linear_predictions, pdf_test, "LinearCox"
    )
    log.info("  LinearCox overall metrics:")
    for _, r in linear_metrics[linear_metrics["stratum_type"] == "all"].iterrows():
        log.info(
            f"    T={int(r['horizon']):3d}: "
            f"AUC={r['auc']:.4f}, Brier={r['brier']:.4f}, "
            f"LogLoss={r['log_loss']:.4f}, n={int(r['n']):,}"
        )

    # ──────────────────────────────────────────────────────────
    # Build outputs
    # ──────────────────────────────────────────────────────────
    log_section(log, "Building output files")

    # 10a. D_predictions.parquet — aligned with C_predictions row order
    out = pd.DataFrame({
        "LoanSequenceNumber": pdf_test["LoanSequenceNumber"].values,
        "VintageYear": pdf_test["VintageYear"].values.astype(np.int16),
        "VintageBucket": pdf_test["VintageBucket"].values,
        "FICO_bucket": pdf_test["FICO_bucket"].values,
        "LTV_bucket": pdf_test["LTV_bucket"].values,
        "LoanPurpose": pdf_test["LoanPurpose"].values,
        "Duration": pdf_test["Duration"].values.astype(np.int16),
        "Event_Prepay": pdf_test["Event_Prepay"].values.astype(np.int8),
        "MaxObsMonths": pdf_test["MaxObsMonths"].values.astype(np.int16),
    })
    for T in HORIZONS:
        out[f"Target_T{T}"] = pdf_test[f"Target_T{T}"].values
        out[f"pred_DeepCox_T{T}"] = deep_predictions[T]
        out[f"pred_LinearCox_T{T}"] = linear_predictions[T]

    with log_step(log, "Writing D_predictions.parquet"):
        pl.from_pandas(out).write_parquet(
            RESULTS_CD / "D_predictions.parquet",
            compression="zstd", compression_level=8,
        )
    log.info(
        f"  D_predictions.parquet: {len(out):,} rows, {out.shape[1]} cols"
    )

    # 10b. D_metrics.parquet — Deep + Linear concatenated
    metrics_all = pd.concat([deep_metrics, linear_metrics], ignore_index=True)
    with log_step(log, "Writing D_metrics.parquet"):
        pl.from_pandas(metrics_all).write_parquet(
            RESULTS_CD / "D_metrics.parquet",
            compression="zstd", compression_level=8,
        )
    log.info(f"  D_metrics.parquet: {len(metrics_all):,} rows")

    # 10c. D_calibration.parquet
    calib_all = pd.concat(
        [c for c in [deep_calib, linear_calib] if len(c)],
        ignore_index=True,
    ) if len(deep_calib) or len(linear_calib) else pd.DataFrame()
    if len(calib_all):
        with log_step(log, "Writing D_calibration.parquet"):
            pl.from_pandas(calib_all).write_parquet(
                RESULTS_CD / "D_calibration.parquet",
                compression="zstd", compression_level=8,
            )
        log.info(f"  D_calibration.parquet: {len(calib_all):,} rows")

    # 10d. D_training_history.parquet
    if len(deep_history) or len(linear_history):
        history_all = pd.concat(
            [h for h in [deep_history, linear_history] if len(h)],
            ignore_index=True,
        )
        with log_step(log, "Writing D_training_history.parquet"):
            pl.from_pandas(history_all).write_parquet(
                RESULTS_CD / "D_training_history.parquet",
                compression="zstd", compression_level=8,
            )
        log.info(f"  D_training_history.parquet: {len(history_all):,} rows")

    # 10e. D_linear_vs_deep.parquet — D(v) summary table (per-horizon Δ)
    overall_d = deep_metrics[deep_metrics["stratum_type"] == "all"].set_index("horizon")
    overall_l = linear_metrics[linear_metrics["stratum_type"] == "all"].set_index("horizon")
    ablation_rows: list[dict] = []
    for T in HORIZONS:
        if T not in overall_d.index or T not in overall_l.index:
            continue
        rd = overall_d.loc[T]
        rl = overall_l.loc[T]
        ablation_rows.append({
            "horizon": T,
            "auc_deep": float(rd["auc"]),
            "auc_linear": float(rl["auc"]),
            "auc_delta": float(rd["auc"]) - float(rl["auc"]),
            "brier_deep": float(rd["brier"]),
            "brier_linear": float(rl["brier"]),
            "brier_delta": float(rd["brier"]) - float(rl["brier"]),
            "logloss_deep": float(rd["log_loss"]),
            "logloss_linear": float(rl["log_loss"]),
            "logloss_delta": float(rd["log_loss"]) - float(rl["log_loss"]),
            "n": int(rd["n"]),
            "n_events": int(rd["n_events"]),
        })
    ablation_df = pd.DataFrame(ablation_rows)
    with log_step(log, "Writing D_linear_vs_deep.parquet"):
        pl.from_pandas(ablation_df).write_parquet(
            RESULTS_CD / "D_linear_vs_deep.parquet",
            compression="zstd", compression_level=8,
        )
    log.info(f"  D_linear_vs_deep.parquet: {len(ablation_df):,} rows")
    log.info("  D(v) ablation — non-linear gain (DeepCox − LinearCox):")
    for r in ablation_rows:
        log.info(
            f"    T={r['horizon']:3d}: "
            f"ΔAUC={r['auc_delta']:+.4f}  "
            f"ΔBrier={r['brier_delta']:+.4f}  "
            f"ΔLogLoss={r['logloss_delta']:+.4f}"
        )

    # 10f. Save model state dicts and metadata
    torch = stack["torch"]
    with log_step(log, "Saving DeepCox state dict"):
        torch.save(
            deep_model.net.state_dict(), RESULTS_CD / "D_model.pt"
        )
    with log_step(log, "Saving LinearCox state dict"):
        torch.save(
            linear_model.net.state_dict(),
            RESULTS_CD / "D_linear_model.pt",
        )

    meta = {
        "feature_names": feature_names,
        "input_dim": len(feature_names),
        "scaler_means": scaler["means"].to_dict(),
        "scaler_stds": scaler["stds"].to_dict(),
        "scaler_columns": scaler["columns"],
        "arch_deep": ARCH_DEEP,
        "arch_linear": ARCH_LINEAR,
        "training_config": {
            "epochs_max": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "val_frac": VAL_FRAC,
            "random_seed": RANDOM_SEED,
            "sample_frac": SAMPLE_FRAC_CD,
        },
        "target_mode": target_mode,
        "drop_vintage_year": bool(args.drop_vintage_year),
        "horizons": HORIZONS,
        "deep_train_time_s": float(deep_time),
        "linear_train_time_s": float(linear_time),
        "n_train_inner": int(len(pdf_train)),
        "n_val": int(len(pdf_val)),
        "n_test": int(len(pdf_test)),
        "n_events_train_inner": int(e_tr.sum()),
        "n_events_val": int(e_val.sum()),
        "n_events_test": int(pdf_test["Event_Prepay"].sum()),
    }
    with log_step(log, "Saving D_model_meta.json"):
        with (RESULTS_CD / "D_model_meta.json").open("w") as f:
            json.dump(meta, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────
    total_time = time.time() - t_start
    log_section(log, "DONE")
    log.info(f"  Total runtime: {total_time:.1f}s ({total_time / 60:.1f}m)")
    log.info(f"  Output dir: {RESULTS_CD}/")
    log.info("  Files produced:")
    for f in sorted(RESULTS_CD.glob("D_*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            log.info(f"    {f.name:40s} ({size_kb:.1f} KB)")
    extra = RESULTS_CD / "canonical_split_with_val.parquet"
    if extra.exists():
        size_kb = extra.stat().st_size / 1024
        log.info(f"    {extra.name:40s} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
