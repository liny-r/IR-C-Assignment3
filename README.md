# MTH9877 — Assignment 3: Mortgage Prepayment & Credit Risk Modeling

**Course:** MTH9877 Interest Rate & Credit Models, Baruch MFE  
**Data:** Freddie Mac Single-Family Loan-Level Dataset (1999–2025)  
**Dataset size:** 34,013,469 loans

---

## Repository Structure

```
Assignment3.ipynb          # Main notebook (all parts A–E)
processed/
  survival_loans.parquet   # Cached survival dataset (built once, ~20–40 min)
  macro_monthly.parquet    # FRED macro covariates (MORTGAGE30US, UNRATE, CPI, HPI)
  panel_monthly.parquet    # Discrete-time ML panel (one row per loan-month)
  A1_km_hazard.png         # KM survival curve + implied hazard rate
  A2_km_*.png              # Stratified KM by LTV / FICO / vintage
  B1_cox_hr.png            # Cox PH hazard ratios
  B2_schoenfeld.png        # Schoenfeld residuals (PH assumption check)
  C_feature_importance.png # XGBoost / LightGBM / RF feature importance
  D_training_loss.png      # Deep Cox training loss curve
  D_gradient_sensitivity.png # Gradient sensitivity by feature
  E1_competing_risks_cif.png # Aalen-Johansen competing risks CIF
  E2_scenario_analysis.png   # Interest rate shock scenario analysis
```

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total loans | 34,013,469 |
| Prepaid | 21,970,748 (64.6%) |
| Defaulted | 532,563 (1.6%) |
| Censored | 11,510,158 (33.8%) |
| Vintage range | 1999–2025 |
| Median duration | 32 months |

---

## Computational Note — Subsampling

Parts B–E use a **stratified 100K-loan subsample** (0.29% of full dataset) drawn proportionally by vintage year. The subsample is validated to be representative:

| Metric | Full (34M) | Subsample (100K) | Rel. diff |
|--------|-----------|-----------------|-----------|
| Prepay rate | 0.64594 | 0.64565 | −0.045% |
| Default rate | 0.01566 | 0.01571 | +0.350% |
| Median FICO | 751 | 752 | +0.133% |
| Median LTV | 79 | 79 | 0.000% |
| Median rate (%) | 4.875 | 4.875 | 0.000% |
| Median DTI | 35 | 35 | 0.000% |

All relative differences < 1%.

---

## Part A — Exploratory Survival Analysis

**A.1 Kaplan-Meier (full 34M loans)**

| Horizon | Survival probability |
|---------|---------------------|
| 10 years | 15.4% |
| 20 years | 3.6% |

Median prepayment time: **50 months**

**A.2 Stratified KM** — LTV buckets show minimal differentiation; FICO and vintage year produce strong separation (high-FICO and post-2010 loans prepay faster, consistent with refinancing incentives).

---

## Part B — Classical Cox Proportional Hazards

### B.1 Static Cox Model (100K subsample)

Fit on 99,986 loans with 64,556 observed prepayments.

| Covariate | HR (exp coef) | 95% CI |
|-----------|:---:|--------|
| CreditScore | 1.056 | [1.048, 1.064] |
| OriginalInterestRate | **1.332** | [1.318, 1.346] |
| OriginalUPB | 1.202 | [1.192, 1.212] |
| OriginalLoantoValueLTV | 0.973 | [0.965, 0.980] |
| VintageYear | 0.946 | [0.935, 0.956] |
| OriginalDebttoIncomeRatio | 0.993 | [0.986, 1.000] |

### B.2 Proportional Hazards Test (Schoenfeld residuals)

Key violations (p < 0.001): **OriginalInterestRate** (test stat 190), **OriginalLTV** (62), **CreditScore** (43), **LoanPurpose_P** (33), **VintageYear** (23) — confirming the PH assumption is violated and motivating time-varying extensions.

### B.3 Time-Varying Cox (300K panel rows)

Key time-varying coefficients:

| Covariate | HR | Interpretation |
|-----------|:--:|----------------|
| rate_incentive | 1.033 | Higher refinancing incentive → faster prepayment |
| orig_rate | 1.023 | Higher original rate → faster prepayment |
| mortgage_rate | 0.984 | Higher current rate → slower prepayment |
| unemployment | 1.013 | Higher unemployment → slightly faster prepayment |
| hpi_yoy | 1.003 | Rising house prices → marginally faster prepayment |

---

## Part C — Machine Learning Models

Discrete-time binary classification (prepaid this month) with train/val/test split by vintage year (train ≤ 2016, val 2017–2019, test ≥ 2020).

| Model | AUC | Brier Score | C-index |
|-------|:---:|:-----------:|:-------:|
| XGBoost | **0.7119** | 0.1076 | 0.6758 |
| LightGBM | 0.6684 | 0.0096 | 0.6415 |
| Random Forest | 0.7114 | 0.0875 | **0.6856** |
| Elastic Net | 0.6537 | **0.0070** | 0.8166 |

Top features (XGBoost): `rate_incentive`, `loan_age`, `orig_rate`, `mortgage_rate`, `FICO`.

---

## Part D — Deep Cox Model

Neural network replacing the Cox linear predictor: `λ(t|x) = λ₀(t) · exp(f_θ(x))`, with architecture [128 → 64 → 32 → 1].

Training loss converged from ~4.363 → **4.345** over 40 epochs (MPS device).

| Model | C-index |
|-------|:-------:|
| Cox PH (static) | 0.6106 |
| XGBoost | 0.6758 |
| LightGBM | 0.6415 |
| Random Forest | 0.6856 |
| Elastic Net | 0.8166 |
| **Deep Cox** | **0.6499** |

**Gradient sensitivity** (avg |∂f_θ/∂xⱼ|):

| Feature | Sensitivity |
|---------|:-----------:|
| VintageYear | 1.964 |
| OriginalInterestRate | 0.583 |
| mortgage_rate | 0.242 |
| hpi_yoy | 0.240 |
| OriginalLTV | 0.165 |
| DTI | 0.158 |
| UPB | 0.152 |
| CreditScore | 0.145 |

---

## Part E — Extensions

### E.1 Competing Risks (Aalen-Johansen, 100K subsample)

| Event | 10-year CIF |
|-------|:-----------:|
| Prepayment | **83.3%** |
| Default | **2.23%** |

### E.2 Interest Rate Shock Scenario Analysis

| Shock | Deep Cox log-HR | XGBoost avg prob |
|-------|:--------------:|:----------------:|
| −200 bp | 0.113 | 0.492 |
| −100 bp | 0.189 | 0.493 |
| 0 bp (baseline) | 0.215 | 0.457 |
| +100 bp | 0.211 | 0.315 |
| +200 bp | 0.217 | 0.253 |

Rate cuts increase prepayment probability (refinancing incentive). XGBoost shows a clear monotone response; Deep Cox log-HR is relatively stable across shocks due to the static feature design.

---

## Setup

```bash
pip install polars lifelines lightgbm xgboost scikit-learn torch fredapi
```

Set your FRED API key in the notebook (`FRED_API_KEY`) before running Step 2.  
All intermediate datasets are cached in `processed/` — re-running is fast after the first build.
