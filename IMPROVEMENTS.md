# Notebook Improvement Tracker
_Based on Chen 2024 — An Introduction to Deep Survival Analysis Models_

## Status key: ✅ Done | 🔄 In progress | ⬜ Backlog

---

## Priority 1 — Methodological Correctness

### ✅ #1 · Fix loan-level C-index for ML models
**Problem:** `dur_te = test["loan_age"]` computes C-index at the panel-row level, not the
loan level. Chen Def 2.1 requires total observed survival time Y_i per loan.  
**Fix:** Aggregate test panel → one row per loan (max loan_age = duration, max
prepaid_month = event). Use mean monthly predicted probability as loan-level risk score.  
**Cell:** After C backtest cell.

---

## Priority 2 — Model Improvements

### ✅ #3 · Cox-Time: remove PH assumption (Chen §3.4)
**Problem:** Deep Cox still imposes PH on static features, despite B.2 showing strong
violations (OrigInterestRate p < 10⁻⁴³).  
**Fix:** Concatenate normalised loan_age `t/T_max` to feature vector. Model learns
`f_θ(x, t)` — hazard ratio varies with loan age, not just covariates.  
**Reference:** Kvamme et al. 2019, Chen §3.4  
**Cell:** After Deep Cox eval cell.

### ✅ #4 · Breslow baseline hazard → individual survival curves S(t|x) (Chen §3.3 eq 43–46)
**Problem:** Deep Cox only outputs a log-hazard score. Chen §3.3 shows the full
prediction target is the survival curve `S(t|x) = exp(-exp(f_θ(x)) · H_0(t))`.  
**Fix:** Estimate Breslow baseline cumulative hazard H_0(t) on training data.
Plot S(t|x) for low/median/high risk loan profiles.  
**Reference:** Chen eq (43)–(46), Figure 4  
**Cell:** After Deep Cox eval, before Cox-Time.

---

## Priority 3 — Additional Metrics

### ⬜ #2 · Integrated Brier Score (IBS)
**Problem:** Current Brier score is binary classification BS, not survival BS with IPCW
weights (Chen Def 2.5–2.6).  
**Fix:** After predicting S(t|x) from Deep Cox/Cox-Time, integrate BS(t) across
horizons [12, 60, 120] months using KM censoring weights. Use `scikit-survival`
or implement from Chen eq (Definition 2.5).

### ⬜ #5 · D-Calibration (chi-squared test, Haider et al. 2020)
**Problem:** No calibration check. Chen §2.5.3 defines D-Cal: checks whether
predicted S(T_i|X_i) at observed event time is uniform over [0,1].  
**Fix:** Compute D-Cal chi-squared p-value for Deep Cox and Cox-Time.  
**Reference:** Chen §2.5.3, Haider et al. 2020

### ⬜ #6 · Time-dependent AUC at key horizons (t = 12, 24, 36, 60 months)
**Problem:** Single AUC collapses time dimension.  
**Fix:** Use `lifelines` `concordance_index` at specific horizons, or
`scikit-survival` `cumulative_dynamic_auc`. Report AUC(t) as a time-series.

---

## Priority 4 — Advanced Models

### ⬜ #7 · DeepHit for competing risks (Chen §6.1, Lee et al. 2018)
**Problem:** Part E.1 uses Aalen-Johansen (classical). DeepHit jointly models
prepayment and default sub-distribution hazards.  
**Fix:** Implement DeepHit with two output heads (cause-specific CIF per month).
Compare CIF curves to Aalen-Johansen.  
**Reference:** Chen §6.1, Lee et al. 2018 (pycox implementation available)
