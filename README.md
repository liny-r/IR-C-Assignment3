# MTH9877 — Assignment 3: Mortgage Prepayment & Credit Risk Modeling

**Course:** MTH9877 Interest Rate & Credit Models, Baruch MFE, Spring 2026
**Authors:** Rose Lin, Paolo Smalhout, Daniel Tuzes
**Data:** Freddie Mac Single-Family Loan-Level Dataset, 1999–2025 (31.7M loans)

---

## Submission artifacts

| File | Description |
|------|-------------|
| `report.pdf` | Combined written report (Parts A–E), 18 pages |
| `slides.pdf` | Combined presentation slides (Parts A–E + Q&A appendix), 54 pages |
| `report.tex`, `slides.tex` | LaTeX sources |
| `slides_speaker_notes.txt`, `slides_speaker_notes.docx` | Speaker scripts for the Part E section of the talk |

---

## Reproducible code

| File | Part | Run order | Type |
|------|------|-----------|------|
| `build_survival_table.py` | — | 1 | Builds the 31.7M-row survival table from Freddie Mac monthly performance files. Run once. |
| `part_a_survival_analysis.py` | A | 2 | Kaplan–Meier survival on the full population, stratified by FICO, LTV, vintage, purpose. |
| `part_b_cox_model.py` | B | 3 | Cox proportional hazards (5% sample), Schoenfeld PH diagnostic, vintage-macro extension. |
| `part_c_ml_models.py` | C | 4 | LogReg / Random Forest / LightGBM / Cox baseline at horizons T ∈ {12, 24, 36, 60}. |
| `part_d_deep_cox.py` | D | 5 | DeepSurv (3×64 ReLU) + linear-Cox-via-SGD ablation. |
| `PartE_i_CompetingRisks.ipynb` | E(i) | 6 | KM vs Aalen–Johansen, cause-specific Cox (prepayment vs default). |
| `PartE_ii_TimeVaryingCovariates.ipynb` | E(ii) | 7 | Andersen–Gill time-varying Cox on the loan-month panel. |
| `PartE_iii_NeuralModels.ipynb` | E(iii) | 8 | LongitudinalDeepHit (Transformer encoder, competing-risk CIFs). |
| `PartE_iv_ScenarioAnalysis.ipynb` | E(iv) | 9 | MBS scenario analysis: WAL, price, negative convexity under ±300bp shocks. |
| `utilities.py` | shared | — | Canonical-split helpers, encoder persistence, scaler stats. |

Each Python script writes parquet outputs only (no plotting). Plotting is handled in per-part notebooks under `results_a/`, `results_b/`, `results_cd/`.

---

## How to run

### 1. Environment

```bash
pip install polars pandas numpy scikit-learn lifelines lightgbm \
            torch pycox torchtuples fredapi matplotlib seaborn \
            statsmodels
```

Tested on Python 3.11 / macOS (Apple Silicon) and Linux x86_64. CPU-only is fine — total runtime is ~50 minutes.

### 2. Data prerequisites

- **Freddie Mac data** must be placed under `Origination_Historical_Data/` and `Monthly_Performance_historical_data_time/` (not in repo — request from Freddie Mac).
- **FRED API key** required for macro covariates. Export it before running:
  ```bash
  export FRED_API_KEY="your_key_here"
  ```

### 3. Build the survival table (one-time, ~20–40 min)

```bash
python build_survival_table.py
```

Writes `survival_table.parquet` (~530 MB) plus `processed/macro_monthly.parquet` and `processed/panel_monthly.parquet`.

### 4. Run Parts A–D (scripts)

```bash
python part_a_survival_analysis.py    # ~5 min — KM curves, hazards, stratifications
python part_b_cox_model.py            # ~3 min — Cox PH + Schoenfeld test
python part_c_ml_models.py            # ~15 min — LogReg / RF / LightGBM / Cox at 4 horizons
python part_d_deep_cox.py             # ~17 min — DeepSurv + linear ablation
```

Each script writes parquet predictions and metrics into `results_a/`, `results_b/`, `results_cd/`. Open the matching plotting notebook in each `results_*/` folder to regenerate figures into `plots/`.

### 5. Run Part E (notebooks)

```bash
jupyter notebook
# then open each in order:
PartE_i_CompetingRisks.ipynb           # ~5 min
PartE_ii_TimeVaryingCovariates.ipynb   # ~10 min
PartE_iii_NeuralModels.ipynb           # ~25 min CPU, ~6 min MPS/GPU
PartE_iv_ScenarioAnalysis.ipynb        # ~3 min
```

All Part E figures land in `processed/` and are picked up automatically by `report_final.tex`.

### 6. Rebuild the report and slides

```bash
pdflatex report.tex
pdflatex slides.tex
```

`slides.tex` reads figures from `figures/` (A–D) and `processed/` (E). `report.tex` reads from both via `\graphicspath`.

---

## Folder structure

```
Assignment3/
├── README.md                          ← this file
├── report.tex, report.pdf             ← combined report (18 pp)
├── slides.tex, slides.pdf             ← combined slides (54 pp)
├── slides_speaker_notes.txt           ← speaker script (Part E)
├── slides_speaker_notes.docx          ← dot-point script
│
├── build_survival_table.py            ← code
├── part_{a,b,c,d}_*.py                ←
├── PartE_{i,ii,iii,iv}_*.ipynb        ←
├── utilities.py                       ←
│
├── figures/                           ← A–D figures
├── processed/                         ← Part E figures + parquet panels
├── results_a/, results_b/, results_cd/ ← intermediate outputs + plotting notebooks
├── samples/                           ← sample data (small, in repo)
├── survival_table.parquet             ← built by step 3 (not in repo)
│
└── References
    ├── IR&C_Assignment3_2026.pdf      ← assignment spec
    ├── Chen2024_*.pdf                 ← Chen 2024 survival models monograph
    └── Sadhwani2021_*.pdf             ← Sadhwani et al. 2021 (anchor paper)
```

---

## Key methodological notes

- **Train/test split:** by vintage — train ≤ 2014, test 2015–2019. Persisted as `canonical_split.parquet` so Parts C and D evaluate on identical rows.
- **Encoders & scalers** are fit on the training set only and persisted (`encoders.pkl`) for downstream parts to reuse — prevents test-set leakage.
- **Subsampling:** Part A uses the full 31.7M loans; Part B uses 5% (1.58M); Parts C and D share a 10% sample (1.85M train / 597K test); Part E(ii) and E(iii) draw from a 100K-loan panel with full monthly performance history.
- **Competing risks:** Parts A–D treat default as right-censoring (cause-specific framing). Part E(i) explicitly models default as a competing risk via Aalen–Johansen and cause-specific Cox.
- **Time-varying covariates:** Parts B and D use only origination features. Part E(ii) introduces four time-varying features (rate_incentive, ELTV, unemployment, hpi_yoy) via the Andersen–Gill extension.

---

## Headline results

| Model | Scope | T=12 AUC | T=60 AUC |
|-------|-------|:--:|:--:|
| Cox PH (Part B) | static | 0.66 | 0.67 |
| LightGBM (Part C) | static | 0.68 | 0.68 |
| LogReg (Part C) | static | 0.69 | 0.70 |
| **Deep Cox (Part D)** | static | 0.68 | **0.73** |
| TV Cox (Part E(ii)) | time-varying | **0.68** | 0.65 |
| LongitudinalDeepHit (Part E(iii)) | sequence | 0.69 (default) | 0.73 (default) |

**Part E(i):** 10-year Aalen–Johansen CIF: 83.3% prepaid, 2.2% defaulted, 14.5% active. KM overstates prepayment by up to +2.2 pp over 20 years.

**Part E(iv):** ±300bp rate shock shifts MBS price by +$9.25 / −$9.30 (negative convexity).
