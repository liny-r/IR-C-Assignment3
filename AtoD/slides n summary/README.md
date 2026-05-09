# IR&C Assignment 3 Deliverables

Two LaTeX source files plus a `plots/` folder.

## Files

- `report.tex` — 10-page report (PDF style). Compile with `pdflatex` (or `xelatex`).
- `slides.tex` — 22 Beamer slides (1 title + 20 content + 1 thank-you). 16:9
  aspect ratio. Compile with `xelatex` (recommended for Metropolis theme).
- `plots/` — 14 PNG files referenced by both documents. Currently
  populated with placeholder images; replace with real exports from your
  notebooks before submitting.

## Plot export workflow

The 14 figures referenced in both report and slides are:

| Filename                       | Source notebook       | What to export |
|--------------------------------|----------------------|----------------|
| `A_km_overall.png`             | part_a_plot.ipynb     | Overall KM survival curve with CI ribbon (cell ~5) |
| `A_hazard_overall.png`         | part_a_plot.ipynb     | Discrete monthly hazard + Nelson–Aalen cumulative hazard panel |
| `A_km_by_vintage.png`          | part_a_plot.ipynb     | Stratified KM by vintage cohort |
| `A_km_by_fico.png`             | part_a_plot.ipynb     | Stratified KM by FICO bucket |
| `B_hazard_ratios_base.png`     | part_b_plot.ipynb     | Base-model HR forest plot |
| `B_hazard_ratios_macro.png`    | part_b_plot.ipynb     | Macro-extension HR forest plot |
| `B_ph_test.png`                | part_b_plot.ipynb     | Schoenfeld p-value bars |
| `C_auc_by_horizon.png`         | part_c_plot.ipynb     | AUC by horizon, all 4 Part C models |
| `C_calibration.png`            | part_c_plot.ipynb     | Decile calibration, all 4 Part C models |
| `C_top_decile_lift.png`        | part_c_plot.ipynb     | Top-decile lift, all 4 Part C models |
| `D_ablation_bars.png`          | part_d_plot.ipynb     | Linear vs Deep AUC/Brier/log-loss bars (D(v)) |
| `D_auc_by_horizon.png`         | part_d_plot.ipynb     | All-6-models AUC head-to-head (D(iv)) |
| `D_calibration.png`            | part_d_plot.ipynb     | Decile calibration for DeepCox + LinearCox |
| `D_top_decile_lift.png`        | part_d_plot.ipynb     | Top-decile lift, all 6 models |

Save each as 8 × 5 inches at 110-150 dpi for crisp embedding without
ballooning the PDF.

## Compile

```
pdflatex report.tex          # report
xelatex slides.tex           # slides (Metropolis theme)
```
