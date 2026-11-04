# Report figures

All figures are drawn at their printed size with one shared style (Times, 8 pt
labels, 7 pt ticks, vector PDF), so include them at `\columnwidth` (data plots)
or unscaled (`model_architecture.pdf`, 17.7 cm, fits `\textwidth`).

- `make_figures.py` renders `fig1_rent_timeline.pdf`, `fig2_flat_type_iqr.pdf`,
  `fig4_validation_by_month.pdf` and, when the local
  `results/validation_predictions.csv` exists, `fig5_predicted_vs_recorded.pdf`.
  Run from the project root: `python report/latex/photo/make_figures.py`.
- `model_architecture.tex` is the TikZ source of the architecture figure:
  `pdflatex model_architecture.tex`.

`fig5_predicted_vs_recorded.png` is the older export from `src/scripts/plot_results.py`;
it stays in use until the per-row predictions are available to regenerate it.
No CS5322 diagrams or logs are included.
