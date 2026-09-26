# Stage-one EDA

Run from the project root using Python 3.12:

```bash
python3 -m venv .venv-eda
.venv-eda/bin/python -m pip install -r notebooks/requirements-eda.txt
.venv-eda/bin/python notebooks/run_eda.py
```

The tested Python version is 3.12.14. The entry point also accepts `--root /absolute/project/path` and locates this project automatically if omitted. No external data or network is used during analysis. `--skip-auxiliary` is a debugging option that regenerates core tables/figures only and does not update the report.

For this session the bundled Python already provides pandas/NumPy. Matplotlib and its missing dependencies were installed in `/private/tmp/cs5228-eda-deps`, without changing the bundled environment:

```bash
PYTHONPATH=/private/tmp/cs5228-eda-deps /Users/moyu114/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 notebooks/run_eda.py
```

Use the venv command for a fresh machine; the temporary installation is not portable. Matplotlib/font caches go to `/tmp/cs5228-eda-*` if cache variables are unset.

Modules:

- `eda_core.py`: field profiles, deterministic text normalization, rent summaries, descriptive shift and repeated-record/building audits.
- `eda_auxiliary.py`: all six auxiliary tables, normalized HDB key audit, timing and sentinel-value checks; may also run standalone with `python notebooks/eda_auxiliary.py --root .`.
- `eda_figures.py`: seven full-sample PNG charts with date scope and units.
- `eda_report.py`: builds `docs/EDA_REPORT.md` from saved results and documented interpretations.
- `run_eda.py`: runs the whole pipeline and verifies protected-file hashes.

Outputs are `experiments/eda/core/`, `experiments/eda/auxiliary/`, `experiments/eda/run_manifest.json`, `proposal/figures/eda_*.png`, and `docs/EDA_REPORT.md`. Re-running overwrites these derived files. Table column names `n`, `mean`, `median`, `q25`, `q75` denote record count and rent statistics in SGD/month. Table source-row identifiers are zero-based excluding the CSV header.

The script never modifies source CSVs, fits a model, creates predictions or submissions, or edits proposal/final-report documents. All rent statistics use train only. Test X and future auxiliary dates are inspected solely to audit distribution/availability boundaries, never to construct prediction features. Normalization operates on copies and uses no learned vocabulary. No tail filtering or de-duplication is performed.

The report's suggested preprocessing, outlier and temporal-validation decisions require team review; descriptive full-train statistics are not fitted transforms for future validation folds. Missing auxiliary effective dates remain unresolved.
