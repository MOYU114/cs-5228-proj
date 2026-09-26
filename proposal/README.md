# CS5228 Progress Report

The progress-report deck covers the **whole project in 14 slides, including the title slide**. Pages **1–6** contain the drafted English problem description and EDA. Pages **7–14** contain only labelled outlines and owners for the team to complete later. This is a working draft, with no model performance results.

| File | Purpose |
| --- | --- |
| [SLIDE_PLAN.md](SLIDE_PLAN.md) | Agreed page budget, each slide's purpose, owners, completion status and handoff rules |
| [main.md](main.md) | Detailed English narrative with statistics, figures and audit links; unchanged in this slide restructuring |
| [slides.md](slides.md) | Editable Pandoc Beamer source with speaker notes and evidence paths |
| [slides.pdf](slides.pdf) | 14-page working draft; pages 13–14 reserve the roadmap through project completion |
| [render_eda_figures.py](render_eda_figures.py) | Reproducible, presentation-sized figures from existing EDA aggregate CSVs |
| [build_manifest.json](build_manifest.json) | Reviewed build commands, dependencies, source hashes and preservation checks |

## Page allocation

| Content | Pages |
| --- | --- |
| Project goals and questions, including title | 1–2 |
| Data understanding and selected EDA | 3–6 |
| Methods and experimental design | 7–10 |
| Experimental progress and challenges | 11–12 |
| Remaining work through the end of the project | 13–14 |

Start with [SLIDE_PLAN.md](SLIDE_PLAN.md) before adding material. Replace each unfinished brief within its allocated page and update the status table after review. The `Outline — pending completion` label must remain until the page is substantively completed. Team name, member names and student IDs remain unconfirmed.

## Rebuild the PDF

From the project root, using the supplied figures:

```bash
cd proposal
pandoc slides.md --slide-level=1 -t beamer --pdf-engine=xelatex -o slides.pdf
```

Tested with Pandoc 3.8.2.1 and XeLaTeX from TeX Live 2025. The TeX installation needs Beamer and the standard packages used by Pandoc's template. The deck retains the existing English 16:9 Boadilla layout at 11pt. Source citations remain in editable Markdown speaker notes and are hidden in the PDF. Figure sample ranges and units stay visible; per-slide source footers are omitted.

To regenerate the two slide figures, use the environment described in [EDA reproduction instructions](../notebooks/README_EDA.md), then run from the project root:

```bash
.venv-eda/bin/python proposal/render_eda_figures.py
```

The figure script requires Matplotlib (tested 3.10.6 with Python 3.12.14). It reads only `monthly_counts.csv`, `rent_by_month.csv` and `rent_by_flat_type.csv` from `experiments/eda/core/`. It writes `figures/slides_eda_rent_timeline.png` and `figures/slides_eda_flat_type_iqr.png`; the original `eda_*.png` figures remain unchanged. Sample ranges and units appear on the slides, with additional caveats in speaker notes. No EDA recomputation, model libraries or network access are needed.

After changes, rebuild and inspect every changed page. Confirm 14 pages, visible outline labels on pages 7–14, readable plots and no clipping. The build manifest describes the reviewed build only; regenerate its hashes and verification record when publishing a later revision.

## Supporting material

- [EDA report](../docs/EDA_REPORT.md), [audit outputs](../experiments/eda/) and [main.md](main.md) retain detailed statistics beyond the slide budget.
- [Research index](../docs/research/README.md) contains candidate evidence for the later methods pages.
- [Course description](../docs/Project%20Desc.md), [team workflow](../docs/TEAM_WORKFLOW.md) and [progress plan](../docs/PROGRESS_PLAN.md) establish deliverables and responsibilities.

Earlier EDA and research manifests are historical stage snapshots. Their proposal hashes predate later authorized document edits. The current build manifest verifies preservation of raw CSVs, EDA code and outputs, `main.md` and final-report files during this restructuring. The `experiments/` subfolder remains reserved for later experiment summaries.
