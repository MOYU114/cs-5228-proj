# Progress Report Slide Plan

Agreed structure: 26 September 2026. **14 slides in total, including the title slide.** This is the page plan for the whole project progress report, not a standalone EDA presentation. The [course requirement](../docs/Project%20Desc.md) calls for approximately 10–15 slides, covering progress and obstacles, with the final 1–2 slides devoted to work through the end of the project.

## Page budget and current scope

| Section | Pages | Budget |
| --- | --- | ---: |
| Project identity, goals and questions | 1–2 | 2 |
| Data understanding and selected EDA evidence | 3–6 | 4 |
| Methods and experimental design | 7–10 | 4 |
| Experimental progress and challenges | 11–12 | 2 |
| Remaining work through project completion | 13–14 | 2 |

**Current completion:** pages 1–6 are drafted in English from verified evidence and await team review. Team identifiers on the title page remain unconfirmed. Pages 7–14 are deliberately incomplete content briefs, each labelled `Outline — pending completion`. Their owner labels identify who should prepare the material; they do not claim that experiments have been assigned or run. The PDF remains a working draft.

## Page-by-page plan

| Page | English title | Purpose and content limit | Owner | Current status |
| ---: | --- | --- | --- | --- |
| 1 | HDB Rental Rate Prediction | Project title, course and date. Add confirmed team name/member identifiers later. No abstract. | D / all | Draft complete; identifiers pending |
| 2 | Problem, Goals and Research Questions | Prediction record, inputs, target and original-scale RMSE. Three questions: property differences, future generalization, auxiliary value. | A | Draft complete |
| 3 | Data and Prediction Setting | One train/test overview table, grouped fields and up to three observations. Explain the training cutoff, 16-month horizon and boundary-month overlap. | A | Draft complete |
| 4 | EDA: Rent Levels and Changes over Time | One figure combining month coverage and observed training rent. Up to three observations about rent quantiles and temporal uncertainty. | A | Draft complete |
| 5 | EDA: Rent Differences across Properties | One flat-type median/IQR figure. Two observations cover town composition and area/lease-age associations. No causal or predictive-effect claim. | A | Draft complete |
| 6 | EDA: Data Quality and Auxiliary Readiness | One compact table covering spelling, repeated inputs and HDB joins. Include the six-source audit's historical-availability constraint. | A | Draft complete |
| 7 | Feature Design and Data Preparation | One proposed data-flow diagram. Define cleaning, building/calendar features, auxiliary admission and fold-specific fitting. | A / B / C | Outline only |
| 8 | Candidate Methods and Priorities | One compact baseline/candidate comparison, grounded in EDA hypotheses, priorities and resources. No page-per-paper literature review. | A / B / C | Outline only |
| 9 | Temporal Validation and Evaluation | One agreed split diagram. Define RMSE, diagnostic groups, tuning boundaries and limitations. | B / C | Outline only |
| 10 | Experiment Design and Ablations | One experimental matrix: common baseline, feature additions and temporal variants. Each row states its question and controlled factors. | B / C | Outline only |
| 11 | Progress and Preliminary Results | Actual run status and, when available, one results table with experiment IDs, validation RMSE, cost and interpretation. Currently no model results exist. | B / C / D | Outline only |
| 12 | Challenges and Open Decisions | Main obstacles, their impact, proposed resolution and owner. Cover temporal extrapolation, auxiliary history, data boundaries and resources selectively. | All | Outline only |
| 13 | Next Steps: Model Development | Work after the progress report: baselines, candidate tests, ablations and error analysis. Attach owners and deliverables to milestones. | All | Outline only |
| 14 | Next Steps: Final Evaluation and Delivery | Extend the roadmap through model confirmation, justified ensemble checks, authorized Kaggle checks, reproducibility, final writing and review. Confirm deadlines before inserting dates. | All | Outline only |

## How teammates should fill the remaining pages

1. **Agree before comparing.** B/C first confirm a shared information cutoff, validation protocol and metric. Existing research windows and priorities are proposals; the slide outline does not freeze them.
2. **Prepare evidence outside the deck.** Record experiments using the [experiment template](../docs/EXPERIMENT_RECORD_TEMPLATE.md). A supplies data interpretation, B/C supply methods and measured results, and D assembles the narrative and verifies references. Follow the [team workflow](../docs/TEAM_WORKFLOW.md).
3. **Replace the content brief within its allocated page.** Remove `Outline — pending completion` only when that page has substantive, reviewed content. Keep hypothetical method benefits labelled as hypotheses. Never insert illustrative scores or invented runtime estimates as results.
4. **Update this status table with the deck.** D tracks which pages are drafted and reviewed. If results remain unavailable, page 11 should eventually describe actual progress and the lack of results clearly; it must not imply that model testing has occurred.
5. **Preserve the budget.** Add results within pages 10–11 rather than appending pages or expanding EDA. Pages 13–14 remain the closing roadmap through project completion. Any change to the agreed 14-page allocation requires a team decision.

## Evidence placement and editing rules

- Pages 3–6 each contain one principal table or figure and no more than three conclusions. Captions identify the sample, dates and units; IQR bars describe observations, not forecast uncertainty.
- Detailed field profiles, six-table schemas, anomaly lists, all-town charts and additional distribution plots remain in [main.md](main.md) and the [EDA report](../docs/EDA_REPORT.md). They are source material, not automatic extra slides.
- Candidate literature and evaluation rationale remain in the [research index](../docs/research/README.md). Page 8 summarizes the decision questions without reproducing the survey.
- Preserve the English 16:9 Boadilla format. Use one main visual per evidence slide; shorten text before reducing the template's font sizes.
- Keep source citations and evidence paths in Markdown speaker notes, without per-slide source footers. Retain sample ranges and units beside the figures. Notes are hidden in the review PDF. Do not expose implementation instructions as finished scientific claims.
- The current change does not expand methods or experiment prose in `main.md`, train models, edit raw CSVs, edit the final report, or authorize Kaggle/Git submission.

## Review gate

Before circulating a revised PDF, verify exactly 14 pages, matching Markdown/PDF content, correct status labels, source-backed numbers and working references. Render and inspect every changed page for clipping and legibility. Confirm that the last two pages cover the remaining project, rather than only work before the progress report.

Use the [build instructions](README.md) and update [build_manifest.json](build_manifest.json) to describe the reviewed artifact. A successful PDF build does not by itself establish factual or layout correctness.
