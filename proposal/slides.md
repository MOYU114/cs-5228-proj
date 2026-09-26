---
title: "HDB Rental Rate Prediction"
subtitle: "CS5228 Progress Report · Working Draft"
author: ""
date: "26 September 2026"
theme: Boadilla
colortheme: default
aspectratio: 169
fontsize: 11pt
lang: en
header-includes:
  - \setbeamertemplate{navigation symbols}{}
  - \setbeamertemplate{caption}{\raggedright\insertcaption\par}
---

# Problem, Goals and Research Questions

**Task:** predict monthly rent for an HDB rental approval record from property attributes and information available at the prediction cutoff.

**Goal:** support rental comparisons and evaluate generalization to future months. The target is `MONTHLY_RENT`, in **SGD/month**.

1. How does rent differ across property characteristics?
2. Can models generalize across future months as rent levels change?
3. Do historically available auxiliary features improve prediction?

**Metric:** original-scale RMSE in SGD/month; lower is better.

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2}
$$

::: notes
Sources: course project description and team task guide. Model performance has not been measured.

Sources: docs/Project Desc.md and docs/PROJECT_TASK_GUIDE.md. RMSE follows the recorded competition specification; the recent research pass did not independently reopen the evaluation page. Reconfirm it before experiments/submission. Team name, member names and student IDs remain unconfirmed. Slides 1–6 contain the completed problem description and EDA. Slides 7–14 are explicitly unfinished outlines.
:::

# Data and Prediction Setting

| Split | Records | Approval months | Columns |
|:---|---:|:---|:---|
| Train | 150,000 | Jan 2021–Mar 2025 | 10 inputs + rent |
| Test | 50,000 | Mar 2025–Jul 2026 | 10 inputs, no rent |

- **Inputs:** month, address, flat type/model, area and lease year, plus two constants (`FURNISHED=yes`, `FEE=0`). Core fields contain no parsed missing values or blank strings.
- **Prediction setting:** training ends in March 2025; test extends up to **16 months** later. The same month also contains **380 test rows**, with unknown within-month order.
- **Audit scope:** all core records and six auxiliary tables. No unit identifier is supplied, and no test rent is available.

::: notes
Source: experiments/eda/core/summary.json and field_overview.csv. Actual test coverage extends one month beyond the course description.

All counts use supplied local CSVs, not the abbreviated course description. The core files have 11 and 10 columns respectively. The course description mentions nine attributes and an endpoint of June 2026. No source CSV was changed. A record cannot be assumed to uniquely identify a physical unit or contract.
:::

# EDA: Rent Levels and Changes over Time

\begin{center}
{\footnotesize Train: Jan 2021–Mar 2025 (n=150,000); test: Mar 2025–Jul 2026 (n=50,000)}
\includegraphics[width=0.98\linewidth,height=4.0cm,keepaspectratio]{figures/slides_eda_rent_timeline.png}
\end{center}

- Training median: **2,700 SGD/month**; P25–P75: **2,100–3,200**; P1–P99: **1,300–4,600**.
- **99.24%** of test rows follow the cutoff, at horizons up to **16 months**. Observed rent changes do not establish the future trend.

::: notes
Train Jan 2021–Mar 2025, n=150,000; test Mar 2025–Jul 2026, n=50,000. Sources: core/monthly_counts.csv, rent_by_month.csv and summary.json.

All records are included. The rent panel uses only training labels. The band is the observed interquartile range, not a forecast confidence interval. January 2021 mean is 2,048.90 SGD/month (n=3,129); March 2025 mean is 3,162.71 (n=3,032). These are unadjusted summaries affected by composition. The March 2025 bar stacks the two input files. The maximum horizon is a month difference, despite test containing 17 calendar months. Full rent-distribution and yearly summaries remain in main.md and docs/EDA_REPORT.md.
:::

# EDA: Rent Differences across Properties

\begin{center}
{\footnotesize All training records: Jan 2021–Mar 2025 (n=150,000)}
\includegraphics[width=0.96\linewidth,height=4.0cm,keepaspectratio]{figures/slides_eda_flat_type_iqr.png}
\end{center}

- Town rent distributions overlap and sample sizes differ. Town and flat-type comparisons mix property characteristics and time.
- Rent correlates with **area (+0.329)** and **lease-based age (−0.178)**. These marginal associations do not establish causal effects.

::: notes
Train Jan 2021–Mar 2025, n=150,000. Sources: core/rent_by_flat_type.csv, rent_by_town.csv and summary.json. Rent unit: SGD/month.

Flat types use deterministic spelling normalization. All six types are shown, including the 42 one-room records. Dots are medians and bars are P25–P75; the original full data and figures remain unchanged. Correlations are Pearson correlations. Area is in m². Lease-based age is approval year minus lease commencement year, in years; it is not exact age since construction. The flagged negative-age record remains included. Town summaries and area/age plots are available in main.md and the EDA report.
:::

# EDA: Data Quality and Auxiliary Readiness

\begingroup
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.16\linewidth}@{\hspace{0.025\linewidth}}>{\raggedright\arraybackslash}p{0.40\linewidth}@{\hspace{0.025\linewidth}}>{\raggedright\arraybackslash}p{0.39\linewidth}@{}}
\toprule
\textbf{Audit} & \textbf{Evidence} & \textbf{Proposed handling} \\
\midrule
Categories & Training categories:\newline Streets: \textbf{1,146 to 576}\newline Flat types: \textbf{11 to 6} & Normalize spelling; preserve block suffixes. \\
\addlinespace[0.8em]
Repeated inputs & Same inputs, different rents\newline \textbf{17,900 groups}\newline \textbf{39,540 training rows} & Retain records; duplication cannot be confirmed without unit or contract IDs. \\
\addlinespace[0.8em]
Normalized HDB join & \textbf{100\%} of train and test rows matched\newline \textbf{Zero} ambiguous keys & Validate many-to-one joins; check missing attributes. \\
\bottomrule
\end{tabular}
\endgroup

**Auxiliary timing:** all six sources were audited. Facility snapshots lack effective dates; future observed COE or stock values cannot become prediction inputs.

::: notes
Sources: core/summary.json, identical_X_multiple_targets.csv and auxiliary/summary.json. Complete matching does not establish historical validity.

HDB normalized keys combine town, block and street. Raw row-match rates were 79.48% for train and 79.04% for test; normalization resolves the failures, with no unmatched keys. The six auxiliary tables are HDB, MRT, schools, malls, COE and stocks. Postal NIL and floor/completion-year -1 values require derived missing indicators, not raw CSV edits. Current open status does not establish historical operation. A one-month economic lag does not make post-cutoff observations available to a fixed-origin forecast. The full auxiliary schema, anomaly and duplicate audits remain in docs/EDA_REPORT.md.
:::

# Feature Design and Data Preparation

**Outline — pending completion**

**Owners:** A / B / C

- Translate the EDA into an agreed cleaning and feature specification, including building and calendar inputs.
- Specify auxiliary admission rules, unknown-value handling and preprocessing fitted within each training fold.
- Add one data-flow diagram from source records to legal model inputs.

::: notes
Required inputs: reviewed cleaning rules from docs/EDA_REPORT.md and a shared experimental protocol. This page is a content brief, not an implemented pipeline or approved feature set. See SLIDE_PLAN.md for page budget and handoff rules.
:::

# Candidate Methods and Priorities

**Outline — pending completion**

**Owners:** A / B / C

- Summarize the common baselines and modern candidates in one compact comparison.
- Connect each candidate to an EDA-motivated hypothesis and explain the proposed attempt order.
- State resource assumptions and the evidence needed to retain or reject a candidate.

::: notes
Use docs/research/PRIORITY_PLAN.md and its linked literature as source material. Complete this page only in the methods-writing phase. Avoid one page per paper, predicted RMSE claims or treating recency as evidence of superiority.
:::

# Temporal Validation and Evaluation

**Outline — pending completion**

**Owners:** B / C

- Add the agreed temporal split diagram, with training cutoffs and forecast horizons.
- Define full-window RMSE and horizon, tail and property-group diagnostics.
- Document preprocessing boundaries, tuning rules and overlap or holdout limitations.

::: notes
Existing window counts in experiments/research/validation_windows.csv are proposals, not a frozen protocol. Keep all learned transformations within the corresponding training segment. The page requires team agreement before being marked complete.
:::

# Experiment Design and Ablations

**Outline — pending completion**

**Owners:** B / C

- Add one comparison matrix covering common baselines, feature additions and temporal variants.
- Identify the hypothesis tested by each experiment and the factors held constant.
- Specify experiment IDs, shared budgets and the results to record.

::: notes
Use docs/EXPERIMENT_RECORD_TEMPLATE.md. Keep design separate from execution status and measured results. No training or ablation has been performed in this document update.
:::

# Progress and Preliminary Results

**Outline — pending completion**

**Owners:** B / C / D

**No model results are available yet.**

- Summarize actual execution status and link completed work to reproducible records.
- Once experiments run, report IDs, validation RMSE, runtime/resource use and the matching baseline.
- Explain findings and failed attempts using measured evidence.

::: notes
Do not insert illustrative scores, estimated gains or empty numeric tables that could be mistaken for results. The completed work currently consists of data auditing and research documentation. This slide remains an outline until the experimental-progress section is written.
:::

# Challenges and Open Decisions

**Outline — pending completion**

**Owners:** A / B / C / D

- Select the most consequential obstacles: temporal extrapolation, auxiliary history, data boundaries and resources.
- State each obstacle's impact, proposed resolution and responsible owner.
- Separate unresolved data questions from observed experiment failures.

::: notes
Source material includes the March 2025 overlap, July 2026 coverage discrepancy, undated facilities and unconfirmed compute. These need a reviewed issues summary, not another full EDA audit. Do not invent experiments or failure outcomes.
:::

# Next Steps: Model Development

**Outline — pending completion**

**Owners:** A / B / C / D

- Plan work after the progress report: common baselines, prioritized candidate tests and feature ablations.
- Define the error analyses and evidence needed before selecting a model.
- Assign an owner and concrete deliverable to each development milestone.

::: notes
This is the first of the two closing next-step pages. Cover remaining model development after the progress report, not a retrospective list of completed EDA tasks. Confirm responsibilities with the team before turning the outline into commitments.
:::

# Next Steps: Final Evaluation and Delivery

**Outline — pending completion**

**Owners:** A / B / C / D

- Plan final model confirmation and any justified ensemble comparison under the agreed protocol.
- Schedule authorized Kaggle checks, reproducibility packaging and final-report preparation.
- Define final review and submission responsibilities once the project deadline is confirmed.

::: notes
The plan must extend through the end of the project. No final deadline is invented. Kaggle activity and final-report editing remain future work requiring the relevant authorization. This outline and the previous page reserve the final two slides for the course-required next steps.
:::
