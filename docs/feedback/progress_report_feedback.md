# Progress-report feedback from the tutor (Chris), Team 58

Recorded 2026-11-04. The original text is kept verbatim below; the response check after it
compares each point with the repository state on that date (main at `3573e8f add:baselines`
plus the untracked report draft in `report/latex/`).

## Original text

> Hello Team 58,
>
> Thanks for your progress report! You're certainly on the right track. Please find below a couple of comments that might be useful regarding your next steps.
>
> A little disclaimer: I never train a model for the project myself; I only roughly know what the data looks like. This means that I don't know which preprocessing steps or which models etc. work best. This is intentional as I don't want to be biased. So please treat my comments more as suggestions and ideas, and not anything you have to do.
>
> All the best,
>
> Chris
>
> Slide 02: Framing the project using explicit research questions is generally a good idea.
>
> The EDA and preprocessing steps look quite alright. For the report, you want to provide sufficient but concise details about what happened to some features, at least the ones from the core dataset. For example, I can't tell for sure what happened to "street". Nothing wrong with ignoring it, but it should be mentioned explicitly. A good rule of thumb is to assume that the reader does not know the dataset.
>
> Slide 07: Organizing the features with respect to their "nature" is quite useful – did you try training 1 or 2 of your best models using different subset of features to see which matter most/least?
>
> Not critical comment: You "only" use or planning to use rather sophisticated models. You may also want to consider some simpler ones and compare all models using a simplified setup. Comparing results across different models can give you some interesting insights into the data. For example, it's not obvious if we actually need these sophisticated boost methods at all.
>
> Slide 12: Your roadmap looks meaningful. In general, you may also want to consider any insights you might get beyond the RSME value – as indicated by Point 3. There is no need to bend over backwards and tune the model to squeeze out a small improvement. At some point, the improvements will be marginal and no longer worth the effort. Understanding the result and the model itself is equally important than the raw RSME values.
>
> Lastly, as a general comment, always keep an eye on the deadline. You want to start early with your final report to avoid rushing it. After all, the final report is the main assessment component. There is no need to spend too much time just to squeeze out a slightly better ranking in Kaggle.

## Response check (2026-11-04)

| # | Comment | Addressed by new work? | Report draft status | Action for the report |
|---|---|---|---|---|
| 1 | Frame with explicit research questions | n/a | Chapter 1 states "three bounded contributions", no explicit RQs | Consider stating 2-3 RQs (e.g. model complexity vs. baselines; which feature groups matter; where errors concentrate) and answer them in Sec. VI/VII |
| 2 | Say explicitly what happened to each core feature, e.g. `STREET` | Code: `STREET` is normalized, used in the `(TOWN, BLOCK, STREET)` join key, kept as `STREET_ID`, and `TOWN\|BLOCK\|STREET` forms `BUILDING_ID` (`src/preprocess/stage3_encode.py`, `src/bert_rent/data.py`); `FURNISHED`/`FEE` dropped as constants | Normalization and join are described, but Table II never names the 8 categorical IDs, so the reader cannot tell that street enters the model as `STREET_ID` | List the 8 IDs (`TOWN, STREET, BUILDING, SUBZONE, PLANNING_AREA, REGION, FLAT_MODEL, FLAT_TYPE_ORD`) and give a one-line fate for every core column, including dropped `RENT_APPROVAL_DATE` string, `BLOCK`, `FURNISHED`, `FEE` |
| 3 | Train 1-2 best models on feature subsets | Yes, for Ridge and CatBoost (strict protocol): removing street/building IDs +8.52 / +2.95 RMSE, facilities -0.08 / +0.84, time group +85.64 / +232.26 (`results/baselines/round_a/ablation.csv`). Not run for BERT | Sec. V/VI say no tree or comparison run exists | Add an ablation table and discussion; state BERT ablation was not run |
| 4 | Compare simpler models in a simplified setup | Yes: global mean 842.54, recent 6-month town×type mean 497.97, Ridge 606.82, shallow CatBoost 468.11 vs. BERT 468.34 RMSE (historical protocol, 35,058 rows; `scores.csv`) | Sec. V/VI explicitly claim no baseline exists, now outdated | Add a model-comparison table; note the strict protocol (35,158 rows) has no BERT result, so cross-protocol numbers must not be mixed |
| 5 | Insights beyond RMSE | Partly: per-month bias (`by_month.csv`, `monthly_bias.png`), seen vs. unseen buildings (`by_building_seen.csv`: CatBoost strict 475.50 vs. 548.26, n=151), Ridge over-extrapolation 218 → 532 SGD | Sec. VI already has month-wise error and the 168.04 decomposition | Fold baseline per-month / unseen-building findings into Error Analysis; BERT per-row predictions are still absent from the repo, so BERT cannot be split by building |
| 6 | Start the final report early; don't chase Kaggle rank | n/a | Draft exists | Prioritize report completeness over further tuning |
