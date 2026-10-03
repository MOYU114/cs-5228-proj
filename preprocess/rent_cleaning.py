"""Training-rent cleaning, the last step of stage 1, so the cleaned CSV can be checked by hand.

1. Outliers. Each rent is compared with the median of its comparables: other rentals of the same unit (TOWN, BLOCK,
   STREET, FLAT_TYPE, FLAT_MODEL, FLOOR_AREA_SQM, LEASE_COMMENCE_DATE) approved within 2 months, excluding itself.
   With at least 3 comparables, a rent is an outlier when it is more than twice or less than half that median and
   differs from it by at least 800 SGD. Within a unit the most extreme outlier is removed first and the medians are
   recomputed, so outliers cannot drag a normal rent's median (ties go to the earlier ROW_ID). Rents with fewer than
   3 comparables are kept unjudged.
2. Duplicates. Remaining rentals with identical features (same unit, same month) all get the mean of their rents.
   Under squared error this trains exactly like the separate rents (the spread around the mean is a constant), but
   without conflicting labels.

Only train is cleaned (test has no rents), including the months later used for validation. Removed rows leave gaps
in ROW_ID; every kept row keeps its original rent in MONTHLY_RENT_ORIGINAL.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .common import TARGET

UNIT = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]
ORIGINAL = "MONTHLY_RENT_ORIGINAL"
WINDOW_MONTHS = 2
MIN_COMPARABLES = 3
MAX_RATIO = 2.0
MIN_DIFFERENCE = 800


def comparables(rent: np.ndarray, month: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row of one unit: which active rows are its comparables, how many, and their median rent."""
    near = (np.abs(month[:, None] - month[None, :]) <= WINDOW_MONTHS) & active[None, :]
    np.fill_diagonal(near, False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows without comparables get a NaN median
        median = np.nanmedian(np.where(near, rent[None, :], np.nan), axis=1)
    return near, near.sum(axis=1), median


def month_number(train: pd.DataFrame) -> np.ndarray:
    dates = train["RENT_APPROVAL_DATE"]
    return (dates.str.slice(0, 4).astype("int64") * 12 + dates.str.slice(5, 7).astype("int64")).to_numpy()


def initial_comparable_counts(train: pd.DataFrame) -> np.ndarray:
    """Comparables per row before any removal, to report how many rows the rule could judge."""
    month, counts = month_number(train), np.zeros(len(train), dtype=np.int64)
    for positions in train.groupby(UNIT, sort=False, dropna=False).indices.values():
        m = month[positions]
        counts[positions] = (np.abs(m[:, None] - m[None, :]) <= WINDOW_MONTHS).sum(axis=1) - 1
    return counts


def find_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Rows to remove, with the evidence at the moment each was removed."""
    month = month_number(train)
    rent = train[TARGET].to_numpy(dtype=np.float64)
    row_id = train["ROW_ID"].to_numpy()
    records = []
    for positions in train.groupby(UNIT, sort=False, dropna=False).indices.values():
        y, m = rent[positions], month[positions]
        active = np.ones(len(positions), dtype=bool)
        removal_round = 0
        while True:
            near, count, median = comparables(y, m, active)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = y / median
            flagged = (active & (count >= MIN_COMPARABLES) & ((ratio > MAX_RATIO) | (ratio < 1 / MAX_RATIO))
                       & (np.abs(y - median) >= MIN_DIFFERENCE))
            if not flagged.any():
                break
            candidates = np.flatnonzero(flagged)
            j = candidates[np.argmax(np.abs(np.log(ratio[candidates])))]
            active[j] = False
            removal_round += 1
            records.append({"position": positions[j], "REMOVAL_ROUND": removal_round, "COMPARABLES": int(count[j]),
                            "COMPARABLE_MEDIAN": float(median[j]), "RATIO": round(float(ratio[j]), 4),
                            "DIFFERENCE": float(y[j] - median[j]),
                            "COMPARABLE_ROW_IDS": ";".join(map(str, row_id[positions[near[j]]]))})
    return pd.DataFrame(records, columns=["position", "REMOVAL_ROUND", "COMPARABLES", "COMPARABLE_MEDIAN", "RATIO",
                                          "DIFFERENCE", "COMPARABLE_ROW_IDS"])


def clean_train_rents(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return (cleaned train, removed outliers, rows whose rent was replaced, summary)."""
    evidence = find_outliers(train)
    removed_mask = np.zeros(len(train), dtype=bool)
    removed_mask[evidence["position"].to_numpy(dtype=np.int64)] = True
    context = ["ROW_ID", "RENT_APPROVAL_DATE"] + UNIT
    removed = pd.concat([train.iloc[evidence["position"]][context + [TARGET]].reset_index(drop=True),
                         evidence.drop(columns="position").reset_index(drop=True)], axis=1)
    removed = removed.sort_values("ROW_ID").reset_index(drop=True)

    kept = train.loc[~removed_mask].copy()
    group = kept.groupby(["RENT_APPROVAL_DATE"] + UNIT, sort=False)[TARGET]
    mean, size, distinct = group.transform("mean"), group.transform("size"), group.transform("nunique")
    duplicate_groups, varied_groups = int((group.size() > 1).sum()), int((group.nunique() > 1).sum())
    kept.insert(kept.columns.get_loc(TARGET) + 1, ORIGINAL, kept[TARGET].astype("int64"))
    kept[TARGET] = mean.round(2).astype("float64")

    varied = (distinct > 1).to_numpy()
    replaced = kept.loc[varied, context + [ORIGINAL, TARGET]].assign(GROUP_SIZE=size[varied].astype("int64"))
    replaced = replaced.sort_values(["RENT_APPROVAL_DATE"] + UNIT + ["ROW_ID"]).reset_index(drop=True)
    change = (kept[TARGET] - kept[ORIGINAL]).abs()
    summary = {
        "parameters": {"unit": UNIT, "window_months": WINDOW_MONTHS, "min_comparables": MIN_COMPARABLES,
                       "max_ratio": MAX_RATIO, "min_difference_sgd": MIN_DIFFERENCE,
                       "removal": "most extreme first per unit, medians recomputed after each removal",
                       "duplicates": "rent of each remaining same-unit same-month group replaced by its mean (2 dp)"},
        "rows_before": len(train),
        "rows_removed": len(removed),
        "removed_below_median": int((removed["RATIO"] < 1).sum()),
        "removed_above_median": int((removed["RATIO"] > 1).sum()),
        "rows_judged": int((initial_comparable_counts(train) >= MIN_COMPARABLES).sum()),
        "rows_after": len(kept),
        "duplicate_groups_after_removal": duplicate_groups,
        "groups_with_different_rents": varied_groups,
        "rows_in_those_groups": int(varied.sum()),
        "rows_with_rent_changed": int((change > 0).sum()),
        "max_rent_change": float(change.max()),
        "note": "Validation months are cleaned too; scores on them measure cleaned rents, not raw ones like test.",
    }
    return kept, removed, replaced, summary
