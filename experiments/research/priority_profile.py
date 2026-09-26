"""Summarize known date coverage for research priorities; never read targets.

Run from the project root: python3 experiments/research/priority_profile.py
Requires Python 3 standard library and the existing EDA monthly_counts.csv.
This produces descriptive counts, not fitted weights or model scores.
"""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/eda/core/monthly_counts.csv"
OUTPUT = ROOT / "experiments/research/priority_profile.json"


def month_index(month):
    year, month_number = map(int, month.split("-"))
    return 12 * year + month_number - 1


def main():
    with SOURCE.open() as handle:
        rows = list(csv.DictReader(handle))
    train_rows = sum(int(row["train"]) for row in rows)
    test_rows = sum(int(row["test"]) for row in rows)
    assert train_rows == 150_000 and test_rows == 50_000
    cutoff = max(row["month"] for row in rows if int(row["train"]))
    horizons = []
    for row in rows:
        count = int(row["test"])
        if count:
            horizon = month_index(row["month"]) - month_index(cutoff)
            assert horizon >= 0
            horizons.append({"month": row["month"], "horizon": horizon, "rows": count})
    bands = []
    for label, lower, upper in (("h0", 0, 0), ("h1_6", 1, 6), ("h7_16", 7, 16)):
        count = sum(row["rows"] for row in horizons if lower <= row["horizon"] <= upper)
        bands.append({"band": label, "rows": count, "share_percent": round(100 * count / test_rows, 3)})
    assert sum(row["rows"] for row in bands) == test_rows
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "scope": "Date coverage only; no target access, fitting, predictions or score weighting",
        "train_rows": train_rows,
        "train_months": sum(int(row["train"]) > 0 for row in rows),
        "train_cutoff": cutoff,
        "test_rows": test_rows,
        "test_months": len(horizons),
        "max_horizon_months": max(row["horizon"] for row in horizons),
        "test_horizon_bands": bands,
        "test_horizons": horizons,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "bands": bands}, ensure_ascii=False))


if __name__ == "__main__":
    main()
