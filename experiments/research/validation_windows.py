"""Count proposed time windows from EDA output; no model or target access.

Run from any directory: python3 experiments/research/validation_windows.py
Requires only the Python standard library and the existing EDA monthly counts.
The proposed windows are for team review, not a frozen validation protocol.
"""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WINDOWS = (
    ("V1_recent_development", "2024-03", "2024-04", "2024-09"),
    ("V2_recent_confirmation", "2024-09", "2024-10", "2025-03"),
    ("VL_long_horizon", "2023-05", "2023-06", "2024-09"),
)


def main():
    with (ROOT / "experiments/eda/core/monthly_counts.csv").open() as handle:
        counts = {row["month"]: int(row["train"]) for row in csv.DictReader(handle)}
    assert sum(counts.values()) == 150_000
    rows = []
    for name, cutoff, start, end in WINDOWS:
        training = [m for m, n in counts.items() if m <= cutoff and n]
        validation = [m for m, n in counts.items() if start <= m <= end and n]
        assert training and validation and max(training) < min(validation)
        rows.append({
            "window": name,
            "train_start": min(training),
            "train_end": max(training),
            "train_months": len(training),
            "train_rows": sum(counts[m] for m in training),
            "validation_start": min(validation),
            "validation_end": max(validation),
            "validation_months": len(validation),
            "validation_rows": sum(counts[m] for m in validation),
        })
    output = ROOT / "experiments/research/validation_windows.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.read_text(), end="")


if __name__ == "__main__":
    main()
