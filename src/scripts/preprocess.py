"""Rebuild ../data_preprocess/stage1, stage2 and stage3 from ../data.

stage1  format unification of every original table (case, whitespace, spellings, missing markers, number formats),
        then training-rent cleaning (outliers removed, identical-feature rentals averaged; see stage1/cleaning/)
stage2  rental rows joined with HDB block facts and nearby MRT/school/mall features
stage3  every feature as a number: category IDs, date encodings, imputation, log1p, standardization

Each stage reads only the previous stage's folder, so a single stage can be rerun with --stage.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from preprocess.common import DATA_DIR, OUTPUT_DIR
from preprocess.stage1_normalize import run_stage1
from preprocess.stage2_enrich import run_stage2
from preprocess.stage3_encode import run_stage3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    stage1, stage2, stage3 = (args.output_dir / f"stage{i}" for i in (1, 2, 3))
    summary = {}
    if args.stage in ("1", "all"):
        report = run_stage1(args.data_dir, stage1)
        summary["stage1"] = {name: t["rows"] for name, t in report["tables"].items()}
        cleaning = report["rent_cleaning"]
        summary["stage1"]["train"] = cleaning["rows_after"]
        summary["stage1_rent_cleaning"] = {k: cleaning[k] for k in ["rows_before", "rows_removed", "rows_after",
                                                                     "rows_with_rent_changed"]}
    if args.stage in ("2", "all"):
        report = run_stage2(stage1, stage2)
        summary["stage2"] = {s: [r["rows"], r["columns"]] for s, r in report["splits"].items()}
    if args.stage in ("3", "all"):
        report = run_stage3(stage1, stage2, stage3)
        summary["stage3"] = {s: [r["rows"], r["columns"]] for s, r in report["splits"].items()}
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
