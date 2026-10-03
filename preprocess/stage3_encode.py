"""Stage 3: turn every feature into a number.

- Categories become integer IDs 1..N from fixed vocabularies (0 = unknown). Area and building vocabularies come
  from the HDB block table, so they depend on neither labels nor the train/validation split; FLAT_MODEL's comes
  from the training rows; FLAT_TYPE becomes its size order (1-room=1 ... 5-room=5, executive=6).
- The approval month becomes MONTH_INDEX (2021-01 = 0) plus sine/cosine of the calendar month; AGE_YEARS and
  COMPLETION_AGE are measured from the approval year.
- Missing building facts (11 blocks) are filled per row: YEAR_COMPLETED from the lease year, MAX_FLOOR from the
  median of blocks on the same street (then subzone, town, all blocks); HDB_INFO_MISSING flags those rows.
- Facility distances and counts are log1p-transformed; continuous columns are standardized with the mean and
  population standard deviation of all training rows (features only, never MONTHLY_RENT).
- Text columns, the constant FURNISHED/FEE and MONTHLY_RENT_ORIGINAL are dropped. MONTHLY_RENT is the stage-1
  cleaned rent in SGD.

encoder.json records every vocabulary and statistic, so each value can be decoded back to stage 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import TARGET, check_output_dir, dtypes, read_splits, read_table, reject, save_json, write_table
from .stage2_enrich import MRT_COLUMNS, STATIC_COLUMNS

FLAT_TYPE_ORDER = ["1-room", "2-room", "3-room", "4-room", "5-room", "executive"]
# (output column, stage 2 source column); BUILDING_ID's source is the TOWN|BLOCK|STREET key.
HDB_VOCABULARIES = [("TOWN_ID", "TOWN"), ("STREET_ID", "STREET"), ("BUILDING_ID", "BUILDING_KEY"),
                    ("SUBZONE_ID", "SUBZONE"), ("PLANNING_AREA_ID", "PLANNING_AREA"), ("REGION_ID", "REGION")]
ID_COLUMNS = [c for c, _ in HDB_VOCABULARIES] + ["FLAT_MODEL_ID", "FLAT_TYPE_ORD"]
LOG1P_COLUMNS = MRT_COLUMNS + STATIC_COLUMNS
STANDARDIZED = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "AGE_YEARS", "MONTH_INDEX", "LATITUDE", "LONGITUDE",
                "MAX_FLOOR", "YEAR_COMPLETED", "COMPLETION_AGE"] + LOG1P_COLUMNS
UNSCALED = ["MONTH_SIN", "MONTH_COS", "HDB_INFO_MISSING"]
FEATURES = ID_COLUMNS + STANDARDIZED + UNSCALED
DROPPED = ["RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FURNISHED", "FEE",
           "SUBZONE", "PLANNING_AREA", "REGION", "MONTHLY_RENT_ORIGINAL"]
# Median of LEASE_COMMENCE_DATE - YEAR_COMPLETED over rows where both are known (correlation 0.989).
LEASE_AFTER_COMPLETION_YEARS = 1
FLOOR_FALLBACK = ["STREET", "SUBZONE", "TOWN"]


def building_key(frame: pd.DataFrame) -> pd.Series:
    return frame["TOWN"] + "|" + frame["BLOCK"] + "|" + frame["STREET"]


def vocabularies(blocks: pd.DataFrame, train: pd.DataFrame) -> dict[str, list[str]]:
    blocks = blocks.assign(BUILDING_KEY=building_key(blocks))
    vocab = {column: sorted(blocks[source].dropna().unique().tolist()) for column, source in HDB_VOCABULARIES}
    vocab["FLAT_MODEL_ID"] = sorted(train["FLAT_MODEL"].dropna().unique().tolist())
    vocab["FLAT_TYPE_ORD"] = FLAT_TYPE_ORDER
    return vocab


def encode_ids(frame: pd.DataFrame, vocab: dict[str, list[str]]) -> pd.DataFrame:
    sources = dict(HDB_VOCABULARIES, FLAT_MODEL_ID="FLAT_MODEL", FLAT_TYPE_ORD="FLAT_TYPE")
    frame = frame.assign(BUILDING_KEY=building_key(frame))
    return pd.DataFrame({column: frame[sources[column]].map({v: i + 1 for i, v in enumerate(vocab[column])})
                         .fillna(0).astype("int64") for column in ID_COLUMNS}, index=frame.index)


def floor_medians(blocks: pd.DataFrame) -> dict:
    known = blocks[blocks["MAX_FLOOR"].notna()]
    levels = {column: known.groupby(column)["MAX_FLOOR"].median().to_dict() for column in FLOOR_FALLBACK}
    return {"levels": levels, "overall": float(known["MAX_FLOOR"].median())}


def raw_numbers(frame: pd.DataFrame, medians: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Numeric features before standardization, plus where MAX_FLOOR came from on imputed rows."""
    year = frame["RENT_APPROVAL_DATE"].str.slice(0, 4).astype("int64")
    month = frame["RENT_APPROVAL_DATE"].str.slice(5, 7).astype("int64")
    missing = frame["MAX_FLOOR"].isna()
    if not missing.equals(frame["YEAR_COMPLETED"].isna()):
        raise ValueError("MAX_FLOOR and YEAR_COMPLETED are expected to be missing on the same rows")

    max_floor = frame["MAX_FLOOR"].copy()
    floor_source = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in FLOOR_FALLBACK:
        fill = max_floor.isna() & frame[column].map(medians["levels"][column]).notna()
        max_floor[fill] = frame.loc[fill, column].map(medians["levels"][column])
        floor_source[fill] = f"{column.lower()} median"
    fill = max_floor.isna()
    max_floor[fill] = medians["overall"]
    floor_source[fill] = "all-block median"
    year_completed = frame["YEAR_COMPLETED"].fillna(frame["LEASE_COMMENCE_DATE"] - LEASE_AFTER_COMPLETION_YEARS)

    angle = 2 * np.pi * (month - 1) / 12
    out = pd.DataFrame({
        "FLOOR_AREA_SQM": frame["FLOOR_AREA_SQM"],
        "LEASE_COMMENCE_DATE": frame["LEASE_COMMENCE_DATE"],
        "AGE_YEARS": year - frame["LEASE_COMMENCE_DATE"],
        "MONTH_INDEX": (year - 2021) * 12 + month - 1,
        "LATITUDE": frame["LATITUDE"],
        "LONGITUDE": frame["LONGITUDE"],
        "MAX_FLOOR": max_floor,
        "YEAR_COMPLETED": year_completed,
        "COMPLETION_AGE": year - year_completed,
        **{column: np.log1p(frame[column]) for column in LOG1P_COLUMNS},
        "MONTH_SIN": np.sin(angle),
        "MONTH_COS": np.cos(angle),
        "HDB_INFO_MISSING": missing.astype("int64"),
    }, index=frame.index)
    for column in out.columns.drop("HDB_INFO_MISSING"):
        out[column] = out[column].astype("float64")
    return out, floor_source[missing]


def check_constants(frames: dict[str, pd.DataFrame]) -> None:
    for split, frame in frames.items():
        reject(frame["FURNISHED"].ne("yes").fillna(True), split, "FURNISHED is no longer always 'yes'; it cannot be dropped")
        reject(frame["FEE"].ne(0), split, "FEE is no longer always 0; it cannot be dropped")
        reject(frame[[c for c in LOG1P_COLUMNS]].lt(0).any(axis=1), split, "facility distances/counts must be >= 0")


def describe(frame: pd.DataFrame) -> dict:
    return {c: {"min": float(frame[c].min()), "max": float(frame[c].max()), "mean": float(frame[c].mean())}
            for c in frame.columns if c != "ROW_ID"}


def column_schema(vocab: dict[str, list[str]], stats: dict) -> list[dict]:
    sources = dict(HDB_VOCABULARIES, FLAT_MODEL_ID="FLAT_MODEL", FLAT_TYPE_ORD="FLAT_TYPE")
    columns = [{"name": "ROW_ID", "role": "id", "transform": "original row position; not a model input"}]
    for column in ID_COLUMNS:
        columns.append({"name": column, "role": "ordinal" if column == "FLAT_TYPE_ORD" else "categorical",
                        "cardinality": len(vocab[column]) + 1, "source": sources[column],
                        "transform": "1..N by vocabulary order in encoder.json; 0 = unknown"})
    for column in STANDARDIZED:
        log = "log1p, then " if column in LOG1P_COLUMNS else ""
        columns.append({"name": column, "role": "numeric", "transform": f"{log}z-score (mean {stats[column]['mean']:.6g}, "
                                                                       f"std {stats[column]['std']:.6g})"})
    columns += [{"name": "MONTH_SIN", "role": "cyclic", "transform": "sin(2*pi*(month-1)/12)"},
                {"name": "MONTH_COS", "role": "cyclic", "transform": "cos(2*pi*(month-1)/12)"},
                {"name": "HDB_INFO_MISSING", "role": "binary",
                 "transform": "1 if MAX_FLOOR/YEAR_COMPLETED/COMPLETION_AGE were imputed"},
                {"name": TARGET, "role": "target",
                 "transform": "stage-1 cleaned rent in SGD per month (outliers removed, identical-feature rentals "
                              "averaged); train only"}]
    return columns


def run_stage3(stage1_dir: Path, stage2_dir: Path, out_dir: Path) -> dict:
    stage1_dir, stage2_dir, out_dir = (Path(p).resolve() for p in (stage1_dir, stage2_dir, out_dir))
    check_output_dir(out_dir, stage1_dir, stage2_dir)
    frames = read_splits(stage2_dir)
    blocks = read_table(stage1_dir, "auxiliary/sg-hdb-block")
    check_constants(frames)
    vocab = vocabularies(blocks, frames["train"])
    medians = floor_medians(blocks)

    ids, raw, floor_sources = {}, {}, {}
    for split, frame in frames.items():
        ids[split] = encode_ids(frame, vocab)
        raw[split], floor_sources[split] = raw_numbers(frame, medians)
    fit = raw["train"][STANDARDIZED]
    std = fit.std(ddof=0).replace(0, 1.0)
    stats = {c: {"mean": float(fit[c].mean()), "std": float(std[c])} for c in STANDARDIZED}

    encoded, report = {}, {"stage": 3, "input_dir": str(stage2_dir), "splits": {}}
    for split, frame in frames.items():
        scaled = raw[split].copy()
        for column in STANDARDIZED:
            scaled[column] = (scaled[column] - stats[column]["mean"]) / stats[column]["std"]
        parts = [frame[["ROW_ID"]], ids[split], scaled[STANDARDIZED + UNSCALED]]
        if split == "train":
            parts.append(frame[[TARGET]])
        out = pd.concat(parts, axis=1)

        values = out.to_numpy(dtype=np.float64)
        if not all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes) or not np.isfinite(values).all():
            raise ValueError(f"{split}: stage 3 output must be finite numbers only")
        for column, source in dict(HDB_VOCABULARIES, FLAT_MODEL_ID="FLAT_MODEL", FLAT_TYPE_ORD="FLAT_TYPE").items():
            original = building_key(frame) if source == "BUILDING_KEY" else frame[source]
            known = out[column].to_numpy() > 0
            decoded = np.asarray([None] + vocab[column], dtype=object)[out[column].to_numpy()]
            if not np.array_equal(decoded[known], original.to_numpy(dtype=object)[known]):
                raise ValueError(f"{split}: {column} does not decode back to {source}")
        encoded[split] = out

        imputed = frame.loc[floor_sources[split].index].assign(BUILDING_KEY=building_key)
        report["splits"][split] = {
            "rows": len(out), "columns": len(out.columns),
            "unknown_ids": {c: int(out[c].eq(0).sum()) for c in ID_COLUMNS},
            "imputed_rows": len(imputed),
            "imputed_buildings": {key: {"rows": int(len(g)), "MAX_FLOOR": float(raw[split].loc[g.index[0], "MAX_FLOOR"]),
                                        "MAX_FLOOR_from": str(floor_sources[split][g.index[0]]),
                                        "YEAR_COMPLETED": float(raw[split].loc[g.index[0], "YEAR_COMPLETED"])}
                                  for key, g in imputed.groupby("BUILDING_KEY")},
            "before_scaling": describe(raw[split][STANDARDIZED]),
            "after_encoding": describe(out),
        }
    if encoded["train"].columns.drop(TARGET).tolist() != encoded["test"].columns.tolist():
        raise ValueError("Train and test feature columns differ")

    encoder = {
        "stage": 3,
        "feature_columns": FEATURES,
        "categorical": {c: {"source": s, "vocabulary_from": "stage1/auxiliary/sg-hdb-block.csv", "unknown_id": 0,
                            "values": vocab[c]} for c, s in HDB_VOCABULARIES},
        "flat_model": {"column": "FLAT_MODEL_ID", "source": "FLAT_MODEL", "vocabulary_from": "stage2/train.csv",
                       "unknown_id": 0, "values": vocab["FLAT_MODEL_ID"]},
        "flat_type": {"column": "FLAT_TYPE_ORD", "source": "FLAT_TYPE", "unknown_id": 0, "order": FLAT_TYPE_ORDER},
        "date": {"source": "RENT_APPROVAL_DATE", "MONTH_INDEX": "(year - 2021) * 12 + month - 1",
                 "MONTH_SIN": "sin(2*pi*(month-1)/12)", "MONTH_COS": "cos(2*pi*(month-1)/12)",
                 "AGE_YEARS": "year - LEASE_COMMENCE_DATE", "COMPLETION_AGE": "year - YEAR_COMPLETED (after imputation)"},
        "imputation": {"flag": "HDB_INFO_MISSING",
                       "YEAR_COMPLETED": f"LEASE_COMMENCE_DATE - {LEASE_AFTER_COMPLETION_YEARS}",
                       "MAX_FLOOR": "median MAX_FLOOR of known blocks on the same STREET, else SUBZONE, else TOWN, else all blocks",
                       "medians_from": "stage1/auxiliary/sg-hdb-block.csv"},
        "log1p": LOG1P_COLUMNS,
        "standardize": {"fit_rows": "all training rows, features only", "fit_row_count": len(fit), "ddof": 0,
                        "formula": "(value - mean) / std", "columns": stats},
        "dropped": DROPPED,
    }
    for split, frame in encoded.items():
        write_table(out_dir, split, frame, float_format="%.10g")
    save_json(out_dir / "encoder.json", encoder)
    save_json(out_dir / "schema.json", {"stage": 3, "tables": {s: dtypes(f) for s, f in encoded.items()},
                                        "columns": column_schema(vocab, stats)})
    save_json(out_dir / "report.json", report)
    return report
