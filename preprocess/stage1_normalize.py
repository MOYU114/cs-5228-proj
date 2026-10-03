"""Stage 1: unify formats, then clean the training rents.

Every original table is rewritten with one spelling per value: NFKC, trimmed, lowercase text with single
spaces, "N room" flat types written "N-room", missing-value markers ("NIL" postal codes, -1 building facts)
turned into empty fields, money/count strings ("$40,609", "1,280") turned into numbers and 5-digit school
postal codes zero-padded. ROW_ID (original row position) is added to train/test so rows can be traced through
every stage. Finally the training rents are cleaned (rent_cleaning.py): outliers are removed and rentals with
identical features get their mean rent; cleaning/ lists every removed row and every replaced rent for review.
Nothing is derived, joined or encoded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import TARGET, check_output_dir, dtypes, normalize_text, numbers, read_raw, reject, save_json, write_table
from .rent_cleaning import clean_train_rents

MAIN_COLUMNS = ["RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL",
                "FLOOR_AREA_SQM", "FURNISHED", "LEASE_COMMENCE_DATE", "FEE"]
MAIN_TEXT = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FURNISHED"]
AUXILIARY = ["sg-hdb-block", "sg-mrt-stations", "sg-schools", "sg-shopping-malls", "sg-coe-prices", "sg-stock-prices"]


def normalize_columns(frame: pd.DataFrame, columns: list[str]) -> dict:
    """Normalize text columns in place and report what changed, with a few before -> after examples."""
    report = {}
    for column in columns:
        before = frame[column].astype("string")
        after = before.map(normalize_text).astype("string")
        changed = ~(before.eq(after).fillna(False) | (before.isna() & after.isna())).to_numpy(dtype=bool)
        pairs = pd.DataFrame({"before": before[changed], "after": after[changed]}).drop_duplicates().head(5)
        report[column] = {"unique_before": int(before.nunique()), "unique_after": int(after.nunique()),
                          "changed_rows": int(changed.sum()),
                          "examples": [f"{b!r} -> {a!r}" for b, a in pairs.itertuples(index=False)]}
        frame[column] = after
    return report


def normalize_main(raw: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict]:
    expected = MAIN_COLUMNS + ([TARGET] if split == "train" else [])
    if sorted(raw.columns) != sorted(expected) or not raw.columns.is_unique:
        raise ValueError(f"{split}: expected columns {expected}, found {list(raw.columns)}")
    if raw.empty:
        raise ValueError(f"{split}: no rows")
    out = raw.loc[:, expected].reset_index(drop=True).copy()
    text = normalize_columns(out, MAIN_TEXT)

    spelled = out["FLAT_TYPE"]
    out["FLAT_TYPE"] = spelled.str.replace(r"^(\d+)\s*[- ]\s*room$", r"\1-room", regex=True)
    respelled = ~spelled.eq(out["FLAT_TYPE"]).fillna(True).to_numpy(dtype=bool)
    flat_type_spelling = {f"{b} -> {a}": int(n) for (b, a), n in
                          pd.DataFrame({"b": spelled[respelled], "a": out["FLAT_TYPE"][respelled]}).value_counts().sort_index().items()}

    dates = out["RENT_APPROVAL_DATE"].str.strip()
    reject(~dates.str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").fillna(False), split, "RENT_APPROVAL_DATE must be YYYY-MM")
    out["RENT_APPROVAL_DATE"] = dates
    out["FLOOR_AREA_SQM"] = numbers(out["FLOOR_AREA_SQM"], split, positive=True)
    out["LEASE_COMMENCE_DATE"] = numbers(out["LEASE_COMMENCE_DATE"], split, integer=True, positive=True).astype("int64")
    out["FEE"] = numbers(out["FEE"], split)
    if split == "train":
        out[TARGET] = numbers(out[TARGET], split, integer=True, positive=True).astype("int64")
    out.insert(0, "ROW_ID", np.arange(len(out), dtype=np.int64))

    report = {"rows": len(out), "columns": len(out.columns), "text": text,
              "flat_type_spelling": flat_type_spelling,
              "missing": {c: int(n) for c, n in out.isna().sum().items()},
              "distinct_values": {"FURNISHED": sorted(out["FURNISHED"].dropna().unique().tolist()),
                                  "FEE": sorted(out["FEE"].unique().tolist())},
              "duplicate_rows_excluding_row_id": int(out.drop(columns="ROW_ID").duplicated().sum())}
    return out, report


def coordinates(t: pd.DataFrame, where: str) -> None:
    t["LATITUDE"] = numbers(t["LATITUDE"], where)
    t["LONGITUDE"] = numbers(t["LONGITUDE"], where)
    reject((t["LATITUDE"].abs() > 90) | (t["LONGITUDE"].abs() > 180), where, "coordinates out of range")


def normalize_hdb(t: pd.DataFrame) -> dict:
    where = "sg-hdb-block"
    text = normalize_columns(t, ["TOWN", "BLOCK", "STREET", "POSTAL_CODE", "SUBZONE", "PLANNING_AREA", "REGION"])
    nil = t["POSTAL_CODE"].eq("nil").fillna(False).to_numpy(dtype=bool)
    t.loc[nil, "POSTAL_CODE"] = pd.NA
    reject(~t["POSTAL_CODE"].str.fullmatch(r"\d{6}").fillna(True), where, "POSTAL_CODE must be 6 digits")
    coordinates(t, where)
    sentinels = {}
    for column in ["MAX_FLOOR", "YEAR_COMPLETED"]:
        unknown = t[column].str.strip().eq("-1").fillna(False)
        sentinels[column] = int(unknown.sum())
        t[column] = numbers(t[column].mask(unknown), where, integer=True, positive=True, nullable=True)
    return {"text": text, "postal_code_nil_to_missing": int(nil.sum()), "minus_one_to_missing": sentinels}


def normalize_mrt(t: pd.DataFrame) -> dict:
    text = normalize_columns(t, ["CODE", "NAME", "STATUS", "SUBZONE", "PLANNING_AREA", "REGION"])
    coordinates(t, "sg-mrt-stations")
    return {"text": text}


def normalize_schools(t: pd.DataFrame) -> dict:
    where = "sg-schools"
    # URL is left byte-for-byte: URL paths are case-sensitive.
    text = normalize_columns(t, ["NAME", "STREET", "POSTAL_CODE", "MRT_STATIONS", "BUS_LINES", "PLANNING_AREA",
                                 "REGION", "TYPE_CODE", "NATRUE_CODE", "SESSION_CODE", "MAINLEVEL_CODE"])
    short = t["POSTAL_CODE"].str.fullmatch(r"\d{1,5}").fillna(False).to_numpy(dtype=bool)
    t.loc[short, "POSTAL_CODE"] = t.loc[short, "POSTAL_CODE"].str.zfill(6)
    reject(~t["POSTAL_CODE"].str.fullmatch(r"\d{6}").fillna(True), where, "POSTAL_CODE must be 6 digits")
    coordinates(t, where)
    return {"text": text, "postal_code_zero_padded": int(short.sum())}


def normalize_malls(t: pd.DataFrame) -> dict:
    text = normalize_columns(t, ["NAME"])
    coordinates(t, "sg-shopping-malls")
    return {"text": text}


def normalize_coe(t: pd.DataFrame) -> dict:
    where = "sg-coe-prices"
    text = normalize_columns(t, ["CATEGORY", "MONTH"])
    for column in ["YEAR", "ROUND"]:
        t[column] = numbers(t[column], where, integer=True, positive=True).astype("int64")
    for column in ["QUOTA_PREMIUM", "PREVAILING_QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]:
        t[column] = numbers(t[column].str.replace(r"[$,\s]", "", regex=True), where, integer=True).astype("int64")
    return {"text": text, "currency_and_thousands_separators_removed": ["QUOTA_PREMIUM", "PREVAILING_QUOTA_PREMIUM",
                                                                        "QUOTA", "BIDS_RECEIVED"]}


def normalize_stocks(t: pd.DataFrame) -> dict:
    where = "sg-stock-prices"
    text = normalize_columns(t, ["NAME", "SYMBOL"])
    t["DATE"] = t["DATE"].str.strip()
    reject(~t["DATE"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False), where, "DATE must be YYYY-MM-DD")
    for column in ["OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"]:
        t[column] = numbers(t[column], where, nullable=True)
    return {"text": text, "missing_price_rows": int(t["CLOSE"].isna().sum())}


NORMALIZERS = {"sg-hdb-block": normalize_hdb, "sg-mrt-stations": normalize_mrt, "sg-schools": normalize_schools,
               "sg-shopping-malls": normalize_malls, "sg-coe-prices": normalize_coe, "sg-stock-prices": normalize_stocks}


def run_stage1(data_dir: Path, out_dir: Path) -> dict:
    """Validate and normalize every table before writing any of them."""
    data_dir, out_dir = Path(data_dir).resolve(), Path(out_dir).resolve()
    check_output_dir(out_dir, data_dir)
    tables, report = {}, {"stage": 1, "input_dir": str(data_dir), "tables": {}}
    for split in ["train", "test"]:
        tables[split], report["tables"][split] = normalize_main(read_raw(data_dir / f"{split}.csv"), split)
    if tables["train"].columns.drop(TARGET).tolist() != tables["test"].columns.tolist():
        raise ValueError("Train and test columns differ")
    tables["train"], removed, replaced, report["rent_cleaning"] = clean_train_rents(tables["train"])
    tables["cleaning/removed_outliers"], tables["cleaning/rent_replaced"] = removed, replaced
    for name in AUXILIARY:
        table = read_raw(data_dir / "auxiliary" / f"{name}.csv")
        details = NORMALIZERS[name](table)
        tables[f"auxiliary/{name}"] = table
        report["tables"][f"auxiliary/{name}"] = {"rows": len(table), "columns": len(table.columns), **details}

    for name, table in tables.items():
        write_table(out_dir, name, table)
    save_json(out_dir / "schema.json", {"stage": 1, "tables": {name: dtypes(t) for name, t in tables.items()}})
    save_json(out_dir / "report.json", report)
    return report
