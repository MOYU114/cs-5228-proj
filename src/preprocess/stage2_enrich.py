"""Stage 2: attach auxiliary information to every rental row.

Each row is matched exactly to its HDB block by (TOWN, BLOCK, STREET) and gains the block's area names,
coordinates and building facts, plus Haversine distances to and counts of nearby MRT stations, schools and
malls. MRT stations count only from their first full operating month (explicit calendar in mrt_calendar.json),
evaluated in the row's approval month; schools and malls are static snapshots. Rental columns pass through
unchanged; nothing is encoded, imputed or dropped (missing building facts stay empty).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import KEY, SPLITS, check_output_dir, dtypes, read_splits, read_table, save_json, write_table
from .mrt_calendar import active_stations, load_calendar

HDB_COLUMNS = ["SUBZONE", "PLANNING_AREA", "REGION", "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"]
MRT_COLUMNS = ["MRT_NEAREST_M", "MRT_NEAREST3_MEAN_M", "MRT_COUNT_500M", "MRT_COUNT_1000M"]
STATIC_COLUMNS = ["PRIMARY_NEAREST_M", "SECONDARY_NEAREST_M", "PRIMARY_COUNT_1000M",
                  "MALL_NEAREST_M", "MALL_COUNT_1000M", "MALL_COUNT_2000M"]
COUNT_COLUMNS = ["MRT_COUNT_500M", "MRT_COUNT_1000M", "PRIMARY_COUNT_1000M", "MALL_COUNT_1000M", "MALL_COUNT_2000M"]
ADDED_COLUMNS = HDB_COLUMNS + MRT_COLUMNS + STATIC_COLUMNS


def coordinates(frame: pd.DataFrame) -> np.ndarray:
    points = frame[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("Facility coordinates must be present and finite")
    return points


def distances_m(origin: np.ndarray, destinations: np.ndarray) -> np.ndarray:
    """Pairwise Haversine distances using mean Earth radius 6,371,008.8 m."""
    a, b = np.radians(origin), np.radians(destinations)
    delta = a[:, None, :] - b[None, :, :]
    h = np.sin(delta[:, :, 0] / 2) ** 2 + np.cos(a[:, None, 0]) * np.cos(b[None, :, 0]) * np.sin(delta[:, :, 1] / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def within(distances: np.ndarray, radius: float) -> np.ndarray:
    return (distances <= radius).sum(axis=1)


def school_groups(schools: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mixed-level schools count for every level they teach: "(p1-s4)" is both primary and secondary,
    "(s1-jc2)" is secondary; junior-college-only schools are neither."""
    schools = schools.drop_duplicates(["NAME", "MAINLEVEL_CODE", "LATITUDE", "LONGITUDE"])
    if schools["NAME"].isna().any() or schools["NAME"].duplicated().any():
        raise ValueError("School names must identify one school location and level")
    level = schools["MAINLEVEL_CODE"]
    primary = (level.eq("primary").fillna(False) | level.str.contains(r"\bp\d", na=False)).to_numpy(dtype=bool)
    secondary = level.str.contains(r"\bs\d", na=False).to_numpy(dtype=bool)
    return schools[primary], schools[secondary]


def static_features(origin: np.ndarray, schools: pd.DataFrame, malls: pd.DataFrame) -> tuple[dict, dict]:
    primary, secondary = school_groups(schools)
    if malls["NAME"].isna().any():
        raise ValueError("Mall name is missing")
    malls = malls.drop_duplicates(["NAME", "LATITUDE", "LONGITUDE"])
    if min(len(primary), len(secondary), len(malls)) == 0:
        raise ValueError("Need nonempty primary-school, secondary-school and mall groups")
    dp, ds, dc = [distances_m(origin, coordinates(t)) for t in [primary, secondary, malls]]
    features = {"PRIMARY_NEAREST_M": dp.min(axis=1), "SECONDARY_NEAREST_M": ds.min(axis=1),
                "PRIMARY_COUNT_1000M": within(dp, 1000), "MALL_NEAREST_M": dc.min(axis=1),
                "MALL_COUNT_1000M": within(dc, 1000), "MALL_COUNT_2000M": within(dc, 2000)}
    return features, {"primary_schools": len(primary), "secondary_schools": len(secondary), "malls": len(malls)}


def mrt_features(origin: np.ndarray, stations: pd.DataFrame) -> np.ndarray:
    if len(stations) < 3:
        raise ValueError("Need at least 3 open MRT stations")
    d = distances_m(origin, coordinates(stations))
    return np.column_stack([d.min(axis=1), np.partition(d, 2, axis=1)[:, :3].mean(axis=1),
                            within(d, 500), within(d, 1000)])


def building_index(frame: pd.DataFrame, blocks: pd.DataFrame, split: str) -> np.ndarray:
    """Exact key match only; every rental row must find exactly one block."""
    joined = frame[KEY].merge(blocks[KEY + ["BUILDING_INDEX"]], on=KEY, how="left", sort=False, validate="many_to_one")
    missing = joined["BUILDING_INDEX"].isna().to_numpy()
    if missing.any():
        raise ValueError(f"{split}: unmatched HDB keys at ROW_ID {frame['ROW_ID'].to_numpy()[missing][:10].tolist()}")
    return joined["BUILDING_INDEX"].to_numpy(dtype=np.int64)


def run_stage2(stage1_dir: Path, out_dir: Path) -> dict:
    stage1_dir, out_dir = Path(stage1_dir).resolve(), Path(out_dir).resolve()
    check_output_dir(out_dir, stage1_dir)
    frames = read_splits(stage1_dir)
    blocks = read_table(stage1_dir, "auxiliary/sg-hdb-block")
    mrt = read_table(stage1_dir, "auxiliary/sg-mrt-stations")
    schools = read_table(stage1_dir, "auxiliary/sg-schools")
    malls = read_table(stage1_dir, "auxiliary/sg-shopping-malls")
    if blocks[KEY].isna().any().any() or blocks.duplicated(KEY).any():
        raise ValueError("HDB keys must be nonmissing and unique")
    for column in KEY:
        if pd.concat([blocks[column]] + [f[column] for f in frames.values()]).str.contains("|", regex=False).any():
            raise ValueError(f"{column} contains '|', which later stages reserve as the building-key separator")
    blocks = blocks.reset_index(drop=True)
    blocks["BUILDING_INDEX"] = np.arange(len(blocks))
    origin = coordinates(blocks)

    static, facility_counts = static_features(origin, schools, malls)
    months = sorted(set().union(*(set(f["RENT_APPROVAL_DATE"]) for f in frames.values())))
    monthly, station_counts = {}, {}
    for month in months:
        stations = active_stations(mrt, month)
        station_counts[month] = len(stations)
        monthly[month] = mrt_features(origin, stations)

    enriched, report = {}, {"stage": 2, "input_dir": str(stage1_dir), "facilities": facility_counts, "splits": {}}
    for split in SPLITS:
        frame = frames[split]
        index = building_index(frame, blocks, split)
        out = frame.copy()
        for column in HDB_COLUMNS:
            out[column] = blocks[column].to_numpy()[index]
        mrt_values = np.empty((len(frame), len(MRT_COLUMNS)))
        month_of_row = frame["RENT_APPROVAL_DATE"].to_numpy(dtype=object)
        for month in months:
            rows = month_of_row == month
            mrt_values[rows] = monthly[month][index[rows]]
        for position, column in enumerate(MRT_COLUMNS):
            out[column] = mrt_values[:, position]
        for column in STATIC_COLUMNS:
            out[column] = static[column][index]
        for column in COUNT_COLUMNS:
            out[column] = out[column].astype("int64")
        for column in HDB_COLUMNS[:3]:
            out[column] = out[column].astype("string")
        pd.testing.assert_frame_equal(out[frame.columns], frame)
        enriched[split] = out
        report["splits"][split] = {"rows": len(out), "columns": len(out.columns), "hdb_match_rate": 1.0,
                                   "added_columns": ADDED_COLUMNS,
                                   "missing": {c: int(out[c].isna().sum()) for c in ADDED_COLUMNS},
                                   "rows_with_missing_building_facts": int(out["MAX_FLOOR"].isna().sum())}

    calendar = load_calendar()
    report["mrt_calendar"] = {k: v for k, v in calendar.items() if k != "stations"}
    report["mrt_station_counts_by_month"] = station_counts
    for split, frame in enriched.items():
        write_table(out_dir, split, frame)
    save_json(out_dir / "schema.json", {"stage": 2, "tables": {s: dtypes(f) for s, f in enriched.items()}})
    save_json(out_dir / "report.json", report)
    return report
