"""Read-only auxiliary-data audit; no model fitting or feature dataset creation.

Run: python notebooks/eda_auxiliary.py --root .
Dependencies: Python 3.10+, pandas, numpy. Outputs only under the chosen directory.
CSV row indices in audit outputs are zero-based data rows, excluding the header.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["TOWN", "BLOCK", "STREET"]
PRICE = ["OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE"]


def normalize_text(value: object) -> str:
    """Conservative normalization; no abbreviations, fuzzy matching or key guessing."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).strip().lower())


def keys(df: pd.DataFrame, columns: list[str], normalize: bool = True) -> pd.Series:
    values = df[columns].astype(str)
    if normalize:
        values = values.map(normalize_text)
    assert not values.map(lambda v: "|" in v).any().any(), "Key separator occurs in data"
    return values.agg("|".join, axis=1)


def save_json(value: object, path: Path) -> None:
    def convert(x: object):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (pd.Timestamp, pd.Period, Path)):
            return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False) + "\n")


def md_table(df: pd.DataFrame) -> str:
    def cell(v: object) -> str:
        return str(v).replace("|", " / ").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(cell, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
        *["| " + " | ".join(map(cell, row)) + " |" for row in df.itertuples(index=False, name=None)],
    ])


def profile(data: dict[str, pd.DataFrame], out: Path) -> dict:
    fields, frequencies, summaries = [], [], {}
    for name, df in data.items():
        summaries[name] = {
            "rows": len(df), "columns": len(df.columns),
            "full_duplicate_excess": int(df.duplicated().sum()),
            "null_cells": int(df.isna().sum().sum()), "column_names": list(df.columns),
            "constant_fields": [c for c in df if df[c].nunique(dropna=False) <= 1],
        }
        for col in df:
            s = df[col]
            record = dict(file=name, field=col, inferred_dtype=str(s.dtype), rows=len(s),
                          missing=int(s.isna().sum()), unique_non_null=int(s.nunique()),
                          constant=bool(s.nunique(dropna=False) <= 1))
            if pd.api.types.is_numeric_dtype(s):
                record.update(minimum=s.min(), maximum=s.max(), median=s.median(),
                              zero_count=int(s.eq(0).sum()), negative_count=int(s.lt(0).sum()),
                              infinite_count=int(np.isinf(s).sum()))
            else:
                record.update(minimum=s.dropna().min(), maximum=s.dropna().max())
                record["range_interpretation"] = "lexicographic; not numerical"
                record["blank_count"] = int(s.astype("string").str.strip().eq("").sum())
                record["explicit_nil_count"] = int(s.astype("string").str.strip().str.upper().eq("NIL").sum())
                for value, count in s.value_counts(dropna=False).items():
                    frequencies.append(dict(file=name, field=col, value=str(value), count=int(count)))
            fields.append(record)
    pd.DataFrame(fields).to_csv(out / "field_overview.csv", index=False)
    pd.DataFrame(frequencies).to_csv(out / "categorical_frequencies.csv", index=False)
    pd.DataFrame(summaries).T.to_csv(out / "file_overview.csv", index_label="file")
    return summaries


def hdb_audit(hdb: pd.DataFrame, core: dict[str, pd.DataFrame], out: Path) -> dict:
    modes = {"raw_town_block_street": (KEY, False),
             "normalized_town_block_street": (KEY, True),
             "normalized_block_street": (["BLOCK", "STREET"], True)}
    match_rows, conflict_tables, crosswalks, unresolved = [], [], [], []
    h = hdb.copy()
    h.insert(0, "hdb_row_index", h.index)
    h["normalized_key"] = keys(h, KEY)
    table_stats = {}
    for mode, (columns, norm) in modes.items():
        hk = keys(h, columns, norm)
        counts = hk.value_counts()
        conflict_keys = counts[counts > 1].index
        conflicts = h.loc[hk.isin(conflict_keys)].copy()
        conflicts["mode"] = mode
        conflicts["audited_key"] = hk.loc[conflicts.index]
        conflict_tables.append(conflicts)
        table_stats[mode] = {"unique_keys": int(hk.nunique()), "one_to_many_keys": len(conflict_keys),
                             "rows_in_conflicts": len(conflicts)}
        for split, df in core.items():
            dk = keys(df, columns, norm)
            n = dk.map(counts).fillna(0).astype(int)
            unique_n = dk.drop_duplicates().map(counts).fillna(0).astype(int)
            match_rows.append(dict(split=split, mode=mode, rows=len(df),
                matched_rows=int(n.gt(0).sum()), unique_match_rows=int(n.eq(1).sum()),
                ambiguous_rows=int(n.gt(1).sum()), unmatched_rows=int(n.eq(0).sum()),
                row_match_pct=float(100 * n.gt(0).mean()), unique_keys=len(unique_n),
                matched_unique_keys=int(unique_n.gt(0).sum()),
                unique_key_match_pct=float(100 * unique_n.gt(0).mean())))
    # Fail closed: the current data should support a strict many-to-one join.
    # Any future normalized collisions must be resolved explicitly, never keep-first.
    hc = h["normalized_key"].value_counts()
    h_unique = h.loc[h["normalized_key"].map(hc).eq(1)].set_index("normalized_key")
    validation = {}
    for split, df in core.items():
        d = df.copy()
        d["normalized_key"] = keys(d, KEY)
        d["raw_key"] = keys(d, KEY, False)
        d["raw_matches"] = d["raw_key"].isin(keys(h, KEY, False))
        d["normalized_candidates"] = d["normalized_key"].map(hc).fillna(0).astype(int)
        d["reason"] = np.select([d.normalized_candidates.eq(0), d.normalized_candidates.gt(1),
                                  ~d.raw_matches],
                                 ["normalized_key_absent", "ambiguous_hdb_key", "format_only_recovered"],
                                 default="raw_exact_match")
        d["hdb_row_index"] = d.normalized_key.map(h_unique.hdb_row_index)
        grouped = d.groupby(KEY + ["normalized_key", "raw_key", "reason"], dropna=False).agg(
            core_rows=("raw_key", "size"), hdb_row_index=("hdb_row_index", "min"),
            normalized_candidates=("normalized_candidates", "max")).reset_index()
        grouped.insert(0, "split", split)
        crosswalks.append(grouped)
        unresolved.append(grouped.loc[grouped.reason.isin(["normalized_key_absent", "ambiguous_hdb_key"])])
        joined = d.join(h_unique[["POSTAL_CODE", "MAX_FLOOR", "YEAR_COMPLETED"]],
                        on="normalized_key", validate="many_to_one")
        rent_year = pd.to_datetime(joined.RENT_APPROVAL_DATE, format="%Y-%m").dt.year
        valid_completion = joined.YEAR_COMPLETED.gt(0)
        after_lease = valid_completion & joined.YEAR_COMPLETED.gt(joined.LEASE_COMMENCE_DATE)
        discrepancy = joined.loc[after_lease, ["normalized_key", "YEAR_COMPLETED", "LEASE_COMMENCE_DATE"]].copy()
        discrepancy["completion_minus_lease_years"] = discrepancy.YEAR_COMPLETED - discrepancy.LEASE_COMMENCE_DATE
        discrepancy.value_counts().reset_index(name="core_rows").to_csv(
            out / f"hdb_{split}_completion_after_lease.csv", index=False)
        validation[split] = {
            "rows_joined": len(joined), "rows_postal_nil": int(joined.POSTAL_CODE.eq("NIL").sum()),
            "rows_floor_minus_one": int(joined.MAX_FLOOR.eq(-1).sum()),
            "rows_completion_minus_one": int(joined.YEAR_COMPLETED.eq(-1).sum()),
            "rows_completion_after_approval_year": int((valid_completion & joined.YEAR_COMPLETED.gt(rent_year)).sum()),
            "rows_completion_after_lease_year": int(after_lease.sum()),
            "buildings_completion_after_lease_year": int(discrepancy.normalized_key.nunique()),
            "max_completion_minus_lease_years": int(discrepancy.completion_minus_lease_years.max()) if len(discrepancy) else 0,
        }
        pd.DataFrame({"split": split, "reason": d.reason}).value_counts().reset_index(name="rows").to_csv(
            out / f"hdb_{split}_match_reasons.csv", index=False)
    matches = pd.DataFrame(match_rows)
    matches.to_csv(out / "hdb_match_rates.csv", index=False)
    pd.concat(conflict_tables, ignore_index=True).to_csv(out / "hdb_key_conflicts.csv", index=False)
    pd.concat(crosswalks, ignore_index=True).to_csv(out / "hdb_key_crosswalk.csv", index=False)
    pd.concat(unresolved, ignore_index=True).to_csv(out / "hdb_unmatched_or_ambiguous.csv", index=False)
    for split, d in core.items():
        h[f"{split}_core_rows"] = h.normalized_key.map(keys(d, KEY).value_counts()).fillna(0).astype(int)
    h.to_csv(out / "hdb_normalized_inventory.csv", index=False)
    sentinel = h.POSTAL_CODE.eq("NIL") | h.MAX_FLOOR.eq(-1) | h.YEAR_COMPLETED.eq(-1)
    h.loc[sentinel].to_csv(out / "hdb_sentinel_rows.csv", index=False)
    return {"key_table_stats": table_stats, "match_rates": match_rows, "joined_quality": validation,
            "postal_nil_rows": int(h.POSTAL_CODE.eq("NIL").sum()),
            "floor_minus_one_rows": int(h.MAX_FLOOR.eq(-1).sum()),
            "completion_minus_one_rows": int(h.YEAR_COMPLETED.eq(-1).sum()),
            "valid_completion_min": int(h.loc[h.YEAR_COMPLETED.gt(0), "YEAR_COMPLETED"].min()),
            "valid_completion_max": int(h.YEAR_COMPLETED.max()),
            "valid_floor_min": int(h.loc[h.MAX_FLOOR.gt(0), "MAX_FLOOR"].min()),
            "valid_floor_max": int(h.MAX_FLOOR.max())}


def facility_audit(data: dict[str, pd.DataFrame], out: Path) -> dict:
    result = {}
    for name in ["sg-mrt-stations.csv", "sg-schools.csv", "sg-shopping-malls.csv"]:
        d = data[name]
        geo_valid = d.LATITUDE.between(-90, 90) & d.LONGITUDE.between(-180, 180)
        result[name] = {
            "rows": len(d), "unique_names": int(d.NAME.nunique()),
            "duplicate_name_excess": int(d.NAME.duplicated().sum()),
            "unique_coordinates": len(d[["LATITUDE", "LONGITUDE"]].drop_duplicates()),
            "duplicate_coordinate_excess": int(d.duplicated(["LATITUDE", "LONGITUDE"]).sum()),
            "invalid_global_coordinate_rows": int((~geo_valid).sum()),
            "latitude_min": float(d.LATITUDE.min()), "latitude_max": float(d.LATITUDE.max()),
            "longitude_min": float(d.LONGITUDE.min()), "longitude_max": float(d.LONGITUDE.max()),
            "has_effective_date": False,
        }
        duplicate = d.NAME.duplicated(keep=False) | d.duplicated(["LATITUDE", "LONGITUDE"], keep=False)
        d.loc[duplicate].to_csv(out / (name.removesuffix(".csv") + "_shared_names_or_coordinates.csv"), index=False)
    mrt, school = data["sg-mrt-stations.csv"], data["sg-schools.csv"]
    result["sg-mrt-stations.csv"]["status_counts"] = mrt.STATUS.value_counts().to_dict()
    result["sg-mrt-stations.csv"]["unique_codes"] = int(mrt.CODE.nunique())
    result["sg-mrt-stations.csv"]["names_with_mixed_status"] = int(mrt.groupby("NAME").STATUS.nunique().gt(1).sum())
    result["sg-schools.csv"]["level_counts"] = school.MAINLEVEL_CODE.value_counts().to_dict()
    return result


def economy_audit(data: dict[str, pd.DataFrame], core: dict[str, pd.DataFrame], out: Path) -> dict:
    coe = data["sg-coe-prices.csv"].copy()
    coe["date"] = pd.to_datetime(coe.YEAR.astype(str) + "-" + coe.MONTH, format="%Y-%b", errors="coerce")
    coe["month"] = coe.date.dt.to_period("M").astype(str)
    parsed = []
    for col in ["QUOTA_PREMIUM", "PREVAILING_QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]:
        numbers = pd.to_numeric(coe[col].str.replace(r"[$,]", "", regex=True), errors="coerce")
        parsed.append(dict(field=col, parse_failures=int(numbers.isna().sum()), minimum=float(numbers.min()),
                           maximum=float(numbers.max()), zero_count=int(numbers.eq(0).sum()),
                           negative_count=int(numbers.lt(0).sum())))
        coe[col + "_numeric"] = numbers
    pd.DataFrame(parsed).to_csv(out / "coe_numeric_parse_audit.csv", index=False)
    coe.groupby(["month", "CATEGORY"]).agg(rows=("ROUND", "size"), rounds=("ROUND", "nunique")).reset_index().to_csv(
        out / "coe_month_category_coverage.csv", index=False)
    months = pd.period_range(coe.date.min(), coe.date.max(), freq="M").astype(str)
    expected = pd.MultiIndex.from_product([months, sorted(coe.CATEGORY.unique()), [1, 2]],
                                          names=["month", "CATEGORY", "ROUND"])
    seen = pd.MultiIndex.from_frame(coe[["month", "CATEGORY", "ROUND"]])
    absent = expected.difference(seen).to_frame(index=False)
    absent.to_csv(out / "coe_missing_month_category_rounds.csv", index=False)
    stock = data["sg-stock-prices.csv"].copy()
    stock["parsed_date"] = pd.to_datetime(stock.DATE, format="%Y-%m-%d", errors="coerce")
    stock["month"] = stock.parsed_date.dt.to_period("M").astype(str)
    stock["missing_price"] = stock[PRICE].isna().any(axis=1)
    symbol = stock.groupby(["SYMBOL", "NAME"]).agg(first_date=("DATE", "min"), last_date=("DATE", "max"),
        rows=("DATE", "size"), months=("month", "nunique"), missing_price_rows=("missing_price", "sum")).reset_index()
    symbol.to_csv(out / "stock_symbol_coverage.csv", index=False)
    monthly = stock.groupby("month").agg(rows=("SYMBOL", "size"), symbols=("SYMBOL", "nunique"),
                                         missing_price_rows=("missing_price", "sum")).reset_index()
    monthly.to_csv(out / "stock_month_coverage.csv", index=False)
    stock.loc[stock.missing_price].to_csv(out / "stock_missing_rows.csv", index=False)
    high_bad = stock.HIGH.lt(stock[["OPEN", "CLOSE", "LOW"]].max(axis=1))
    low_bad = stock.LOW.gt(stock[["OPEN", "CLOSE", "HIGH"]].min(axis=1))
    stock.loc[high_bad | low_bad].assign(high_below_open_close_or_low=high_bad,
        low_above_open_close_or_high=low_bad).to_csv(out / "stock_ohlc_inconsistencies.csv", index=False)
    train_end = pd.Period(core["train"].RENT_APPROVAL_DATE.max(), freq="M")
    test_end = pd.Period(core["test"].RENT_APPROVAL_DATE.max(), freq="M")
    temporal = {}
    for name, df in [("coe", coe), ("stock", stock)]:
        mm = pd.PeriodIndex(df.month, freq="M")
        temporal[name] = {"rows_after_train_end_month": int((mm > train_end).sum()),
                          "rows_after_test_end_month": int((mm > test_end).sum()),
                          "train_missing_months": sorted(set(core["train"].RENT_APPROVAL_DATE) - set(df.month)),
                          "test_missing_months": sorted(set(core["test"].RENT_APPROVAL_DATE) - set(df.month))}
    return {"coe": {"rows": len(coe), "first_month": coe.month.min(), "last_month": coe.month.max(),
        "months": int(coe.month.nunique()), "categories": coe.CATEGORY.value_counts().to_dict(),
        "date_parse_failures": int(coe.date.isna().sum()),
        "duplicate_month_category_round_excess": int(coe.duplicated(["month", "CATEGORY", "ROUND"]).sum()),
        "missing_round_cells_within_global_range": len(absent), "numeric_parse": parsed,
        "zero_prevailing_by_category": coe.loc[coe.PREVAILING_QUOTA_PREMIUM_numeric.eq(0), "CATEGORY"].value_counts().to_dict()},
        "stock": {"rows": len(stock), "symbols": int(stock.SYMBOL.nunique()),
        "first_date": stock.DATE.min(), "last_date": stock.DATE.max(), "months": int(stock.month.nunique()),
        "date_parse_failures": int(stock.parsed_date.isna().sum()),
        "duplicate_symbol_date_excess": int(stock.duplicated(["SYMBOL", "DATE"]).sum()),
        "missing_price_rows": int(stock.missing_price.sum()), "zero_volume_rows": int(stock.VOLUME.eq(0).sum()),
        "high_below_open_close_or_low": int(high_bad.sum()), "low_above_open_close_or_high": int(low_bad.sum()),
        "ohlc_inconsistent_rows": int((high_bad | low_bad).sum()),
        "symbols_starting_after_train_end": int(pd.to_datetime(symbol.first_date).dt.to_period("M").gt(train_end).sum()),
        "symbols_with_all_global_months": int(symbol.months.eq(stock.month.nunique()).sum()),
        "monthly_symbol_min": int(monthly.symbols.min()), "monthly_symbol_max": int(monthly.symbols.max()),
        "missing_price_dates": stock.loc[stock.missing_price, "DATE"].value_counts().to_dict()},
        "temporal_relation": temporal}


def write_report(results: dict, out: Path) -> None:
    h, f, e = results["hdb"], results["facilities"], results["economy"]
    mrt, school, mall = (f[n] for n in ["sg-mrt-stations.csv", "sg-schools.csv", "sg-shopping-malls.csv"])
    coe, stock = e["coe"], e["stock"]
    inventory = []
    for name, d in results["files"].items():
        coverage = {"sg-coe-prices.csv": f'{coe["first_month"]}–{coe["last_month"]}，{coe["months"]} 个月',
                    "sg-stock-prices.csv": f'{stock["first_date"]}–{stock["last_date"]}，{stock["months"]} 个月'}.get(name, "快照；无生效/采集日期")
        inventory.append([name, d["rows"], d["columns"], d["null_cells"], d["full_duplicate_excess"], coverage])
    match = pd.DataFrame(h["match_rates"])
    match = match[["split", "mode", "matched_rows", "rows", "row_match_pct", "matched_unique_keys", "unique_keys", "unique_key_match_pct"]]
    for col in ["row_match_pct", "unique_key_match_pct"]:
        match[col] = match[col].map(lambda x: f"{x:.4f}%")
    schema = "\n".join(f'- `{name}`：`' + "`, `".join(d["column_names"]) + "`。" for name, d in results["files"].items())
    text = f"""## 辅助数据审计（实际 CSV；仅用于质量审计）

统计来源：`experiments/eda/auxiliary/summary.json`、`field_overview.csv` 和各专项表。全量辅助文件用于检查覆盖边界，未据此创建预测特征、填补训练数据或使用测试目标。原始文件 SHA-256 在运行前后核对一致。

{md_table(pd.DataFrame(inventory, columns=['文件','行数','列数','空值单元格','整行重复多余行','时间覆盖']))}

字段明细（逐列推断类型、范围、缺失、唯一值、零/负值和常量标记在 `field_overview.csv`；字符串范围按字典序，完整频数在 `categorical_frequencies.csv`）：

{schema}

### HDB 楼栋连接与占位缺失

仅对内存中的 `TOWN`、`BLOCK`、`STREET` 做 Unicode NFKC、首尾空白去除、小写化、连续空白压为一个空格。楼号保留字母后缀；不转数值，不猜街道别名，不模糊连接。以 `TOWN + BLOCK + STREET` 为主键，另外审计 `BLOCK + STREET` 的 fallback 可能性。

{md_table(match)}

原始模式的“唯一键”分母保留大小写写法差异；规范化模式分母为规范化楼栋数，两者不能直接当作同一楼栋分母比较。两套规范化键均达到全量匹配，因此没有必要启用 fallback。三种模式在 HDB 表的重复键、一对多键及冲突行均为 0；规范化后未匹配和歧义楼栋均为 0。原始失配全部经格式规范化恢复，未观察到需要街道改名或城镇猜测的证据。

`hdb_key_crosswalk.csv` 保留原始键、规范化键、输入 split、交易行数和匹配 HDB 零基行号；`hdb_normalized_inventory.csv` 记录每栋在 train/test 的使用行数。`hdb_key_conflicts.csv`、`hdb_unmatched_or_ambiguous.csv` 保留表头，即使目前为空也作为下次运行的审计接口。后续生产连接必须使用 `validate='many_to_one'`，遇到冲突停止并保留全部候选，不可静默选首条。

HDB 表没有 pandas 空值，但 `POSTAL_CODE='NIL'` 有 {h['postal_nil_rows']} 行，`MAX_FLOOR=-1` 和 `YEAR_COMPLETED=-1` 各 {h['floor_minus_one_rows']} 行。有效竣工年份为 {h['valid_completion_min']}–{h['valid_completion_max']}，有效最大楼层数为 {h['valid_floor_min']}–{h['valid_floor_max']}。这些哨兵值应在派生数据中转缺失并保留缺失指示；邮编保持字符串。匹配成功不代表每个属性都有有效值。

{md_table(pd.DataFrame(h['joined_quality']).T.reset_index(names='split'))}

上述竣工年晚于起租契年份的记录已单列 `hdb_train_completion_after_lease.csv` / `hdb_test_completion_after_lease.csv`，属于跨表口径核查项，不能未经查证就用一个年份覆盖另一个。`YEAR_COMPLETED` 是竣工年份，不能视为该辅助记录的生效时间。文件缺少历史版本、拆除和地址变更日期；地理位置、楼层等是否可追溯到每个批准月仍需确认。

### MRT、学校、商场：快照不是历史可用性证明

- **MRT**：{mrt['rows']} 条线路/站代码记录，`STATUS` 为 open {mrt['status_counts'].get('open', 0)}、planned {mrt['status_counts'].get('planned', 0)}；只有 {mrt['unique_names']} 个站名和 {mrt['unique_coordinates']} 个坐标，按坐标重复的多余记录 {mrt['duplicate_coordinate_excess']} 条；{mrt['names_with_mixed_status']} 个站名同时含 open 与 planned。多代码可能代表换乘线路，不是整行重复，不能按记录数直接计“站点数”。没有开放/规划公布日期，当前 open 不代表 2021 年已开放。历史“最近已开放站”或线路数量必须先取得可靠生效日期；否则只能作为明确注明静态快照假设的待验证特征，不得声称严格时点有效。planned 不应混入已开放距离。
- **学校**：{school['rows']} 所，`MAINLEVEL_CODE='primary'` 仅 {school['level_counts'].get('primary', 0)} 所，文件还包含中学、初院和混合学段，不能将全表称为小学。{school['unique_coordinates']} 个坐标，坐标重复多余 {school['duplicate_coordinate_excess']} 条可能为共址，需按目标定义决定计数口径。可审议距离、学段及学校类型；开校、迁址、合并日期缺失。`NATRUE_CODE` 是原文件列名拼写，派生层若改名需明确映射；`REGION` 的学校分区（east/north/south/west）与 HDB/MRT 地理分区词表不一致，不能直接互换或连接。
- **商场**：{mall['rows']} 条、{mall['unique_names']} 个名称，整行/坐标均无重复；只有名称与经纬度。可审议距离或邻域数量，无法单凭此表判定历史月份的开业、关业与改造状态。

三表的经纬度均通过全球合法范围检查（纬度 ±90、经度 ±180）；这只排除范围错误，不能证明地理定位正确。设施距离计算须使用球面距离并标注米/千米，且明确站点去重和生效日期口径。

### COE 与股票：时间覆盖和泄漏边界

**COE 事实**：{coe['rows']} 行，覆盖 {coe['first_month']}–{coe['last_month']} 共 {coe['months']} 个月，类别为 {', '.join(coe['categories'])}（原文件无 d）。按“月×类别×轮次”无重复；若在全局月份内假定每类别每月两轮，缺 {coe['missing_round_cells_within_global_range']} 个格子，清单见 `coe_missing_month_category_rounds.csv`，不能据此断言数据错误（末月可能不完整）。美元符号和千分位逗号去掉后，四个数值列解析失败均为 0；解析后的范围见 `coe_numeric_parse_audit.csv`。`PREVAILING_QUOTA_PREMIUM` 的零值按类别为 {json.dumps(coe['zero_prevailing_by_category'], ensure_ascii=False)}，零不能未经确认直接当作经济意义上的零或统一填补。

**股票事实**：{stock['rows']} 行、{stock['symbols']} 个 symbol，{stock['first_date']}–{stock['last_date']} 共 {stock['months']} 个月，`SYMBOL + DATE` 无重复。每月 symbol 数为 {stock['monthly_symbol_min']}–{stock['monthly_symbol_max']}；{stock['symbols_with_all_global_months']} 个覆盖全部 {stock['months']} 个月，{stock['symbols_starting_after_train_end']} 个首条日期晚于训练结束月。价格各列和 VOLUME 各缺 {stock['missing_price_rows']} 行（相同行），日期分布为 {json.dumps(stock['missing_price_dates'], ensure_ascii=False)}；不能把周末/节假日的自然无交易记录等同缺失，逐 symbol 日历还需结合交易所。VOLUME=0 有 {stock['zero_volume_rows']} 行，应与未知区分。HIGH 低于 OPEN/CLOSE/LOW 的行数为 {stock['high_below_open_close_or_low']}，LOW 高于 OPEN/CLOSE/HIGH 的行数为 {stock['low_above_open_close_or_high']}，合计 {stock['ohlc_inconsistent_rows']} 个不同记录，见 `stock_ohlc_inconsistencies.csv`；原因待核查，未自动修复。

经济表覆盖与核心数据的关系：

{md_table(pd.DataFrame(e['temporal_relation']).T.reset_index(names='table'))}

**风险与待决策**：租赁只有批准月份，没有日。COE 两轮结果、当月股票收盘或月均不保证在该月某笔批准前已知，不能直接把当月汇总批准为无泄漏输入。若团队采用“每个月开始时滚动预测”，可测试仅用此前已经公布的滞后指标；还需核实发布时间，严禁反向填补。若任务定义为“2025-03 一次性预测至 2026-07”，则 2025-04 起实际观测的经济指标在该起点全部属于未来，简单滞后一个月也不够，须在每个验证截止点冻结可用历史。COE 2026-08 记录甚至晚于最后测试月，不得用于任何测试月特征。

股票 `ADJUSTED_CLOSE` 的历史复权值可能包含后续分红/拆股调整；其他价格是否做过拆股处理、公司名单是否按未来市值筛选，以及退市/新股带来的存活偏差，均不能由当前 CSV 证明或排除。严格时点回放应核对数据来源和 as-of 历史版本；当前阶段只记录风险，不从全期间选择“最佳股票”，不利用验证后走势选择特征。不同交易所、币种、上市日期及缺失模式也可能使简单等权平均缺少一致含义。

### 交给 B/C 的辅助数据事项

先使用本脚本保守主键规则和 many-to-one 校验，保留原始 CSV；将 HDB 哨兵值显式标记缺失。确认设施历史生效日期与换乘站计数口径，确认经济指标在长跨度固定起点和逐月滚动两种预测语境中的可用边界。楼栋/位置、历史可用设施、滞后经济指标都只是待测试假设，应在团队统一时间切分和 RMSE 下逐组消融；尚无任何预测效果结论。
"""
    (out / "report_section.md").write_text(text, encoding="utf-8")


def run(root: Path, output_dir: Path) -> dict:
    root, out = Path(root).resolve(), Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((root / "data" / "auxiliary").glob("*.csv"))
    source_paths = paths + [root / "data" / f"{s}.csv" for s in ["train", "test"]]
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    data = {p.name: pd.read_csv(p, low_memory=False) for p in paths}
    # No target is needed for the auxiliary audit; train labels are not loaded here.
    core = {s: pd.read_csv(root / "data" / f"{s}.csv", dtype={"BLOCK": str},
                          usecols=KEY + ["RENT_APPROVAL_DATE", "LEASE_COMMENCE_DATE"])
            for s in ["train", "test"]}
    result = {"files": profile(data, out), "hdb": hdb_audit(data["sg-hdb-block.csv"], core, out),
              "facilities": facility_audit(data, out), "economy": economy_audit(data, core, out),
              "normalization": "Unicode NFKC + strip + lower + collapse whitespace; no fuzzy rules",
              "versions": {"pandas": pd.__version__, "numpy": np.__version__},
              "input_sha256": before}
    after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    assert before == after, "Original CSV changed during audit"
    result["source_hashes_unchanged"] = True
    save_json(result, out / "summary.json")
    write_report(result, out)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    destination = args.output_dir or args.root / "experiments" / "eda" / "auxiliary"
    result = run(args.root, destination)
    print(json.dumps({"output_dir": str(destination.resolve()), "files": len(result["files"]),
                      "source_hashes_unchanged": result["source_hashes_unchanged"]}, ensure_ascii=False))
