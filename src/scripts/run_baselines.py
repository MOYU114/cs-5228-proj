"""Evaluate simple rent estimators on the stored BERT split and a train-only-cleaned split."""
import argparse
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from preprocess.common import DATA_DIR, TARGET, read_raw, read_table, save_json, write_table
from preprocess.stage1_normalize import run_stage1
from preprocess.stage2_enrich import run_stage2
from preprocess.stage3_encode import FEATURES, ID_COLUMNS, LOG1P_COLUMNS, run_stage3

DATE = "RENT_APPROVAL_DATE"
CUTOFF = "2024-03"
GROUP = ["TOWN_ID", "FLAT_TYPE_ORD"]
RIDGE = dict(alpha=10.0, solver="lsqr", tol=1e-6, max_iter=3000)
TREE = dict(iterations=500, depth=4, learning_rate=0.1, random_seed=42,
            loss_function="RMSE", max_ctr_complexity=1, thread_count=3,
            task_type="CPU", allow_writing_files=False, verbose=False)
ABLATIONS = {"full": [], "no_street_building": ["STREET_ID", "BUILDING_ID"],
             "no_facilities": LOG1P_COLUMNS,
             "no_explicit_time": ["MONTH_INDEX", "MONTH_SIN", "MONTH_COS", "AGE_YEARS", "COMPLETION_AGE"]}


def prepare(protocol, data, work, raw):
    train_mask = raw[DATE] <= CUTOFF
    work.mkdir(parents=True, exist_ok=False)
    source = data
    if protocol == "strict":
        # Let the existing pipeline treat validation as *unlabelled* test rows.
        # Its cleaning, flat-model vocabulary and scaling then fit only on the past.
        source = work / "raw_fold"
        source.mkdir()
        (source / "auxiliary").symlink_to(data / "auxiliary", target_is_directory=True)
        raw.loc[train_mask].to_csv(source / "train.csv", index=False)
        raw.loc[~train_mask].drop(columns=TARGET).to_csv(source / "test.csv", index=False)
    s1, s2, s3 = (work / f"stage{i}" for i in (1, 2, 3))
    for label, fn, args in [(1, run_stage1, (source, s1)), (2, run_stage2, (s1, s2)),
                            (3, run_stage3, (s1, s2, s3))]:
        print(f"{protocol}: preprocessing stage {label}", flush=True)
        fn(*args)
    encoded = read_table(s3, "train")
    if protocol == "strict":
        train, val = encoded, read_table(s3, "test")
        # The preprocessing ROW_ID points into the temporary fold CSV. Restore original IDs.
        train["ROW_ID"] = raw.index[train_mask].to_numpy()[train.ROW_ID]
        val["ROW_ID"] = raw.index[~train_mask].to_numpy()[val.ROW_ID]
        if not np.array_equal(val.ROW_ID, raw.index[~train_mask]):
            raise ValueError("Strict validation must cover every raw held-out row")
    else:
        before = encoded.ROW_ID.map(train_mask)
        train, val = encoded.loc[before].copy(), encoded.loc[~before].copy()
        import json
        reference = json.loads((ROOT / "results/split.json").read_text())
        if (len(train), len(val)) != (reference["train"]["rows"], reference["validation"]["rows"]):
            raise ValueError("Historical split differs from the recorded BERT run")
    for frame in (train, val):
        if not frame.ROW_ID.is_unique or not np.isfinite(frame[FEATURES]).all().all():
            raise ValueError("Duplicate IDs or nonfinite features")
        frame[DATE] = frame.ROW_ID.map(raw[DATE])
        frame.sort_values([DATE, "ROW_ID"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if train[DATE].max() != CUTOFF or val[DATE].min() != "2024-04" or val[DATE].max() != "2025-03":
        raise ValueError("Unexpected date range")
    val["recorded"] = pd.to_numeric(val.ROW_ID.map(raw[TARGET])).astype(float)
    val["building_seen"] = val.BUILDING_ID.isin(train.BUILDING_ID)
    return train, val


def recent_means(train, val):
    start = str(pd.Period(CUTOFF, freq="M") - 5)
    recent = train[train[DATE] >= start]
    groups = recent.groupby(GROUP)[TARGET].agg(["size", "mean"])
    usable = groups.loc[groups["size"] >= 10, "mean"]
    est = pd.Series(usable.reindex(pd.MultiIndex.from_frame(val[GROUP])).to_numpy(), index=val.index)
    types = val.FLAT_TYPE_ORD.map(recent.groupby("FLAT_TYPE_ORD")[TARGET].mean())
    counts = {"town_type": int(est.notna().sum()), "type": int((est.isna() & types.notna()).sum()),
              "global": int((est.isna() & types.isna()).sum())}
    return est.fillna(types).fillna(recent[TARGET].mean()).to_numpy(), counts


def predict(model, train, val, features):
    if model == "B0_mean":
        return np.full(len(val), train[TARGET].mean()), {}
    if model == "B1_recent":
        return recent_means(train, val)
    cats = [c for c in features if c in ID_COLUMNS]
    if model == "B2_ridge":
        transform = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore"), cats)],
                                      remainder="passthrough", sparse_threshold=1)
        estimator = make_pipeline(transform, Ridge(**RIDGE))
    else:
        estimator = CatBoostRegressor(**TREE, cat_features=cats)
    estimator.fit(train[features], train[TARGET])
    return estimator.predict(val[features]), {}


def score(rows):
    error = rows["Predicted"].to_numpy() - rows["recorded"].to_numpy()
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite prediction or target")
    return {"rows": len(error), "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))}


def render_monthly(output, monthly):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    scores = json.loads((ROOT / "results/validation_summary.json").read_text())["by_month"]
    for ax, protocol in zip(axes, ("historical", "strict")):
        data = monthly[(monthly.protocol == protocol) & (monthly.features == "full")]
        for model, group in data.groupby("model"):
            ax.plot(group[DATE], group.bias, marker=".", label=model)
        if protocol == "historical":
            ax.plot(sorted(scores), [scores[m]["mean_error"] for m in sorted(scores)], "--", label="BERT (recorded)")
        ax.axhline(0, color="0.35", linewidth=.8)
        ax.tick_params(axis="x", labelrotation=60)
        ax.set_title(protocol.capitalize())
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean predicted minus recorded rent (SGD/month)")
    fig.tight_layout()
    fig.savefig(output / "monthly_bias.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = read_raw(data / "train.csv")
    runs, monthly, buildings = [], [], []
    metadata = {"base_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "command": sys.argv, "python": sys.version, "versions": {p: version(p) for p in
                ["pandas", "numpy", "scikit-learn", "scipy", "catboost", "matplotlib"]},
                "cutoff": CUTOFF, "B1": {"months": 6, "min_group_rows": 10}, "ridge": RIDGE, "tree": TREE,
                "ablation_drops": ABLATIONS, "folds": {}}
    for protocol in ("historical", "strict"):
        train, val = prepare(protocol, data, args.work_dir / protocol, raw)
        metadata["folds"][protocol] = {"train_rows": len(train), "validation_rows": len(val)}
        print(f"{protocol}: {len(train)} train, {len(val)} validation", flush=True)
        for model in ("B0_mean", "B1_recent", "B2_ridge", "B3_catboost"):
            variants = ABLATIONS if protocol == "strict" and model in ("B2_ridge", "B3_catboost") else ["full"]
            for variant in variants:
                columns = [c for c in FEATURES if c not in ABLATIONS[variant]]
                start = perf_counter()
                estimate, detail = predict(model, train, val, columns)
                seconds = perf_counter() - start
                pred = val[["ROW_ID", DATE, "recorded", "building_seen"]].assign(Predicted=estimate)
                key = dict(protocol=protocol, model=model, features=variant)
                overall = score(pred)
                if protocol == "historical" and model == "B0_mean" and round(overall["rmse"], 1) != 842.5:
                    raise ValueError(f"B0 failed BERT reference check: {overall}")
                runs.append({**key, **overall, "fit_predict_seconds": seconds})
                monthly.extend({**key, DATE: m, **score(rows)} for m, rows in pred.groupby(DATE))
                buildings.extend({**key, "building_seen": seen, **score(rows)}
                                 for seen, rows in pred.groupby("building_seen"))
                if detail:
                    metadata["folds"][protocol]["B1_sources"] = detail
                write_table(output, f"predictions/{protocol}_{model}_{variant}", pred)
                print({**key, **overall, "seconds": round(seconds, 1)}, flush=True)
            for name, records in [("scores", runs), ("by_month", monthly), ("by_building_seen", buildings)]:
                write_table(output, name, pd.DataFrame(records))
            save_json(output / "manifest.json", metadata)
    scores = pd.DataFrame(runs)
    strict = scores[scores.protocol.eq("strict") & scores.model.isin(["B2_ridge", "B3_catboost"])].copy()
    full = strict[strict.features == "full"].set_index("model").rmse
    strict["delta_rmse_vs_full"] = strict.rmse - strict.model.map(full)
    write_table(output, "ablation", strict)
    render_monthly(output, pd.DataFrame(monthly))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
