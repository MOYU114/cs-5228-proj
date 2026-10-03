"""Shared helpers for the three preprocessing stages: paths, text normalization, validation and I/O."""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# __file__ lives in <workspace>/cs5228-proj/preprocess/.
WORKSPACE = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE / "data"
OUTPUT_DIR = WORKSPACE / "data_preprocess"
SPLITS = ("train", "test")
TARGET = "MONTHLY_RENT"
KEY = ["TOWN", "BLOCK", "STREET"]


def normalize_text(value: object) -> object:
    """NFKC, trim, lowercase and collapse whitespace; nulls and blanks become missing."""
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).strip().lower())
    return text if text else pd.NA


def reject(mask, where: str, message: str) -> None:
    """Raise with the zero-based positions of offending rows (equal to ROW_ID for train/test)."""
    positions = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(positions):
        raise ValueError(f"{where}: {message}; {len(positions)} row(s), first positions: {positions[:20].tolist()}")


def numbers(text: pd.Series, where: str, *, integer: bool = False, positive: bool = False,
            nullable: bool = False) -> pd.Series:
    """Parse a text column into float64; anything unparseable, infinite or out of contract is rejected."""
    present = text.notna().to_numpy()
    raw = pd.Series(text.to_numpy(dtype=object, na_value=None), index=text.index, dtype=object)
    values = pd.to_numeric(raw, errors="coerce").astype("float64")
    finite = np.isfinite(values.to_numpy())
    reject(present & ~finite, where, f"{text.name} must be finite numbers")
    if not nullable:
        reject(~present, where, f"{text.name} must not be missing")
    if integer:
        reject(finite & (values.to_numpy() % 1 != 0), where, f"{text.name} must be whole numbers")
    if positive:
        reject(finite & (values.to_numpy() <= 0), where, f"{text.name} must be positive")
    return values


def check_output_dir(output_dir: Path, *inputs: Path) -> None:
    """Refuse to write into the original data or into a directory this stage reads from."""
    output_dir = Path(output_dir).resolve()
    for protected in {DATA_DIR.resolve(), *(Path(p).resolve() for p in inputs)}:
        if output_dir == protected or protected in output_dir.parents:
            raise ValueError(f"Output directory {output_dir} must be outside input {protected}")


def atomic_write(path: Path, writer) -> None:
    """Write through a sibling temp file, which inherits normal Windows ACLs unlike mkstemp/mkdtemp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Refusing to replace non-regular output: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path: Path, value) -> None:
    atomic_write(path, lambda p: p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                              encoding="utf-8"))


def write_table(folder: Path, name: str, frame: pd.DataFrame, float_format: str | None = None) -> None:
    atomic_write(folder / f"{name}.csv", lambda p: frame.to_csv(p, index=False, encoding="utf-8", lineterminator="\n",
                                                                float_format=float_format))


def dtypes(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}


def read_raw(path: Path) -> pd.DataFrame:
    """Read an original CSV entirely as text; only empty fields count as missing."""
    return pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[""], encoding="utf-8")


def read_table(folder: Path, name: str) -> pd.DataFrame:
    """Reload a stage output with the dtypes recorded in that stage's schema.json (BLOCK stays text)."""
    schema = json.loads((folder / "schema.json").read_text(encoding="utf-8"))
    if name not in schema["tables"]:
        raise ValueError(f"{folder / 'schema.json'} does not describe {name}")
    frame = pd.read_csv(folder / f"{name}.csv", dtype=schema["tables"][name], keep_default_na=False,
                        na_values=[""], encoding="utf-8")
    if list(frame.columns) != list(schema["tables"][name]):
        raise ValueError(f"{folder / name}.csv columns differ from schema.json")
    return frame


def read_splits(folder: Path) -> dict[str, pd.DataFrame]:
    frames = {split: read_table(folder, split) for split in SPLITS}
    for split, frame in frames.items():
        # Train has gaps where stage 1 removed rent outliers; test rows must all stay, in submission order.
        ids = frame["ROW_ID"]
        if not (ids.is_unique and ids.is_monotonic_increasing):
            raise ValueError(f"{split}: ROW_ID must be unique and in original row order")
        if split == "test" and not np.array_equal(ids.to_numpy(), np.arange(len(frame))):
            raise ValueError("test: every original row must be present")
    if TARGET in frames["test"]:
        raise ValueError("Test targets are forbidden")
    return frames
