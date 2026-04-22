#!/usr/bin/env python3
"""
prepare_project2_data.py — materialize HO-safe split CSVs from the
project-2 data lake into agent-multi/data/project2/<source>/{d4,d5,d6}.csv.

Each output CSV has columns DATE_TIME, OPEN, HIGH, LOW, CLOSE, VOLUME
plus the 12-feature set from II-7.3 if --compute_features is passed.

Hard invariant: d4 (train) never includes any row with DATE_TIME >=
2020-01-01. A negative self-test asserts this before writing.

Usage:
    python tools/prepare_project2_data.py --source eurusd_1h
    python tools/prepare_project2_data.py --source btcusdt_1h --compute_features
    python tools/prepare_project2_data.py --all --compute_features
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "project2_manifest.json"
OUTPUT_ROOT = REPO_ROOT / "data" / "project2"


def _load_source(meta: Dict[str, Any]) -> pd.DataFrame:
    path = meta["path"]
    fmt = meta["format"]
    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported format: {fmt}")

    dt_col = meta["datetime_col"]
    if dt_col not in df.columns:
        # parquet with DatetimeIndex — reset
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": dt_col})
        else:
            raise KeyError(f"datetime column '{dt_col}' not in {path}")

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=[dt_col])
    df = df.rename(columns=meta.get("rename", {}))
    df = df.rename(columns={dt_col: "DATE_TIME"})

    for col in ("OPEN", "HIGH", "LOW", "CLOSE"):
        if col not in df.columns:
            raise KeyError(f"missing column {col} in {path}")
    if "VOLUME" not in df.columns:
        df["VOLUME"] = 0.0

    df = df[["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
    df = df.sort_values("DATE_TIME").drop_duplicates(subset=["DATE_TIME"]).reset_index(drop=True)
    return df


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 12-feature set from II-7.3 (mean-reversion validated)."""
    out = df.copy()
    close = out["CLOSE"].astype(float)
    high = out["HIGH"].astype(float)
    low = out["LOW"].astype(float)
    volume = out["VOLUME"].astype(float).replace(0, np.nan)

    out["returns"] = close.pct_change()
    out["momentum_5"] = close.pct_change(5)
    out["momentum_20"] = close.pct_change(20)
    out["volatility_5"] = out["returns"].rolling(5).std()
    out["volatility_20"] = out["returns"].rolling(20).std()

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    out["atr_norm"] = atr / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    out["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal

    vol_mean = volume.rolling(20).mean()
    out["volume_ratio"] = (volume / vol_mean).fillna(1.0)

    return out


def _split(df: pd.DataFrame, train_start: str) -> Dict[str, pd.DataFrame]:
    ho = pd.Timestamp("2020-01-01")
    val_end = pd.Timestamp("2022-12-31 23:59:59")
    test_start = pd.Timestamp("2023-01-01")
    test_end = pd.Timestamp("2025-12-31 23:59:59")
    t0 = pd.Timestamp(train_start)

    d4 = df[(df["DATE_TIME"] >= t0) & (df["DATE_TIME"] < ho)].copy()
    d5 = df[(df["DATE_TIME"] >= ho) & (df["DATE_TIME"] <= val_end)].copy()
    d6 = df[(df["DATE_TIME"] >= test_start) & (df["DATE_TIME"] <= test_end)].copy()

    # Hard guard: train must not touch HO
    if not d4.empty and d4["DATE_TIME"].max() >= ho:
        raise RuntimeError(
            f"HO contamination: d4 train contains rows >= {ho.isoformat()}"
        )
    return {"d4": d4, "d5": d5, "d6": d6}


def _prepare_one(source_key: str, manifest: Dict[str, Any], compute_features: bool) -> Dict[str, int]:
    meta = manifest["sources"][source_key]
    print(f"\n=== {source_key} ===", flush=True)
    print(f"Loading {meta['path']}", flush=True)
    df = _load_source(meta)
    print(f"  rows={len(df)}  range=[{df['DATE_TIME'].min()} .. {df['DATE_TIME'].max()}]", flush=True)

    if compute_features:
        df = _compute_features(df)

    splits = _split(df, meta["train_start"])
    out_dir = OUTPUT_ROOT / source_key
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for name, split_df in splits.items():
        if split_df.empty:
            print(f"  {name}: EMPTY (skipped)", flush=True)
            counts[name] = 0
            continue
        # Drop warmup NaNs introduced by rolling features (only in training slice
        # to avoid leaking across boundaries)
        if compute_features and name == "d4":
            split_df = split_df.dropna().reset_index(drop=True)
        path = out_dir / f"{name}.csv"
        split_df.to_csv(path, index=False)
        counts[name] = len(split_df)
        print(
            f"  {name}: rows={len(split_df)} "
            f"range=[{split_df['DATE_TIME'].min()} .. {split_df['DATE_TIME'].max()}] "
            f"→ {path.relative_to(REPO_ROOT)}",
            flush=True,
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="manifest source key (e.g. eurusd_1h)")
    parser.add_argument("--all", action="store_true", help="process every source in manifest")
    parser.add_argument("--compute_features", action="store_true", help="add the 12-feature set columns")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    keys = list(manifest["sources"].keys()) if args.all else ([args.source] if args.source else [])
    if not keys:
        parser.error("specify --source <key> or --all")

    summary: Dict[str, Dict[str, int]] = {}
    for key in keys:
        if key not in manifest["sources"]:
            print(f"[error] unknown source: {key}", file=sys.stderr)
            return 2
        summary[key] = _prepare_one(key, manifest, args.compute_features)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
