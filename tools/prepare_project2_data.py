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
from typing import Any, Dict, List, Tuple

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


_FEATURE_COLS_12: List[str] = [
    "returns",
    "momentum_5",
    "momentum_20",
    "volatility_5",
    "volatility_20",
    "atr_norm",
    "bb_pos",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "volume_ratio",
]


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


def _load_aux(meta: Dict[str, Any]) -> pd.DataFrame:
    """Load an auxiliary data source (macro / funding / on-chain)."""
    fmt = meta.get("format", "csv")
    if fmt == "csv":
        df = pd.read_csv(meta["path"])
    elif fmt == "parquet":
        df = pd.read_parquet(meta["path"])
    else:
        raise ValueError(f"unsupported aux format: {fmt}")

    dt_col = meta["datetime_col"]
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=[dt_col])

    keep = [dt_col] + list(meta.get("columns", []))
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"aux source {meta['path']} missing columns {missing}")
    df = df[keep].rename(columns=meta.get("rename", {}))
    df = df.rename(columns={dt_col: "DATE_TIME"})
    df = df.sort_values("DATE_TIME").drop_duplicates(subset=["DATE_TIME"])
    return df.reset_index(drop=True)


def _merge_aux(
    df: pd.DataFrame,
    manifest: Dict[str, Any],
    source_key: str,
    preset_name: str,
    ffill_cap_days: int = 7,
) -> pd.DataFrame:
    """Merge aux columns onto `df` via forward-fill with a cap."""
    presets = manifest.get("feature_presets", {})
    preset = presets.get(preset_name, {})
    aux_sources = manifest.get("aux_sources", {})

    aux_keys: List[str] = list(preset.get("aux", []))
    aux_keys += list(preset.get("aux_by_source", {}).get(source_key, []))
    if not aux_keys:
        return df

    merged = df.copy().sort_values("DATE_TIME").reset_index(drop=True)
    cap = pd.Timedelta(days=ffill_cap_days)

    for aux_key in aux_keys:
        if aux_key not in aux_sources:
            raise KeyError(f"aux source '{aux_key}' not in manifest.aux_sources")
        aux_df = _load_aux(aux_sources[aux_key]).sort_values("DATE_TIME").reset_index(drop=True)
        merged = pd.merge_asof(
            merged,
            aux_df,
            on="DATE_TIME",
            direction="backward",
            tolerance=cap,
        )
    return merged


def _fit_scaler_and_transform(
    splits: Dict[str, pd.DataFrame],
    non_feature_cols: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Fit a StandardScaler on d4 feature cols and transform d4/d5/d6.

    Returns (normalized_splits, scaler_dict). The scaler_dict is JSON-safe.
    """
    from sklearn.preprocessing import StandardScaler

    d4 = splits.get("d4")
    if d4 is None or d4.empty:
        raise RuntimeError("--normalize requires a non-empty d4 split")

    feature_cols = [c for c in d4.columns if c not in non_feature_cols]
    # Drop columns that are all-NaN on d4 (they can't be normalized)
    feature_cols = [c for c in feature_cols if d4[c].notna().any()]

    scaler = StandardScaler()
    scaler.fit(d4[feature_cols].dropna().values)

    normed: Dict[str, pd.DataFrame] = {}
    for name, df in splits.items():
        if df is None or df.empty:
            normed[name] = df
            continue
        out = df.copy()
        values = out[feature_cols].values
        # Safe transform: replace NaN inputs with column mean (of d4) before scaling
        col_mean = d4[feature_cols].mean().values
        mask = np.isnan(values)
        if mask.any():
            # broadcast column means
            col_mean_b = np.broadcast_to(col_mean, values.shape).copy()
            values = np.where(mask, col_mean_b, values)
        out.loc[:, feature_cols] = scaler.transform(values)
        normed[name] = out

    scaler_dict = {
        "feature_cols": feature_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "fit_rows": int(len(d4)),
    }
    return normed, scaler_dict


def _prepare_one(
    source_key: str,
    manifest: Dict[str, Any],
    compute_features: bool,
    preset: str,
    normalize: bool,
) -> Dict[str, int]:
    meta = manifest["sources"][source_key]
    print(f"\n=== {source_key} (preset={preset}, normalize={normalize}) ===", flush=True)
    print(f"Loading {meta['path']}", flush=True)
    df = _load_source(meta)
    print(f"  rows={len(df)}  range=[{df['DATE_TIME'].min()} .. {df['DATE_TIME'].max()}]", flush=True)

    if compute_features:
        df = _compute_features(df)

    # Merge auxiliary data (macro / funding / on-chain) per preset
    if preset != "twelve":
        before_cols = set(df.columns)
        df = _merge_aux(df, manifest, source_key, preset)
        added = sorted(set(df.columns) - before_cols)
        print(f"  preset '{preset}' added columns: {added}", flush=True)

    splits = _split(df, meta["train_start"])
    out_dir = OUTPUT_ROOT / source_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop warmup NaNs introduced by rolling features (only in d4).
    if compute_features and "d4" in splits and not splits["d4"].empty:
        splits["d4"] = splits["d4"].dropna(subset=_FEATURE_COLS_12).reset_index(drop=True)

    counts: Dict[str, int] = {}
    for name, split_df in splits.items():
        if split_df is None or split_df.empty:
            print(f"  {name}: EMPTY (skipped)", flush=True)
            counts[name] = 0
            continue
        path = out_dir / f"{name}.csv"
        split_df.to_csv(path, index=False)
        counts[name] = len(split_df)
        print(
            f"  {name}: rows={len(split_df)} "
            f"range=[{split_df['DATE_TIME'].min()} .. {split_df['DATE_TIME'].max()}] "
            f"→ {path.relative_to(REPO_ROOT)}",
            flush=True,
        )

    if normalize:
        non_feat = ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        normed, scaler_dict = _fit_scaler_and_transform(splits, non_feat)
        for name, ndf in normed.items():
            if ndf is None or ndf.empty:
                continue
            npath = out_dir / f"{name}_norm.csv"
            ndf.to_csv(npath, index=False)
            print(f"  {name}_norm: {len(ndf)} rows → {npath.relative_to(REPO_ROOT)}", flush=True)
        scaler_path = out_dir / "scaler.json"
        with open(scaler_path, "w", encoding="utf-8") as fh:
            json.dump(scaler_dict, fh, indent=2)
        print(f"  scaler → {scaler_path.relative_to(REPO_ROOT)}", flush=True)

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="manifest source key (e.g. eurusd_1h)")
    parser.add_argument("--all", action="store_true", help="process every source in manifest")
    parser.add_argument("--compute_features", action="store_true", help="add the 12-feature set columns")
    parser.add_argument(
        "--features",
        default="twelve",
        choices=["twelve", "twelve_macro", "twelve_funding", "twelve_onchain"],
        help="feature preset (requires --compute_features for non-default)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="fit StandardScaler on d4 feature cols and emit d{4,5,6}_norm.csv + scaler.json",
    )
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if args.features != "twelve" and not args.compute_features:
        parser.error(f"--features {args.features} requires --compute_features")

    keys = list(manifest["sources"].keys()) if args.all else ([args.source] if args.source else [])
    if not keys:
        parser.error("specify --source <key> or --all")

    summary: Dict[str, Dict[str, int]] = {}
    for key in keys:
        if key not in manifest["sources"]:
            print(f"[error] unknown source: {key}", file=sys.stderr)
            return 2
        summary[key] = _prepare_one(key, manifest, args.compute_features, args.features, args.normalize)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
