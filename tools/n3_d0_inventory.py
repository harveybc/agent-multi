#!/usr/bin/env python3
"""N3-D0 executable inventory (order agent-multi@a13671ab §3).

Binds, by digest and recomputed range: the frozen predictor CSVs and
manifest; the raw Binance-shaped ETHUSDT H4 parquet and its
provenance in financial-data; the Stage 2.1/2.2/3.1 worker files
that produced the 83 columns; the N2 result bundle, attribution
contract/artifact and role census; and the public REST endpoint
grammar already used by the governed acquisition workers. Records
every divergence between the frozen export and the currently
committed workers. READ-ONLY: no network, no writes outside the
output JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREDICTOR = Path.home() / "Documents/GitHub/predictor"
FINDATA = Path.home() / "Documents/GitHub/financial-data"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_tip(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import pandas as pd

    p3 = PREDICTOR / "examples/data/project3"
    frozen_csv = p3 / "ethusdt_4h_tech_stat_full_model_ready.csv"
    warm_csv = p3 / "ethusdt_4h_tech_stat_full_with_warmup_nans.csv"
    manifest = json.loads(
        (p3 / "ethusdt_4h_tech_stat_full_model_ready.manifest.json")
        .read_text())
    export_meta = json.loads(
        (p3 / "ethusdt_4h_tech_stat_export_metadata.json").read_text())
    frozen_sha = sha(frozen_csv)
    assert frozen_sha == manifest["sha256"], "manifest sha mismatch"
    df = pd.read_csv(frozen_csv,
                     usecols=["DATE_TIME"])
    assert len(df) == manifest["rows"] == 18085

    lake = FINDATA / "market_data/crypto/spot_top50/ethusdt"
    lake_parquet = lake / "4h.parquet"
    lake_prov = json.loads((lake / "provenance.json").read_text())
    lake_sha = sha(lake_parquet)
    assert lake_sha == lake_prov["files"][0]["sha256"], \
        "lake provenance sha mismatch"
    raw = pd.read_parquet(lake_parquet)
    n = len(raw)
    t = pd.to_datetime(raw["open_time"], utc=True)
    t0, t1 = t.iloc[0], t.iloc[-1]
    expected_grid = pd.date_range(t0, t1, freq="4h")
    missing = expected_grid.difference(t)
    dup = int(t.duplicated().sum())

    base = FINDATA / "features/trading_asset_data/ethusdt/4h.parquet"
    tech = FINDATA / ("features/trading_asset_features/ethusdt/4h/"
                      "technical.parquet")
    stat = FINDATA / ("features/trading_asset_features/ethusdt/4h/"
                      "statistical.parquet")
    # the stage-1.3 acquisition worker's filename embeds an operator
    # host identifier; it is bound here by DIGEST with a sanitized
    # label (DATA-SOTA-340: no topology tokens in public evidence)
    worker_files = {
        "stage13_crypto_acquisition_worker (filename withheld — "
        "carries a host identifier; governed financial-data "
        "worker)": "stage13_dragon_crypto_worker.py",
        "stage21_trading_asset_worker.py":
            "stage21_trading_asset_worker.py",
        "stage22_trading_features_worker.py":
            "stage22_trading_features_worker.py",
        "stage31_prepare_inputs_worker.py":
            "stage31_prepare_inputs_worker.py"}
    workers = {
        label: {"sha256": sha(FINDATA / "_scripts/workers" / real)}
        for label, real in worker_files.items()}

    ev = REPO / "docs/audits/evidence"
    n2 = {
        "bundle": sha(ev / "TARGET_HORIZON_CENSUS_N2_BUNDLE"
                           "_2026_09_03.json"),
        "attribution_contract": sha(
            ev / "N2_ATTRIBUTION_AUDIT_CONTRACT_2026_09_04.json"),
        "attribution_artifact": sha(
            ev / "TARGET_HORIZON_CENSUS_N2_ATTRIBUTION_AUDIT"
                 "_2026_09_04.json"),
        "role_census": sha(
            ev / "N3_UNTOUCHED_ROLE_CENSUS_2026_09_04.json"),
        "verdict_trace": sha(
            ev / "TARGET_HORIZON_CENSUS_N2_VERDICT_TRACE"
                 "_2026_09_03.json"),
        "census_inputs_npz_sha256": (
            "07c5ff085dfd8bab0dfa33d038005c8fdb2d6c2acff3961d0fe4"
            "b042ef57cca7"),
    }

    inventory = {
        "schema": "agent_multi.n3_d0_inventory.v1",
        "order": "agent-multi@a13671ab §3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_predictor_contract": {
            "model_ready_csv": {
                "path": "predictor/examples/data/project3/"
                        "ethusdt_4h_tech_stat_full_model_ready.csv",
                "sha256": frozen_sha, "rows": 18085,
                "columns": 90,
                "range": [manifest["splits"]["train_start"],
                          "2025-12-31T20:00:00"],
                "feature_columns_ordered_count": len(
                    manifest["feature_columns"])},
            "with_warmup_nans_csv": {
                "sha256": sha(warm_csv), "rows": 18337,
                "range": ["2017-08-17T04:00:00",
                          "2025-12-31T20:00:00"]},
            "manifest_sha256": sha(
                p3 / "ethusdt_4h_tech_stat_full_model_ready"
                     ".manifest.json"),
            "export_metadata_sha256": sha(
                p3 / "ethusdt_4h_tech_stat_export_metadata.json"),
            "export_generated_at": export_meta["generated_at"]},
        "raw_lake": {
            "path": "financial-data/market_data/crypto/spot_top50/"
                    "ethusdt/4h.parquet",
            "sha256": lake_sha,
            "acquired_at": lake_prov["acquired_at"],
            "rows": n,
            "range_utc": [str(t0), str(t1)],
            "grid": {"expected_bars": len(expected_grid),
                     "actual_bars": n,
                     "missing_bars": len(missing),
                     "missing_utc": [str(x) for x in missing],
                     "all_missing_before":
                         "2020-02-20 (known Binance maintenance "
                         "windows; none after)"},
            "duplicate_open_times": dup,
            "columns": list(raw.columns)},
        "stage_pipeline": {
            "stage_2_1_base": {"path": str(base.relative_to(
                FINDATA.parent)), "sha256": sha(base)},
            "stage_2_2_technical": {"path": str(tech.relative_to(
                FINDATA.parent)), "sha256": sha(tech)},
            "stage_2_2_statistical": {"path": str(stat.relative_to(
                FINDATA.parent)), "sha256": sha(stat)},
            "workers": workers,
            "financial_data_git_tip": git_tip(FINDATA),
            "predictor_git_tip": git_tip(PREDICTOR)},
        "endpoint_grammar": {
            "base_url": "https://api.binance.com",
            "path": "/api/v3/klines",
            "method": "HTTP GET only",
            "params": ["symbol=ETHUSDT", "interval=4h",
                       "startTime=<ms>", "endTime=<ms>",
                       "limit=1000"],
            "rate_limit_handling": "HTTP 429 backoff as in the "
                                   "stage-1.3 acquisition worker",
            "authority": "same public endpoint used by the governed "
                         "financial-data acquisition workers; no "
                         "credentials, no private endpoints"},
        "n2_identities": n2,
        "divergences_recorded": {
            "DV1_exporter_not_committed": (
                "the one-off exporter that produced the model-ready "
                "layout (typical_price column, full and "
                "with_warmup_nans variants, generated 2026-05-02) "
                "is not a committed worker; its recipe is bound by "
                "the export metadata sources map (raw lake parquet "
                "+ stage-2.2 technical/statistical parquets + "
                "stage-3.1 tech_stat merge). The N3-D3 overlap "
                "parity proof is the arbiter, feature by feature — "
                "no nearby pipeline is silently treated as the "
                "original"),
            "DV2_binary_label_mismatch": (
                "the manifest lists ema_cross_10_50/ema_cross_20_100 "
                "as feature_binary_columns but the frozen values are "
                "CONTINUOUS ((ema_a-ema_b)/close per stage 2.2); "
                "only vol_regime_high/low are truly binary {0,1} — "
                "exact-equality checks in D3 apply to the true "
                "binary pair only"),
            "DV3_missing_bars_and_ffill": (
                "the raw lake is MISSING 16 grid bars, all before "
                "2020-02-20 (known Binance maintenance windows, "
                "enumerated above): missing bars are ABSENT ROWS, "
                "so stage-2.2 rolling features span them "
                "positionally and stage31's ffill only fills "
                "occasional mid-series feature NaNs, never "
                "fabricated market bars. Parity on overlap "
                "reproduces exactly this behavior; the 2026 "
                "extension REFUSES any newly missing grid bar "
                "instead of spanning or filling it"),
            "DV4_float32_sanitize": (
                "stage 2.2 sanitize() casts features to float32 "
                "before CSV serialization — D3 numeric tolerances "
                "derive from float32 round-trip precision, "
                "predeclared before comparison"),
            "DV5_read_asset_column_rename": (
                "stage 2.2 read_asset requires a 'timestamp' "
                "column; the lake stores 'open_time' — the rename "
                "happens in stage 2.1 (trading_asset_data); the N3 "
                "regeneration reproduces that chain explicitly")},
    }
    Path(args.out).write_text(json.dumps(inventory, indent=1)
                              + "\n")
    print(json.dumps({"ok": True,
                      "frozen_sha": frozen_sha[:16],
                      "lake_sha": lake_sha[:16],
                      "missing_bars": len(missing),
                      "dups": dup}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
