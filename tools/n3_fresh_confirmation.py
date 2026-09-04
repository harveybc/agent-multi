#!/usr/bin/env python3
"""N3_FRESH_DATA_CONFIRMATION (order agent-multi@a13671ab §§4-8;
contract sealed in N3_FRESH_CONFIRMATION_CONTRACT_2026_09_04.json
BEFORE the first network request and before any 2026 target/score).

Subcommands:
  acquire     bounded public GET of ETHUSDT 4h klines (overlap+2026)
              into a NEW restricted staging root, with receipts,
              validation and field-exact overlap continuity.
  regenerate  full-history feature regeneration through the bound
              Stage 2.1/2.2/3.1 chain + overlap parity proof.
  execute     the five frozen arms on the four 2026 blocks (CPU),
              bundle + decision trace.
  verify      offline verifier: rederives the decision from the
              bundle alone and refuses the ten ordered adversaries.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import sha_file, sha_obj  # noqa: E402
from agent_plugins.paired_inference import holm_adjust  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "target_horizon_census_n2",
    REPO / "tools" / "target_horizon_census_n2.py")
tcn2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcn2)

FINDATA = Path.home() / "Documents/GitHub/financial-data"
PREDICTOR = Path.home() / "Documents/GitHub/predictor"
CONTRACT = ("docs/audits/evidence/"
            "N3_FRESH_CONFIRMATION_CONTRACT_2026_09_04.json")
LAKE_PARQUET = (FINDATA
                / "market_data/crypto/spot_top50/ethusdt/4h.parquet")
LAKE_SHA = ("7a6b79833355d7c22a3db30e6494ced078d628338d76e671af320"
            "a06b35fc9e5")
FROZEN_CSV = (PREDICTOR / "examples/data/project3/"
              "ethusdt_4h_tech_stat_full_model_ready.csv")
FROZEN_SHA = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc7357476"
              "28f8d0435ebe440f")
BAR_MS = 14_400_000
H_MAX = 12
STRIDE = 4
WINDOW = 64
BOOT_SEED = 707
BOOT_B = 2000
BLOCK_LEN = 6
MARGIN_SCALE = 0.01
MARGIN_REPR = 0.005
SUPPORT_MIN = 15
TARGETS = {"bar_h6": 6, "bar_h12": 12}
ARMS = ("arm1", "arm2", "arm3", "arm4", "arm5")
CONTRAST_FAMILY = (("arm2", "arm1"), ("arm3", "arm2"),
                   ("arm4", "arm1"), ("arm5", "arm2"))
REPRESENTATION = {("arm3", "arm2"), ("arm4", "arm1"),
                  ("arm5", "arm2")}
BLOCKS = (("B1_JanFeb", "2026-01-01 00:00", "2026-02-28 20:00", 354),
          ("B2_MarApr", "2026-03-01 00:00", "2026-04-30 20:00", 366),
          ("B3_MayJun", "2026-05-01 00:00", "2026-06-30 20:00", 366),
          ("B4_JulAug", "2026-07-01 00:00", "2026-08-31 20:00", 372))
ROLE_FIT = ("2017-09-28 04:00", "2024-12-31 20:00")
ROLE_CAL = ("2025-01-01 00:00", "2025-12-31 20:00")
CONF_START = "2026-01-01 00:00"
CONF_END = "2026-08-31 20:00"


class FreshRefusal(ValueError):
    """Typed refusal for any D2-D4 boundary violation."""


def _utc(ts: str):
    return datetime.strptime(ts, "%Y-%m-%d %H:%M").replace(
        tzinfo=timezone.utc)


def role_ledger() -> dict:
    return {"schema": "agent_multi.n3_role_ledger.v1",
            "roles": {"history_fit": list(ROLE_FIT),
                      "calibration": list(ROLE_CAL),
                      "confirmation": [CONF_START, CONF_END]},
            "blocks": {name: [start, end, bars]
                       for name, start, end, bars in BLOCKS},
            "purge_bars": H_MAX, "stride": STRIDE,
            "window": WINDOW}


def scoring_anchor_offsets(block_bars: int) -> list:
    """Stride-4 offsets from the block's first bar, tail-purged so
    labels (a, a+12] stay inside the block."""
    return [i for i in range(0, block_bars, STRIDE)
            if i + H_MAX < block_bars]


def decide(contrast_stats: dict, blocks_complete: bool,
           licenses_ok: bool) -> str:
    """Sealed decision table, pure and total. contrast_stats:
    {(target, a, b): {pooled_skill, all_blocks_positive,
    holm_p}}."""
    if not blocks_complete:
        return "FRESH_CONFIRMATION_INSUFFICIENT"
    if not licenses_ok:
        return "FRESH_CONFIRMATION_INCONCLUSIVE"

    def ok(key, margin):
        s = contrast_stats[key]
        return (s["all_blocks_positive"]
                and s["pooled_skill"] >= margin
                and s["holm_p"] < 0.05)
    if any(ok((t, a, b), MARGIN_REPR)
           for t in TARGETS for (a, b) in REPRESENTATION):
        return "INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA"
    if all(ok((t, "arm2", "arm1"), MARGIN_SCALE) for t in TARGETS):
        return ("TARGET_SCALE_EFFECT_CONFIRMED_NO_REPRESENTATION"
                "_SIGNAL")
    return "TARGET_SCALE_EFFECT_NOT_CONFIRMED"


# ------------------------------------------------------------------ #
# D2: bounded acquisition                                            #
# ------------------------------------------------------------------ #

def acquire(staging: Path) -> dict:
    import pandas as pd
    import requests
    staging.mkdir(parents=True, exist_ok=True)
    start_ms = int(_utc("2025-01-01 00:00").timestamp() * 1000)
    end_open_ms = int(_utc(CONF_END).timestamp() * 1000)
    acquired_at = datetime.now(timezone.utc)
    if end_open_ms + BAR_MS >= acquired_at.timestamp() * 1000:
        raise FreshRefusal("terminal confirmation bar not yet "
                           "closed at acquisition time")
    receipts = []
    rows = []
    cursor = start_ms
    page = 0
    while cursor <= end_open_ms:
        params = {"symbol": "ETHUSDT", "interval": "4h",
                  "startTime": cursor,
                  "endTime": end_open_ms + BAR_MS - 1,
                  "limit": 1000}
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params=params, timeout=30)
        except Exception as exc:
            raise FreshRefusal(
                f"PUBLIC_DATA_ACQUISITION_BLOCKED: {exc}") from exc
        if resp.status_code == 429:
            time.sleep(10)
            continue
        if resp.status_code != 200:
            raise FreshRefusal(
                f"PUBLIC_DATA_ACQUISITION_BLOCKED: HTTP "
                f"{resp.status_code}")
        raw = resp.content
        payload = json.loads(raw)
        if not payload:
            break
        page_path = staging / f"page_{page:03d}.json"
        page_path.write_bytes(raw)
        receipts.append({
            "page": page,
            "request": {"symbol": "ETHUSDT", "interval": "4h",
                        "startTime": params["startTime"],
                        "endTime": params["endTime"],
                        "limit": 1000},
            "status": resp.status_code,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "acquired_at_utc": datetime.now(
                timezone.utc).isoformat(),
            "first_open_time": payload[0][0],
            "last_open_time": payload[-1][0],
            "n_rows": len(payload)})
        for k in payload:
            if len(k) != 12:
                raise FreshRefusal("schema drift: kline row does "
                                   f"not have 12 fields: {len(k)}")
            rows.append(k)
        last_open = payload[-1][0]
        cursor = last_open + BAR_MS
        page += 1
        time.sleep(0.4)
    # ---- validation ----
    opens = [r[0] for r in rows]
    if len(set(opens)) != len(opens):
        raise FreshRefusal("duplicate open_time in acquisition")
    for prev, cur in zip(opens, opens[1:]):
        if cur - prev != BAR_MS:
            raise FreshRefusal(
                f"grid gap/overlap between {prev} and {cur}")
    if opens[0] != start_ms:
        raise FreshRefusal("acquisition does not start at the "
                           "contract start bar")
    if opens[-1] != end_open_ms:
        raise FreshRefusal(
            "confirmation interval incomplete: last acquired open "
            f"{opens[-1]} != {end_open_ms}")
    for r in rows:
        o, h, low, c = (float(r[1]), float(r[2]), float(r[3]),
                        float(r[4]))
        vol = float(r[5])
        if not all(map(math.isfinite, (o, h, low, c, vol))):
            raise FreshRefusal("non-finite OHLCV")
        if min(o, h, low, c) <= 0 or vol < 0:
            raise FreshRefusal("non-positive OHLC or negative "
                               "volume")
        if h < max(o, c) or low > min(o, c):
            raise FreshRefusal("invalid OHLC geometry")
        if r[6] != r[0] + BAR_MS - 1:
            raise FreshRefusal("close_time != open_time + 4h - 1ms")
        if r[6] >= acquired_at.timestamp() * 1000:
            raise FreshRefusal("partially open terminal bar")
    # ---- overlap continuity vs the frozen lake ----
    if sha_file(LAKE_PARQUET) != LAKE_SHA:
        raise FreshRefusal("frozen lake parquet digest changed")
    lake = pd.read_parquet(LAKE_PARQUET)
    lake_ms = (pd.to_datetime(lake["open_time"], utc=True)
               .astype("int64") // 10 ** 6)
    lake_2025 = lake[lake_ms >= start_ms].reset_index(drop=True)
    overlap = [r for r in rows if r[0] <= int(lake_ms.iloc[-1])]
    if len(overlap) != len(lake_2025):
        raise FreshRefusal(
            "SOURCE_CONTINUITY_NOT_DEMONSTRATED: overlap row count "
            f"{len(overlap)} != frozen {len(lake_2025)}")
    field_map = [(1, "open"), (2, "high"), (3, "low"),
                 (4, "close"), (5, "volume"), (7, "quote_volume"),
                 (9, "taker_buy_base_volume"),
                 (10, "taker_buy_quote_volume")]
    for i, r in enumerate(overlap):
        if r[0] != int(lake_ms.iloc[i]):
            raise FreshRefusal(
                "SOURCE_CONTINUITY_NOT_DEMONSTRATED: timestamp "
                f"order mismatch at overlap row {i}")
        for j, col in field_map:
            if float(r[j]) != float(lake_2025[col].iloc[i]):
                raise FreshRefusal(
                    "SOURCE_CONTINUITY_NOT_DEMONSTRATED: field "
                    f"{col} revised at open_time {r[0]}: api "
                    f"{r[j]} vs frozen {lake_2025[col].iloc[i]}")
        if int(overlap[i][8]) != int(lake_2025["trade_count"]
                                     .iloc[i]):
            raise FreshRefusal(
                "SOURCE_CONTINUITY_NOT_DEMONSTRATED: trade_count "
                f"revised at {r[0]}")
    receipt = {"schema": "agent_multi.n3_acquisition_receipt.v1",
               "contract": CONTRACT,
               "acquired_at_utc": acquired_at.isoformat(),
               "source_identity": "Binance Spot public "
                                  "/api/v3/klines",
               "pages": receipts,
               "rows_total": len(rows),
               "rows_overlap_verified_exact": len(overlap),
               "rows_2026": len(rows) - len(overlap),
               "last_closed_open_time": opens[-1],
               "verdict": "SOURCE_CONTINUITY_DEMONSTRATED"}
    (staging / "acquisition_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    table = pd.DataFrame(
        rows, columns=["open_time", "open", "high", "low", "close",
                       "volume", "close_time", "quote_volume",
                       "trade_count", "taker_buy_base_volume",
                       "taker_buy_quote_volume", "ignore"])
    table.to_parquet(staging / "acquired.parquet")
    receipt["acquired_parquet_sha256"] = sha_file(
        staging / "acquired.parquet")
    (staging / "acquisition_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    return receipt


# ------------------------------------------------------------------ #
# D3: full-history regeneration + overlap parity                     #
# ------------------------------------------------------------------ #

def _load_stage22():
    spec = importlib.util.spec_from_file_location(
        "stage22", FINDATA / "_scripts/workers/"
        "stage22_trading_features_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def regenerate(staging: Path) -> dict:
    import numpy as np
    import pandas as pd
    stage22 = _load_stage22()
    lake = pd.read_parquet(LAKE_PARQUET)
    acq = pd.read_parquet(staging / "acquired.parquet")
    acq_ms = pd.to_numeric(acq["open_time"])
    lake_last_ms = int(
        (pd.to_datetime(lake["open_time"], utc=True)
         .astype("int64") // 10 ** 6).iloc[-1])
    ext = acq[acq_ms > lake_last_ms].copy()
    ext["open_time"] = pd.to_datetime(
        pd.to_numeric(ext["open_time"]), unit="ms", utc=True)
    ext["close_time"] = pd.to_datetime(
        pd.to_numeric(ext["close_time"]), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base_volume",
                "taker_buy_quote_volume"):
        ext[col] = pd.to_numeric(ext[col])
    ext["trade_count"] = pd.to_numeric(ext["trade_count"])
    full_raw = pd.concat(
        [lake, ext[lake.columns]], ignore_index=True)
    # stage 2.1 semantics: open_time -> timestamp, sorted, deduped,
    # numeric OHLCV (mirrors stage22.read_asset)
    base = full_raw.rename(columns={"open_time": "timestamp"})
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    base = (base.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True))
    for col in ("open", "high", "low", "close", "volume"):
        base[col] = pd.to_numeric(base[col], errors="coerce")
    # extension refuses any newly missing grid bar
    t26 = base["timestamp"]
    conf = base[(t26 >= _utc(CONF_START))
                & (t26 <= _utc(CONF_END))]
    if len(conf) != 1458:
        raise FreshRefusal(
            f"confirmation grid incomplete: {len(conf)} != 1458")
    tech = stage22.compute_technical(base)
    stat = stage22.compute_statistical(base)
    stat = stat.rename(columns={"log_return_1":
                                "statistical__log_return_1"})
    merged = pd.DataFrame({
        "DATE_TIME": base["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"),
        "typical_price": ((base["high"] + base["low"]
                           + base["close"]) / 3),
        "OPEN": base["open"], "HIGH": base["high"],
        "LOW": base["low"], "CLOSE": base["close"],
        "VOLUME": base["volume"]})
    frozen_cols = json.loads(
        (PREDICTOR / "examples/data/project3/"
         "ethusdt_4h_tech_stat_export_metadata.json").read_text()
    )["columns"]
    feature_cols = [c for c in frozen_cols
                    if c not in merged.columns]
    for col in feature_cols:
        if col in tech.columns:
            merged[col] = tech[col].to_numpy()
        elif col in stat.columns:
            merged[col] = stat[col].to_numpy()
        else:
            raise FreshRefusal(f"feature {col} produced by neither "
                               "stage-2.2 table")
    merged = merged[frozen_cols]
    numeric = [c for c in frozen_cols if c != "DATE_TIME"]
    merged[numeric] = merged[numeric].replace(
        [np.inf, -np.inf], np.nan).ffill()
    model_ready = merged.dropna(
        subset=[c for c in feature_cols]).reset_index(drop=True)
    # ---- overlap parity ----
    if sha_file(FROZEN_CSV) != FROZEN_SHA:
        raise FreshRefusal("frozen model-ready CSV digest changed")
    frozen = pd.read_csv(FROZEN_CSV)
    n_over = len(frozen)
    regen_over = model_ready.iloc[:n_over]
    if list(regen_over["DATE_TIME"]) != list(frozen["DATE_TIME"]):
        raise FreshRefusal("SOURCE_OR_PIPELINE_DRIFT: DATE_TIME "
                           "sequence mismatch on overlap")
    if list(model_ready.columns) != list(frozen.columns):
        raise FreshRefusal("SOURCE_OR_PIPELINE_DRIFT: column order "
                           "mismatch")
    exact_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
                  "vol_regime_high", "vol_regime_low"]
    parity = {}
    drift = []
    for col in [c for c in frozen_cols if c != "DATE_TIME"]:
        a = regen_over[col].to_numpy(dtype="float64")
        b = frozen[col].to_numpy(dtype="float64")
        if col in exact_cols:
            ok = bool(np.array_equal(a, b))
            parity[col] = {"class": "exact", "equal": ok}
            if not ok:
                drift.append(col)
            continue
        a32 = a.astype("float32")
        b32 = b.astype("float32")
        bit = a32 == b32
        both_nan = np.isnan(a) & np.isnan(b)
        mism = ~(bit | both_nan)
        absdev = np.abs(a - b)
        with np.errstate(divide="ignore", invalid="ignore"):
            reldev = absdev / np.maximum(np.abs(b), 1e-30)
        inside = (absdev <= 1e-6) | (reldev <= 1e-5)
        bad = mism & ~inside & ~both_nan
        parity[col] = {
            "class": "float32_envelope",
            "bit_exact_frac": round(
                float((bit | both_nan).mean()), 6),
            "max_abs_dev": float(np.nanmax(absdev))
            if len(absdev) else 0.0,
            "max_rel_dev": float(np.nanmax(reldev))
            if len(reldev) else 0.0,
            "cells_outside_envelope": int(bad.sum())}
        if bad.sum():
            drift.append(col)
    verdict = ("SOURCE_OR_PIPELINE_DRIFT" if drift
               else "OVERLAP_PARITY_DEMONSTRATED")
    report = {"schema": "agent_multi.n3_parity_report.v1",
              "contract": CONTRACT,
              "overlap_rows": n_over,
              "regenerated_rows": len(model_ready),
              "rows_2026": len(model_ready) - n_over,
              "verdict": verdict,
              "drifted_features": drift,
              "per_feature": parity}
    (staging / "parity_report.json").write_text(
        json.dumps(report, indent=1))
    if drift:
        raise FreshRefusal(f"SOURCE_OR_PIPELINE_DRIFT: {drift[:5]}")
    model_ready.to_parquet(staging / "model_ready_extended.parquet")
    report["extended_sha256"] = sha_file(
        staging / "model_ready_extended.parquet")
    (staging / "parity_report.json").write_text(
        json.dumps(report, indent=1))
    return report


# ------------------------------------------------------------------ #
# shared contrast rederivation (single bootstrap implementation)     #
# ------------------------------------------------------------------ #

def _rederive(units):
    """Contrasts, bootstrap p, Holm and decision inputs derived
    purely from unit payloads. Returns (contrasts_out,
    contrast_stats, complete)."""
    import numpy as np
    pvals, stats = {}, {}
    complete = True
    for tkey in TARGETS:
        tunits = {u["block"]: u for u in units
                  if u["unit"].startswith(tkey)}
        for (a, b) in CONTRAST_FAMILY:
            per_block = {}
            diffs = []
            for name, _, _, _ in BLOCKS:
                u = tunits.get(name)
                if u is None or a not in u.get("arms", {}) \
                        or b not in u.get("arms", {}):
                    complete = False
                    continue
                la = np.asarray(u["arms"][a]["per_obs_logloss"])
                lb = np.asarray(u["arms"][b]["per_obs_logloss"])
                per_block[name] = round(
                    1.0 - float(la.sum() / lb.sum()), 6)
                diffs.append(lb - la)
            if len(per_block) < 4:
                complete = False
                continue
            pooled = round(1.0 - float(
                sum(np.asarray(
                    tunits[nm]["arms"][a]["per_obs_logloss"]).sum()
                    for nm, _, _, _ in BLOCKS)
                / sum(np.asarray(
                    tunits[nm]["arms"][b]["per_obs_logloss"]).sum()
                    for nm, _, _, _ in BLOCKS)), 6)
            rng = np.random.default_rng(BOOT_SEED)
            n_low = 0
            for _ in range(BOOT_B):
                parts = []
                for d in diffs:
                    m = len(d)
                    n_blocks = math.ceil(m / BLOCK_LEN)
                    starts = rng.integers(0, m, size=n_blocks)
                    idx = (starts[:, None]
                           + np.arange(BLOCK_LEN)[None, :]
                           ).reshape(-1) % m
                    parts.append(d[idx[:m]])
                if float(np.concatenate(parts).mean()) <= 0.0:
                    n_low += 1
            p = (1 + n_low) / (BOOT_B + 1)
            ckey = f"{tkey}:{a}-vs-{b}"
            pvals[ckey] = min(1.0, p)
            stats[(tkey, a, b)] = {
                "pooled_skill": pooled,
                "per_block_skill": per_block,
                "all_blocks_positive": all(
                    v > 0 for v in per_block.values()),
                "bootstrap_p": ("<= 1/2001"
                                if p <= 1 / (BOOT_B + 1) + 1e-12
                                else round(p, 6))}
    holm = holm_adjust(pvals) if pvals else {}
    contrast_stats, contrasts_out = {}, {}
    for (tkey, a, b), s in stats.items():
        ckey = f"{tkey}:{a}-vs-{b}"
        s["holm_p"] = round(holm[ckey], 6)
        contrast_stats[(tkey, a, b)] = {
            "pooled_skill": s["pooled_skill"],
            "all_blocks_positive": s["all_blocks_positive"],
            "holm_p": s["holm_p"]}
        contrasts_out[ckey] = s
    return contrasts_out, contrast_stats, complete


# ------------------------------------------------------------------ #
# D4: frozen five-arm execution                                      #
# ------------------------------------------------------------------ #

def _anchor_indices(df, start, end, purge_after=None):
    import pandas as pd
    ts = pd.to_datetime(df["DATE_TIME"])
    lo = ts.searchsorted(pd.Timestamp(_utc(start).replace(
        tzinfo=None)))
    hi = ts.searchsorted(pd.Timestamp(_utc(end).replace(
        tzinfo=None)), side="right")
    rows = list(range(lo, hi))
    if purge_after is not None:
        rows = [r for r in rows if r + H_MAX < purge_after]
    return rows


def execute(staging: Path, out_bundle: Path) -> dict:
    import numpy as np
    import pandas as pd
    from agent_plugins.branch_pretraining import barrier_hit_labels
    started = time.time()
    df = pd.read_parquet(staging / "model_ready_extended.parquet")
    n = len(df)
    closes = df["CLOSE"].to_numpy()
    returns = np.diff(np.log(closes))
    # labels for ALL rows with full forward coverage
    max_a = n - H_MAX - 1
    steps = list(range(WINDOW + 4 * 3 + 1, max_a + 2))
    labels = barrier_hit_labels(
        df["OPEN"].to_numpy(), df["HIGH"].to_numpy(),
        df["LOW"].to_numpy(), closes, steps, [6, 12], 64, 2.0, 2.0,
        1e-8)
    label_row = {s - 1: i for i, s in enumerate(steps)}
    # barrier-scale lags (vectorized trailing mean square)
    sq = np.concatenate([[0.0], np.cumsum(returns ** 2)])
    scale = np.full((n, 4), np.nan)
    for k in range(4):
        a = np.arange(WINDOW + 4 * 3 + 1, n)
        a2 = a - 4 * k
        valid = a2 >= 65
        av = a[valid]
        a2v = a2[valid]
        # window returns[a2-64 : a2] -> sq[a2] - sq[a2-64]
        scale[av, k] = np.sqrt(
            (sq[a2v] - sq[a2v - 64]) / 64.0) + 1e-8
    # 249 causal summary via the audited observation pipeline
    from agent_plugins.branch_pretraining import (
        collect_preprocessed_windows, validate_contract)
    from agent_plugins.pretrained_branch_loader import verify_source
    pretrain_dir = (Path.home() / ".local/share/agent-multi/"
                    "restricted_evidence/"
                    "candidate_full5_pcgrad_o2022_20260828")
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json")
        .read_text())
    source = verify_source(pretrain_dir, REPO,
                           Path(split_contract["source_csv"]))
    contract = source["contract"]
    env_source = contract["observation_pipeline"]["source_config"]
    env_config = json.loads(
        (Path(env_source) if Path(env_source).is_absolute()
         else REPO / env_source).read_text())
    contract_w = {**contract, "window_size": WINDOW}
    env_w = {**env_config, "window_size": WINDOW}

    def summary_for(rows):
        steps_w = [a + 1 for a in rows]
        win = collect_preprocessed_windows(df, contract_w, env_w,
                                           steps_w)
        return np.concatenate(
            [win[:, -1, :], win.mean(axis=1), win.std(axis=1)],
            axis=1).astype("float64")

    # role anchors: stride from each role's first bar, tail-purged
    # so no label crosses the role/confirmation boundary
    def stride_rows(start, end):
        rows = _anchor_indices(df, start, end)
        boundary = rows[-1] + 1
        return [r for r in rows[::STRIDE]
                if r + H_MAX < boundary and r in label_row]
    fit_rows = stride_rows(*ROLE_FIT)
    cal_rows = stride_rows(*ROLE_CAL)
    block_rows = {}
    blocks_complete = True
    for name, start, end, bars in BLOCKS:
        rows = _anchor_indices(df, start, end)
        if len(rows) != bars:
            blocks_complete = False
        offs = scoring_anchor_offsets(len(rows))
        block_rows[name] = [rows[i] for i in offs
                            if rows[i] in label_row]
    ledger = role_ledger()
    ledger["anchor_counts"] = {
        "fit": len(fit_rows), "cal": len(cal_rows),
        **{k: len(v) for k, v in block_rows.items()}}
    score_rows = [r for v in block_rows.values() for r in v]
    all_rows = fit_rows + cal_rows + score_rows
    summ = summary_for(all_rows)
    summaries = {r: summ[i] for i, r in enumerate(all_rows)}

    def xmat(rows, arm):
        if arm == "arm2":
            return np.array([[scale[r, 0]] for r in rows])
        if arm == "arm3":
            return np.array([scale[r] for r in rows])
        if arm == "arm4":
            return np.array([summaries[r] for r in rows])
        return np.array(
            [np.concatenate([scale[r], summaries[r]])
             for r in rows])

    units = []
    licenses_ok = True
    for tkey, h in TARGETS.items():
        hcol = 0 if h == 6 else 1
        y = {r: int(labels[label_row[r], hcol]) for r in all_rows}
        yf = np.array([y[r] for r in fit_rows])
        yc = np.array([y[r] for r in cal_rows])
        counts = np.bincount(np.concatenate([yf, yc]),
                             minlength=3)
        prior = np.clip(counts / counts.sum(), 1e-12, None)
        prior = prior / prior.sum()
        # fit each arm ONCE per target; score all blocks jointly
        arm_probs = {}
        arm_recs = {}
        degenerate = None
        for arm in ARMS:
            if arm == "arm1":
                arm_probs[arm] = np.tile(prior,
                                         (len(score_rows), 1))
                arm_recs[arm] = {"prior_from": "fit+calibration"}
                continue
            probs, rec = tcn2._logistic(
                xmat(fit_rows, arm), yf, xmat(cal_rows, arm), yc,
                xmat(score_rows, arm))
            if probs is None:
                degenerate = arm
                break
            arm_probs[arm] = probs
            arm_recs[arm] = rec
        row_pos = {r: i for i, r in enumerate(score_rows)}
        for name, _, _, _ in BLOCKS:
            rows_s = block_rows[name]
            ys = np.array([y[r] for r in rows_s])
            payload = {"unit": f"{tkey}:{name}",
                       "horizon": h, "block": name,
                       "n_score": len(rows_s),
                       "anchor_datetimes": [
                           str(df["DATE_TIME"].iloc[r])
                           for r in rows_s],
                       "fit_cal_label_histogram": [
                           int(c) for c in counts],
                       "class_support_score": {
                           str(c): int((ys == c).sum())
                           for c in (0, 1, 2)},
                       "arms": {}}
            if min(payload["class_support_score"]["0"],
                   payload["class_support_score"]["1"]) \
                    < SUPPORT_MIN:
                licenses_ok = False
                payload["license_failure"] = "class_support"
            if degenerate is not None:
                licenses_ok = False
                payload["license_failure"] = \
                    f"degenerate:{degenerate}"
                units.append(payload)
                continue
            idx = [row_pos[r] for r in rows_s]
            for arm in ARMS:
                probs = arm_probs[arm][idx]
                lm = -np.log(np.clip(
                    probs[np.arange(len(ys)), ys], 1e-12, None))
                p_hit = np.clip(probs[:, 0] + probs[:, 1],
                                1e-12, 1 - 1e-12)
                is_hit = ys < 2
                l_hit = np.where(is_hit, -np.log(p_hit),
                                 -np.log(1 - p_hit))
                payload["arms"][arm] = {
                    "record": arm_recs[arm],
                    "multiclass_logloss_mean": round(
                        float(lm.mean()), 6),
                    "hit_vs_censored_mean": round(
                        float(l_hit.mean()), 6),
                    "per_obs_logloss": [round(float(v), 8)
                                        for v in lm]}
            units.append(payload)
    # ---- contrasts + decision (shared rederivation) ----
    contrasts_out, contrast_stats, complete = _rederive(units)
    if not complete:
        licenses_ok = False
    verdict = decide(contrast_stats, blocks_complete, licenses_ok)
    bundle = {
        "schema": "agent_multi.n3_fresh_bundle.v1",
        "contract": CONTRACT,
        "contract_sha256": sha_file(REPO / CONTRACT),
        "role_ledger": ledger,
        "digests": {
            "acquired_parquet": sha_file(
                staging / "acquired.parquet"),
            "model_ready_extended": sha_file(
                staging / "model_ready_extended.parquet"),
            "frozen_csv": FROZEN_SHA,
            "lake_parquet": LAKE_SHA,
            "code": sha_obj({
                "n3_fresh_confirmation.py": sha_file(
                    REPO / "tools/n3_fresh_confirmation.py"),
                "target_horizon_census_n2.py": sha_file(
                    REPO / "tools/target_horizon_census_n2.py"),
                "paired_inference.py": sha_file(
                    REPO / "agent_plugins/paired_inference.py"),
                "branch_pretraining.py": sha_file(
                    REPO / "agent_plugins/branch_pretraining.py")})},
        "blocks_complete": blocks_complete,
        "licenses_ok": licenses_ok,
        "units": units,
        "contrasts": contrasts_out,
        "verdict": verdict,
        "elapsed_s": round(time.time() - started, 1),
        "decision_constants": {
            "margin_scale": MARGIN_SCALE,
            "margin_repr": MARGIN_REPR,
            "support_min": SUPPORT_MIN,
            "boot_b": BOOT_B, "boot_seed": BOOT_SEED,
            "block_len": BLOCK_LEN}}
    for u in units:
        u["payload_sha256"] = sha_obj(
            {k: v for k, v in u.items() if k != "payload_sha256"})
    out_bundle.write_text(json.dumps(bundle, indent=1,
                                     default=float) + "\n")
    return bundle


# ------------------------------------------------------------------ #
# offline verifier — refuses the ten ordered adversaries             #
# ------------------------------------------------------------------ #

def verify(bundle_path: Path) -> dict:
    import numpy as np
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("schema") != "agent_multi.n3_fresh_bundle.v1":
        raise FreshRefusal("unknown bundle schema")
    sealed = json.loads((REPO / CONTRACT).read_text())
    ledger = bundle["role_ledger"]
    # adversary 2: boundary moved after acquisition
    def _norm(block):
        return [str(block[0]).replace("T", " "),
                str(block[1]).replace("T", " "), block[2]]
    for name, spec in sealed["role_ledger"]["blocks_utc"].items():
        got = ledger["blocks"].get(name)
        if got is None or _norm(got) != _norm(spec):
            raise FreshRefusal(
                f"role boundary moved: block {name} differs from "
                "the sealed contract")
    expected_units = {f"{t}:{b[0]}" for t in TARGETS
                      for b in BLOCKS}
    seen = [u["unit"] for u in bundle["units"]]
    if len(seen) != len(set(seen)):
        raise FreshRefusal("duplicate units")
    if set(seen) != expected_units:
        raise FreshRefusal(
            f"missing/extra units: {sorted(set(seen) ^ expected_units)[:4]}")
    conf_lo = _utc(CONF_START).replace(tzinfo=None)
    conf_hi = _utc(CONF_END).replace(tzinfo=None)
    for u in bundle["units"]:
        claimed = u.get("payload_sha256")
        if sha_obj({k: v for k, v in u.items()
                    if k != "payload_sha256"}) != claimed:
            raise FreshRefusal(
                f"unit {u['unit']}: payload altered (digest)")
        if u.get("license_failure"):
            raise FreshRefusal(
                f"unit {u['unit']}: license failure "
                f"{u['license_failure']} beside the decision")
        blo, bhi, bars = [
            (s, e, n) for name, s, e, n in BLOCKS
            if name == u["block"]][0]
        lo = _utc(blo).replace(tzinfo=None)
        hi = _utc(bhi).replace(tzinfo=None)
        for t in u["anchor_datetimes"]:
            dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            # adversary 1: a 2025 row used as confirmation
            if dt < conf_lo:
                raise FreshRefusal(
                    f"pre-2026 anchor {t} used as confirmation")
            # adversary 3: future row beyond the sealed interval
            if dt > conf_hi:
                raise FreshRefusal(
                    f"anchor {t} beyond the sealed confirmation "
                    "end")
            if not (lo <= dt <= hi):
                raise FreshRefusal(
                    f"anchor {t} outside its block {u['block']}")
        # adversary 8: prior vs fitted label histories
        hist = u["fit_cal_label_histogram"]
        total = sum(hist)
        prior = np.clip(np.array(hist) / total, 1e-12, None)
        prior = prior / prior.sum()
        arm1 = u["arms"]["arm1"]
        ys_loss = np.asarray(arm1["per_obs_logloss"])
        candidate = -np.log(prior)
        if not all(any(abs(v - c) < 5e-7 for c in candidate)
                   for v in ys_loss[:20]):
            raise FreshRefusal(
                f"unit {u['unit']}: arm1 losses inconsistent with "
                "the bundled fit+cal label histogram — different "
                "label histories")
    # rederive contrasts + decision (adversaries 9, 10)
    contrasts_out, contrast_stats, complete = _rederive(
        bundle["units"])
    if not complete:
        raise FreshRefusal(
            "missing or failed unit beside the decision — cannot "
            "rederive all eight contrasts")
    verdict = decide(contrast_stats, bundle["blocks_complete"],
                     bundle["licenses_ok"])
    if verdict != bundle["verdict"]:
        raise FreshRefusal(
            f"report edited: rederived verdict {verdict} != "
            f"bundled {bundle['verdict']}")
    for ckey, s in contrasts_out.items():
        rec = bundle["contrasts"].get(ckey)
        if rec is None or \
                rec["pooled_skill"] != s["pooled_skill"] or \
                rec["per_block_skill"] != s["per_block_skill"]:
            raise FreshRefusal(
                f"report edited: contrast {ckey} numbers do not "
                "rederive from unit payloads")
    return {"verdict": "N3_BUNDLE_VERIFIED",
            "rederived_decision": verdict,
            "units_verified": len(bundle["units"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("acquire", "regenerate", "execute"):
        sp = sub.add_parser(name)
        sp.add_argument("--staging", required=True)
        if name == "execute":
            sp.add_argument("--out-bundle", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--bundle", required=True)
    args = parser.parse_args()
    try:
        if args.cmd == "acquire":
            out = acquire(Path(args.staging))
            print(json.dumps({"verdict": out["verdict"],
                              "rows": out["rows_total"],
                              "pages": len(out["pages"])}))
        elif args.cmd == "regenerate":
            out = regenerate(Path(args.staging))
            print(json.dumps({"verdict": out["verdict"],
                              "rows_2026": out["rows_2026"]}))
        elif args.cmd == "execute":
            out = execute(Path(args.staging),
                          Path(args.out_bundle))
            print(json.dumps({"verdict": out["verdict"],
                              "elapsed_s": out["elapsed_s"]}))
        else:
            print(json.dumps(verify(Path(args.bundle)), indent=1))
        return 0
    except FreshRefusal as refusal:
        print(json.dumps({"refusal": str(refusal)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
