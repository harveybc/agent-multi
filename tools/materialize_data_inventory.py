#!/usr/bin/env python3
"""WP-DATA (Data-First SOTA Multibranch order 2026-08-26 §2).

Machine-readable input inventory: every field of the CURRENT 83-column
ETH H4 contract measured from the pinned dataset, plus a typed
inventory of the ABSENT families measured against what this program
ALREADY collects (read-only against the lts evidence stores). Nothing
is invented: unmeasurable facts are typed UNKNOWN/REQUIRES_COLLECTOR.
"""
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agent_plugins.feature_families import semantic_feature_families  # noqa

import os
DATA = Path(os.environ.get(
    "AGENT_MULTI_ETH_CSV",
    str(Path.home() / "Documents/GitHub/predictor/examples/data/"
        "project3/ethusdt_4h_tech_stat_full_model_ready.csv")))
DATA_SHA = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc"
            "735747628f8d0435ebe440f")
V2_SYSTEM = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
             "systems/ethusdt_4h_l1_system_v2.json")
ALPACA_DB = Path(os.environ.get(
    "LTS_ALPACA_LAB_DB",
    str(Path.home() / ".local/state/lts/alpaca-paper-lab.sqlite")))
SIMLIVE_DB = Path(os.environ.get(
    "LTS_SIMLIVE_DB",
    str(Path.home() / ".local/state/lts/sim-vs-live-comparison.sqlite")))

# DATA-SOTA-330: committed artifacts carry LOGICAL source ids + hashes,
# never operator filesystem topology.
SRC_DATASET = ("dataset:ethusdt_4h_tech_stat_full_model_ready"
               f"@{'1b447c66'}")
SRC_ALPACA = "store:lts/alpaca_paper_lab#quote_observations"
SRC_SIMLIVE = "store:lts/sim_vs_live_comparison#comparison_rows"
SRC_SPY = "store:lts/live_data#alpaca_iex_spy_1d"


def current_83_inventory() -> dict:
    df = pd.read_csv(DATA, parse_dates=["DATE_TIME"])
    obs = json.loads(V2_SYSTEM.read_text())["observation"]
    cols = obs["feature_columns"]
    families = semantic_feature_families(cols)
    fam_of = {c: f for f, cs in families.items() for c in cs}
    fields = {}
    for c in cols:
        s = df[c]
        fields[c] = {
            "semantic_family": fam_of.get(c, "UNASSIGNED"),
            "units": ("normalized technical/statistical value "
                      "(model-ready export; source units baked in)"),
            "sampling_timestamp": "H4 bar close (DATE_TIME column)",
            "publication_delay": {
                "claim": "~0 (computed at bar close from closed OHLCV)",
                "status": "UNVERIFIED_UNTIL_MEASURED"},
            "historical_source": SRC_DATASET,
            "historical_status": "HISTORICAL_MEASURED",
            "first_timestamp": str(df["DATE_TIME"].iloc[
                int(s.first_valid_index() or 0)]),
            "last_timestamp": str(df["DATE_TIME"].iloc[-1]),
            "missing_fraction": round(float(s.isna().mean()), 6),
            "causal_transformation": ("precomputed at export "
                                      "(max_feature_lookback baked); "
                                      "runtime rolling_zscore window "
                                      "256 fitted on training data "
                                      "only"),
            "live_status": "LIVE_DERIVABLE_UNVERIFIED",
            "live_claim": ("derivable at bar close from venue bars; "
                           "NOT measured — no same-bar historical-vs-"
                           "live comparison exists yet for this field"),
            "live_venues": {
                "alpaca_ethusd": {"freshness": "UNKNOWN_UNTIL_COLLECTOR",
                                  "coverage": "UNKNOWN_UNTIL_COLLECTOR"},
                "mt5_ethusd": {"freshness": "UNKNOWN_UNTIL_COLLECTOR",
                               "coverage": "UNKNOWN_UNTIL_COLLECTOR"}},
            "v3_eligible": False,
            "v3_eligibility_rule": ("LIVE_PARITY_VERIFIED required: "
                                    "same-bar value reproduces within "
                                    "declared tolerance per venue"),
            "license": "own computation over exchange OHLCV",
        }
    return {"count": len(cols),
            "feature_columns_sha256": obs["feature_columns_sha256"],
            "dataset_sha256": DATA_SHA,
            "rows": int(len(df)),
            "range": [str(df['DATE_TIME'].iloc[0]),
                      str(df['DATE_TIME'].iloc[-1])],
            "families": {k: len(v) for k, v in families.items()},
            "fields": fields}


def measured_absent_families() -> dict:
    out = {}
    # --- bid/ask spread + depth: ALREADY COLLECTED (Alpaca paper) ----
    con = sqlite3.connect(f"file:{ALPACA_DB}?mode=ro", uri=True)
    row = con.execute(
        "SELECT COUNT(*), MIN(observed_at), MAX(observed_at) FROM "
        "quote_observations WHERE symbol='ETH/USD'").fetchone()
    out["bid_ask_spread_and_depth"] = {
        "status": "COLLECTED_ALPACA_PAPER",
        "source": SRC_ALPACA,
        "fields": ["bid", "ask", "mid", "spread", "spread_bps",
                   "bid_size", "ask_size"],
        "eth_rows": int(row[0]), "first": row[1], "last": row[2],
        "coverage_caveat": ("session-scoped preflight sampling, NOT a "
                            "continuous H4-aligned series; a scheduled "
                            "collector is REQUIRED for training-grade "
                            "history"),
        "mt5_side": "quoted_spread present in n=3 attributed fills "
                    "(sparse); MT5 quote collector REQUIRED",
        "historical_depth": "REQUIRES_EXTERNAL_SOURCE (exchange L2 "
                            "history is licensed; e.g. Tardis/Kaiko)",
    }
    cross = {}
    for sym in ("BTC/USD",):
        r = con.execute(
            "SELECT COUNT(*) FROM quote_observations WHERE symbol=?",
            (sym,)).fetchone()
        cross[sym] = int(r[0])
    con.close()
    # --- realized slippage ------------------------------------------
    con = sqlite3.connect(f"file:{SIMLIVE_DB}?mode=ro", uri=True)
    n = con.execute("SELECT COUNT(*) FROM comparison_rows WHERE "
                    "venue='mt5_demo' AND symbol='ETHUSD'").fetchone()[0]
    con.close()
    out["realized_slippage"] = {
        "status": "COLLECTOR_EXISTS_SPARSE",
        "source": SRC_SIMLIVE,
        "eth_rows": int(n),
        "note": "entry_slippage_vs_mid journaled per live decision; "
                "sample grows only with live decisions"}
    # --- finer OHLCV -------------------------------------------------
    out["finer_ohlcv_intraday"] = {
        "status": "REQUIRES_COLLECTOR",
        "candidates": ["Alpaca crypto bars API (1m/5m/15m, free, "
                       "history via REST)",
                       "MT5 CopyRates M1/M15 (demo terminal)"],
        "license": "exchange data via broker APIs (redistribution "
                   "restricted; internal use OK)",
        "storage_estimate": "ETH 1m OHLCV 2017-2025 ~ 4.7M rows x 6 "
                            "cols ~ 250 MB csv / ~60 MB parquet"}
    # --- derivatives family -----------------------------------------
    out["funding_open_interest_basis_liquidations"] = {
        "status": "REQUIRES_COLLECTOR_AND_SOURCE_DECISION",
        "note": ("SPOT venues (Alpaca, MT5 CFD) carry none of these; "
                 "they are PERP-market context (cross-venue). Public "
                 "candidates: Binance/Bybit/OKX REST (funding, OI); "
                 "liquidations feeds are partial/licensed. LIVE "
                 "availability at H4 close is measurable only after a "
                 "collector runs; publication delay typically <1min "
                 "(UNVERIFIED until measured)"),
        "typed_mask_required": True,
        "license": "per-exchange ToS; internal analytics generally "
                   "permitted (VERIFY per source before collection)"}
    # --- cross-asset context ----------------------------------------
    out["cross_asset_context"] = {
        "status": "PARTIALLY_COLLECTED",
        "collected": {"BTC/USD quotes (alpaca paper)": cross["BTC/USD"],
                      "SPY 1d bars": SRC_SPY},
        "needed": "BTC H4 OHLCV history collector (same pipeline as "
                  "ETH; Alpaca/Binance REST)"}
    # --- calendar/session -------------------------------------------
    out["calendar_session_event"] = {
        "status": "DERIVABLE_NO_COLLECTOR",
        "note": ("hour-of-day/day-of-week/session flags derive from "
                 "DATE_TIME causally; macro EVENT calendars require an "
                 "external licensed source (typed mask if adopted)")}
    # --- COP FX for the CDT hurdle ----------------------------------
    out["usdcop_fx_for_cdt_hurdle"] = {
        "status": "REQUIRES_COLLECTOR",
        "purpose": ("REPORTING ONLY (owner hurdle: 10% nominal annual "
                    "COP CDT). Never enters training fitness; never "
                    "compared directly with USD returns"),
        "candidates": ["Banco de la República TRM (official daily, "
                       "public)", "exchange FX feeds"],
        "provenance_requirement": "timestamped FX rate per conversion",
    }
    return out


def multiresolution_proposal() -> dict:
    return {
        "principle": ("exact windows become BOUNDED GENES after "
                      "coverage and live latency are measured — not "
                      "arbitrary constants"),
        "short_intraday": {"windows_candidate_bars": [16, 64],
                           "resolution": "1m-15m",
                           "blocked_on": "finer_ohlcv_intraday "
                                         "collector + parity tests"},
        "medium_h4": {"windows_candidate_bars": [32, 180, 540],
                      "resolution": "H4",
                      "status": "AVAILABLE NOW (current dataset); 180="
                                "30d and 540=90d align with the doc-40 "
                                "TSMOM horizons"},
        "long_daily": {"windows_candidate_bars": [90, 365],
                       "resolution": "1d (causal resample of H4)",
                       "status": "DERIVABLE NOW; resample test required"},
    }


def main() -> int:
    out_path = Path(sys.argv[sys.argv.index("--output") + 1]) if \
        "--output" in sys.argv else Path("DATA_INVENTORY_V1.json")
    actual = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if actual != DATA_SHA:
        raise SystemExit("REFUSED: pinned dataset drifted")
    inv = {
        "schema": "agent_multi.data_inventory.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "order": "DATA_FIRST_SOTA_MULTIBRANCH_2026_08_26 §2",
        "current_contract_83": current_83_inventory(),
        "absent_families": measured_absent_families(),
        "multiresolution_windows": multiresolution_proposal(),
        "missing_data_policy": {
            "rule": ("a feature that exists only historically is "
                     "INADMISSIBLE; live-unavailable inputs carry a "
                     "TYPED MASK channel (value + availability bit) "
                     "fitted into the branch input contract; masks are "
                     "part of the observation identity digest")},
        "acceptance_tests_planned": [
            "leakage: per-field shift test (mutating t must not change "
            "any feature at t computed causally)",
            "historical/live parity: same bar, collector vs dataset "
            "field equality within declared tolerance",
            "coverage: first/last/missingness asserted against this "
            "inventory before any training role materializes"],
        "storage_estimate": {
            "current_dataset_mb": round(DATA.stat().st_size / 1e6, 1),
            "eth_1m_history": "~60 MB parquet",
            "btc_h4_history": "~5 MB",
            "quotes_continuous_collector": "~1 MB/day at 1 obs/min "
                                           "per symbol"},
    }
    body = json.dumps(inv, indent=1)
    out_path.write_text(body)
    print(json.dumps({"written": str(out_path),
                      "sha256": hashlib.sha256(body.encode()).hexdigest(),
                      "fields_83": inv["current_contract_83"]["count"],
                      "absent_families": list(inv["absent_families"])},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
