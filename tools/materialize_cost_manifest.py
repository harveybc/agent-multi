#!/usr/bin/env python3
"""C4 (Screen B correction order): versioned economic cost canon from
READ-ONLY Demo evidence. Never writes to any venue store.

Terms and sources:
- commission_fraction_per_side: evidenced from filled Demo lifecycle
  events (demo-execution-l0.sqlite) — every fill reports commission.
- half_spread_fraction_per_side: evidenced from Alpaca paper-lab quote
  observations (ETH/USD bid/ask), median of full-spread bps / 2.
- slippage_fraction_per_side: NOT directly evidenced on this host —
  DECLARED conservative bound, labeled as such.
The stress contract uses spread p95 and an external published taker-fee
bound (labeled external, not local evidence).
"""
import hashlib
import json
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ALPACA_DB = Path.home() / ".local/state/lts/alpaca-paper-lab.sqlite"
L0_DB = Path.home() / ".local/state/lts/demo-execution-l0.sqlite"


def spread_stats() -> dict:
    con = sqlite3.connect(f"file:{ALPACA_DB}?mode=ro", uri=True)
    v = sorted(r[0] for r in con.execute(
        "SELECT spread_bps FROM quote_observations WHERE symbol='ETH/USD'"
        " AND spread_bps IS NOT NULL"))
    con.close()
    if not v:
        raise SystemExit("REFUSED: no ETH/USD quote evidence")
    def pct(p):
        return v[min(len(v) - 1, int(p * len(v)))]
    return {"n_quotes": len(v),
            "full_spread_bps_median": round(statistics.median(v), 3),
            "full_spread_bps_p95": round(pct(0.95), 3),
            "source": str(ALPACA_DB),
            "table": "quote_observations", "symbol": "ETH/USD"}


def commission_evidence() -> dict:
    con = sqlite3.connect(f"file:{L0_DB}?mode=ro", uri=True)
    rows = [r[0] for r in con.execute(
        "SELECT report_json FROM lifecycle_events WHERE state='filled'")]
    con.close()
    commissions = []
    for raw in rows:
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        def walk(x):
            if isinstance(x, dict):
                for k, val in x.items():
                    if "commission" in k.lower() and isinstance(
                            val, (int, float)):
                        commissions.append(float(val))
                    walk(val)
            elif isinstance(x, list):
                for i in x:
                    walk(i)
        walk(d)
    return {"n_filled_events": len(rows),
            "n_commission_fields": len(commissions),
            "max_commission_observed": max(commissions) if commissions
            else None,
            "source": str(L0_DB), "table": "lifecycle_events"}


def mt5_evidence() -> dict:
    """Venue+instrument+timestamp attributed MT5 demo ETHUSD facts."""
    import statistics
    con = sqlite3.connect(
        "file:" + str(Path.home() / ".local/state/lts/"
                      "sim-vs-live-comparison.sqlite") + "?mode=ro",
        uri=True)
    con.row_factory = sqlite3.Row
    rows = []
    for r in con.execute(
            "SELECT recorded_at, payload_json FROM comparison_rows "
            "WHERE venue='mt5_demo' AND symbol='ETHUSD'"):
        b = json.loads(r["payload_json"])["broker"]
        rows.append({"venue": "mt5_demo", "instrument": "ETHUSD",
                     "timestamp": r["recorded_at"],
                     "fill_price": b["fill_price"],
                     "quoted_spread": b["quoted_spread"],
                     "entry_slippage_vs_mid": b["entry_slippage_vs_mid"],
                     "fees_field": b["fees"]})
    con.close()
    if not rows:
        raise SystemExit("REFUSED: no attributed MT5 ETHUSD evidence")
    sp = [x["quoted_spread"] / x["fill_price"] * 1e4 for x in rows]
    sl = [abs(x["entry_slippage_vs_mid"]) / x["fill_price"] * 1e4
          for x in rows]
    return {"rows": rows, "n": len(rows),
            "full_spread_bps_median": round(statistics.median(sp), 2),
            "full_spread_bps_max": round(max(sp), 2),
            "slippage_vs_mid_bps_median": round(statistics.median(sl), 2),
            "slippage_vs_mid_bps_max": round(max(sl), 2)}


def main() -> int:
    out = Path(sys.argv[sys.argv.index("--output") + 1]) if "--output" \
        in sys.argv else Path("cost_manifest_eth_h4_v2.json")
    sp = spread_stats()          # Alpaca ETH/USD quotes (attributed)
    mt5 = mt5_evidence()         # MT5 demo ETHUSD fills (attributed)
    comm = commission_evidence() # L0: observability ONLY (unattributed)
    alp_half = sp["full_spread_bps_median"] / 2.0
    mt5_half = mt5["full_spread_bps_median"] / 2.0
    mt5_slip_extra = max(0.0, mt5["slippage_vs_mid_bps_max"] - mt5_half)
    manifest = {
        "schema": "agent_multi.cost_manifest.v2",
        "version": "eth_h4_v2_venue_specific",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "alpaca_ethusd": {
            "instrument": "ETH/USD", "venue": "alpaca",
            "commission_fraction_per_side": 0.0025,
            "commission_basis": (
                "PUBLISHED_REAL_FEE_SCHEDULE: Alpaca crypto Tier-1 "
                "(lowest volume) TAKER fee 25 bp; the Paper simulator "
                "charges 0, which is a SIMULATOR OMISSION, not business "
                "economics"),
            "fee_schedule_source": {
                "url": "https://docs.alpaca.markets/docs/crypto-fees",
                "tier": "Tier 1 (monthly volume < $100k)",
                "assumption": "TAKER (market orders cross the spread)",
                "retrieved": "2026-08-26",
                "verified_by": ("Musashi audit 2026-08-26: 'agrees with "
                                "Alpaca's current official fee "
                                "schedule'")},
            "paper_simulator_charge_fraction": 0.0,
            "half_spread_fraction_per_side": round(alp_half / 1e4, 8),
            "half_spread_basis": (
                f"EVIDENCED: median full spread "
                f"{sp['full_spread_bps_median']} bps over "
                f"{sp['n_quotes']} venue+instrument-attributed ETH/USD "
                f"paper quotes"),
            "slippage_fraction_per_side": 0.0001,
            "slippage_basis": "DECLARED_BOUND_NOT_EVIDENCED: 1 bp",
            "env_binding": {"commission": round(0.0025 + alp_half / 1e4,
                                                8),
                            "slippage_perc": 0.0001},
        },
        "mt5_ethusd": {
            "instrument": "ETHUSD", "venue": "mt5_demo",
            "commission_fraction_per_side": 0.0,
            "commission_basis": (
                "DECLARED_BROKER_MODEL: spread-based CFD pricing; the "
                "demo journal reports fees 'unavailable: not journaled "
                "per effect' — commission-zero is a declared model, "
                "NOT venue-journaled evidence"),
            "half_spread_fraction_per_side": round(mt5_half / 1e4, 8),
            "half_spread_basis": (
                f"EVIDENCED (n={mt5['n']}, small): median quoted full "
                f"spread {mt5['full_spread_bps_median']} bps from "
                f"attributed mt5_demo ETHUSD fills"),
            "slippage_fraction_per_side": round(
                max(0.0001, mt5_slip_extra / 1e4), 8),
            "slippage_basis": (
                f"EVIDENCED worst |fill-mid| {mt5['slippage_vs_mid_bps_max']} "
                f"bps minus half-spread, floored at 1 bp declared"),
            "financing_swap": {
                "status": "REQUIRED_BEFORE_MT5_PRIMARY_G1",
                "reason": ("not journaled on this host; H4 positions "
                           "cross the charge boundary — the broker "
                           "schedule must be evidenced before this "
                           "contract can govern G1")},
            "env_binding": {"commission": round(mt5_half / 1e4, 8),
                            "slippage_perc": round(
                                max(0.0001, mt5_slip_extra / 1e4), 8)},
        },
        "zero_cost": {"role": "VENUE_NEUTRAL_DIAGNOSTIC_ONLY",
                      "env_binding": {"commission": 0.0,
                                      "slippage_perc": 0.0}},
        "l0_lifecycle_rows": {
            "classification": ("OBSERVABILITY_ONLY: rows carry neither "
                               "venue nor instrument (broker_ids empty) "
                               "and cannot establish any commission"),
            "n_filled_events": comm["n_filled_events"]},
        "evidence": {"alpaca_quotes": sp, "mt5_fills": mt5},
        "g1_authority": ("per-venue primaries pending Musashi "
                         "ratification; mt5_ethusd additionally blocked "
                         "by the financing evidence gap; zero_cost "
                         "diagnostic"),
    }
    body = json.dumps(manifest, indent=1)
    out.write_text(body)
    print(json.dumps({
        "written": str(out),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "alpaca_per_side_bps": round((manifest["alpaca_ethusd"][
            "env_binding"]["commission"] + 0.0001) * 1e4, 2),
        "mt5_per_side_bps": round((manifest["mt5_ethusd"]["env_binding"][
            "commission"] + manifest["mt5_ethusd"]["env_binding"][
            "slippage_perc"]) * 1e4, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
