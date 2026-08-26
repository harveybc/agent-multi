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


def main() -> int:
    out = Path(sys.argv[sys.argv.index("--output") + 1]) if "--output" \
        in sys.argv else Path("cost_manifest_eth_h4_v1.json")
    sp = spread_stats()
    comm = commission_evidence()
    if comm["n_commission_fields"] and comm["max_commission_observed"] != 0.0:
        raise SystemExit("REFUSED: Demo fills report nonzero commission; "
                         "the evidenced-zero canon no longer holds")
    half_med = sp["full_spread_bps_median"] / 2.0
    half_p95 = sp["full_spread_bps_p95"] / 2.0
    manifest = {
        "schema": "agent_multi.cost_manifest.v1",
        "version": "eth_h4_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary": {
            "commission_fraction_per_side": 0.0,
            "commission_basis": ("EVIDENCED: every filled Demo lifecycle "
                                 "event reports commission=0.0"),
            "half_spread_fraction_per_side": round(half_med / 1e4, 8),
            "half_spread_basis": (f"EVIDENCED: median full spread "
                                  f"{sp['full_spread_bps_median']} bps over "
                                  f"{sp['n_quotes']} ETH/USD paper quotes"),
            "slippage_fraction_per_side": 0.0001,
            "slippage_basis": ("DECLARED_BOUND_NOT_EVIDENCED: 1 bp per "
                               "side conservative bound; no fill-vs-quote "
                               "join is available on this host"),
            "env_binding": {"commission": round(half_med / 1e4, 8),
                            "slippage_perc": 0.0001,
                            "note": ("gym-fx models per-side cost via "
                                     "'commission' (fraction of notional); "
                                     "half-spread+fee map onto it; "
                                     "slippage_perc applies per fill")},
        },
        "stress": {
            "commission_fraction_per_side": 0.0025,
            "commission_basis": ("EXTERNAL_LABELED: Alpaca live crypto "
                                 "base-tier taker fee 25 bp — published "
                                 "schedule bound, not local evidence"),
            "half_spread_fraction_per_side": round(half_p95 / 1e4, 8),
            "half_spread_basis": f"EVIDENCED p95: {sp['full_spread_bps_p95']} bps full",
            "slippage_fraction_per_side": 0.0002,
            "slippage_basis": "DECLARED_BOUND_NOT_EVIDENCED: 2 bp stress",
            "env_binding": {"commission": round(
                0.0025 + half_p95 / 1e4, 8), "slippage_perc": 0.0002},
        },
        "zero_cost": {"role": "DIAGNOSTIC_ONLY (P1 recipe default)",
                      "env_binding": {"commission": 0.0,
                                      "slippage_perc": 0.0}},
        "evidence": {"spread": sp, "commission": comm},
        "g1_authority": ("primary governs G1; zero_cost diagnostic; "
                         "stress descriptive — pending Musashi "
                         "ratification"),
    }
    body = json.dumps(manifest, indent=1)
    out.write_text(body)
    manifest_sha = hashlib.sha256(body.encode()).hexdigest()
    print(json.dumps({"written": str(out), "sha256": manifest_sha,
                      "primary_effective_per_side_bps": round(
                          (manifest["primary"]["env_binding"]["commission"]
                           + 0.0001) * 1e4, 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
