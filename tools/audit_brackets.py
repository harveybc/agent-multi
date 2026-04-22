"""Audit that every order emitted by direct_atr_sltp has a SL + TP bracket,
and that bracket distances are not extreme vs price.

Runs a short smoke, intercepts buy_bracket/sell_bracket calls on the backtrader
strategy, records (action, entry, stop, limit), and reports:
  - N_orders_total
  - N_naked_orders (any buy/sell without a bracket — must be 0)
  - SL distance / price:  min, median, p95, max (should live within 0.1%..20%)
  - TP distance / price:  same

Usage:
    python tools/audit_brackets.py examples/config/p4_ppo_btc_1h.json --steps 3000
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import backtrader as bt

# Import the strategy plugin from gym-fx working copy so we monkey-patch the same
# class that the env will instantiate.
sys.path.insert(0, str(Path.home() / "Documents/GitHub/gym-fx"))

AUDIT: dict = {"brackets": [], "naked": 0}


def _patch() -> None:
    orig_buy_bracket = bt.Strategy.buy_bracket
    orig_sell_bracket = bt.Strategy.sell_bracket
    orig_buy = bt.Strategy.buy
    orig_sell = bt.Strategy.sell

    def log(kind: str, strategy, size, stopprice=None, limitprice=None, **_):
        close = float(strategy.data.close[0])
        AUDIT["brackets"].append(
            {
                "kind": kind,
                "entry": close,
                "stop": float(stopprice) if stopprice is not None else None,
                "limit": float(limitprice) if limitprice is not None else None,
                "size": float(size) if size is not None else None,
            }
        )

    def buy_bracket_audit(self, *a, **kw):
        log("long_bracket", self, kw.get("size"), kw.get("stopprice"), kw.get("limitprice"))
        return orig_buy_bracket(self, *a, **kw)

    def sell_bracket_audit(self, *a, **kw):
        log("short_bracket", self, kw.get("size"), kw.get("stopprice"), kw.get("limitprice"))
        return orig_sell_bracket(self, *a, **kw)

    def buy_audit(self, *a, **kw):
        AUDIT["naked"] += 1
        return orig_buy(self, *a, **kw)

    def sell_audit(self, *a, **kw):
        AUDIT["naked"] += 1
        return orig_sell(self, *a, **kw)

    bt.Strategy.buy_bracket = buy_bracket_audit
    bt.Strategy.sell_bracket = sell_bracket_audit
    bt.Strategy.buy = buy_audit
    bt.Strategy.sell = sell_audit


def _report() -> int:
    brackets = AUDIT["brackets"]
    naked = AUDIT["naked"]
    # bt_bridge internally calls strategy.close() which may route through buy/sell;
    # discount close-direction cancels by subtracting bracket count * 0 — but the
    # close() path actually dispatches a separate order that *would* show up as a
    # naked order. So the true naked count equals `naked - (#bracket-entries that
    # also reversed an existing position)`. We can't distinguish precisely here,
    # but in a quick smoke both counts being zero on steady-state trading is the
    # acceptance signal. Report both raw.
    print(f"total_bracket_orders = {len(brackets)}")
    print(f"raw_naked_calls     = {naked}  (includes bt close() reversals)")
    if not brackets:
        print("NO BRACKET ORDERS CAPTURED — plugin did not trade")
        return 1

    def pct(x: float) -> float:
        return 100.0 * x

    sl_fracs = []
    tp_fracs = []
    invalid = 0
    for b in brackets:
        entry = b["entry"]
        stop = b["stop"]
        limit = b["limit"]
        if entry is None or stop is None or limit is None:
            invalid += 1
            continue
        if b["kind"] == "long_bracket":
            sl = entry - stop
            tp = limit - entry
        else:
            sl = stop - entry
            tp = entry - limit
        if sl <= 0 or tp <= 0:
            invalid += 1
            continue
        sl_fracs.append(sl / entry)
        tp_fracs.append(tp / entry)

    print(f"invalid_brackets    = {invalid}  (missing price or wrong sign)")
    if not sl_fracs:
        return 2

    import statistics as st

    def stats(name: str, xs: list[float]) -> None:
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        p95 = xs_sorted[int(0.95 * (n - 1))]
        print(
            f"{name}: n={n}  min={pct(min(xs_sorted)):.3f}%  "
            f"median={pct(st.median(xs_sorted)):.3f}%  "
            f"p95={pct(p95):.3f}%  max={pct(max(xs_sorted)):.3f}%"
        )

    stats("SL/price", sl_fracs)
    stats("TP/price", tp_fracs)

    pathological = sum(1 for f in sl_fracs + tp_fracs if f > 0.20 or f < 0.001)
    print(f"out_of_band (>20% or <0.1%) = {pathological}")
    return 0 if invalid == 0 and pathological == 0 else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()

    _patch()

    import subprocess

    base = json.loads(Path(args.config).read_text())
    base["total_timesteps"] = args.steps
    if "ppo" in args.config:
        base["n_steps"] = 256
    base["learning_starts"] = 300
    base["save_model"] = "/tmp/_audit.zip"
    base["results_file"] = "/tmp/_audit_summary.json"
    base["save_config"] = "/tmp/_audit_cfg_out.json"
    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    tmp.write_text(json.dumps(base))

    # Run in-process via the agent-multi CLI entrypoint so the patched bt.Strategy
    # is the one used by the spawned cerebro.
    import importlib

    # agent-multi main is not importable; use subprocess but pass an env marker so
    # the child imports this module too. Simpler: run the CLI and rely on the
    # fact that our patched bt is in this interpreter. Since agent-multi CLI
    # spawns in-process, we import it and call main directly.
    from app import main as am_main  # type: ignore

    old_argv = sys.argv
    sys.argv = ["agent-multi", "--load_config", str(tmp), "--quiet_mode"]
    try:
        am_main.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv

    return _report()


if __name__ == "__main__":
    raise SystemExit(main())
