"""Audit every order emitted by direct_atr_sltp.

Sets GYMFX_BRACKET_AUDIT to a log file, runs a short smoke, then analyses
the emitted JSONL records to verify:
  - every entry has both stop and limit prices
  - SL and TP distances as fraction of entry price fall within sane bounds

Usage:
    python tools/audit_brackets.py examples/config/p4_ppo_btc_1h.json --steps 5000
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import tempfile
from pathlib import Path

MIN_OK_FRAC = 0.0005
MAX_OK_FRAC = 0.20


def _run_smoke(cfg_path: str, steps: int, log_path: str) -> int:
    base = json.loads(Path(cfg_path).read_text())
    base["total_timesteps"] = steps
    if "ppo" in cfg_path:
        base["n_steps"] = 256
    base["learning_starts"] = 300
    base["save_model"] = "/tmp/_audit.zip"
    base["results_file"] = "/tmp/_audit_summary.json"
    base["save_config"] = "/tmp/_audit_cfg_out.json"
    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    tmp.write_text(json.dumps(base))

    env = dict(os.environ)
    env["GYMFX_BRACKET_AUDIT"] = log_path
    proc = subprocess.run(
        ["agent-multi", "--load_config", str(tmp), "--quiet_mode"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )
    if proc.returncode != 0:
        print("SMOKE FAILED rc=", proc.returncode)
        print(proc.stderr[-1500:])
    return proc.returncode


def _analyse(log_path: str) -> int:
    records = []
    path = Path(log_path)
    if not path.exists():
        print("NO AUDIT LOG WRITTEN - plugin never hit the bracket code path")
        return 1
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass

    if not records:
        print("log present but empty")
        return 1

    total = len(records)
    missing_bracket = sum(
        1 for r in records if r.get("stop") is None or r.get("limit") is None
    )
    zero_size = sum(1 for r in records if (r.get("size") or 0) <= 0)

    sl_fracs: list[float] = []
    tp_fracs: list[float] = []
    invalid_sign = 0
    for r in records:
        entry = r["entry"]
        stop = r["stop"]
        limit = r["limit"]
        if r["kind"] == "long_bracket":
            sl = entry - stop
            tp = limit - entry
        else:
            sl = stop - entry
            tp = entry - limit
        if sl <= 0 or tp <= 0 or entry <= 0:
            invalid_sign += 1
            continue
        sl_fracs.append(sl / entry)
        tp_fracs.append(tp / entry)

    def q(xs: list[float], p: float) -> float:
        xs_sorted = sorted(xs)
        return xs_sorted[int(p * (len(xs_sorted) - 1))]

    def fmt(xs: list[float]) -> str:
        return (
            f"n={len(xs)} min={100*min(xs):.3f}% "
            f"p50={100*st.median(xs):.3f}% p95={100*q(xs,0.95):.3f}% "
            f"max={100*max(xs):.3f}%"
        )

    print(f"total_bracket_orders  = {total}")
    print(f"missing_sl_or_tp      = {missing_bracket}   (must be 0)")
    print(f"zero_or_neg_size      = {zero_size}   (must be 0)")
    print(f"invalid_sign          = {invalid_sign}   (must be 0)")
    if sl_fracs:
        print(f"SL/price              {fmt(sl_fracs)}")
    if tp_fracs:
        print(f"TP/price              {fmt(tp_fracs)}")

    tol = 1e-9
    extreme = sum(
        1 for f in sl_fracs + tp_fracs
        if f < MIN_OK_FRAC - tol or f > MAX_OK_FRAC + tol
    )
    print(f"out_of_band (<{MIN_OK_FRAC*100:.2f}% or >{MAX_OK_FRAC*100:.1f}%) = {extreme}   (must be 0)")

    ok = (missing_bracket == 0 and zero_size == 0 and invalid_sign == 0 and extreme == 0)
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=5000)
    args = ap.parse_args()

    log = Path(tempfile.mkstemp(suffix=".jsonl", prefix="bracket_audit_")[1])
    log.write_text("")
    rc = _run_smoke(args.config, args.steps, str(log))
    if rc != 0:
        return rc
    return _analyse(str(log))


if __name__ == "__main__":
    raise SystemExit(main())
