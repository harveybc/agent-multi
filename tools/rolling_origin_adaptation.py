#!/usr/bin/env python3
"""RT0/RT1: rolling-origin, test-then-train adaptation runner (WP7).

LOCAL-ONLY and zero-network: no DOIN, no sockets, no venue calls. It
measures the two facts finding 134 demands before any cadence can be
frozen: (RT0) can an update finish inside its cadence on this hardware,
and (RT1) does periodic fine-tuning help or hurt the NEXT deployment
interval, scored strictly before its rows may be trained on.

Prequential discipline (Hyndman & Athanasopoulos tscv; Bifet et al.
MOA): at origin t the incumbent policy — trained only on bars <= t —
is scored on (t, t+h]; only afterwards may those bars enter the
update that produces the policy for the next origin.

Cadences are BAR-ALIGNED: 2/3/6/18/42 bars (8/12/24/72/168 h); 1 bar
(4 h) is feasibility-only. Restart-safe: each origin writes a
content-addressed record and completed origins are skipped only on
exact identity. Every record lands in one OLAP SQLite for comparison.
The deadline guard (p95 update <= 2/3 cadence) is reported as
`proposed_pending_owner_ratification`, never enforced as accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

DATA_FILE = ("/home/harveybc/Documents/GitHub/predictor/examples/data/"
             "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
DATA_SHA256 = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435"
               "ebe440f")
ETH_BASE = (REPO / "examples/results/"
            "project3_ethusdt_4h_sac_train_val_test_v2/config_out.json")
RUNNER_VERSION = "rolling_origin_adaptation.v1"
BARS_PER_DAY = 6
ALLOWED_CADENCES = (1, 2, 3, 6, 18, 42)      # 1 bar = feasibility only
WARMUP_BARS = 256                            # rolling scaling context


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_probe() -> dict:
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if probe.returncode != 0:
            return {"unavailable": f"exit {probe.returncode}"}
        first = probe.stdout.strip().splitlines()[0].split(",")
        return {"temperature_c": int(first[0].strip()),
                "memory_used": first[1].strip()}
    except Exception as exc:
        return {"unavailable": str(exc)[:120]}


def _olap(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS rt_intervals (
            record_id TEXT PRIMARY KEY,
            run_id TEXT, phase TEXT, cadence_bars INTEGER,
            lookback TEXT, seed INTEGER, origin_index INTEGER,
            origin_time TEXT, interval_end_time TEXT,
            interval_return REAL, interval_trades INTEGER,
            interval_max_drawdown_fraction REAL,
            equity_before REAL, equity_after REAL,
            update_seconds REAL, deadline_seconds REAL,
            deadline_miss INTEGER, model_age_bars INTEGER,
            new_bars INTEGER, update_steps INTEGER,
            peak_rss_mb REAL, gpu_json TEXT,
            model_sha256 TEXT, created_at TEXT
        )""")
    return con


def _slice_csv(df: pd.DataFrame, start: int, end: int,
               out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.iloc[start:end].to_csv(out, index=False)
    return out


def _build_env(config: dict, csv_path: Path):
    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)
    from importlib.metadata import entry_points
    cfg = dict(config)
    cfg["input_data_file"] = str(csv_path)
    env_plugin = _load_env_plugin(
        str(cfg.get("env_plugin", "gym_fx_env")), cfg)
    env = env_plugin.make_env(cfg)
    agent_ep = next(e for e in entry_points().select(
        group="agent.plugins")
        if e.name == cfg.get("agent_plugin",
                             "project3_sac_actor_critic_agent"))
    agent = agent_ep.load()()
    wrap = getattr(agent, "wrap_env", None)
    if callable(wrap):
        env = wrap(env, cfg)
    return env


def _score_interval(model, env) -> dict:
    """Deterministic rollout over one deployment interval slice."""
    import numpy as np
    obs, _ = env.reset()
    equities = []
    trades = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        if isinstance(info, dict):
            equity = info.get("economic_equity", info.get("equity"))
            if equity is not None:
                equities.append(float(equity))
            trades = int(info.get("trades_total", trades) or trades)
        if terminated or truncated:
            break
    if not equities:
        return {"unavailable": "no equity telemetry in rollout"}
    initial, final = equities[0], equities[-1]
    peak = -float("inf")
    max_dd = 0.0
    for value in equities:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    return {
        "interval_return": (final / initial - 1.0) if initial else None,
        "equity_before": initial, "equity_after": final,
        "max_drawdown_fraction": max_dd, "trades": trades,
        "bars": len(equities),
    }


def run(args) -> int:
    assert _sha_file(Path(DATA_FILE)) == DATA_SHA256, "dataset drift"
    assert args.cadence_bars in ALLOWED_CADENCES, (
        f"cadence {args.cadence_bars} bars is not bar-aligned/allowed")
    config = json.loads(ETH_BASE.read_text())
    config["quiet_mode"] = True
    config["env_mode"] = "training"
    config["solvency_mode"] = "normal_realistic"
    config["evaluate_test_split"] = False

    df = pd.read_csv(DATA_FILE)
    dates = pd.to_datetime(df["DATE_TIME"])
    block_start = int((dates >= args.block_start).idxmax())
    block_bars = args.block_days * BARS_PER_DAY
    # 2025 must never enter scoring or adaptation.
    guard_2025 = int((dates >= "2025-01-01").idxmax())
    assert block_start + block_bars <= guard_2025, (
        "block would cross into the disclosed 2025 period")

    run_identity = {
        "runner_version": RUNNER_VERSION, "phase": args.phase,
        "cadence_bars": args.cadence_bars, "lookback": args.lookback,
        "seed": args.seed, "block_start": args.block_start,
        "block_days": args.block_days,
        "update_steps": args.update_steps,
        "data_sha256": DATA_SHA256,
    }
    run_id = hashlib.sha256(json.dumps(
        run_identity, sort_keys=True).encode()).hexdigest()[:16]
    out_dir = args.output_root / f"{args.phase}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_identity.json").write_text(
        json.dumps(run_identity, indent=1) + "\n")
    con = _olap(args.output_root / "rt_adaptation.sqlite")

    from stable_baselines3 import SAC
    model_path = out_dir / "incumbent.zip"
    scratch = out_dir / "slices"

    def lookback_start(origin: int) -> int:
        if args.lookback == "expanding":
            return 0
        years = int(args.lookback.rstrip("y"))
        return max(0, origin - years * 365 * BARS_PER_DAY)

    origins = list(range(block_start,
                         block_start + block_bars - args.cadence_bars,
                         args.cadence_bars))
    if args.max_origins:
        origins = origins[:args.max_origins]
    cadence_seconds = args.cadence_bars * 4 * 3600.0
    update_times = []
    model_age_bars = 0

    for index, origin in enumerate(origins):
        record_id = hashlib.sha256(
            f"{run_id}:{index}:{origin}".encode()).hexdigest()[:20]
        if con.execute("SELECT 1 FROM rt_intervals WHERE record_id=?",
                       (record_id,)).fetchone():
            print(f"[rt] origin {index} already recorded — skip",
                  flush=True)
            model_age_bars += args.cadence_bars
            continue

        # ---- fit/refresh incumbent on bars <= origin (strictly) ----
        started = time.monotonic()
        fit_start = lookback_start(origin)
        fit_csv = _slice_csv(
            df, max(0, fit_start - WARMUP_BARS), origin,
            scratch / f"fit_{index}.csv")
        fit_env = _build_env({**config, "train_seed": args.seed},
                             fit_csv)
        if model_path.exists():
            model = SAC.load(str(model_path), env=fit_env,
                             device=args.device)
        else:
            model = SAC("MlpPolicy", fit_env, seed=args.seed,
                        device=args.device,
                        policy_kwargs={"net_arch": [256, 256]})
            model.learn(total_timesteps=args.initial_steps,
                        progress_bar=False)
        if index > 0 or args.update_first_origin:
            model.learn(total_timesteps=args.update_steps,
                        progress_bar=False)
        model.save(str(model_path))
        fit_env.close()
        update_seconds = time.monotonic() - started
        update_times.append(update_seconds)
        model_age_bars = 0

        # ---- score the NEXT interval before its rows may train ----
        eval_csv = _slice_csv(
            df, max(0, origin - WARMUP_BARS),
            origin + args.cadence_bars,
            scratch / f"eval_{index}.csv")
        eval_env = _build_env(
            {**config, "eval_seed": args.seed, "env_mode": "training"},
            eval_csv)
        score = _score_interval(model, eval_env)
        eval_env.close()

        peak_rss = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024.0
        record = {
            "record_id": record_id, "run_id": run_id,
            "phase": args.phase, "cadence_bars": args.cadence_bars,
            "lookback": args.lookback, "seed": args.seed,
            "origin_index": index,
            "origin_time": str(dates.iloc[origin]),
            "interval_end_time": str(
                dates.iloc[origin + args.cadence_bars - 1]),
            "interval_return": score.get("interval_return"),
            "interval_trades": score.get("trades"),
            "interval_max_drawdown_fraction": score.get(
                "max_drawdown_fraction"),
            "equity_before": score.get("equity_before"),
            "equity_after": score.get("equity_after"),
            "update_seconds": update_seconds,
            "deadline_seconds": cadence_seconds,
            "deadline_miss": int(update_seconds > cadence_seconds),
            "model_age_bars": model_age_bars,
            "new_bars": args.cadence_bars,
            "update_steps": (args.initial_steps if index == 0
                             else args.update_steps),
            "peak_rss_mb": peak_rss,
            "gpu_json": json.dumps(_gpu_probe()),
            "model_sha256": _sha_file(model_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        con.execute(
            f"INSERT INTO rt_intervals ({','.join(record)}) VALUES"
            f" ({','.join('?' * len(record))})",
            list(record.values()))
        con.commit()
        print(json.dumps({k: record[k] for k in (
            "origin_index", "interval_return", "update_seconds",
            "deadline_miss")}), flush=True)

    ordered = sorted(update_times)

    def _pct(q: float):
        if not ordered:
            return None
        index = min(len(ordered) - 1,
                    max(0, round(q * (len(ordered) - 1))))
        return ordered[int(index)]

    p50 = _pct(0.50)
    p95 = _pct(0.95)
    summary = {
        "schema": "agent_multi.rt_adaptation_summary.v1",
        "run_id": run_id, "identity": run_identity,
        "origins_completed": len(origins),
        "update_seconds_p50": p50, "update_seconds_p95": p95,
        "deadline_seconds": cadence_seconds,
        "deadline_guard": {
            "rule": "p95_update <= (2/3) * cadence",
            "value": (p95 is not None
                      and p95 <= cadence_seconds * 2 / 3),
            "status": "proposed_pending_owner_ratification",
        },
        "note": ("RT0 measures runtime feasibility only; profit/risk"
                 " promotion requires RT1's paired multi-block"
                 " evidence, never a one-month pilot"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps(summary, indent=1, default=str))
    con.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("RT0", "RT1"),
                        required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cadence-bars", type=int, required=True)
    parser.add_argument("--lookback", default="1y",
                        help="e.g. 1y, 2y, 4y or 'expanding'")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--block-start", default="2024-02-01")
    parser.add_argument("--block-days", type=int, default=28)
    parser.add_argument("--initial-steps", type=int, default=20000)
    parser.add_argument("--update-steps", type=int, default=2000)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--update-first-origin", action="store_true")
    parser.add_argument("--device", default="cuda")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
