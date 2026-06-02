#!/usr/bin/env python3
"""Project 3 no-trade preflight.

Runs the exact configured gym-fx environment with forced long/short actions.
This is a first-layer execution-path check: if forced actions cannot produce
orders or closed trades, do not waste GPU training a policy on that config.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_plugins.gym_fx_env import Plugin as EnvPlugin  # noqa: E402


SCHEMA_VERSION = "project3_no_trade_preflight_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def forced_action(config: dict[str, Any], step: int, hold_bars: int) -> Any:
    side = 1 if ((step // max(1, hold_bars)) % 2 == 0) else 2
    if str(config.get("action_space_mode") or "").lower() == "continuous":
        return [1.0] if side == 1 else [-1.0]
    return side


def diagnose(payload: dict[str, Any]) -> str:
    if payload["closed_trades"] >= payload["min_trades"]:
        return "execution_path_closes_trades"
    if payload["entry_orders_submitted"] > 0:
        return "orders_submitted_but_no_closed_trades_in_preflight_window"
    if payload["entry_actions_seen"] <= 0:
        return "forced_actions_not_seen_by_strategy"
    blocked = {
        "atr_warmup": payload["blocked_atr_warmup"],
        "session_filter": payload["blocked_session_filter"],
        "non_positive_atr": payload["blocked_non_positive_atr"],
        "non_positive_size": payload["blocked_non_positive_size"],
        "non_positive_price": payload["blocked_non_positive_price"],
    }
    reason = max(blocked, key=lambda key: blocked[key])
    if blocked[reason] > 0:
        return f"execution_guard_blocked_all_forced_actions:{reason}"
    return "forced_actions_no_orders_unknown"


def run_preflight(config: dict[str, Any], *, max_steps: int, hold_bars: int, min_trades: int) -> dict[str, Any]:
    env_plugin = EnvPlugin(config)
    env = env_plugin.make_env(config)
    try:
        obs, info = env.reset(seed=safe_int(config.get("eval_seed", config.get("train_seed", 0))))
        done = False
        steps = 0
        last_info = dict(info or {})
        first_trade_step: int | None = None
        first_order_step: int | None = None
        while not done and steps < max_steps:
            action = forced_action(config, steps, hold_bars)
            obs, reward, terminated, truncated, last_info = env.step(action)
            steps += 1
            exec_diag = last_info.get("execution_diagnostics") if isinstance(last_info, dict) else {}
            trades = safe_int(last_info.get("trades") if isinstance(last_info, dict) else 0)
            orders = safe_int(exec_diag.get("entry_orders_submitted") if isinstance(exec_diag, dict) else 0)
            if orders > 0 and first_order_step is None:
                first_order_step = steps
            if trades >= min_trades:
                first_trade_step = steps
                break
            done = bool(terminated or truncated)
        summary = env.summary() if hasattr(env, "summary") else {}
    finally:
        try:
            env_plugin.close()
        except Exception:
            pass

    exec_diag = {}
    action_diag = {}
    if isinstance(last_info, dict):
        exec_diag = last_info.get("execution_diagnostics") or {}
        action_diag = last_info.get("action_diagnostics") or {}
    live_trades = safe_int(last_info.get("trades", 0) if isinstance(last_info, dict) else 0)
    analyzer_trades = safe_int(summary.get("trades_total"))
    closed_trades = max(live_trades, analyzer_trades)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "run_id": config.get("run_id") or "",
        "config_file": "",
        "asset": config.get("asset") or "",
        "timeframe": config.get("timeframe") or "",
        "agent_plugin": config.get("agent_plugin") or "",
        "features_preset": config.get("features_preset") or "",
        "input_data_file": config.get("input_data_file") or "",
        "max_steps": int(max_steps),
        "steps_run": int(steps),
        "hold_bars": int(hold_bars),
        "min_trades": int(min_trades),
        "closed_trades": closed_trades,
        "live_closed_trades": live_trades,
        "analyzer_closed_trades": analyzer_trades,
        "entry_actions_seen": safe_int(exec_diag.get("entry_actions_seen")),
        "entry_orders_submitted": safe_int(exec_diag.get("entry_orders_submitted")),
        "blocked_atr_warmup": safe_int(exec_diag.get("blocked_atr_warmup")),
        "blocked_session_filter": safe_int(exec_diag.get("blocked_session_filter")),
        "blocked_non_positive_atr": safe_int(exec_diag.get("blocked_non_positive_atr")),
        "blocked_non_positive_size": safe_int(exec_diag.get("blocked_non_positive_size")),
        "blocked_non_positive_price": safe_int(exec_diag.get("blocked_non_positive_price")),
        "action_non_hold_rate": (
            safe_int(action_diag.get("non_hold_actions")) / max(1, safe_int(action_diag.get("steps")))
            if isinstance(action_diag, dict) else 1.0
        ),
        "first_order_step": first_order_step,
        "first_trade_step": first_trade_step,
        "final_equity": safe_float(summary.get("final_equity")),
        "total_return": safe_float(summary.get("total_return")),
    }
    payload["preflight_passed"] = bool(payload["closed_trades"] >= payload["min_trades"])
    payload["execution_path_orders_ok"] = bool(payload["entry_orders_submitted"] > 0)
    payload["diagnosis"] = diagnose(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--hold-bars", type=int, default=16)
    parser.add_argument("--min-trades", type=int, default=1)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        payload = run_preflight(
            config,
            max_steps=max(1, int(args.max_steps)),
            hold_bars=max(1, int(args.hold_bars)),
            min_trades=max(1, int(args.min_trades)),
        )
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "run_id": config.get("run_id") or "",
            "config_file": str(config_path),
            "asset": config.get("asset") or "",
            "timeframe": config.get("timeframe") or "",
            "agent_plugin": config.get("agent_plugin") or "",
            "features_preset": config.get("features_preset") or "",
            "input_data_file": config.get("input_data_file") or "",
            "max_steps": max(1, int(args.max_steps)),
            "hold_bars": max(1, int(args.hold_bars)),
            "min_trades": max(1, int(args.min_trades)),
            "closed_trades": 0,
            "entry_actions_seen": 0,
            "entry_orders_submitted": 0,
            "preflight_passed": False,
            "execution_path_orders_ok": False,
            "diagnosis": "preflight_env_construction_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    payload["config_file"] = str(config_path)
    output = Path(args.output).resolve()
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload["preflight_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
