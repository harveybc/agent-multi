#!/usr/bin/env python3
"""Periodic CPU-only portfolio supervisor runner for Project 3.

This process keeps the portfolio layer fresh while GPU workers continue pulling
weekly SAC subjobs. It intentionally writes stable ``auto_latest_*`` run ids so
the SQLite database does not accumulate endless timestamped portfolio sweeps.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from project3_portfolio_supervisor import (
    DEFAULT_DB,
    DEFAULT_OUTPUT_DIR,
    _write_db,
    _write_outputs,
    fetch_strategy_weeks,
    simulate_portfolio,
    utc_now,
)


DEFAULT_SWEEP: list[dict[str, Any]] = [
    {
        "label": "equal_weight_top_k_capped",
        "method": "equal_weight_top_k_capped",
    },
    {
        "label": "score_weight_top_k_capped",
        "method": "score_weight_top_k_capped",
    },
    {
        "label": "score_inverse_vol_top_k_capped",
        "method": "score_inverse_vol_top_k_capped",
    },
    {
        "label": "score_inverse_cvar_top_k_capped",
        "method": "score_inverse_cvar_top_k_capped",
    },
    {
        "label": "score_inverse_vol_turnover_1bps",
        "method": "score_inverse_vol_turnover_penalty",
        "turnover_penalty_bps": 1.0,
    },
    {
        "label": "score_inverse_cvar_turnover_1bps",
        "method": "score_inverse_cvar_turnover_penalty",
        "turnover_penalty_bps": 1.0,
    },
    {
        "label": "score_inverse_vol_target_60bps",
        "method": "score_inverse_vol_portfolio_vol_target",
        "portfolio_vol_target": 0.006,
    },
]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _selected_specs(methods: list[str] | None) -> list[dict[str, Any]]:
    if not methods:
        return list(DEFAULT_SWEEP)
    wanted = {_slug(method) for method in methods}
    selected = [spec for spec in DEFAULT_SWEEP if _slug(str(spec["label"])) in wanted or _slug(str(spec["method"])) in wanted]
    missing = sorted(wanted - {_slug(str(spec["label"])) for spec in selected} - {_slug(str(spec["method"])) for spec in selected})
    if missing:
        raise SystemExit(f"Unknown portfolio method labels: {', '.join(missing)}")
    return selected


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = args.output_dir / "auto_latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    conn = sqlite3.connect(args.db, timeout=args.sqlite_timeout_sec)
    conn.row_factory = sqlite3.Row
    try:
        weeks = fetch_strategy_weeks(
            conn,
            min_test_start=args.min_test_start,
            require_trade_gate=not args.allow_failed_trade_gate,
        )
        for spec in _selected_specs(args.methods):
            config = {
                "method": spec["method"],
                "max_assets": args.max_assets,
                "min_signal": args.min_signal,
                "lookback_weeks": args.lookback_weeks,
                "risk_floor": args.risk_floor,
                "max_weight": args.max_weight,
                "min_test_start": args.min_test_start,
                "require_trade_gate": not args.allow_failed_trade_gate,
                "min_observations": args.min_observations,
                "max_drawdown_threshold": args.max_drawdown_threshold,
                "drawdown_lookback_weeks": args.drawdown_lookback_weeks,
                "activation_min_signal": args.activation_min_signal,
                "min_train_tail_trades": args.min_train_tail_trades,
                "min_validation_trades": args.min_validation_trades,
                "min_action_entropy": args.min_action_entropy,
                "max_cost_to_gross_edge": args.max_cost_to_gross_edge,
                "require_no_broker_violations": args.require_no_broker_violations,
                "require_no_friday_violations": args.require_no_friday_violations,
                "max_asset_weight": args.max_asset_weight,
                "max_cluster_weight": args.max_cluster_weight,
                "turnover_penalty_bps": spec.get("turnover_penalty_bps", args.turnover_penalty_bps),
                "portfolio_vol_target": spec.get("portfolio_vol_target", args.portfolio_vol_target),
                "runner_label": spec["label"],
            }
            result = simulate_portfolio(
                weeks,
                method=str(config["method"]),
                max_assets=max(1, int(config["max_assets"])),
                min_signal=float(config["min_signal"]),
                lookback_weeks=max(0, int(config["lookback_weeks"])),
                risk_floor=max(1.0e-12, float(config["risk_floor"])),
                max_weight=min(1.0, max(0.01, float(config["max_weight"]))),
                min_observations=max(0, int(config["min_observations"])),
                max_drawdown_threshold=config["max_drawdown_threshold"],
                drawdown_lookback_weeks=max(0, int(config["drawdown_lookback_weeks"])),
                activation_min_signal=config["activation_min_signal"],
                min_train_tail_trades=max(0, int(config["min_train_tail_trades"])),
                min_validation_trades=max(0, int(config["min_validation_trades"])),
                min_action_entropy=config["min_action_entropy"],
                max_cost_to_gross_edge=config["max_cost_to_gross_edge"],
                require_no_broker_violations=bool(config["require_no_broker_violations"]),
                require_no_friday_violations=bool(config["require_no_friday_violations"]),
                max_asset_weight=min(1.0, max(0.01, float(config["max_asset_weight"]))),
                max_cluster_weight=min(1.0, max(0.01, float(config["max_cluster_weight"]))),
                turnover_penalty_bps=max(0.0, float(config["turnover_penalty_bps"] or 0.0)),
                portfolio_vol_target=config["portfolio_vol_target"],
            )
            run_id = f"{args.run_prefix}_{_slug(str(spec['label']))}"
            result["run_id"] = run_id
            result["generated_at"] = utc_now()
            result["config"] = config
            _write_outputs(output_dir, run_id, result)
            if args.write_db:
                _write_db(conn, run_id, config, result)
            summaries.append(
                {
                    "run_id": run_id,
                    "method": config["method"],
                    "label": spec["label"],
                    "summary": result["summary"],
                }
            )
    finally:
        conn.close()
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-prefix", default="auto_latest")
    parser.add_argument("--method", dest="methods", action="append", default=None, help="method or runner label; repeatable")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--sleep-sec", type=int, default=1800)
    parser.add_argument("--sqlite-timeout-sec", type=float, default=60.0)
    parser.add_argument("--write-db", action="store_true", default=True)
    parser.add_argument("--max-assets", type=int, default=5)
    parser.add_argument("--min-signal", type=float, default=0.0)
    parser.add_argument("--lookback-weeks", type=int, default=12)
    parser.add_argument("--risk-floor", type=float, default=1.0e-4)
    parser.add_argument("--max-weight", type=float, default=0.50)
    parser.add_argument("--max-asset-weight", type=float, default=0.60)
    parser.add_argument("--max-cluster-weight", type=float, default=0.80)
    parser.add_argument("--turnover-penalty-bps", type=float, default=0.0)
    parser.add_argument("--portfolio-vol-target", type=float, default=None)
    parser.add_argument("--min-test-start")
    parser.add_argument("--allow-failed-trade-gate", action="store_true")
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--max-drawdown-threshold", type=float, default=0.20)
    parser.add_argument("--drawdown-lookback-weeks", type=int, default=0)
    parser.add_argument("--activation-min-signal", type=float, default=0.0)
    parser.add_argument("--min-train-tail-trades", type=int, default=0)
    parser.add_argument("--min-validation-trades", type=int, default=0)
    parser.add_argument("--min-action-entropy", type=float, default=None)
    parser.add_argument("--max-cost-to-gross-edge", type=float, default=None)
    parser.add_argument("--require-no-broker-violations", action="store_true")
    parser.add_argument("--require-no-friday-violations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        summaries = run_sweep(args)
        print(json.dumps({"generated_at": utc_now(), "event": "portfolio_supervisor_sweep", "runs": summaries}, sort_keys=True))
        if args.once:
            return
        time.sleep(max(60, int(args.sleep_sec)))


if __name__ == "__main__":
    main()
