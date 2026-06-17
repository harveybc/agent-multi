#!/usr/bin/env python3
"""Weekly portfolio supervisor simulation for Project 3 pool results.

The per-asset RL jobs create weekly strategy streams. This tool simulates the
upper layer that a production service needs: every weekend select active
asset/strategy streams and allocate order-size weights for the next week using
only information available before that next week.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DB = Path(
    "/home/harveybc/Documents/GitHub/financial-data/experiments/weekly_walkforward_pool/project3_weekly_pool.sqlite"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/harveybc/Documents/GitHub/financial-data/experiments/portfolio_supervisor"
)


@dataclass(frozen=True)
class StrategyWeek:
    job_id: str
    asset: str
    timeframe: str
    asset_key: str
    weekly_anchor_id: str
    test_start: str
    test_end: str
    composite_signal: float
    train_tail_return: float
    validation_return: float
    realized_test_return: float
    trade_gate_passed: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _parse_result(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    var = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(max(0.0, var))


def _cvar(values: list[float], alpha: float = 0.20) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = max(1, math.ceil(len(ordered) * alpha))
    return _mean(ordered[:n])


def _max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns:
        equity *= 1.0 + ret
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)
    return max_dd


def fetch_strategy_weeks(
    conn: sqlite3.Connection,
    *,
    min_test_start: str | None = None,
    require_trade_gate: bool = True,
) -> list[StrategyWeek]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            j.external_id AS job_id,
            j.asset,
            j.timeframe,
            s.weekly_anchor_id,
            s.test_start,
            s.test_end,
            s.result_json
        FROM subjobs s
        JOIN jobs j ON j.id=s.job_id
        WHERE s.status='done'
          AND s.result_json IS NOT NULL
        ORDER BY s.test_start, j.asset, j.timeframe, j.external_id
        """
    ).fetchall()
    out: list[StrategyWeek] = []
    for row in rows:
        if min_test_start and str(row["test_start"]) < min_test_start:
            continue
        result = _parse_result(row["result_json"])
        trade_gate_passed = bool(result.get("trade_gate_passed", False))
        if require_trade_gate and not trade_gate_passed:
            continue
        composite = _safe_float(result.get("train_validation_composite_score"))
        train_tail = _safe_float(result.get("train_tail_total_return"))
        validation = _safe_float(result.get("validation_total_return"))
        test_ret = _safe_float(result.get("test_total_return"))
        asset_key = f"{row['asset']}_{row['timeframe']}"
        out.append(
            StrategyWeek(
                job_id=str(row["job_id"]),
                asset=str(row["asset"]),
                timeframe=str(row["timeframe"]),
                asset_key=asset_key,
                weekly_anchor_id=str(row["weekly_anchor_id"]),
                test_start=str(row["test_start"]),
                test_end=str(row["test_end"]),
                composite_signal=composite,
                train_tail_return=train_tail,
                validation_return=validation,
                realized_test_return=test_ret,
                trade_gate_passed=trade_gate_passed,
            )
        )
    return out


def _select_candidates_for_week(
    candidates: Iterable[StrategyWeek],
    *,
    max_assets: int,
    min_signal: float,
) -> list[StrategyWeek]:
    best_by_asset: dict[str, StrategyWeek] = {}
    for item in candidates:
        if item.composite_signal < min_signal:
            continue
        current = best_by_asset.get(item.asset_key)
        if current is None or item.composite_signal > current.composite_signal:
            best_by_asset[item.asset_key] = item
    return sorted(
        best_by_asset.values(),
        key=lambda row: row.composite_signal,
        reverse=True,
    )[:max_assets]


def _historical_returns(
    by_job: dict[str, list[StrategyWeek]],
    *,
    job_id: str,
    current_test_start: str,
    lookback_weeks: int,
) -> list[float]:
    previous = [
        row.realized_test_return
        for row in by_job.get(job_id, [])
        if row.test_start < current_test_start
    ]
    return previous[-lookback_weeks:] if lookback_weeks > 0 else previous


def _normalize(weights: dict[str, float], *, max_weight: float) -> dict[str, float]:
    cleaned = {key: max(0.0, value) for key, value in weights.items()}
    if not cleaned or sum(cleaned.values()) <= 0.0:
        n = max(1, len(cleaned))
        cleaned = {key: 1.0 / n for key in cleaned}
    total = sum(cleaned.values())
    normalized = {key: value / total for key, value in cleaned.items()}
    capped = {key: min(max_weight, value) for key, value in normalized.items()}
    total = sum(capped.values())
    if total <= 0.0:
        n = max(1, len(capped))
        return {key: 1.0 / n for key in capped}
    return {key: value / total for key, value in capped.items()}


def allocate_week(
    selected: list[StrategyWeek],
    *,
    by_job: dict[str, list[StrategyWeek]],
    method: str,
    lookback_weeks: int,
    risk_floor: float,
    max_weight: float,
) -> dict[str, float]:
    if not selected:
        return {}
    if method == "equal_weight":
        return {row.job_id: 1.0 / len(selected) for row in selected}
    raw: dict[str, float] = {}
    for row in selected:
        history = _historical_returns(
            by_job,
            job_id=row.job_id,
            current_test_start=row.test_start,
            lookback_weeks=lookback_weeks,
        )
        signal = max(0.0, row.composite_signal)
        if method == "score_weight":
            raw[row.job_id] = signal + risk_floor
        elif method == "score_inverse_vol":
            raw[row.job_id] = (signal + risk_floor) / max(risk_floor, _std(history))
        elif method == "score_inverse_cvar":
            downside = abs(min(0.0, _cvar(history)))
            raw[row.job_id] = (signal + risk_floor) / max(risk_floor, downside)
        else:
            raise ValueError(
                "method must be one of equal_weight, score_weight, "
                "score_inverse_vol, score_inverse_cvar"
            )
    return _normalize(raw, max_weight=max_weight)


def simulate_portfolio(
    weeks: list[StrategyWeek],
    *,
    method: str,
    max_assets: int,
    min_signal: float,
    lookback_weeks: int,
    risk_floor: float,
    max_weight: float,
) -> dict[str, Any]:
    by_week: dict[str, list[StrategyWeek]] = {}
    by_job: dict[str, list[StrategyWeek]] = {}
    for row in weeks:
        by_week.setdefault(row.test_start, []).append(row)
        by_job.setdefault(row.job_id, []).append(row)

    allocations: list[dict[str, Any]] = []
    weekly: list[dict[str, Any]] = []
    for test_start in sorted(by_week):
        selected = _select_candidates_for_week(
            by_week[test_start],
            max_assets=max_assets,
            min_signal=min_signal,
        )
        weights = allocate_week(
            selected,
            by_job=by_job,
            method=method,
            lookback_weeks=lookback_weeks,
            risk_floor=risk_floor,
            max_weight=max_weight,
        )
        portfolio_return = 0.0
        for row in selected:
            weight = weights.get(row.job_id, 0.0)
            portfolio_return += weight * row.realized_test_return
            allocations.append(
                {
                    "test_start": row.test_start,
                    "test_end": row.test_end,
                    "asset_key": row.asset_key,
                    "job_id": row.job_id,
                    "weight": weight,
                    "composite_signal": row.composite_signal,
                    "train_tail_return": row.train_tail_return,
                    "validation_return": row.validation_return,
                    "realized_test_return": row.realized_test_return,
                }
            )
        weekly.append(
            {
                "test_start": test_start,
                "selected_assets": len(selected),
                "portfolio_return": portfolio_return,
            }
        )

    returns = [row["portfolio_return"] for row in weekly]
    cumulative = 1.0
    for ret in returns:
        cumulative *= 1.0 + ret
    summary = {
        "method": method,
        "weeks": len(weekly),
        "allocations": len(allocations),
        "max_assets": max_assets,
        "min_signal": min_signal,
        "lookback_weeks": lookback_weeks,
        "risk_floor": risk_floor,
        "max_weight": max_weight,
        "mean_weekly_return": _mean(returns),
        "std_weekly_return": _std(returns),
        "sharpe_like_weekly": (_mean(returns) / _std(returns)) if _std(returns) > 0 else 0.0,
        "cvar_20_weekly": _cvar(returns, 0.20),
        "max_drawdown": _max_drawdown(returns),
        "cumulative_return": cumulative - 1.0,
    }
    return {"summary": summary, "weekly": weekly, "allocations": allocations}


def _write_outputs(output_dir: Path, run_id: str, result: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}.json"
    weekly_path = output_dir / f"{run_id}_weekly.csv"
    allocation_path = output_dir / f"{run_id}_allocations.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with weekly_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_start", "selected_assets", "portfolio_return"])
        writer.writeheader()
        writer.writerows(result["weekly"])
    with allocation_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "test_start",
            "test_end",
            "asset_key",
            "job_id",
            "weight",
            "composite_signal",
            "train_tail_return",
            "validation_return",
            "realized_test_return",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["allocations"])
    return {"json": str(json_path), "weekly_csv": str(weekly_path), "allocations_csv": str(allocation_path)}


def _write_db(conn: sqlite3.Connection, run_id: str, config: dict[str, Any], result: dict[str, Any]) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS portfolio_runs (
            run_id TEXT PRIMARY KEY,
            generated_at TEXT NOT NULL,
            config_json TEXT NOT NULL,
            summary_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolio_weekly_returns (
            run_id TEXT NOT NULL,
            test_start TEXT NOT NULL,
            selected_assets INTEGER NOT NULL,
            portfolio_return REAL NOT NULL,
            PRIMARY KEY(run_id, test_start)
        );
        CREATE TABLE IF NOT EXISTS portfolio_allocations (
            run_id TEXT NOT NULL,
            test_start TEXT NOT NULL,
            asset_key TEXT NOT NULL,
            job_id TEXT NOT NULL,
            weight REAL NOT NULL,
            composite_signal REAL NOT NULL,
            train_tail_return REAL NOT NULL,
            validation_return REAL NOT NULL,
            realized_test_return REAL NOT NULL,
            PRIMARY KEY(run_id, test_start, asset_key, job_id)
        );
        """
    )
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_runs(run_id, generated_at, config_json, summary_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, utc_now(), json.dumps(config, sort_keys=True), json.dumps(result["summary"], sort_keys=True)),
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO portfolio_weekly_returns(
                run_id, test_start, selected_assets, portfolio_return
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (run_id, row["test_start"], row["selected_assets"], row["portfolio_return"])
                for row in result["weekly"]
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO portfolio_allocations(
                run_id, test_start, asset_key, job_id, weight, composite_signal,
                train_tail_return, validation_return, realized_test_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    row["test_start"],
                    row["asset_key"],
                    row["job_id"],
                    row["weight"],
                    row["composite_signal"],
                    row["train_tail_return"],
                    row["validation_return"],
                    row["realized_test_return"],
                )
                for row in result["allocations"]
            ],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--method",
        choices=("equal_weight", "score_weight", "score_inverse_vol", "score_inverse_cvar"),
        default="score_inverse_vol",
    )
    parser.add_argument("--max-assets", type=int, default=5)
    parser.add_argument("--min-signal", type=float, default=0.0)
    parser.add_argument("--lookback-weeks", type=int, default=12)
    parser.add_argument("--risk-floor", type=float, default=1.0e-4)
    parser.add_argument("--max-weight", type=float, default=0.50)
    parser.add_argument("--min-test-start")
    parser.add_argument("--allow-failed-trade-gate", action="store_true")
    parser.add_argument("--write-db", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    config = {
        "method": args.method,
        "max_assets": args.max_assets,
        "min_signal": args.min_signal,
        "lookback_weeks": args.lookback_weeks,
        "risk_floor": args.risk_floor,
        "max_weight": args.max_weight,
        "min_test_start": args.min_test_start,
        "require_trade_gate": not args.allow_failed_trade_gate,
    }
    weeks = fetch_strategy_weeks(
        conn,
        min_test_start=args.min_test_start,
        require_trade_gate=not args.allow_failed_trade_gate,
    )
    result = simulate_portfolio(
        weeks,
        method=args.method,
        max_assets=max(1, args.max_assets),
        min_signal=args.min_signal,
        lookback_weeks=max(0, args.lookback_weeks),
        risk_floor=max(1.0e-12, args.risk_floor),
        max_weight=min(1.0, max(0.01, args.max_weight)),
    )
    run_id = args.run_id or f"portfolio_{args.method}_{utc_now().replace(':', '').replace('-', '')}"
    result["run_id"] = run_id
    result["generated_at"] = utc_now()
    result["config"] = config
    paths = _write_outputs(args.output_dir, run_id, result)
    if args.write_db:
        _write_db(conn, run_id, config, result)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "summary": result["summary"],
                "paths": paths,
                "write_db": bool(args.write_db),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
