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

try:  # numpy enables the covariance-aware allocators; we degrade gracefully.
    import numpy as np
except Exception:  # pragma: no cover - exercised only without numpy
    np = None


ALLOCATION_METHODS = (
    "equal_weight",
    "score_weight",
    "score_inverse_vol",
    "score_inverse_cvar",
    "min_variance",
    "min_semivariance",
    "equal_weight_top_k_capped",
    "score_weight_top_k_capped",
    "score_inverse_vol_top_k_capped",
    "score_inverse_cvar_top_k_capped",
    "score_inverse_vol_turnover_penalty",
    "score_inverse_cvar_turnover_penalty",
    "score_inverse_vol_portfolio_vol_target",
)


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
    train_tail_trades: int = 0
    validation_trades: int = 0
    action_entropy: float = 0.0
    validation_cost_to_gross_edge: float = 0.0
    broker_policy_violations: int = 0
    friday_force_close_violations: int = 0
    asset_cluster: str = ""


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_result(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _nested_get(mapping: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


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
        train_tail_trades = _safe_int(
            result.get(
                "train_tail_trades_total",
                _nested_get(result, ("splits", "train_tail", "trades_total"), 0),
            )
        )
        validation_trades = _safe_int(
            result.get(
                "validation_trades_total",
                _nested_get(result, ("splits", "validation", "trades_total"), 0),
            )
        )
        action_entropy = _safe_float(
            result.get(
                "validation_action_entropy",
                result.get("action_entropy", result.get("policy_action_entropy", 0.0)),
            )
        )
        validation_cost_to_gross_edge = _safe_float(
            result.get(
                "validation_cost_to_gross_edge",
                result.get("cost_to_gross_edge", 0.0),
            )
        )
        broker_policy_violations = _safe_int(
            result.get("broker_policy_violations", result.get("validation_broker_policy_violations", 0))
        )
        friday_force_close_violations = _safe_int(
            result.get(
                "friday_force_close_violations",
                result.get("validation_friday_force_close_violations", 0),
            )
        )
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
                train_tail_trades=train_tail_trades,
                validation_trades=validation_trades,
                action_entropy=action_entropy,
                validation_cost_to_gross_edge=validation_cost_to_gross_edge,
                broker_policy_violations=broker_policy_violations,
                friday_force_close_violations=friday_force_close_violations,
                asset_cluster=str(result.get("asset_cluster") or row["asset"]),
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
    """Project non-negative raw weights onto the long-only capped simplex.

    Negatives are clipped to zero and the result sums to 1. The ``max_weight``
    cap is enforced by water-filling (iteratively pin over-cap names and
    redistribute the remainder), so no returned weight exceeds the cap whenever
    the cap is feasible (``max_weight * n >= 1``); if the cap is infeasible it is
    relaxed to ``1/n`` (equal weight), the tightest feasible cap.
    """
    cleaned = {key: max(0.0, value) for key, value in weights.items()}
    n = len(cleaned)
    if n == 0:
        return {}
    if sum(cleaned.values()) <= 0.0:
        cleaned = {key: 1.0 for key in cleaned}
    cap = max(max_weight, 1.0 / n)
    total = sum(cleaned.values())
    current = {key: value / total for key, value in cleaned.items()}
    for _ in range(n + 1):
        over = {key for key, value in current.items() if value > cap + 1.0e-12}
        if not over:
            break
        remaining = 1.0 - cap * len(over)
        under = {key: value for key, value in current.items() if key not in over}
        under_total = sum(under.values())
        updated: dict[str, float] = {}
        for key, value in current.items():
            if key in over:
                updated[key] = cap
            elif under_total > 0.0:
                updated[key] = under[key] / under_total * remaining
            else:
                updated[key] = remaining / max(1, len(under))
        current = updated
    return current


def _aligned_returns_matrix(
    selected: list[StrategyWeek],
    *,
    by_job: dict[str, list[StrategyWeek]],
    current_test_start: str,
    lookback_weeks: int,
) -> tuple[list[str], list[list[float]]]:
    """Return (job_ids, matrix) of prior weekly returns aligned by week.

    Only weeks strictly before ``current_test_start`` are used and only weeks for
    which *every* selected stream has an observation, so the covariance is built
    from genuinely contemporaneous, leak-free history.
    """
    jobs = [row.job_id for row in selected]
    per_job: dict[str, dict[str, float]] = {}
    for row in selected:
        per_job[row.job_id] = {
            prior.test_start: prior.realized_test_return
            for prior in by_job.get(row.job_id, [])
            if prior.test_start < current_test_start
        }
    if not per_job:
        return jobs, []
    common = set.intersection(*(set(d.keys()) for d in per_job.values()))
    weeks = sorted(common)
    if lookback_weeks > 0:
        weeks = weeks[-lookback_weeks:]
    matrix = [[per_job[job][week] for job in jobs] for week in weeks]
    return jobs, matrix


def _diagonal_risk_weights(
    selected: list[StrategyWeek],
    *,
    by_job: dict[str, list[StrategyWeek]],
    lookback_weeks: int,
    risk_floor: float,
    max_weight: float,
    kind: str,
) -> dict[str, float]:
    """Robust inverse-(semi)variance fallback using each stream's own history."""
    raw: dict[str, float] = {}
    for row in selected:
        history = _historical_returns(
            by_job,
            job_id=row.job_id,
            current_test_start=row.test_start,
            lookback_weeks=lookback_weeks,
        )
        if kind == "semi":
            downside = [min(0.0, r) for r in history]
            variance = _mean([d * d for d in downside]) if downside else 0.0
        else:
            variance = _std(history) ** 2
        raw[row.job_id] = 1.0 / max(risk_floor, variance)
    return _normalize(raw, max_weight=max_weight)


def _covariance_weights(
    selected: list[StrategyWeek],
    *,
    by_job: dict[str, list[StrategyWeek]],
    lookback_weeks: int,
    risk_floor: float,
    max_weight: float,
    kind: str,
    min_obs: int = 3,
) -> dict[str, float]:
    """Long-only minimum-(semi)variance Markowitz-style weights.

    Builds a covariance (``kind='full'``) or semicovariance (``kind='semi'``,
    downside relative to zero) matrix from aligned prior weekly returns and
    solves the closed-form minimum-variance portfolio, then projects to the
    long-only simplex (clip negatives, cap, renormalize via :func:`_normalize`).

    Falls back to :func:`_diagonal_risk_weights` (inverse-variance) when numpy
    is unavailable, when there is too little aligned history, or when the
    covariance matrix is singular. This keeps the allocator dependency-light and
    always reproducible.
    """
    current = selected[0].test_start
    jobs, matrix = _aligned_returns_matrix(
        selected,
        by_job=by_job,
        current_test_start=current,
        lookback_weeks=lookback_weeks,
    )
    if np is None or len(matrix) < max(min_obs, 2) or len(jobs) < 2:
        return _diagonal_risk_weights(
            selected,
            by_job=by_job,
            lookback_weeks=lookback_weeks,
            risk_floor=risk_floor,
            max_weight=max_weight,
            kind=kind,
        )
    returns = np.asarray(matrix, dtype=np.float64)
    n_assets = returns.shape[1]
    if kind == "semi":
        downside = np.minimum(returns, 0.0)
        cov = (downside.T @ downside) / downside.shape[0]
    else:
        cov = np.cov(returns, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov) + np.eye(n_assets) * max(risk_floor, 1.0e-12)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:  # pragma: no cover - singular fallback
        return _diagonal_risk_weights(
            selected,
            by_job=by_job,
            lookback_weeks=lookback_weeks,
            risk_floor=risk_floor,
            max_weight=max_weight,
            kind=kind,
        )
    raw_vec = inv @ np.ones(n_assets)
    weights = {job: float(value) for job, value in zip(jobs, raw_vec)}
    if all(value <= 0.0 for value in weights.values()):
        return _diagonal_risk_weights(
            selected,
            by_job=by_job,
            lookback_weeks=lookback_weeks,
            risk_floor=risk_floor,
            max_weight=max_weight,
            kind=kind,
        )
    return _normalize(weights, max_weight=max_weight)


def _base_allocation_method(method: str) -> str:
    aliases = {
        "equal_weight_top_k_capped": "equal_weight",
        "score_weight_top_k_capped": "score_weight",
        "score_inverse_vol_top_k_capped": "score_inverse_vol",
        "score_inverse_cvar_top_k_capped": "score_inverse_cvar",
        "score_inverse_vol_turnover_penalty": "score_inverse_vol",
        "score_inverse_cvar_turnover_penalty": "score_inverse_cvar",
        "score_inverse_vol_portfolio_vol_target": "score_inverse_vol",
    }
    return aliases.get(method, method)


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
    method = _base_allocation_method(method)
    if method == "equal_weight":
        return {row.job_id: 1.0 / len(selected) for row in selected}
    if method in ("min_variance", "min_semivariance"):
        return _covariance_weights(
            selected,
            by_job=by_job,
            lookback_weeks=lookback_weeks,
            risk_floor=risk_floor,
            max_weight=max_weight,
            kind="semi" if method == "min_semivariance" else "full",
        )
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
            raise ValueError(f"method must be one of {list(ALLOCATION_METHODS)}")
    return _normalize(raw, max_weight=max_weight)


def _cap_group_exposure(
    weights: dict[str, float],
    selected: list[StrategyWeek],
    *,
    group_getter: str,
    max_group_weight: float,
) -> dict[str, float]:
    """Cap aggregate exposure by asset/cluster while preserving long-only weights.

    The cap is relaxed to the tightest feasible value when the configured value
    is impossible for the number of present groups. This is research accounting:
    better an explicit, deterministic relaxation than a silent sum below one.
    """
    if not weights or max_group_weight >= 1.0:
        return weights
    by_job = {row.job_id: row for row in selected}
    groups = {
        job_id: str(getattr(by_job[job_id], group_getter) or by_job[job_id].asset)
        for job_id in weights
        if job_id in by_job
    }
    group_names = sorted(set(groups.values()))
    if len(group_names) <= 1:
        return weights
    cap = max(max_group_weight, 1.0 / len(group_names))
    current = _normalize(weights, max_weight=1.0)
    for _ in range(len(group_names) + 1):
        totals: dict[str, float] = {}
        for job_id, value in current.items():
            totals[groups[job_id]] = totals.get(groups[job_id], 0.0) + value
        over_groups = {group for group, total in totals.items() if total > cap + 1.0e-12}
        if not over_groups:
            break
        locked_total = cap * len(over_groups)
        free_jobs = [job_id for job_id in current if groups[job_id] not in over_groups]
        free_total = sum(current[job_id] for job_id in free_jobs)
        updated: dict[str, float] = {}
        for job_id, value in current.items():
            group = groups[job_id]
            if group in over_groups:
                group_total = totals[group]
                updated[job_id] = value / group_total * cap if group_total > 0 else 0.0
            elif free_total > 0.0:
                updated[job_id] = value / free_total * max(0.0, 1.0 - locked_total)
            else:
                updated[job_id] = max(0.0, 1.0 - locked_total) / max(1, len(free_jobs))
        current = updated
    return current


def _portfolio_history_vol(
    selected: list[StrategyWeek],
    weights: dict[str, float],
    *,
    by_job: dict[str, list[StrategyWeek]],
    current_test_start: str,
    lookback_weeks: int,
) -> float:
    jobs, matrix = _aligned_returns_matrix(
        selected,
        by_job=by_job,
        current_test_start=current_test_start,
        lookback_weeks=lookback_weeks,
    )
    if len(matrix) < 2:
        return 0.0
    portfolio_returns = [
        sum(weights.get(job, 0.0) * value for job, value in zip(jobs, row))
        for row in matrix
    ]
    return _std(portfolio_returns)


def _apply_portfolio_vol_target(
    selected: list[StrategyWeek],
    weights: dict[str, float],
    *,
    by_job: dict[str, list[StrategyWeek]],
    current_test_start: str,
    lookback_weeks: int,
    portfolio_vol_target: float | None,
) -> tuple[dict[str, float], float, float]:
    if not weights or portfolio_vol_target is None or portfolio_vol_target <= 0.0:
        return weights, 0.0, 0.0
    vol = _portfolio_history_vol(
        selected,
        weights,
        by_job=by_job,
        current_test_start=current_test_start,
        lookback_weeks=lookback_weeks,
    )
    if vol <= 0.0 or vol <= portfolio_vol_target:
        return weights, 0.0, vol
    scale = max(0.0, min(1.0, portfolio_vol_target / vol))
    scaled = {job_id: value * scale for job_id, value in weights.items()}
    return scaled, 1.0 - sum(scaled.values()), vol


def _activation_decisions(
    candidates: list[StrategyWeek],
    *,
    by_job: dict[str, list[StrategyWeek]],
    lookback_weeks: int,
    min_observations: int,
    max_drawdown_threshold: float | None,
    drawdown_lookback_weeks: int,
    activation_min_signal: float | None,
    min_train_tail_trades: int,
    min_validation_trades: int,
    min_action_entropy: float | None,
    max_cost_to_gross_edge: float | None,
    require_no_broker_violations: bool,
    require_no_friday_violations: bool,
) -> tuple[list[StrategyWeek], list[dict[str, Any]]]:
    """Apply the no-trade / activation rule and return (active, decisions).

    A stream is flagged inactive (no-trade for the week) if any rule fires:
      * composite signal at or below the activation threshold;
      * too few prior weekly observations in the lookback;
      * recent drawdown over the lookback exceeds the threshold.
    """
    active: list[StrategyWeek] = []
    decisions: list[dict[str, Any]] = []
    dd_lookback = drawdown_lookback_weeks if drawdown_lookback_weeks > 0 else lookback_weeks
    for row in candidates:
        reasons: list[str] = []
        if activation_min_signal is not None and row.composite_signal <= activation_min_signal:
            reasons.append("composite_signal_at_or_below_threshold")
        if min_train_tail_trades > 0 and row.train_tail_trades < min_train_tail_trades:
            reasons.append("insufficient_train_tail_trades")
        if min_validation_trades > 0 and row.validation_trades < min_validation_trades:
            reasons.append("insufficient_validation_trades")
        if min_action_entropy is not None and row.action_entropy < min_action_entropy:
            reasons.append("action_entropy_below_threshold")
        if (
            max_cost_to_gross_edge is not None
            and row.validation_cost_to_gross_edge > max_cost_to_gross_edge
        ):
            reasons.append("cost_to_gross_edge_above_threshold")
        if require_no_broker_violations and row.broker_policy_violations > 0:
            reasons.append("broker_policy_violations")
        if require_no_friday_violations and row.friday_force_close_violations > 0:
            reasons.append("friday_force_close_violations")
        history = _historical_returns(
            by_job,
            job_id=row.job_id,
            current_test_start=row.test_start,
            lookback_weeks=lookback_weeks,
        )
        if min_observations > 0 and len(history) < min_observations:
            reasons.append("insufficient_lookback_observations")
        recent_drawdown = 0.0
        if max_drawdown_threshold is not None:
            dd_history = _historical_returns(
                by_job,
                job_id=row.job_id,
                current_test_start=row.test_start,
                lookback_weeks=dd_lookback,
            )
            recent_drawdown = _max_drawdown(dd_history)
            if recent_drawdown <= -abs(max_drawdown_threshold):
                reasons.append("recent_drawdown_exceeds_threshold")
        is_active = not reasons
        decisions.append(
            {
                "test_start": row.test_start,
                "asset_key": row.asset_key,
                "job_id": row.job_id,
                "active": is_active,
                "composite_signal": row.composite_signal,
                "observations": len(history),
                "recent_drawdown": recent_drawdown,
                "train_tail_trades": row.train_tail_trades,
                "validation_trades": row.validation_trades,
                "action_entropy": row.action_entropy,
                "validation_cost_to_gross_edge": row.validation_cost_to_gross_edge,
                "broker_policy_violations": row.broker_policy_violations,
                "friday_force_close_violations": row.friday_force_close_violations,
                "reasons": reasons,
            }
        )
        if is_active:
            active.append(row)
    return active, decisions


def simulate_portfolio(
    weeks: list[StrategyWeek],
    *,
    method: str,
    max_assets: int,
    min_signal: float,
    lookback_weeks: int,
    risk_floor: float,
    max_weight: float,
    min_observations: int = 0,
    max_drawdown_threshold: float | None = None,
    drawdown_lookback_weeks: int = 0,
    activation_min_signal: float | None = None,
    min_train_tail_trades: int = 0,
    min_validation_trades: int = 0,
    min_action_entropy: float | None = None,
    max_cost_to_gross_edge: float | None = None,
    require_no_broker_violations: bool = False,
    require_no_friday_violations: bool = False,
    max_asset_weight: float = 1.0,
    max_cluster_weight: float = 1.0,
    turnover_penalty_bps: float = 0.0,
    portfolio_vol_target: float | None = None,
) -> dict[str, Any]:
    by_week: dict[str, list[StrategyWeek]] = {}
    by_job: dict[str, list[StrategyWeek]] = {}
    for row in weeks:
        by_week.setdefault(row.test_start, []).append(row)
        by_job.setdefault(row.job_id, []).append(row)

    allocations: list[dict[str, Any]] = []
    weekly: list[dict[str, Any]] = []
    activations: list[dict[str, Any]] = []
    cutoff_manifests: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    for test_start in sorted(by_week):
        candidates = _select_candidates_for_week(
            by_week[test_start],
            max_assets=max_assets,
            min_signal=min_signal,
        )
        selected, decisions = _activation_decisions(
            candidates,
            by_job=by_job,
            lookback_weeks=lookback_weeks,
            min_observations=min_observations,
            max_drawdown_threshold=max_drawdown_threshold,
            drawdown_lookback_weeks=drawdown_lookback_weeks,
            activation_min_signal=activation_min_signal,
            min_train_tail_trades=min_train_tail_trades,
            min_validation_trades=min_validation_trades,
            min_action_entropy=min_action_entropy,
            max_cost_to_gross_edge=max_cost_to_gross_edge,
            require_no_broker_violations=require_no_broker_violations,
            require_no_friday_violations=require_no_friday_violations,
        )
        activations.extend(decisions)
        weights = allocate_week(
            selected,
            by_job=by_job,
            method=method,
            lookback_weeks=lookback_weeks,
            risk_floor=risk_floor,
            max_weight=max_weight,
        )
        weights = _cap_group_exposure(
            weights,
            selected,
            group_getter="asset",
            max_group_weight=max_asset_weight,
        )
        weights = _cap_group_exposure(
            weights,
            selected,
            group_getter="asset_cluster",
            max_group_weight=max_cluster_weight,
        )
        weights = _cap_group_exposure(
            weights,
            selected,
            group_getter="asset",
            max_group_weight=max_asset_weight,
        )
        cash_weight = 0.0
        realized_prior_vol = 0.0
        if method == "score_inverse_vol_portfolio_vol_target":
            weights, cash_weight, realized_prior_vol = _apply_portfolio_vol_target(
                selected,
                weights,
                by_job=by_job,
                current_test_start=test_start,
                lookback_weeks=lookback_weeks,
                portfolio_vol_target=portfolio_vol_target,
            )
        turnover = sum(
            abs(weights.get(job_id, 0.0) - previous_weights.get(job_id, 0.0))
            for job_id in set(weights) | set(previous_weights)
        )
        rebalance_cost = turnover * max(0.0, turnover_penalty_bps) / 10000.0
        portfolio_gross_return = 0.0
        for row in selected:
            weight = weights.get(row.job_id, 0.0)
            previous_weight = previous_weights.get(row.job_id, 0.0)
            portfolio_gross_return += weight * row.realized_test_return
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
                    "asset": row.asset,
                    "asset_cluster": row.asset_cluster,
                    "previous_weight": previous_weight,
                    "weight_delta": weight - previous_weight,
                    "turnover_contribution": abs(weight - previous_weight),
                }
            )
        portfolio_return = portfolio_gross_return - rebalance_cost
        weekly.append(
            {
                "test_start": test_start,
                "selected_assets": len(selected),
                "inactive_assets": len(decisions) - len(selected),
                "cash_weight": cash_weight,
                "portfolio_gross_return": portfolio_gross_return,
                "portfolio_rebalance_cost": rebalance_cost,
                "portfolio_return": portfolio_return,
                "portfolio_turnover": turnover,
                "portfolio_prior_vol": realized_prior_vol,
            }
        )
        prior_windows = [
            row.test_start
            for row in weeks
            if row.test_end < test_start
        ]
        cutoff_manifests.append(
            {
                "portfolio_rebalance_cutoff_ts": test_start,
                "max_timestamp_used": test_start,
                "same_anchor_test_used": False,
                "future_anchor_used": False,
                "stage_c_access": "DENIED",
                "allocation_method": method,
                "included_history_window_count": len(prior_windows),
                "selected_assets": len(selected),
                "inactive_assets": len(decisions) - len(selected),
            }
        )
        previous_weights = dict(weights)

    returns = [row["portfolio_return"] for row in weekly]
    cumulative = 1.0
    for ret in returns:
        cumulative *= 1.0 + ret
    summary = {
        "method": method,
        "weeks": len(weekly),
        "allocations": len(allocations),
        "inactive_decisions": sum(1 for row in activations if not row["active"]),
        "max_assets": max_assets,
        "min_signal": min_signal,
        "lookback_weeks": lookback_weeks,
        "risk_floor": risk_floor,
        "max_weight": max_weight,
        "min_observations": min_observations,
        "max_drawdown_threshold": max_drawdown_threshold,
        "activation_min_signal": activation_min_signal,
        "min_train_tail_trades": min_train_tail_trades,
        "min_validation_trades": min_validation_trades,
        "min_action_entropy": min_action_entropy,
        "max_cost_to_gross_edge": max_cost_to_gross_edge,
        "require_no_broker_violations": require_no_broker_violations,
        "require_no_friday_violations": require_no_friday_violations,
        "max_asset_weight": max_asset_weight,
        "max_cluster_weight": max_cluster_weight,
        "turnover_penalty_bps": turnover_penalty_bps,
        "portfolio_vol_target": portfolio_vol_target,
        "mean_weekly_gross_return": _mean([row["portfolio_gross_return"] for row in weekly]),
        "mean_rebalance_cost": _mean([row["portfolio_rebalance_cost"] for row in weekly]),
        "mean_turnover": _mean([row["portfolio_turnover"] for row in weekly]),
        "mean_weekly_return": _mean(returns),
        "std_weekly_return": _std(returns),
        "sharpe_like_weekly": (_mean(returns) / _std(returns)) if _std(returns) > 0 else 0.0,
        "cvar_20_weekly": _cvar(returns, 0.20),
        "max_drawdown": _max_drawdown(returns),
        "cumulative_return": cumulative - 1.0,
    }
    return {
        "summary": summary,
        "weekly": weekly,
        "allocations": allocations,
        "activations": activations,
        "cutoff_manifests": cutoff_manifests,
    }


def _write_outputs(output_dir: Path, run_id: str, result: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}.json"
    weekly_path = output_dir / f"{run_id}_weekly.csv"
    allocation_path = output_dir / f"{run_id}_allocations.csv"
    activation_path = output_dir / f"{run_id}_activations.csv"
    cutoff_manifest_path = output_dir / f"{run_id}_cutoff_manifest.jsonl"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with weekly_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "test_start",
                "selected_assets",
                "inactive_assets",
                "cash_weight",
                "portfolio_gross_return",
                "portfolio_rebalance_cost",
                "portfolio_return",
                "portfolio_turnover",
                "portfolio_prior_vol",
            ],
        )
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
            "asset",
            "asset_cluster",
            "previous_weight",
            "weight_delta",
            "turnover_contribution",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["allocations"])
    with activation_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "test_start",
            "asset_key",
            "job_id",
            "active",
            "composite_signal",
            "observations",
            "recent_drawdown",
            "train_tail_trades",
            "validation_trades",
            "action_entropy",
            "validation_cost_to_gross_edge",
            "broker_policy_violations",
            "friday_force_close_violations",
            "reasons",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result.get("activations", []):
            out = dict(row)
            out["reasons"] = ";".join(out.get("reasons", []))
            writer.writerow(out)
    with cutoff_manifest_path.open("w", encoding="utf-8") as handle:
        for row in result.get("cutoff_manifests", []):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "json": str(json_path),
        "weekly_csv": str(weekly_path),
        "allocations_csv": str(allocation_path),
        "activations_csv": str(activation_path),
        "cutoff_manifest_jsonl": str(cutoff_manifest_path),
    }


def _ensure_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: dict[str, str],
) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


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
            inactive_assets INTEGER NOT NULL DEFAULT 0,
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
        CREATE TABLE IF NOT EXISTS portfolio_activations (
            run_id TEXT NOT NULL,
            test_start TEXT NOT NULL,
            asset_key TEXT NOT NULL,
            job_id TEXT NOT NULL,
            active INTEGER NOT NULL,
            composite_signal REAL NOT NULL,
            observations INTEGER NOT NULL,
            recent_drawdown REAL NOT NULL,
            reasons TEXT NOT NULL,
            PRIMARY KEY(run_id, test_start, asset_key, job_id)
        );
        CREATE TABLE IF NOT EXISTS portfolio_cutoff_manifests (
            run_id TEXT NOT NULL,
            portfolio_rebalance_cutoff_ts TEXT NOT NULL,
            max_timestamp_used TEXT NOT NULL,
            same_anchor_test_used INTEGER NOT NULL,
            future_anchor_used INTEGER NOT NULL,
            stage_c_access TEXT NOT NULL,
            allocation_method TEXT NOT NULL,
            included_history_window_count INTEGER NOT NULL,
            selected_assets INTEGER NOT NULL,
            inactive_assets INTEGER NOT NULL,
            manifest_json TEXT NOT NULL,
            PRIMARY KEY(run_id, portfolio_rebalance_cutoff_ts)
        );
        """
    )
    _ensure_columns(
        conn,
        "portfolio_weekly_returns",
        {
            "inactive_assets": "INTEGER NOT NULL DEFAULT 0",
            "cash_weight": "REAL NOT NULL DEFAULT 0.0",
            "portfolio_gross_return": "REAL NOT NULL DEFAULT 0.0",
            "portfolio_rebalance_cost": "REAL NOT NULL DEFAULT 0.0",
            "portfolio_turnover": "REAL NOT NULL DEFAULT 0.0",
            "portfolio_prior_vol": "REAL NOT NULL DEFAULT 0.0",
        },
    )
    _ensure_columns(
        conn,
        "portfolio_allocations",
        {
            "asset": "TEXT",
            "asset_cluster": "TEXT",
            "previous_weight": "REAL NOT NULL DEFAULT 0.0",
            "weight_delta": "REAL NOT NULL DEFAULT 0.0",
            "turnover_contribution": "REAL NOT NULL DEFAULT 0.0",
        },
    )
    _ensure_columns(
        conn,
        "portfolio_activations",
        {
            "train_tail_trades": "INTEGER NOT NULL DEFAULT 0",
            "validation_trades": "INTEGER NOT NULL DEFAULT 0",
            "action_entropy": "REAL NOT NULL DEFAULT 0.0",
            "validation_cost_to_gross_edge": "REAL NOT NULL DEFAULT 0.0",
            "broker_policy_violations": "INTEGER NOT NULL DEFAULT 0",
            "friday_force_close_violations": "INTEGER NOT NULL DEFAULT 0",
        },
    )
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_runs(run_id, generated_at, config_json, summary_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, utc_now(), json.dumps(config, sort_keys=True), json.dumps(result["summary"], sort_keys=True)),
        )
        # Replace prior rows for this run_id so re-runs stay reproducible.
        conn.execute("DELETE FROM portfolio_weekly_returns WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM portfolio_allocations WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM portfolio_activations WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM portfolio_cutoff_manifests WHERE run_id=?", (run_id,))
        conn.executemany(
            """
            INSERT OR REPLACE INTO portfolio_weekly_returns(
                run_id, test_start, selected_assets, inactive_assets, cash_weight,
                portfolio_gross_return, portfolio_rebalance_cost, portfolio_return,
                portfolio_turnover, portfolio_prior_vol
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    row["test_start"],
                    row["selected_assets"],
                    row.get("inactive_assets", 0),
                    row.get("cash_weight", 0.0),
                    row.get("portfolio_gross_return", row["portfolio_return"]),
                    row.get("portfolio_rebalance_cost", 0.0),
                    row["portfolio_return"],
                    row.get("portfolio_turnover", 0.0),
                    row.get("portfolio_prior_vol", 0.0),
                )
                for row in result["weekly"]
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO portfolio_allocations(
                run_id, test_start, asset_key, job_id, weight, composite_signal,
                train_tail_return, validation_return, realized_test_return,
                asset, asset_cluster, previous_weight, weight_delta, turnover_contribution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    row.get("asset"),
                    row.get("asset_cluster"),
                    row.get("previous_weight", 0.0),
                    row.get("weight_delta", 0.0),
                    row.get("turnover_contribution", 0.0),
                )
                for row in result["allocations"]
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO portfolio_activations(
                run_id, test_start, asset_key, job_id, active, composite_signal,
                observations, recent_drawdown, train_tail_trades, validation_trades,
                action_entropy, validation_cost_to_gross_edge, broker_policy_violations,
                friday_force_close_violations, reasons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    row["test_start"],
                    row["asset_key"],
                    row["job_id"],
                    1 if row["active"] else 0,
                    row["composite_signal"],
                    row["observations"],
                    row["recent_drawdown"],
                    row.get("train_tail_trades", 0),
                    row.get("validation_trades", 0),
                    row.get("action_entropy", 0.0),
                    row.get("validation_cost_to_gross_edge", 0.0),
                    row.get("broker_policy_violations", 0),
                    row.get("friday_force_close_violations", 0),
                    ";".join(row.get("reasons", [])),
                )
                for row in result.get("activations", [])
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO portfolio_cutoff_manifests(
                run_id, portfolio_rebalance_cutoff_ts, max_timestamp_used,
                same_anchor_test_used, future_anchor_used, stage_c_access,
                allocation_method, included_history_window_count, selected_assets,
                inactive_assets, manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    row["portfolio_rebalance_cutoff_ts"],
                    row["max_timestamp_used"],
                    1 if row["same_anchor_test_used"] else 0,
                    1 if row["future_anchor_used"] else 0,
                    row["stage_c_access"],
                    row["allocation_method"],
                    row["included_history_window_count"],
                    row["selected_assets"],
                    row["inactive_assets"],
                    json.dumps(row, sort_keys=True),
                )
                for row in result.get("cutoff_manifests", [])
            ],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--method",
        choices=ALLOCATION_METHODS,
        default="score_inverse_vol",
    )
    parser.add_argument("--max-assets", type=int, default=5)
    parser.add_argument("--min-signal", type=float, default=0.0)
    parser.add_argument("--lookback-weeks", type=int, default=12)
    parser.add_argument("--risk-floor", type=float, default=1.0e-4)
    parser.add_argument("--max-weight", type=float, default=0.50)
    parser.add_argument("--max-asset-weight", type=float, default=1.0)
    parser.add_argument("--max-cluster-weight", type=float, default=1.0)
    parser.add_argument("--turnover-penalty-bps", type=float, default=0.0)
    parser.add_argument(
        "--portfolio-vol-target",
        type=float,
        default=None,
        help="weekly realized-vol target for score_inverse_vol_portfolio_vol_target",
    )
    parser.add_argument(
        "--score-floor",
        type=float,
        default=None,
        help="alias for --activation-min-signal; no-trade when composite is at/below this",
    )
    parser.add_argument("--min-test-start")
    parser.add_argument("--allow-failed-trade-gate", action="store_true")
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument(
        "--min-observations",
        type=int,
        default=0,
        help="no-trade if a stream has fewer prior weekly observations than this",
    )
    parser.add_argument(
        "--max-drawdown-threshold",
        type=float,
        default=None,
        help="no-trade if recent drawdown magnitude exceeds this (e.g. 0.20)",
    )
    parser.add_argument(
        "--drawdown-lookback-weeks",
        type=int,
        default=0,
        help="weeks used for the drawdown rule (0 = reuse --lookback-weeks)",
    )
    parser.add_argument(
        "--activation-min-signal",
        type=float,
        default=None,
        help="no-trade if composite signal is at or below this threshold",
    )
    parser.add_argument("--min-train-tail-trades", type=int, default=0)
    parser.add_argument("--min-validation-trades", type=int, default=0)
    parser.add_argument("--min-action-entropy", type=float, default=None)
    parser.add_argument("--max-cost-to-gross-edge", type=float, default=None)
    parser.add_argument("--require-no-broker-violations", action="store_true")
    parser.add_argument("--require-no-friday-violations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    activation_min_signal = (
        args.score_floor if args.score_floor is not None else args.activation_min_signal
    )
    config = {
        "method": args.method,
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
        "activation_min_signal": activation_min_signal,
        "min_train_tail_trades": args.min_train_tail_trades,
        "min_validation_trades": args.min_validation_trades,
        "min_action_entropy": args.min_action_entropy,
        "max_cost_to_gross_edge": args.max_cost_to_gross_edge,
        "require_no_broker_violations": args.require_no_broker_violations,
        "require_no_friday_violations": args.require_no_friday_violations,
        "max_asset_weight": args.max_asset_weight,
        "max_cluster_weight": args.max_cluster_weight,
        "turnover_penalty_bps": args.turnover_penalty_bps,
        "portfolio_vol_target": args.portfolio_vol_target,
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
        min_observations=max(0, args.min_observations),
        max_drawdown_threshold=args.max_drawdown_threshold,
        drawdown_lookback_weeks=max(0, args.drawdown_lookback_weeks),
        activation_min_signal=activation_min_signal,
        min_train_tail_trades=max(0, args.min_train_tail_trades),
        min_validation_trades=max(0, args.min_validation_trades),
        min_action_entropy=args.min_action_entropy,
        max_cost_to_gross_edge=args.max_cost_to_gross_edge,
        require_no_broker_violations=args.require_no_broker_violations,
        require_no_friday_violations=args.require_no_friday_violations,
        max_asset_weight=min(1.0, max(0.01, args.max_asset_weight)),
        max_cluster_weight=min(1.0, max(0.01, args.max_cluster_weight)),
        turnover_penalty_bps=max(0.0, args.turnover_penalty_bps),
        portfolio_vol_target=args.portfolio_vol_target,
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
