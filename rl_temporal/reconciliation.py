"""Episode reconciliation: what the simulator did, reconciled to the cent.

Runs one episode with a scripted or learned policy and returns a record in
which equity, per-step pnl, commissions, closed trades, the terminal position
and the termination cause are tied together. An open position at ``data_end``
stays open: the record says so and no fill is manufactured to close it.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

RECORD_SCHEMA = "rl_temporal.episode_reconciliation.v1"


def _unwrap(env):
    e = env
    while hasattr(e, "env") and not hasattr(e, "bridge"):
        e = e.env
    return getattr(env, "unwrapped", e) if hasattr(getattr(env, "unwrapped", None), "bridge") else e


def _sharpe(step_returns: List[float]) -> Dict[str, Any]:
    arr = np.asarray(step_returns, dtype=np.float64)
    conv = "mean/std of per-bar equity returns, sample std (ddof=1), NOT annualized"
    if arr.size < 2:
        return {"value": None, "convention": conv, "undefined_reason": "fewer than two bars"}
    sd = float(arr.std(ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return {"value": None, "convention": conv, "undefined_reason": "zero or non-finite variance"}
    return {"value": float(arr.mean() / sd), "convention": conv, "undefined_reason": None, "bars": int(arr.size)}


def reconcile_episode(env, *, actions: Callable[[int, Dict[str, Any]], Any], flatten_at: Optional[int] = None,
                      policy_sees_obs: bool = False, seed: int = 0) -> Dict[str, Any]:
    base = _unwrap(env)
    cfg = dict(getattr(base, "config", {}) or {})
    obs, info = env.reset(seed=seed)
    if policy_sees_obs:
        actions.last_obs = obs
    initial_cash = float(getattr(base, "initial_cash", cfg.get("initial_cash", 0.0)))
    equity_prev = float(info.get("equity", initial_cash))
    sum_pnl = 0.0
    sum_cost = 0.0
    step_returns: List[float] = []
    equities: List[float] = [equity_prev]
    turnover = 0.0
    commission_expected = 0.0
    commission_rate = float(cfg.get("commission", 0.0) or 0.0)
    frame = getattr(base, "dataframe", None)
    exposed = 0
    bars = 0
    units_prev = float(info.get("position_units") or 0.0)
    position_before_terminal = int(info.get("position", 0) or 0)
    last_info: Dict[str, Any] = dict(info)
    terminated = truncated = False
    step = 0
    trace: List[Dict[str, Any]] = []
    prefix_bars = 0
    while not (terminated or truncated):
        if flatten_at is not None and step == flatten_at and hasattr(base, "flatten_step"):
            # risk-reducing close through the simulator's own tested path (action 3 +
            # force_flat_request); bars it consumes are accounted from its returned info
            info = base.flatten_step()
            equity = float(info.get("equity", equity_prev))
            advanced = int(info.get("bar_index", 0) or 0) - int(last_info.get("bar_index", 0) or 0)
            bars += max(0, advanced)
            step += max(0, advanced)
            sum_pnl += equity - equity_prev
            sum_cost += float(info.get("trade_cost", 0.0) or 0.0)
            if equity_prev:
                step_returns.append((equity - equity_prev) / equity_prev)
            equities.append(equity)
            units = float(info.get("position_units") or 0.0)
            delta_units = abs(units - units_prev)
            turnover += delta_units
            if delta_units and frame is not None:
                fill_bar = int(info.get("bar_index", 0) or 0) - 1
                if 0 <= fill_bar < len(frame):
                    fill_price = float(frame["OPEN"].iloc[fill_bar]) if "OPEN" in frame.columns else float(info.get("price", 0.0))
                    commission_expected += delta_units * fill_price * commission_rate
            units_prev = units
            equity_prev = equity
            last_info = dict(info)
            terminated = bool(getattr(base.bridge, "terminated", False))
            if terminated:
                break
            continue
        action = actions(step, last_info)
        position_before_terminal = int(last_info.get("position", 0) or 0)
        obs, reward, terminated, truncated, info = env.step(action)
        if policy_sees_obs:
            actions.last_obs = obs
        step += 1
        bars += 1
        equity = float(info.get("equity", equity_prev))
        sum_pnl += float(info.get("pnl", equity - equity_prev))
        sum_cost += float(info.get("trade_cost", 0.0) or 0.0)
        in_prefix = bool(info.get("is_context_prefix", False))
        prefix_bars += int(in_prefix)
        if not in_prefix:   # forced-hold context bars carry no decision: excluded from scored statistics
            step_returns.append((equity - equity_prev) / equity_prev if equity_prev else 0.0)
        equities.append(equity)
        units = float(info.get("position_units") or 0.0)
        delta_units = abs(units - units_prev)
        turnover += delta_units
        if delta_units and frame is not None:
            # a market order submitted on bar N fills at bar N+1's open; bar_index counts
            # bars consumed, so the fill bar is bar_index - 1
            fill_bar = int(info.get("bar_index", 0) or 0) - 1
            if 0 <= fill_bar < len(frame):
                fill_price = float(frame["OPEN"].iloc[fill_bar]) if "OPEN" in frame.columns else float(info.get("price", 0.0))
                commission_expected += delta_units * fill_price * commission_rate
        units_prev = units
        exposed += int(int(info.get("position", 0) or 0) != 0 and not in_prefix)
        trace.append({"step": step, "bar_index": int(info.get("bar_index", 0) or 0),
                      "action": int(info.get("coerced_action", 0) or 0), "raw_action": float(info.get("raw_action_value", 0.0) or 0.0),
                      "position": int(info.get("position", 0) or 0), "position_units": units, "equity": equity,
                      "pnl": float(info.get("pnl", equity - equities[-2])), "trades": int(info.get("trades", 0) or 0),
                      "is_context_prefix": bool(info.get("is_context_prefix", False))})
        equity_prev = equity
        last_info = dict(info)
    final_equity = float(last_info.get("equity", equity_prev))
    cause = last_info.get("termination_cause")
    terminal_position = int(last_info.get("position", 0) or 0)
    manufactured = bool(terminated and cause == "data_end" and position_before_terminal != 0
                        and terminal_position == 0)
    peak = -math.inf
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)
    diag = dict(last_info.get("execution_diagnostics") or {})
    return {
        "schema": RECORD_SCHEMA,
        "initial_cash": initial_cash, "final_equity": final_equity,
        "net_return": (final_equity - initial_cash) / initial_cash if initial_cash else 0.0,
        "sum_step_pnl": sum_pnl, "conservation_gap": final_equity - initial_cash - sum_pnl,
        "commission_paid": float(last_info.get("commission_paid", 0.0) or 0.0),
        "commission_expected_from_fills": commission_expected,
        "commission_reconciled": math.isclose(float(last_info.get("commission_paid", 0.0) or 0.0),
                                              commission_expected, rel_tol=1e-6, abs_tol=1e-9),
        # gym-fx publishes trade_cost per bar from an accumulator that is reset before the
        # next-bar fill is notified, so the per-step sum lags; it is reported, not relied on
        "sum_step_trade_cost": sum_cost,
        "trades_closed": int(last_info.get("trades", 0) or 0),
        "entries_submitted": int(diag.get("default_orders_submitted", 0) or 0) + int(diag.get("entry_orders_submitted", 0) or 0),
        "rejected_orders": int(diag.get("protected_entry_rejections", 0) or 0),
        "terminal_position": terminal_position,
        "terminal_position_units": float(last_info.get("position_units") or 0.0),
        "terminal_fill_manufactured": manufactured,
        "termination_cause": cause, "terminated": bool(terminated), "truncated": bool(truncated),
        "bars_run": bars, "bars_prefix_excluded": prefix_bars, "bars_scored": bars - prefix_bars, "total_bars": int(last_info.get("total_bars", bars) or bars),
        "turnover_units": turnover, "exposure_fraction": exposed / max(1, bars - prefix_bars),
        "max_drawdown_fraction": max_dd, "sharpe": _sharpe(step_returns),
        "costs": {"commission": float(cfg.get("commission", 0.0) or 0.0),
                  "slippage": float(cfg.get("slippage", 0.0) or 0.0),
                  "financing_enabled": bool(cfg.get("financing_enabled", False))},
        "margin": {"min_equity": float(cfg.get("min_equity", 100.0) or 0.0),
                   "leverage": cfg.get("leverage"), "solvency_mode": cfg.get("solvency_mode", "normal_realistic")},
        "execution_diagnostics": diag,
        "trace": trace,
    }
