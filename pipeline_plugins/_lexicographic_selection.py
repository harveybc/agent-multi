"""Transparent constrained/lexicographic selection (ETH order §9).

Selection on VALIDATION only:

1. valid observation/action/protection contract (no simulator or
   evidence failure in the summary);
2. declared minimum activity, WITHOUT a positive-profit gate;
3. maximize mean weekly net simple return;
4. tie-break by lower maximum drawdown, then higher total net return.

The full ordered tuple and its components are persisted; any encoded
scalar exists only as DEAP transport and must never be displayed as
return, profit or champion quality.
"""
from __future__ import annotations

import math
from typing import Any, Dict

SCHEMA = "agent_multi.lexicographic_selection.v1"
METRIC_NAME = "lexicographic_weekly_v1"
INELIGIBLE_TRANSPORT_SCALAR = -1.0e9

# Tie-break weights for the transport encoding only. Chosen so a full
# drawdown difference (1.0) can never outweigh 1e-3 of weekly return,
# preserving the lexicographic ORDER for realistic magnitudes; the
# authoritative comparison is always the persisted tuple.
_DD_EPSILON = 1.0e-4
_TOTAL_EPSILON = 1.0e-8


def evaluate_selection_contract(
    validation_summary: Dict[str, Any],
    *,
    min_trades: int,
) -> Dict[str, Any]:
    reasons = []
    if not isinstance(validation_summary, dict) or not validation_summary:
        reasons.append("missing validation summary")
        validation_summary = {}
    if validation_summary.get("error") or validation_summary.get(
            "simulator_error"):
        reasons.append("simulator or evidence failure")
    if validation_summary.get("evaluation_skipped"):
        reasons.append("validation was not evaluated")

    def _finite(name: str) -> float:
        try:
            value = float(validation_summary.get(name))
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value):
            reasons.append(f"{name} is missing or non-finite")
            return float("nan")
        return value

    mean_weekly = _finite("mean_weekly_return")
    max_drawdown = _finite("max_drawdown_fraction")
    total_return = _finite("total_return")
    trades = validation_summary.get("trades_total")
    try:
        trades = int(trades)
    except (TypeError, ValueError):
        trades = 0
        reasons.append("trades_total missing")
    if trades < int(min_trades):
        reasons.append(
            f"activity below declared minimum ({trades} <"
            f" {int(min_trades)} trades); no profit gate applies")

    eligible = not reasons
    ordered_tuple = (
        [mean_weekly, -max_drawdown, total_return] if eligible else None)
    transport = (
        mean_weekly - _DD_EPSILON * max_drawdown
        + _TOTAL_EPSILON * total_return
        if eligible else INELIGIBLE_TRANSPORT_SCALAR)
    return {
        "schema": SCHEMA,
        "metric": METRIC_NAME,
        "eligible": eligible,
        "ineligible_reasons": reasons,
        "ordered_tuple": ordered_tuple,
        "components": {
            "mean_weekly_net_simple_return": mean_weekly,
            "max_drawdown_fraction": max_drawdown,
            "total_net_return": total_return,
            "trades_total": trades,
            "min_trades_required": int(min_trades),
        },
        "transport_scalar": transport,
        "transport_note": (
            "DEAP transport only — never display as return, profit or"
            " champion quality; the ordered tuple is authoritative"),
    }
