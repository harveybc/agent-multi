"""Authoritative constrained/lexicographic selection (ETH order §9).

Selection on VALIDATION only:

1. valid observation/action/protection contract (no simulator or
   evidence failure in the summary);
2. declared minimum activity, WITHOUT a positive-profit gate;
3. maximize mean weekly net simple return;
4. tie-break by lower maximum drawdown, then higher total net return.

Correction AUD-F1-20260805-112: the previous "transport scalar" was a
weighted sum (``weekly - 1e-4*dd + 1e-8*total``) that can REVERSE the
authoritative order. It is replaced by a preregistered bounded/quantized
integer packing (``encode_order_key``) that preserves the lexicographic
order of the quantized tuple BY CONSTRUCTION: every comparison of the
scalar is exactly a comparison of the quantized tuple, so DEAP fitness,
shared-pool selection, champion comparison and block acceptance — which
all compare one float — become tuple-authoritative without changing
DOIN's decentralized architecture.

Preregistered quantization (bounds clamp, never wrap):

- mean weekly net simple return: step 1e-6, bounds [-0.5, +0.5];
- maximum drawdown fraction:     step 1e-4, bounds [0, 1];
- total net return:              step 1e-4, bounds [-1, +20].

The packed key is an exact integer below 2**53 (float64-exact), strictly
positive for every eligible tuple; ineligible candidates encode to
``INELIGIBLE_ORDER_KEY`` (0.0), below every eligible key. The key is an
ORDER KEY ONLY — it must never be displayed as return, profit or
champion quality; the persisted ordered tuple and components are the
human-facing truth.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence

SCHEMA = "agent_multi.lexicographic_selection.v2"
METRIC_NAME = "lexicographic_weekly_v1"

# Preregistered quantization contract (AUD-F1-20260805-112).
WEEKLY_STEP = 1.0e-6
WEEKLY_MIN, WEEKLY_MAX = -0.5, 0.5
DD_STEP = 1.0e-4
DD_MIN, DD_MAX = 0.0, 1.0
TOTAL_STEP = 1.0e-4
TOTAL_MIN, TOTAL_MAX = -1.0, 20.0

_WEEKLY_LEVELS = int(round((WEEKLY_MAX - WEEKLY_MIN) / WEEKLY_STEP)) + 1
_DD_LEVELS = int(round((DD_MAX - DD_MIN) / DD_STEP)) + 1
_TOTAL_LEVELS = int(round((TOTAL_MAX - TOTAL_MIN) / TOTAL_STEP)) + 1

# Highest packed value must stay float64-exact (< 2**53).
_MAX_KEY = _WEEKLY_LEVELS * _DD_LEVELS * _TOTAL_LEVELS
assert _MAX_KEY < 2 ** 53, "order-key packing exceeds float64 exactness"

INELIGIBLE_ORDER_KEY = 0.0
# Backwards-compatible alias for readers of the v1 field name; the VALUE
# is the new ineligible key, never the old -1e9 sentinel.
INELIGIBLE_TRANSPORT_SCALAR = INELIGIBLE_ORDER_KEY


def _quantize(value: float, minimum: float, maximum: float,
              step: float) -> int:
    clamped = min(max(float(value), minimum), maximum)
    return int(round((clamped - minimum) / step))


def quantized_tuple(mean_weekly: float, max_drawdown: float,
                    total_return: float) -> tuple[int, int, int]:
    """The authoritative comparison domain: quantized, bounded levels.

    Drawdown is stored inverted (lower drawdown -> higher level) so the
    natural tuple order matches "lower drawdown wins".
    """
    return (
        _quantize(mean_weekly, WEEKLY_MIN, WEEKLY_MAX, WEEKLY_STEP),
        (_DD_LEVELS - 1) - _quantize(max_drawdown, DD_MIN, DD_MAX, DD_STEP),
        _quantize(total_return, TOTAL_MIN, TOTAL_MAX, TOTAL_STEP),
    )


def encode_order_key(mean_weekly: float, max_drawdown: float,
                     total_return: float) -> float:
    """Pack the quantized tuple into one float64-exact positive integer.

    Order preservation is structural: the packing is the mixed-radix
    integer of the quantized levels, so scalar comparison IS quantized
    tuple comparison. Eligible keys are >= 1.0 > INELIGIBLE_ORDER_KEY.
    """
    weekly_level, dd_level, total_level = quantized_tuple(
        mean_weekly, max_drawdown, total_return)
    packed = ((weekly_level * _DD_LEVELS) + dd_level) * _TOTAL_LEVELS \
        + total_level
    return float(packed + 1)


def compare_ordered_tuples(a: Sequence[float] | None,
                           b: Sequence[float] | None) -> int:
    """Authoritative comparator: -1 a<b, 0 tie, +1 a>b.

    ``None`` (ineligible) loses to any eligible tuple; two ineligible
    candidates tie. Uses the same preregistered quantization as the
    order key so scalar and tuple comparison can never disagree.
    """
    if a is None and b is None:
        return 0
    if a is None:
        return -1
    if b is None:
        return 1
    qa = quantized_tuple(a[0], -a[1], a[2])
    qb = quantized_tuple(b[0], -b[1], b[2])
    return (qa > qb) - (qa < qb)


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
    order_key = (
        encode_order_key(mean_weekly, max_drawdown, total_return)
        if eligible else INELIGIBLE_ORDER_KEY)
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
        "quantization": {
            "weekly_step": WEEKLY_STEP,
            "weekly_bounds": [WEEKLY_MIN, WEEKLY_MAX],
            "drawdown_step": DD_STEP,
            "drawdown_bounds": [DD_MIN, DD_MAX],
            "total_step": TOTAL_STEP,
            "total_bounds": [TOTAL_MIN, TOTAL_MAX],
        },
        "transport_scalar": order_key,
        "transport_note": (
            "order KEY only (mixed-radix packing of the quantized"
            " tuple; scalar comparison IS tuple comparison) — never"
            " display as return, profit or champion quality"),
    }
