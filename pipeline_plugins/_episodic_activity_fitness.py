"""Order 2026-08-20 WP1: typed episodic activity/economic fitness.

The reproduced failure this corrects
(`WP0_EPISODIC_ORDERING_BEFORE_2026_08_20.json`): under the executing
paired weekly comparator a passive collapse with one trade per split
outranks an active learner with negative return, because trade count
does not improve the weekly utility and a still-learning policy is
usually below zero. Passive collapse is therefore locally attractive.

The historical evidence (gym-fx@2a94cb3 / provenance artifact
`HISTORICAL_FITNESS_PROVENANCE_GYMFX_8088F9E.json`) shows the original
authors met the same attractor and paid for activity INSIDE fitness
(num_orders, sqrt, squared) — the error this module must not revive.
The preserved invariant is EPISODIC: NOP is a valid action and costs
nothing per bar; only an episode that ENDS with zero closed trades
receives the inactivity sentinel.

Every component is published; the scalar is last and least.
"""
from __future__ import annotations

import math
import numbers
from typing import Any, Mapping, Optional

SCHEMA = "agent_multi.episodic_activity_economic_fitness.v1"

#: 4h bars per 365-day year — the annualization denominator.
DEFAULT_BARS_PER_YEAR = 2190

DEFAULT_CONFIG = {
    "zero_trade_sentinel": -100.0,
    # activity curve: steep rise from zero, plateau, gradual decay with
    # a floor so overtrading always beats inactivity. The plateau is a
    # DECLARED candidate pending the WP4 sensitivity table — it is not
    # derived from any outer-validation result.
    "activity_rise_exponent": 0.5,
    "activity_plateau_low_rate": 50.0,
    "activity_plateau_high_rate": 300.0,
    "activity_decay_exponent": 0.5,
    "activity_overtrading_floor": 0.2,
    # branch weights, all bounded by construction
    "loss_scale": 50.0,
    "loss_drawdown_weight": 10.0,
    "loss_activity_relief": 0.5,
    "gain_base_share": 0.25,
    "gain_drawdown_share": 0.5,
    "sharpe_bonus_share": 0.2,
    "sharpe_bonus_cap": 3.0,
}

BRANCH_ZERO_TRADE = "B1_zero_trade_sentinel"
BRANCH_ACTIVE_LOSS = "B2_active_loss_toward_zero"
BRANCH_GAIN_NO_SHARPE = "B3_gain_activity_drawdown"
BRANCH_GAIN_SHARPE = "B4_gain_with_sharpe_bonus"


class EpisodicFitnessError(ValueError):
    """Malformed facts. NaN, infinity or foreign types never become an
    accidental champion — they refuse before any scalar exists."""


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise EpisodicFitnessError(f"{name} must be a real number, got "
                                   f"{value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise EpisodicFitnessError(f"{name} must be finite, got {value!r}")
    return number


def _count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise EpisodicFitnessError(
            f"{name} must be an integer count, got {value!r}")
    number = int(value)
    if number < 0:
        raise EpisodicFitnessError(f"{name} must be >= 0, got {number}")
    return number


def activity_utility(annualized_rate: float,
                     config: Mapping[str, Any]) -> float:
    """Bounded [0, 1]: zero at zero, steep rise, plateau at 1, gradual
    overtrading decay to a strictly positive floor."""
    low = float(config["activity_plateau_low_rate"])
    high = float(config["activity_plateau_high_rate"])
    if annualized_rate <= 0.0:
        return 0.0
    if annualized_rate < low:
        return (annualized_rate / low) ** float(
            config["activity_rise_exponent"])
    if annualized_rate <= high:
        return 1.0
    decayed = (high / annualized_rate) ** float(
        config["activity_decay_exponent"])
    return max(float(config["activity_overtrading_floor"]), decayed)


def evaluate_episode(
    *,
    total_return: Any,
    max_drawdown_fraction: Any,
    sharpe: Any,
    closed_trades: Any,
    scored_rows: Any,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    config: Optional[Mapping[str, Any]] = None,
    production_promotion_contract: Optional[Mapping[str, Any]] = None,
) -> dict:
    """One episode -> every component plus one piecewise scalar.

    Semantics (order section 4): the zero-trade sentinel applies ONCE at
    full-episode evaluation, never per bar. An active losing episode
    ranks by movement toward zero loss with a bounded activity relief,
    always above the sentinel, and a larger loss is always worse — no
    multiplication can reverse ordering because a return is negative.
    Positive return earns a bounded activity multiplier and drawdown
    penalty; a positive Sharpe adds a bounded bonus; an unavailable
    Sharpe is typed, never treated as zero-good.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    trades = _count("closed_trades", closed_trades)
    rows = _count("scored_rows", scored_rows)
    if rows == 0:
        raise EpisodicFitnessError("scored_rows must be > 0")
    ret = _finite("total_return", total_return)
    drawdown = _finite("max_drawdown_fraction", max_drawdown_fraction)
    if not (0.0 <= drawdown <= 1.0):
        raise EpisodicFitnessError(
            f"max_drawdown_fraction must be in [0, 1], got {drawdown}")
    sharpe_value: Optional[float] = None
    if sharpe is not None:
        sharpe_value = _finite("sharpe", sharpe)

    years = rows / float(bars_per_year)
    rate = trades / years
    utility = activity_utility(rate, cfg)
    dd_capped = min(drawdown, 1.0)

    if trades == 0:
        branch = BRANCH_ZERO_TRADE
        scalar = float(cfg["zero_trade_sentinel"])
        economic = 0.0
    elif ret <= 0.0:
        branch = BRANCH_ACTIVE_LOSS
        loss = min(abs(ret), 1.0)
        base = (0.01 + float(cfg["loss_scale"]) * loss
                + float(cfg["loss_drawdown_weight"]) * dd_capped)
        relief = 1.0 - float(cfg["loss_activity_relief"]) * utility
        scalar = -base * relief
        economic = -base
    else:
        base_share = float(cfg["gain_base_share"])
        economic = ret * (base_share + (1.0 - base_share) * utility)
        economic *= (1.0 - float(cfg["gain_drawdown_share"]) * dd_capped)
        if sharpe_value is not None and sharpe_value > 0.0:
            branch = BRANCH_GAIN_SHARPE
            bonus = 1.0 + float(cfg["sharpe_bonus_share"]) * min(
                sharpe_value, float(cfg["sharpe_bonus_cap"])) / float(
                cfg["sharpe_bonus_cap"])
            scalar = economic * bonus
        else:
            branch = BRANCH_GAIN_NO_SHARPE
            scalar = economic

    production_satisfied = False
    if production_promotion_contract is not None:
        from . import _activity_authority as _aa
        floor_rate = production_promotion_contract.get(
            "min_annualized_trade_rate")
        if not isinstance(floor_rate, numbers.Real) or \
                isinstance(floor_rate, bool) or \
                not math.isfinite(float(floor_rate)) or \
                float(floor_rate) <= 0:
            raise EpisodicFitnessError(
                "production promotion contract requires a positive "
                "finite min_annualized_trade_rate")
        production_satisfied = trades > 0 and rate >= float(floor_rate)

    return {
        "schema": SCHEMA,
        "branch": branch,
        "total_return": ret,
        "max_drawdown_fraction": drawdown,
        "sharpe": sharpe_value,
        "sharpe_available": sharpe_value is not None,
        "closed_trades": trades,
        "scored_rows": rows,
        "scored_years": years,
        "annualized_trade_rate": rate,
        "activity_utility": utility,
        "economic_utility": economic,
        "selection_value": scalar,
        "zero_trade_sentinel": float(cfg["zero_trade_sentinel"]),
        "activity_curve": {
            "plateau_low_rate": cfg["activity_plateau_low_rate"],
            "plateau_high_rate": cfg["activity_plateau_high_rate"],
            "rise_exponent": cfg["activity_rise_exponent"],
            "decay_exponent": cfg["activity_decay_exponent"],
            "overtrading_floor": cfg["activity_overtrading_floor"],
            "calibration": "candidate pending WP4 sensitivity table",
        },
        # counterexample 9: experimental mechanical validity (>=1 trade)
        # NEVER implies production promotion; that demands a separately
        # declared calibrated contract.
        "production_promotion_satisfied": production_satisfied,
    }


def assert_handoff_survivable(deterministic_actions,
                              *, normal_threshold: float,
                              tolerance: float = 1e-6) -> dict:
    """Counterexample 10: an easy checkpoint whose actions do not
    survive normal action semantics cannot be selected for handoff."""
    from . import _policy_behavior as _pb
    behavior = _pb.classify_policy_behavior(
        deterministic_actions, threshold=normal_threshold,
        tolerance=tolerance)
    crossings = behavior["threshold_crossings"]
    survivable = crossings > 0 and behavior["classification"] in (
        _pb.DETERMINISTIC_MAPPED_ACTIVITY,)
    return {
        "survivable": survivable,
        "classification": behavior["classification"],
        "threshold_crossings": crossings,
        "normal_threshold": float(normal_threshold),
        "refusal": (None if survivable else
                    "HANDOFF_REFUSED_ACTIONS_DO_NOT_SURVIVE_NORMAL_"
                    "SEMANTICS"),
    }


def verify_handoff_continuity(state_before: Mapping[str, Any],
                              state_after: Mapping[str, Any]) -> dict:
    """Counterexample 11 / WP3: changing difficulty must not change any
    tensor or the topology before the first normal update. Byte-level:
    identical key sets, identical shapes, L1 distance exactly 0.0."""
    import hashlib
    import numpy as np

    def _digest(state):
        h = hashlib.sha256()
        for key in sorted(state):
            arr = np.asarray(state[key])
            h.update(key.encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        return h.hexdigest()

    keys_before, keys_after = set(state_before), set(state_after)
    problems = []
    if keys_before != keys_after:
        problems.append("TOPOLOGY_CHANGED: parameter key sets differ")
    l1_total = 0.0
    for key in sorted(keys_before & keys_after):
        a = np.asarray(state_before[key], dtype=np.float64)
        b = np.asarray(state_after[key], dtype=np.float64)
        if a.shape != b.shape:
            problems.append(f"SHAPE_CHANGED: {key}")
            continue
        l1_total += float(np.abs(a - b).sum())
    if l1_total != 0.0:
        problems.append(f"TENSORS_CHANGED: total L1 {l1_total}")
    return {
        "continuous": not problems,
        "problems": problems,
        "l1_distance_total": l1_total,
        "sha256_before": _digest(state_before),
        "sha256_after": _digest(state_after),
    }
