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
    # activity curve shape. THE PLATEAU HAS NO DEFAULT: the order
    # forbids inventing the target — it must be declared explicitly
    # from the WP4 candidate/sensitivity artifact. evaluate_episode
    # refuses when either bound is absent.
    "activity_rise_exponent": 0.5,
    "activity_plateau_low_rate": None,
    "activity_plateau_high_rate": None,
    "activity_decay_exponent": 0.5,
    "activity_overtrading_floor": 0.2,
    # branch-2 (active loss): activity dominates ACROSS materially
    # different activity levels; loss ranks WITHIN comparable activity.
    "loss_activity_weight": 10.0,
    "loss_economic_weight": 0.1,
    "loss_scale": 50.0,
    "loss_drawdown_weight": 10.0,
    "gain_base_share": 0.25,
    "gain_drawdown_share": 0.5,
    "sharpe_bonus_share": 0.2,
    "sharpe_bonus_cap": 3.0,
}

#: (validator, message) per config key — an invalid configuration is a
#: typed refusal; it can never flip the sign of a branch.
_CONFIG_RULES = {
    "zero_trade_sentinel": (lambda v: v < 0, "must be < 0"),
    "activity_rise_exponent": (lambda v: 0 < v <= 4, "must be in (0,4]"),
    "activity_plateau_low_rate": (lambda v: v > 0, "must be > 0"),
    "activity_plateau_high_rate": (lambda v: v > 0, "must be > 0"),
    "activity_decay_exponent": (lambda v: 0 < v <= 4, "must be in (0,4]"),
    "activity_overtrading_floor": (lambda v: 0 < v <= 1,
                                   "must be in (0,1]"),
    "loss_activity_weight": (lambda v: v > 0, "must be > 0"),
    "loss_economic_weight": (lambda v: v > 0, "must be > 0"),
    "loss_scale": (lambda v: v > 0, "must be > 0"),
    "loss_drawdown_weight": (lambda v: v >= 0, "must be >= 0"),
    "gain_base_share": (lambda v: 0 < v <= 1, "must be in (0,1]"),
    "gain_drawdown_share": (lambda v: 0 <= v <= 1, "must be in [0,1]"),
    "sharpe_bonus_share": (lambda v: 0 <= v <= 1, "must be in [0,1]"),
    "sharpe_bonus_cap": (lambda v: v > 0, "must be > 0"),
}


def validate_config(cfg: Mapping[str, Any]) -> dict:
    out = {}
    for key, (rule, message) in _CONFIG_RULES.items():
        value = cfg.get(key)
        if value is None:
            raise EpisodicFitnessError(
                f"config.{key} is required and has no default"
                if key.startswith("activity_plateau")
                else f"config.{key} must not be None")
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise EpisodicFitnessError(
                f"config.{key}={value!r} is not a real number")
        number = float(value)
        if not math.isfinite(number) or not rule(number):
            raise EpisodicFitnessError(
                f"config.{key}={value!r} invalid: {message}")
        out[key] = number
    if out["activity_plateau_low_rate"] >=             out["activity_plateau_high_rate"]:
        raise EpisodicFitnessError(
            "activity plateau low must be < high")
    # EAF-010 relational invariant: EVERY active-loss scalar lives in a
    # guaranteed open interval above the sentinel. The worst active
    # scalar is -(law + lew*(0.01 + loss_scale + ddw)) because
    # loss_units < 1 and utility >= 0; the sentinel must sit strictly
    # below it, with margin.
    branch2_floor = -(out["loss_activity_weight"]
                      + out["loss_economic_weight"]
                      * (0.01 + out["loss_scale"]
                         + out["loss_drawdown_weight"]))
    if out["zero_trade_sentinel"] >= branch2_floor:
        raise EpisodicFitnessError(
            f"SENTINEL_INVARIANT: zero_trade_sentinel "
            f"{out['zero_trade_sentinel']} must be strictly below the "
            f"worst achievable active-loss scalar {branch2_floor:.4f} —"
            " otherwise a finite active policy could lose to no-trade")
    out["_branch2_floor"] = branch2_floor
    return out

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
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    cfg = validate_config(merged)

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

    if isinstance(bars_per_year, bool) or \
            not isinstance(bars_per_year, numbers.Integral) or \
            int(bars_per_year) <= 0:
        raise EpisodicFitnessError(
            f"bars_per_year must be a positive integer, got "
            f"{bars_per_year!r}")
    years = rows / float(int(bars_per_year))
    rate = trades / years
    utility = activity_utility(rate, cfg)
    dd_capped = min(drawdown, 1.0)

    if trades == 0:
        branch = BRANCH_ZERO_TRADE
        scalar = float(cfg["zero_trade_sentinel"])
        economic = 0.0
    elif ret <= 0.0:
        branch = BRANCH_ACTIVE_LOSS
        # deep losses stay DISTINCT: linear to -100%, logarithmic
        # beyond, strictly monotone forever (-100%/-1000%/-10000% can
        # never alias).
        magnitude = abs(ret)
        # EAF-010: BOUNDED, strictly monotone over the WHOLE finite
        # domain — m/(1+m) in [0,1). No finite loss can ever cross the
        # sentinel, and -1/-10/-100/-1e300 all stay distinct.
        loss_units = magnitude / (1.0 + magnitude)
        loss_term = (0.01 + cfg["loss_scale"] * loss_units
                     + cfg["loss_drawdown_weight"] * dd_capped)
        economic = -loss_term
        # activity dominates across materially different activity
        # levels; loss ranks within comparable activity (the reproduced
        # blocking case: 1 quasi-passive trade must NOT outrank the
        # 40-trade active learner on a 300x smaller loss).
        scalar = -(cfg["loss_activity_weight"] * (1.0 - utility)
                   + cfg["loss_economic_weight"] * loss_term)
        scalar = min(scalar, -1e-9)
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
                              min_normal_crossings: int,
                              min_mapped_changes: int = 2,
                              tolerance: float = 1e-6) -> dict:
    """Counterexample 10: an easy checkpoint whose actions do not
    survive normal action semantics cannot be selected for handoff."""
    from . import _policy_behavior as _pb
    behavior = _pb.classify_policy_behavior(
        deterministic_actions, threshold=normal_threshold,
        tolerance=tolerance)
    if isinstance(min_normal_crossings, bool) or \
            not isinstance(min_normal_crossings, numbers.Integral) or \
            int(min_normal_crossings) < 2:
        raise EpisodicFitnessError(
            "min_normal_crossings must be an integer >= 2 — a single "
            "crossing is mechanical noise, never survivability "
            "(audit 2026-08-20)")
    crossings = behavior["threshold_crossings"]
    mapped_changes = behavior["deterministic"].get(
        "mapped_action_changes", 0)
    survivable = (crossings >= int(min_normal_crossings)
                  and mapped_changes >= int(min_mapped_changes)
                  and behavior["classification"] ==
                  _pb.DETERMINISTIC_MAPPED_ACTIVITY)
    return {
        "survivable": survivable,
        "classification": behavior["classification"],
        "threshold_crossings": crossings,
        "mapped_action_changes": mapped_changes,
        "min_normal_crossings": int(min_normal_crossings),
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
