"""Order 2026-08-21: two independently named and versioned contracts.

Reusing one scalar under two names is a refusal condition. These are
NOT the same number renamed:

- :func:`easy_checkpoint_monitor` (id ``easy_checkpoint_monitor.v1``)
  selects checkpoints and drives patience within ONE training run:
  train-tail/validation economic performance, bounded risk, and their
  gap as a deterioration signal.
- :func:`easy_doin_candidate_fitness`
  (id ``easy_doin_candidate_fitness.v1``) ranks candidates AFTER each
  selected its checkpoint, hierarchically: activity first (zero trades
  loses to every finite active learner; the calibrated band orders
  materially different levels), validation economics within comparable
  activity, gap as a BOUNDED tie-break that can never reverse the
  activity hierarchy, catastrophic loss monotonically worse.

``easy_to_normal_handoff`` remains a third, separate concern
(pipeline_plugins._episodic_activity_fitness.assert_handoff_survivable
+ verify_handoff_continuity) and is NOT re-declared here.
"""
from __future__ import annotations

import math
import numbers
from typing import Any, Mapping, Optional

MONITOR_CONTRACT_ID = "agent_multi.easy_checkpoint_monitor.v1"
FITNESS_CONTRACT_ID = "agent_multi.easy_doin_candidate_fitness.v1"


class EasyContractError(ValueError):
    """Malformed facts or missing contract identity — fail closed."""


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise EasyContractError(f"{name} must be a real number, got "
                                f"{value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise EasyContractError(f"{name} must be finite")
    return number

def _count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise EasyContractError(f"{name} must be an Integral count")
    number = int(value)
    if number < 0:
        raise EasyContractError(f"{name} must be >= 0")
    return number


def _assert_no_test_facts(facts: Mapping[str, Any]) -> None:
    for key in facts:
        if "test" in str(key).lower():
            raise EasyContractError(
                f"REFUSED_TEST_FACT: {key!r} — terminal test evidence "
                "may never enter checkpoint selection, stopping, "
                "ranking or calibration (order 2026-08-21)")


def easy_checkpoint_monitor(*, train_tail_return: Any,
                            validation_return: Any,
                            train_tail_drawdown: Any,
                            validation_drawdown: Any,
                            gap_penalty_beta: float = 0.25,
                            risk_lambda: float = 1.0,
                            **extra_facts) -> dict:
    """Checkpoint selection + patience value for ONE run."""
    _assert_no_test_facts(extra_facts)
    tt = _finite("train_tail_return", train_tail_return)
    vv = _finite("validation_return", validation_return)
    tt_dd = _finite("train_tail_drawdown", train_tail_drawdown)
    vv_dd = _finite("validation_drawdown", validation_drawdown)
    beta = _finite("gap_penalty_beta", gap_penalty_beta)
    lam = _finite("risk_lambda", risk_lambda)
    tt_rap = tt - lam * min(max(tt_dd, 0.0), 1.0)
    vv_rap = vv - lam * min(max(vv_dd, 0.0), 1.0)
    mean = 0.5 * (tt_rap + vv_rap)
    gap = abs(tt_rap - vv_rap)
    value = mean - beta * gap
    return {"contract_id": MONITOR_CONTRACT_ID,
            "value": value,
            "components": {"train_tail_rap": tt_rap,
                           "validation_rap": vv_rap,
                           "mean_rap": mean, "gap": gap,
                           "gap_penalty": beta * gap,
                           "risk_lambda": lam,
                           "gap_penalty_beta": beta}}


def easy_doin_candidate_fitness(*, closed_trades: Any,
                                scored_rows: Any,
                                validation_return: Any,
                                validation_drawdown: Any,
                                train_tail_return: Any,
                                activity_config: Mapping[str, Any],
                                bars_per_year: int = 2190,
                                **extra_facts) -> dict:
    """Hierarchical LEXICOGRAPHIC key for post-selection ranking.

    Returned ``lex_key`` orders by tuple comparison:
    (activity_band, validation_economics, -bounded_gap). Zero trades is
    band -1 (loses to everything active); the gap term is bounded in
    [0, 1] and sits LAST, so it can never reverse activity or
    economics.
    """
    _assert_no_test_facts(extra_facts)
    from . import _episodic_activity_fitness as _ef
    trades = _count("closed_trades", closed_trades)
    rows = _count("scored_rows", scored_rows)
    if rows == 0:
        raise EasyContractError("scored_rows must be > 0")
    vv = _finite("validation_return", validation_return)
    vv_dd = _finite("validation_drawdown", validation_drawdown)
    tt = _finite("train_tail_return", train_tail_return)
    cfg = dict(_ef.DEFAULT_CONFIG)
    cfg.update(activity_config)
    cfg = _ef.validate_config(cfg)
    rate = trades / (rows / float(bars_per_year))
    utility = _ef.activity_utility(rate, cfg)

    if trades == 0:
        band = -1
    elif utility >= 1.0:
        band = 2                      # calibrated target band
    elif utility >= 0.5:
        band = 1                      # material but sub-target
    else:
        band = 0                      # minimal activity
    economics = vv - min(max(vv_dd, 0.0), 1.0)
    # catastrophic monotonicity: economics is strictly decreasing in
    # loss magnitude (no clamp on vv itself)
    gap_bounded = min(abs(tt - vv), 1.0)
    lex_key = (band, economics, -gap_bounded)
    return {"contract_id": FITNESS_CONTRACT_ID,
            "eligible": trades > 0,
            "reason": (None if trades else "ZERO_TRADES"),
            "lex_key": lex_key,
            "components": {"activity_band": band,
                           "annualized_rate": rate,
                           "activity_utility": utility,
                           "validation_economics": economics,
                           "gap_bounded": gap_bounded,
                           "trades": trades}}


def assert_distinct_contracts(monitor: Mapping[str, Any],
                              fitness: Mapping[str, Any]) -> None:
    """Refusal condition from the order: one scalar under two names."""
    if monitor.get("contract_id") == fitness.get("contract_id"):
        raise EasyContractError(
            "REFUSED_SHARED_CONTRACT_IDENTITY: monitor and fitness "
            "must be independently named contracts")
