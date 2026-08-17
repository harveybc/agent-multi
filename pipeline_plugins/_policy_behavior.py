"""WP1 (finding AUD-F1-20260817-277): typed policy-behavior taxonomy.

The defect this exists to name: the training stack pairs a continuous
learner with a hard three-bin environment adapter
(``gym-fx/app/env.py::_coerce_action``) that maps ``Box[-1,1]`` to
``{HOLD, LONG, SHORT}`` through a threshold. Consequences measured on
the live identity:

- at easy threshold ``0.0`` ANY non-zero constant becomes a permanent
  direction, so a behaviorally constant actor produces orders, exposure
  and PnL (+4.38% in one reproduced cell, -3.05% in another) purely
  from market path and SL/TP cycling;
- at normal threshold ``0.1`` the same tiny constants map to permanent
  HOLD and the cell reports zero activity.

**Trades are therefore not evidence of state-conditioned learning, and
zero trades are not evidence of a dead learner.** Both readings need
the action series itself. This module is the single classifier used by
traces, stopping, aggregation and promotion so the four consumers can
never disagree.

Classification compares action VARIATION against a declared numerical
tolerance — exact float equality is explicitly insufficient, because a
policy that jitters in the 1e-9 range is behaviorally constant — and it
always carries threshold counterfactuals, because the same series is a
different behavior under a different adapter setting.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence

SCHEMA = "agent_multi.policy_behavior.v1"

# ── the typed taxonomy (WP1) ──────────────────────────────────────────
STATE_RESPONSIVE_ACTIVE = "STATE_RESPONSIVE_ACTIVE"
STATE_RESPONSIVE_BELOW_THRESHOLD = "STATE_RESPONSIVE_BELOW_THRESHOLD"
CONSTANT_DIRECTIONAL_EXPOSURE = "CONSTANT_DIRECTIONAL_EXPOSURE"
CONSTANT_HOLD = "CONSTANT_HOLD"
STOCHASTIC_ONLY_ACTIVITY = "STOCHASTIC_ONLY_ACTIVITY"
UNAVAILABLE = "UNAVAILABLE"

CLASSIFICATIONS = (
    STATE_RESPONSIVE_ACTIVE,
    STATE_RESPONSIVE_BELOW_THRESHOLD,
    CONSTANT_DIRECTIONAL_EXPOSURE,
    CONSTANT_HOLD,
    STOCHASTIC_ONLY_ACTIVITY,
    UNAVAILABLE,
)

#: Only ONE class may be read as learned, state-conditioned activity.
#: `CONSTANT_DIRECTIONAL_EXPOSURE` is explicitly excluded however many
#: orders it created (WP1, order 2026-08-17).
PROMOTABLE_AS_LEARNED_ACTIVITY = frozenset({STATE_RESPONSIVE_ACTIVE})

#: Declared behavioral-constancy tolerance. A deterministic action
#: series whose spread is at or below this is behaviorally constant,
#: whatever its float bits say. Chosen ~3 orders of magnitude below the
#: smallest adapter threshold in use (0.001) so a real sub-threshold
#: policy is never called constant.
DEFAULT_CONSTANCY_TOLERANCE = 1e-6

#: Threshold counterfactuals persisted with every classification (WP0).
DEFAULT_COUNTERFACTUAL_THRESHOLDS = (0.0, 0.001, 0.01, 0.05, 0.1)

HOLD, LONG, SHORT = 0, 1, 2


class PolicyBehaviorError(ValueError):
    """Raised only for a malformed request, never for a measured
    outcome — an unmeasurable policy is typed ``UNAVAILABLE``."""


def map_action(value: float, threshold: float) -> int:
    """Mirror of ``gym-fx/app/env.py::_coerce_action`` for the
    continuous adapter. Kept deliberately faithful, including the
    ``threshold == 0`` branch where every non-zero value is directional
    and only an exact zero is HOLD."""
    if threshold == 0.0:
        if value > 0.0:
            return LONG
        if value < 0.0:
            return SHORT
        return HOLD
    if value >= threshold:
        return LONG
    if value <= -threshold:
        return SHORT
    return HOLD


def _finite(values: Optional[Iterable[Any]]) -> list[float]:
    if values is None:
        return []
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _spread(values: Sequence[float]) -> float:
    return (max(values) - min(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = math.fsum(values) / len(values)
    var = math.fsum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def threshold_counterfactuals(
    actions: Sequence[float],
    thresholds: Sequence[float] = DEFAULT_COUNTERFACTUAL_THRESHOLDS,
) -> dict[str, dict[str, Any]]:
    """The same action series under every adapter threshold of interest.

    This is what makes a classification auditable: a reader can see that
    a series is HOLD-everywhere at 0.1 and LONG-everywhere at 0.0
    without re-running anything."""
    out: dict[str, dict[str, Any]] = {}
    total = len(actions)
    for threshold in thresholds:
        mapped = [map_action(value, threshold) for value in actions]
        crossings = sum(1 for value in actions
                        if threshold == 0.0 or abs(value) >= threshold)
        changes = sum(1 for i in range(1, len(mapped))
                      if mapped[i] != mapped[i - 1])
        out[f"{threshold:g}"] = {
            "threshold": float(threshold),
            "hold": mapped.count(HOLD),
            "long": mapped.count(LONG),
            "short": mapped.count(SHORT),
            "hold_fraction": (mapped.count(HOLD) / total) if total else None,
            "threshold_crossings": crossings,
            "mapped_action_changes": changes,
        }
    return out


def action_statistics(actions: Sequence[float]) -> dict[str, Any]:
    """Deterministic action distribution facts required by WP0."""
    if not actions:
        return {"count": 0}
    return {
        "count": len(actions),
        "min": min(actions),
        "max": max(actions),
        "mean": math.fsum(actions) / len(actions),
        "std": _std(actions),
        "spread": _spread(actions),
        "unique_count": len(set(actions)),
        "q01": _quantile(actions, 0.01),
        "q50": _quantile(actions, 0.50),
        "q99": _quantile(actions, 0.99),
        "sign_changes": sum(
            1 for i in range(1, len(actions))
            if (actions[i] > 0) != (actions[i - 1] > 0)
            or (actions[i] < 0) != (actions[i - 1] < 0)),
    }


def classify_policy_behavior(
    deterministic_actions: Optional[Sequence[Any]],
    *,
    threshold: float,
    stochastic_actions: Optional[Sequence[Any]] = None,
    tolerance: float = DEFAULT_CONSTANCY_TOLERANCE,
    counterfactual_thresholds: Sequence[float] =
        DEFAULT_COUNTERFACTUAL_THRESHOLDS,
    source: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Classify one policy's behavior on one role's rollout.

    ``threshold`` is the adapter threshold the rollout actually ran
    under. ``tolerance`` is the DECLARED behavioral-constancy bound and
    is recorded in the result, so a later reader can see which bound
    produced the verdict rather than having to trust it.

    Never raises for a measured outcome. Missing or non-finite actions
    are ``UNAVAILABLE``.
    """
    if tolerance < 0.0:
        raise PolicyBehaviorError("tolerance must be non-negative")
    actions = _finite(deterministic_actions)
    stochastic = _finite(stochastic_actions)

    counterfactuals = threshold_counterfactuals(
        actions, counterfactual_thresholds) if actions else {}
    stats = action_statistics(actions)

    key = f"{float(threshold):g}"
    at_threshold = counterfactuals.get(key) or (
        threshold_counterfactuals(actions, [float(threshold)]).get(key)
        if actions else None)
    crossings = int(at_threshold["threshold_crossings"]) if at_threshold \
        else 0
    # The taxonomy is about BEHAVIOR, and behavior is the MAPPED action.
    # At threshold 0 every non-zero value "crosses", so crossings alone
    # would make ACTIVE trivially reachable; and a policy whose numbers
    # vary while its mapped decision never changes is behaviorally
    # constant however much it jitters. Both are settled by counting
    # mapped-action changes.
    mapped_changes = int(at_threshold["mapped_action_changes"]) \
        if at_threshold else 0

    stochastic_crossings = sum(
        1 for value in stochastic
        if float(threshold) == 0.0 or abs(value) >= float(threshold))

    reasons: list[str] = []
    if not actions:
        classification = UNAVAILABLE
        reasons.append("no_finite_deterministic_actions")
    else:
        constant = stats["spread"] <= tolerance
        if constant:
            # A constant series is classified by what the ADAPTER makes
            # of it, not by whether orders happened: the same constant
            # is permanent exposure at threshold 0 and permanent HOLD
            # at 0.1.
            mapped = map_action(actions[0], float(threshold))
            if mapped == HOLD:
                classification = CONSTANT_HOLD
                reasons.append(
                    f"action spread {stats['spread']:.3e} <= tolerance "
                    f"{tolerance:.3e}; constant maps to HOLD at "
                    f"threshold {float(threshold):g}")
            else:
                classification = CONSTANT_DIRECTIONAL_EXPOSURE
                reasons.append(
                    f"action spread {stats['spread']:.3e} <= tolerance "
                    f"{tolerance:.3e}; constant maps to "
                    f"{'LONG' if mapped == LONG else 'SHORT'} at "
                    f"threshold {float(threshold):g} — exposure and any "
                    "resulting orders come from the adapter and market "
                    "path, NOT from state-conditioned learning")
        elif crossings > 0 and mapped_changes > 0:
            classification = STATE_RESPONSIVE_ACTIVE
            reasons.append(
                f"action spread {stats['spread']:.3e} > tolerance, "
                f"{crossings} of {len(actions)} bars cross threshold "
                f"{float(threshold):g}, and the MAPPED decision changes "
                f"{mapped_changes} times — behavior varies, not only "
                "the number")
        elif crossings > 0:
            mapped = map_action(actions[0], float(threshold))
            classification = (CONSTANT_HOLD if mapped == HOLD
                              else CONSTANT_DIRECTIONAL_EXPOSURE)
            reasons.append(
                f"action varies numerically (spread {stats['spread']:.3e}"
                f") but the MAPPED decision never changes across "
                f"{len(actions)} bars at threshold {float(threshold):g}"
                " — behaviorally constant, so varying numbers are not "
                "state-conditioned activity")
        elif stochastic_crossings > 0:
            classification = STOCHASTIC_ONLY_ACTIVITY
            reasons.append(
                "deterministic evaluation never crosses the threshold "
                f"while {stochastic_crossings} of {len(stochastic)} "
                "stochastic draws do — activity would be exploration "
                "noise, not evaluated policy behavior")
        else:
            classification = STATE_RESPONSIVE_BELOW_THRESHOLD
            reasons.append(
                f"action varies (spread {stats['spread']:.3e} > "
                f"tolerance {tolerance:.3e}) but no bar reaches "
                f"threshold {float(threshold):g}")

    if classification == CONSTANT_HOLD and stochastic_crossings > 0:
        # A constant deterministic policy whose stochastic draws trade
        # is still the more informative fact for WP2 question 4.
        classification = STOCHASTIC_ONLY_ACTIVITY
        reasons.append(
            f"{stochastic_crossings} stochastic draws cross the "
            "threshold while deterministic behavior is constant")

    return {
        "schema": SCHEMA,
        "classification": classification,
        "promotable_as_learned_activity":
            classification in PROMOTABLE_AS_LEARNED_ACTIVITY,
        "reasons": reasons,
        "threshold": float(threshold),
        "constancy_tolerance": float(tolerance),
        "deterministic": stats,
        "threshold_crossings": crossings,
        "stochastic": {
            "count": len(stochastic),
            "threshold_crossings": stochastic_crossings,
            "std": _std(stochastic),
        },
        "threshold_counterfactuals": counterfactuals,
        "source": dict(source) if source else None,
    }
