"""WP-A (findings AUD-F1-20260817-277..281): typed policy-behavior authority.

The defect this exists to name: the training stack pairs a continuous
learner with a hard three-bin environment adapter
(``gym-fx/app/env.py::_coerce_action``) that maps ``Box[-1,1]`` to
``{HOLD, LONG, SHORT}`` through a threshold. At easy threshold ``0.0``
ANY non-zero constant becomes permanent direction, so a behaviorally
constant actor produces orders, exposure and PnL (+4.38% in one
reproduced cell, -3.05% in another) purely from market path and SL/TP
cycling. At normal threshold ``0.1`` the same tiny constants map to
permanent HOLD and the cell reports zero activity.

**Trades are not evidence of state-conditioned learning, and zero trades
are not evidence of a dead learner.**

Corrected per the 2026-08-17 order:

- a trace alone can establish *deterministic mapped activity* and never
  *state-responsiveness*: a rollout only visits the states the market
  handed it, so varying actions may simply reflect varying inputs that
  were never controlled for. The promotable class requires paired
  observation evidence bound to a fixed model (see
  :func:`classify_with_observation_evidence`);
- input cardinality is preserved: one malformed, NaN or infinite
  element makes the whole sequence ``UNAVAILABLE`` with index facts,
  never a silently shortened "valid" sequence;
- crossings are derived from the mapping itself
  (``map_action(v, thr) != HOLD``), so an exact zero at threshold zero
  is HOLD and is not a crossing;
- absent stochastic evidence is distinguished from present-but-invalid.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

SCHEMA = "agent_multi.policy_behavior.v2"

# ── the typed taxonomy ────────────────────────────────────────────────
STATE_RESPONSIVE_ACTIVE = "STATE_RESPONSIVE_ACTIVE"
STATE_RESPONSIVE_BELOW_THRESHOLD = "STATE_RESPONSIVE_BELOW_THRESHOLD"
DETERMINISTIC_MAPPED_ACTIVITY = "DETERMINISTIC_MAPPED_ACTIVITY"
CONSTANT_DIRECTIONAL_EXPOSURE = "CONSTANT_DIRECTIONAL_EXPOSURE"
CONSTANT_HOLD = "CONSTANT_HOLD"
STOCHASTIC_ONLY_ACTIVITY = "STOCHASTIC_ONLY_ACTIVITY"
UNAVAILABLE = "UNAVAILABLE"

CLASSIFICATIONS = (
    STATE_RESPONSIVE_ACTIVE,
    STATE_RESPONSIVE_BELOW_THRESHOLD,
    DETERMINISTIC_MAPPED_ACTIVITY,
    CONSTANT_DIRECTIONAL_EXPOSURE,
    CONSTANT_HOLD,
    STOCHASTIC_ONLY_ACTIVITY,
    UNAVAILABLE,
)

#: Only ONE class may be read as learned, state-conditioned activity,
#: and it is unreachable without observation evidence bound to a model.
#: `CONSTANT_DIRECTIONAL_EXPOSURE` is excluded however many orders it
#: created; `DETERMINISTIC_MAPPED_ACTIVITY` is excluded because a trace
#: cannot separate "the policy reacted" from "the inputs moved".
PROMOTABLE_AS_LEARNED_ACTIVITY = frozenset({STATE_RESPONSIVE_ACTIVE})

#: Classes a trace-only measurement is allowed to return.
TRACE_ONLY_CLASSIFICATIONS = frozenset({
    DETERMINISTIC_MAPPED_ACTIVITY,
    STATE_RESPONSIVE_BELOW_THRESHOLD,
    CONSTANT_DIRECTIONAL_EXPOSURE,
    CONSTANT_HOLD,
    STOCHASTIC_ONLY_ACTIVITY,
    UNAVAILABLE,
})

DEFAULT_CONSTANCY_TOLERANCE = 1e-6
DEFAULT_COUNTERFACTUAL_THRESHOLDS = (0.0, 0.001, 0.01, 0.05, 0.1)

HOLD, LONG, SHORT = 0, 1, 2


class PolicyBehaviorError(ValueError):
    """Malformed REQUEST (bad threshold/tolerance/evidence contract).
    A malformed measurement is never an exception — it is typed
    ``UNAVAILABLE``."""


def map_action(value: float, threshold: float) -> int:
    """Mirror of ``gym-fx/app/env.py::_coerce_action`` for the
    continuous adapter, including the ``threshold == 0`` branch where
    every non-zero value is directional and an exact zero is HOLD."""
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


def is_crossing(value: float, threshold: float) -> bool:
    """A crossing is a NON-HOLD mapped decision — derived from the
    mapping, never from ``abs(value) >= threshold``, so exact zero at
    threshold zero is correctly not a crossing."""
    return map_action(value, threshold) != HOLD


def _validate_number(name: str, value: Any, *,
                     allow_negative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PolicyBehaviorError(f"{name} must be a number") from error
    if not math.isfinite(number):
        raise PolicyBehaviorError(f"{name} must be finite, got {value!r}")
    if not allow_negative and number < 0.0:
        raise PolicyBehaviorError(
            f"{name} must be non-negative, got {number!r}")
    return number


def _coerce_sequence(values: Optional[Sequence[Any]]) -> dict:
    """Preserve cardinality. Returns the parsed values plus a typed
    record of every element that could not be parsed — the caller must
    refuse rather than shrink the sequence."""
    if values is None:
        return {"present": False, "count": 0, "values": [],
                "invalid_indices": [], "invalid_count": 0}
    parsed: list[float] = []
    invalid: list[int] = []
    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError):
            invalid.append(index)
            continue
        if not math.isfinite(number):
            invalid.append(index)
            continue
        parsed.append(number)
    return {
        "present": True,
        "count": len(list(values)),
        "values": parsed,
        "invalid_indices": invalid,
        "invalid_count": len(invalid),
    }


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
    """The same action series under every adapter threshold of interest,
    so a reader can see that a series is HOLD-everywhere at 0.1 and
    LONG-everywhere at 0.0 without re-running anything."""
    out: dict[str, dict[str, Any]] = {}
    total = len(actions)
    for raw in thresholds:
        threshold = _validate_number("counterfactual threshold", raw)
        mapped = [map_action(value, threshold) for value in actions]
        out[f"{threshold:g}"] = {
            "threshold": threshold,
            "hold": mapped.count(HOLD),
            "long": mapped.count(LONG),
            "short": mapped.count(SHORT),
            "hold_fraction": (mapped.count(HOLD) / total) if total else None,
            "threshold_crossings": sum(1 for m in mapped if m != HOLD),
            "mapped_action_changes": sum(
                1 for i in range(1, len(mapped))
                if mapped[i] != mapped[i - 1]),
        }
    return out


def action_statistics(actions: Sequence[float], *,
                      declared_count: Optional[int] = None) -> dict:
    """Deterministic action distribution facts. ``declared_count`` is
    the ORIGINAL input cardinality and is always reported, so a
    shortened parse can never masquerade as a complete measurement."""
    count = declared_count if declared_count is not None else len(actions)
    if not actions:
        return {"count": count, "parsed_count": 0}
    return {
        "count": count,
        "parsed_count": len(actions),
        "min": min(actions),
        "max": max(actions),
        "mean": math.fsum(actions) / len(actions),
        "std": _std(actions),
        "spread": max(actions) - min(actions),
        "unique_count": len(set(actions)),
        "q01": _quantile(actions, 0.01),
        "q50": _quantile(actions, 0.50),
        "q99": _quantile(actions, 0.99),
    }


def _unavailable(reason: str, *, threshold: float, tolerance: float,
                 deterministic: dict, stochastic: dict,
                 source: Optional[Mapping[str, Any]]) -> dict:
    return {
        "schema": SCHEMA,
        "classification": UNAVAILABLE,
        "promotable_as_learned_activity": False,
        "evidence_level": "unavailable",
        "reasons": [reason],
        "threshold": threshold,
        "constancy_tolerance": tolerance,
        "deterministic": deterministic,
        "threshold_crossings": 0,
        "stochastic": stochastic,
        "threshold_counterfactuals": {},
        "source": dict(source) if source else None,
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
    """Classify one policy's behavior from a ROLLOUT TRACE alone.

    This function can never return ``STATE_RESPONSIVE_ACTIVE``: a trace
    cannot separate "the policy reacted to its input" from "the input
    moved and the policy followed something else". The strongest
    trace-only verdict is ``DETERMINISTIC_MAPPED_ACTIVITY``, which is
    explicitly NOT promotable. Use
    :func:`classify_with_observation_evidence` for the promotable class.
    """
    threshold = _validate_number("threshold", threshold)
    tolerance = _validate_number("tolerance", tolerance)

    det = _coerce_sequence(deterministic_actions)
    sto = _coerce_sequence(stochastic_actions)
    det_stats = action_statistics(det["values"],
                                  declared_count=det["count"])
    sto_stats = {
        "present": sto["present"],
        "count": sto["count"],
        "parsed_count": len(sto["values"]),
        "invalid_count": sto["invalid_count"],
        "invalid_indices": sto["invalid_indices"],
        "std": _std(sto["values"]),
        "threshold_crossings": sum(
            1 for value in sto["values"] if is_crossing(value, threshold)),
    }

    if not det["present"] or det["count"] == 0:
        return _unavailable(
            "no deterministic action sequence was provided",
            threshold=threshold, tolerance=tolerance,
            deterministic={**det_stats,
                           "invalid_count": det["invalid_count"],
                           "invalid_indices": det["invalid_indices"]},
            stochastic=sto_stats, source=source)
    if det["invalid_count"]:
        return _unavailable(
            f"{det['invalid_count']} of {det['count']} deterministic "
            f"actions are missing, malformed, NaN or infinite at "
            f"indices {det['invalid_indices'][:16]} — a partially "
            "readable sequence is refused, never silently shortened",
            threshold=threshold, tolerance=tolerance,
            deterministic={**det_stats,
                           "invalid_count": det["invalid_count"],
                           "invalid_indices": det["invalid_indices"]},
            stochastic=sto_stats, source=source)
    if sto["present"] and sto["invalid_count"]:
        return _unavailable(
            f"stochastic evidence is present but {sto['invalid_count']} "
            f"of {sto['count']} draws are unreadable — present-but-"
            "invalid evidence is refused, unlike absent evidence",
            threshold=threshold, tolerance=tolerance,
            deterministic={**det_stats, "invalid_count": 0,
                           "invalid_indices": []},
            stochastic=sto_stats, source=source)

    actions = det["values"]
    counterfactuals = threshold_counterfactuals(
        actions, counterfactual_thresholds)
    key = f"{threshold:g}"
    at_threshold = counterfactuals.get(key) or threshold_counterfactuals(
        actions, [threshold])[key]
    crossings = int(at_threshold["threshold_crossings"])
    mapped_changes = int(at_threshold["mapped_action_changes"])

    reasons: list[str] = []
    constant = det_stats["spread"] <= tolerance
    if constant:
        mapped = map_action(actions[0], threshold)
        if mapped == HOLD:
            classification = CONSTANT_HOLD
            reasons.append(
                f"action spread {det_stats['spread']:.3e} <= tolerance "
                f"{tolerance:.3e}; the constant maps to HOLD at "
                f"threshold {threshold:g}")
        else:
            classification = CONSTANT_DIRECTIONAL_EXPOSURE
            reasons.append(
                f"action spread {det_stats['spread']:.3e} <= tolerance "
                f"{tolerance:.3e}; the constant maps to "
                f"{'LONG' if mapped == LONG else 'SHORT'} at threshold "
                f"{threshold:g} — exposure and any resulting orders come "
                "from the adapter and the market path, NOT from "
                "state-conditioned learning")
    elif crossings > 0 and mapped_changes > 0:
        classification = DETERMINISTIC_MAPPED_ACTIVITY
        reasons.append(
            f"the mapped decision changes {mapped_changes} times over "
            f"{len(actions)} bars at threshold {threshold:g}. This is "
            "deterministic mapped ACTIVITY only: a trace cannot show "
            "the policy responded to its observations, so it is not "
            "promotable as learned behavior (observation evidence "
            "required)")
    elif crossings > 0:
        mapped = map_action(actions[0], threshold)
        classification = (CONSTANT_HOLD if mapped == HOLD
                          else CONSTANT_DIRECTIONAL_EXPOSURE)
        reasons.append(
            f"action varies numerically (spread {det_stats['spread']:.3e}"
            f") but the MAPPED decision never changes across "
            f"{len(actions)} bars at threshold {threshold:g} — "
            "behaviorally constant")
    elif sto_stats["threshold_crossings"] > 0:
        classification = STOCHASTIC_ONLY_ACTIVITY
        reasons.append(
            "deterministic evaluation never leaves HOLD while "
            f"{sto_stats['threshold_crossings']} of "
            f"{sto_stats['parsed_count']} stochastic draws do — any "
            "activity would be exploration noise, not evaluated policy "
            "behavior")
    else:
        classification = STATE_RESPONSIVE_BELOW_THRESHOLD
        reasons.append(
            f"action varies (spread {det_stats['spread']:.3e} > "
            f"tolerance {tolerance:.3e}) but no bar leaves HOLD at "
            f"threshold {threshold:g}")

    if classification == CONSTANT_HOLD and \
            sto_stats["threshold_crossings"] > 0:
        classification = STOCHASTIC_ONLY_ACTIVITY
        reasons.append(
            f"{sto_stats['threshold_crossings']} stochastic draws leave "
            "HOLD while deterministic behavior is constant")

    assert classification in TRACE_ONLY_CLASSIFICATIONS, (
        "a trace-only measurement may never return the promotable class")
    return {
        "schema": SCHEMA,
        "classification": classification,
        "promotable_as_learned_activity": False,
        "evidence_level": "trace_only",
        "reasons": reasons,
        "threshold": threshold,
        "constancy_tolerance": tolerance,
        "deterministic": {**det_stats, "invalid_count": 0,
                          "invalid_indices": [],
                          "mapped_action_changes": mapped_changes},
        "threshold_crossings": crossings,
        "stochastic": sto_stats,
        "threshold_counterfactuals": counterfactuals,
        "source": dict(source) if source else None,
    }


REQUIRED_OBSERVATION_EVIDENCE = (
    "model_sha256",
    "observation_contract_sha256",
    "observation_rows",
    "role",
)


def classify_with_observation_evidence(
    deterministic_actions: Optional[Sequence[Any]],
    *,
    threshold: float,
    observation_evidence: Mapping[str, Any],
    repeated_observation_actions: Optional[Sequence[Any]] = None,
    permuted_observation_actions: Optional[Sequence[Any]] = None,
    stochastic_actions: Optional[Sequence[Any]] = None,
    tolerance: float = DEFAULT_CONSTANCY_TOLERANCE,
    counterfactual_thresholds: Sequence[float] =
        DEFAULT_COUNTERFACTUAL_THRESHOLDS,
) -> dict[str, Any]:
    """The ONLY path to ``STATE_RESPONSIVE_ACTIVE``.

    Requires actions produced by a FIXED model over a real-role
    observation batch, plus two controls:

    - ``repeated_observation_actions``: the same observation fed twice
      must give the same action, or the measurement is not a function
      of the observation and is refused;
    - ``permuted_observation_actions``: re-running the batch in a
      different row order must give the same per-row actions, proving
      the output depends on the row and not on its position.

    ``observation_evidence`` must carry the custody fields in
    :data:`REQUIRED_OBSERVATION_EVIDENCE`. Missing custody is a request
    error, not a quiet downgrade.
    """
    missing = [key for key in REQUIRED_OBSERVATION_EVIDENCE
               if not observation_evidence.get(key)]
    if missing:
        raise PolicyBehaviorError(
            "observation evidence is incomplete; missing "
            f"{missing} — the promotable class is unreachable without "
            "custody binding it to a fixed model and a real role")

    base = classify_policy_behavior(
        deterministic_actions, threshold=threshold,
        stochastic_actions=stochastic_actions, tolerance=tolerance,
        counterfactual_thresholds=counterfactual_thresholds,
        source=dict(observation_evidence))
    base["evidence_level"] = "observation_bound"
    base["observation_evidence"] = dict(observation_evidence)

    if base["classification"] != DETERMINISTIC_MAPPED_ACTIVITY:
        return base

    controls: dict[str, Any] = {}
    repeated = _coerce_sequence(repeated_observation_actions)
    permuted = _coerce_sequence(permuted_observation_actions)
    det = _coerce_sequence(deterministic_actions)

    if not repeated["present"] or repeated["invalid_count"] or \
            _std(repeated["values"]) > tolerance:
        controls["identical_observation_control"] = "FAILED_OR_ABSENT"
        base["reasons"].append(
            "the repeated-identical-observation control is absent or "
            "not constant — without it the actions cannot be shown to "
            "be a function of the observation")
        return base
    controls["identical_observation_control"] = "PASSED"

    if not permuted["present"] or permuted["invalid_count"] or \
            len(permuted["values"]) != len(det["values"]) or \
            any(abs(a - b) > tolerance
                for a, b in zip(sorted(permuted["values"]),
                                sorted(det["values"]))):
        controls["row_permutation_control"] = "FAILED_OR_ABSENT"
        base["reasons"].append(
            "the row-permutation control is absent or inconsistent — "
            "the output may depend on row position rather than on the "
            "observation")
        return base
    controls["row_permutation_control"] = "PASSED"

    base["classification"] = STATE_RESPONSIVE_ACTIVE
    base["promotable_as_learned_activity"] = True
    base["controls"] = controls
    base["reasons"].append(
        "actions come from a fixed model over a real-role observation "
        "batch, are reproducible on identical observations and are "
        "invariant to row order — the mapped decision therefore varies "
        "WITH the observation")
    return base
