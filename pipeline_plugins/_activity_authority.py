"""WP1 (order 2026-08-18): the ONE typed activity authority.

Every consumer of trade/activity facts — stopping, checkpoint
eligibility, handoff, aggregation and promotion — evaluates activity
through this module, so no path can reimplement a threshold, invent a
numeric sentinel, or disagree with another path about what "active"
means.

The defect family this closes, with evidence in
``docs/audits/evidence/WP1_ACTIVITY_CONSUMER_MAP_BEFORE_2026_08_18.json``:

- ``selection_min_trades`` defaulted to 0 while ``early_stop_min_trades``
  defaulted to 1, so selection ranked zero-trade candidates the stopper
  refused (the inherited contradiction);
- ``composite = raw - no_trade_penalty`` kept an INELIGIBLE candidate
  RANKABLE at a fixed numeric sentinel (-1e6), which is how sixteen
  constant policies could still be ordered against each other;
- one promotion script re-implemented its own minimum locally;
- the historical gym-fx fitness (provenance artifact
  ``HISTORICAL_FITNESS_PROVENANCE_GYMFX_8088F9E.json``) paid for
  activity INSIDE fitness (num_orders, sqrt, squared), manufacturing
  rank from trade count.

Doctrine, from the order verbatim:

- zero trades on either required role is ineligible;
- missing or malformed activity evidence is unavailable AND ineligible;
- negative return is NOT an activity failure — this module never sees a
  return, so it structurally cannot punish one;
- ineligible candidates carry NO comparable selection score (``None``,
  never a number);
- trade count is never added or multiplied into fitness — this module
  exposes no score field at all;
- the strict nonzero floor is 1; the CALIBRATED activity floor is
  pending WP2 evidence and must arrive as a new threshold contract id,
  never as a silent constant edit.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

SCHEMA = "agent_multi.activity_authority.v1"

#: The declared threshold contract. Floor 1 is the strict nonzero floor
#: the order materializes; a WP2-calibrated floor must ship as a NEW
#: contract id so every artifact names which floor judged it.
THRESHOLD_CONTRACT_ID = "agent_multi.activity_floor.strict_nonzero.v1"
STRICT_NONZERO_FLOOR = 1

# Typed reason codes.
ZERO_TRADES_TRAIN_MONITOR = "ZERO_TRADES_TRAIN_MONITOR"
ZERO_TRADES_INNER_VALIDATION = "ZERO_TRADES_INNER_VALIDATION"
TRADES_UNAVAILABLE_TRAIN_MONITOR = "TRADES_UNAVAILABLE_TRAIN_MONITOR"
TRADES_UNAVAILABLE_INNER_VALIDATION = (
    "TRADES_UNAVAILABLE_INNER_VALIDATION")
BELOW_FLOOR_TRAIN_MONITOR = "BELOW_FLOOR_TRAIN_MONITOR"
BELOW_FLOOR_INNER_VALIDATION = "BELOW_FLOOR_INNER_VALIDATION"


class ActivityAuthorityError(ValueError):
    """Malformed REQUEST (bad floor, contradictory config). A malformed
    MEASUREMENT is never an exception — it is typed unavailable and
    ineligible."""


class IneligibleCandidateError(RuntimeError):
    """Raised by :func:`require_rankable` when a consumer asks for a
    comparable selection score on an ineligible candidate."""


def _coerce_count(value: Any) -> Optional[int]:
    """A trade count is available only as a finite non-negative int.
    Anything else — None, NaN, strings, negatives — is unavailable."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def resolve_floor(config: Mapping[str, Any] | None = None,
                  *, key: str = "activity_floor") -> int:
    """One floor for every consumer.

    Absent -> the strict nonzero floor (1). An explicit value >= 1 is
    honored. An explicit 0 or negative is a TYPED REFUSAL: the
    selection-0-versus-stopping-1 contradiction is eliminated by
    refusing its ingredient, never by silently bumping it.
    """
    if config is None:
        return STRICT_NONZERO_FLOOR
    raw = config.get(key, config.get("selection_min_trades",
                                     config.get("early_stop_min_trades")))
    if raw is None:
        return STRICT_NONZERO_FLOOR
    try:
        floor = int(raw)
    except (TypeError, ValueError) as error:
        raise ActivityAuthorityError(
            f"activity floor {raw!r} is not an integer") from error
    if floor < STRICT_NONZERO_FLOOR:
        raise ActivityAuthorityError(
            f"CONTRADICTORY_ACTIVITY_FLOOR: {floor} is below the strict "
            f"nonzero floor {STRICT_NONZERO_FLOOR} — a zero floor is how "
            "selection ranked zero-trade candidates while early stopping "
            "refused them; refuse the ingredient instead of inheriting "
            "the contradiction (order 2026-08-18 WP1)")
    return floor


def evaluate_role_activity(trades: Any, *, role: str,
                           floor: int = STRICT_NONZERO_FLOOR) -> dict:
    """Single-role primitive, for consumers that only see one role
    (e.g. the lexicographic validation contract). Same floor, same
    typing, same contract id."""
    if floor < STRICT_NONZERO_FLOOR:
        raise ActivityAuthorityError(
            f"floor {floor} below strict nonzero floor")
    count = _coerce_count(trades)
    reasons: list[str] = []
    if count is None:
        reasons.append(f"TRADES_UNAVAILABLE_{role.upper()}")
    elif count == 0:
        reasons.append(f"ZERO_TRADES_{role.upper()}")
    elif count < floor:
        reasons.append(f"BELOW_FLOOR_{role.upper()}")
    return {
        "schema": SCHEMA,
        "role": role,
        "eligible": not reasons,
        "reason_codes": reasons,
        "trades": count,
        "trades_available": count is not None,
        "floor": floor,
        "threshold_contract_id": THRESHOLD_CONTRACT_ID,
    }


def evaluate_activity(
    *,
    train_monitor_trades: Any,
    inner_validation_trades: Any,
    active_weeks: Any = None,
    exposure_fraction: Any = None,
    evidence_refs: Sequence[str] | None = None,
    floor: int = STRICT_NONZERO_FLOOR,
) -> dict:
    """The shared typed activity result (order 2026-08-18 WP1).

    Note what is ABSENT from the signature: any return, profit, Sharpe
    or score. Negative return cannot be an activity failure because
    this authority never sees a return; and no consumer can multiply a
    trade count into fitness through this module because it exposes no
    score at all.
    """
    monitor = evaluate_role_activity(train_monitor_trades,
                                     role="train_monitor", floor=floor)
    validation = evaluate_role_activity(inner_validation_trades,
                                        role="inner_validation",
                                        floor=floor)
    reasons = list(monitor["reason_codes"]) + list(
        validation["reason_codes"])

    weeks = _coerce_count(active_weeks)
    try:
        exposure = float(exposure_fraction)
        if not (0.0 <= exposure <= 1.0):
            exposure = None
    except (TypeError, ValueError):
        exposure = None

    eligible = not reasons
    return {
        "schema": SCHEMA,
        "eligible": eligible,
        "reason_codes": reasons,
        "train_monitor_trades": monitor["trades"],
        "train_monitor_trades_available": monitor["trades_available"],
        "inner_validation_trades": validation["trades"],
        "inner_validation_trades_available":
            validation["trades_available"],
        "active_weeks": weeks,
        "active_weeks_available": weeks is not None,
        "exposure_fraction": exposure,
        "exposure_fraction_available": exposure is not None,
        "threshold_contract_id": THRESHOLD_CONTRACT_ID,
        "floor": floor,
        "calibrated_floor": "pending_wp2_evidence",
        "evidence_refs": list(evidence_refs or []),
        # The load-bearing invariant: an ineligible candidate has NO
        # comparable selection score. Not -1e6. Not raw-minus-penalty.
        # None — and consumers that need a rankable value must call
        # require_rankable, which refuses.
        "selection_score_permitted": eligible,
    }


def require_rankable(result: Mapping[str, Any]) -> None:
    """Consumers call this before ranking. An ineligible candidate has
    no comparable score, so asking for one is a typed refusal — never a
    sentinel value."""
    if not isinstance(result, Mapping) or \
            result.get("schema") != SCHEMA:
        raise ActivityAuthorityError(
            "not an activity-authority result; every ranking path must "
            "evaluate through pipeline_plugins._activity_authority")
    if not result.get("selection_score_permitted"):
        raise IneligibleCandidateError(
            "INELIGIBLE_CANDIDATE_HAS_NO_SELECTION_SCORE: "
            f"{result.get('reason_codes')} — ranking an ineligible "
            "candidate through any numeric sentinel is forbidden "
            "(order 2026-08-18 WP1)")
