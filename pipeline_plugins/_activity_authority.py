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

import numbers
from typing import Any, Mapping, Optional, Sequence

SCHEMA = "agent_multi.activity_authority.v1"

#: The declared threshold contract. Floor 1 is the strict nonzero floor
#: the order materializes; a WP2-calibrated floor must ship as a NEW
#: contract id so every artifact names which floor judged it.
THRESHOLD_CONTRACT_ID = "agent_multi.activity_floor.strict_nonzero.v1"
STRICT_NONZERO_FLOOR = 1

#: C3.4: under the strict-nonzero contract these fields are
#: INFORMATIONAL — they carry no eligibility weight. WP2 may promote
#: them to eligibility-bearing ONLY through a new contract identity.
INFORMATIONAL_FIELDS = ("active_weeks", "exposure_fraction")

# Typed reason codes.
ZERO_TRADES_TRAIN_MONITOR = "ZERO_TRADES_TRAIN_MONITOR"
ZERO_TRADES_INNER_VALIDATION = "ZERO_TRADES_INNER_VALIDATION"
TRADES_UNAVAILABLE_TRAIN_MONITOR = "TRADES_UNAVAILABLE_TRAIN_MONITOR"
TRADES_UNAVAILABLE_INNER_VALIDATION = (
    "TRADES_UNAVAILABLE_INNER_VALIDATION")
BELOW_FLOOR_TRAIN_MONITOR = "BELOW_FLOOR_TRAIN_MONITOR"
BELOW_FLOOR_INNER_VALIDATION = "BELOW_FLOOR_INNER_VALIDATION"


EVIDENCE_DESCRIPTOR_SCHEMA = "agent_multi.activity_evidence_descriptor.v1"

#: D1.2 documented rule: a sha256 is EXACTLY 64 hex characters; upper or
#: mixed case is accepted and NORMALIZED TO LOWERCASE; any other length
#: (including 40-hex SHA-1) or content is refused.
def normalize_sha256(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
        return token
    return None


class ActivityAuthorityError(ValueError):
    """Malformed REQUEST (bad floor, contradictory config). A malformed
    MEASUREMENT is never an exception — it is typed unavailable and
    ineligible."""


class IneligibleCandidateError(RuntimeError):
    """Raised by :func:`require_rankable` when a consumer asks for a
    comparable selection score on an ineligible candidate."""


def _coerce_count(value: Any) -> Optional[int]:
    """A trade count is available ONLY as a non-negative Integral.

    C1 (order 2026-08-19): booleans, strings, containers, fractional
    floats, NaN and infinities are typed unavailable — never truncated,
    never parsed. INTEGRAL FLOATS ARE REFUSED: the one canonical
    persisted representation of a count is an integer type
    (numbers.Integral, which admits numpy integer scalars), so `3.0`
    is a schema violation, not a count. No conversion runs on foreign
    types, so no OverflowError can leak."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, numbers.Integral):
        return None
    number = int(value)
    return number if number >= 0 else None


def validate_floor_value(value: Any, *, source: str) -> int:
    """C1/C4: a declared floor must be a non-bool Integral >= 1.
    Anything else — including an explicit 0 — is a TYPED refusal;
    nothing is coerced, repaired or truncated, and no OverflowError
    can leak because no conversion runs on foreign types."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ActivityAuthorityError(
            f"MALFORMED_ACTIVITY_FLOOR: {source}={value!r} is not an "
            "integer count")
    floor = int(value)
    if floor < STRICT_NONZERO_FLOOR:
        raise ActivityAuthorityError(
            f"CONTRADICTORY_ACTIVITY_FLOOR: {source}={floor} is below "
            f"the strict nonzero floor {STRICT_NONZERO_FLOOR} — a zero "
            "floor is refused, never silently repaired "
            "(order 2026-08-19 C4)")
    return floor


def threshold_contract_for(floor: int,
                           calibrated: Optional[Mapping[str, Any]] = None,
                           ) -> dict:
    """C4: floor 1 is the ONLY floor the strict-nonzero contract id may
    describe. A higher floor demands an explicit calibrated contract —
    id (different from the strict id), matching floor value, units and
    a non-empty evidence reference."""
    if floor == STRICT_NONZERO_FLOOR and calibrated is None:
        return {
            "id": THRESHOLD_CONTRACT_ID,
            "floor": STRICT_NONZERO_FLOOR,
            "units": "trades",
            "informational_fields": list(INFORMATIONAL_FIELDS),
        }
    if calibrated is None:
        raise ActivityAuthorityError(
            f"UNBOUND_FLOOR_CONTRACT: floor {floor} > "
            f"{STRICT_NONZERO_FLOOR} may not reuse "
            f"{THRESHOLD_CONTRACT_ID!r}; declare an explicit calibrated "
            "contract with id, floor, units and evidence_ref "
            "(order 2026-08-19 C4)")
    required = ("id", "floor", "units", "evidence_ref")
    missing = [k for k in required if calibrated.get(k) in (None, "")]
    if missing:
        raise ActivityAuthorityError(
            f"INCOMPLETE_FLOOR_CONTRACT: calibrated contract lacks "
            f"{missing}")
    # D4.1/D4.2: field TYPES are validated, never truthiness — a string
    # or fractional floor refuses through the same validator as every
    # other floor; units and evidence_ref must be non-empty strings.
    for field in ("id", "units", "evidence_ref"):
        if not isinstance(calibrated[field], str) or \
                not calibrated[field].strip():
            raise ActivityAuthorityError(
                f"MALFORMED_FLOOR_CONTRACT: {field} must be a "
                f"non-empty string, got {calibrated[field]!r}")
    contract_floor = validate_floor_value(
        calibrated["floor"], source="calibrated_contract.floor")
    if calibrated["id"] == THRESHOLD_CONTRACT_ID:
        raise ActivityAuthorityError(
            "UNBOUND_FLOOR_CONTRACT: a calibrated floor cannot reuse "
            f"the strict-nonzero id {THRESHOLD_CONTRACT_ID!r}")
    if contract_floor != floor:
        raise ActivityAuthorityError(
            f"FLOOR_CONTRACT_MISMATCH: contract floor "
            f"{contract_floor} != requested floor {floor}")
    # D4.3: the published identity binds the payload, so two different
    # floors can never share one identity string.
    return {**calibrated,
            "id": contract_identity_for(calibrated)
            if not str(calibrated["id"]).startswith(
                "agent_multi.activity_floor.calibrated.v1+")
            else calibrated["id"],
            "declared_id": calibrated["id"],
            "informational_fields": list(INFORMATIONAL_FIELDS)}


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
    return validate_floor_value(raw, source=key)


def evaluate_role_activity(trades: Any, *, role: str,
                           floor: int = STRICT_NONZERO_FLOOR,
                           calibrated_contract:
                               Optional[Mapping[str, Any]] = None) -> dict:
    """Single-role primitive, for consumers that only see one role
    (e.g. the lexicographic validation contract). Same floor, same
    typing, same contract id."""
    floor = validate_floor_value(floor, source=f"{role}.floor")
    contract = threshold_contract_for(floor, calibrated_contract)
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
        "threshold_contract_id": contract["id"],
        "threshold_contract": contract,
    }


def evaluate_activity(
    *,
    train_monitor_trades: Any,
    inner_validation_trades: Any,
    active_weeks: Any = None,
    exposure_fraction: Any = None,
    evidence_refs: Mapping[str, str] | None = None,
    floor: int = STRICT_NONZERO_FLOOR,
    calibrated_contract: Optional[Mapping[str, Any]] = None,
) -> dict:
    """The shared typed activity result (order 2026-08-18 WP1).

    Note what is ABSENT from the signature: any return, profit, Sharpe
    or score. Negative return cannot be an activity failure because
    this authority never sees a return; and no consumer can multiply a
    trade count into fitness through this module because it exposes no
    score at all.
    """
    contract = threshold_contract_for(
        validate_floor_value(floor, source="floor"),
        calibrated_contract)
    monitor = evaluate_role_activity(
        train_monitor_trades, role="train_monitor", floor=floor,
        calibrated_contract=calibrated_contract)
    validation = evaluate_role_activity(
        inner_validation_trades, role="inner_validation", floor=floor,
        calibrated_contract=calibrated_contract)
    reasons = list(monitor["reason_codes"]) + list(
        validation["reason_codes"])

    # D1 (order 2026-08-19): evidence is a TYPED DESCRIPTOR that is
    # loaded, digest-verified and used to DERIVE the role's count. A
    # free-form string, an in-memory digest, a missing artifact, a
    # mutated artifact or a fact that contradicts the caller's count is
    # typed unavailable and ineligible.
    refs = dict(evidence_refs or {})
    verifications: dict = {}
    for role, passed_count in (("train_monitor", monitor),
                               ("inner_validation", validation)):
        verdict = verify_evidence(refs.get(role) or {},
                                  expected_role=role)
        verifications[role] = verdict
        reasons.extend(verdict["reason_codes"])
        if verdict["verified"]:
            derived = verdict["derived_trades"]
            if passed_count["trades_available"] and \
                    passed_count["trades"] != derived:
                reasons.append(f"EVIDENCE_FACT_MISMATCH_{role.upper()}")
            elif not passed_count["trades_available"]:
                # the artifact is the measurement: adopt the derived
                # count and re-run the role judgement on it
                fresh = evaluate_role_activity(
                    derived, role=role, floor=floor,
                    calibrated_contract=calibrated_contract)
                passed_count.update(fresh)
                # drop the stale unavailable reason for this role
                reasons[:] = [r for r in reasons
                              if r != f"TRADES_UNAVAILABLE_{role.upper()}"]
                reasons.extend(fresh["reason_codes"])

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
        "threshold_contract_id": contract["id"],
        "threshold_contract": contract,
        "floor": floor,
        "calibrated_floor": (
            "pending_wp2_evidence"
            if floor == STRICT_NONZERO_FLOOR else contract["id"]),
        "evidence_refs": refs,
        "evidence_verifications": verifications,
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


# ── D1 (order 2026-08-19): re-derivable evidence ──────────────────────

REQUIRED_DESCRIPTOR_FIELDS = (
    "schema", "role", "source_kind", "artifact_locator", "sha256",
    "fact_key", "producer_contract_id",
)


def validate_evidence_descriptor(descriptor: Any, *,
                                 expected_role: str) -> list[str]:
    """Shape validation only; returns typed reason codes (empty = ok).
    Verification against the artifact is :func:`verify_evidence`."""
    role_tag = expected_role.upper()
    if not isinstance(descriptor, Mapping) or not descriptor:
        return [f"EVIDENCE_UNBOUND_{role_tag}"]
    reasons = []
    if descriptor.get("schema") != EVIDENCE_DESCRIPTOR_SCHEMA:
        reasons.append(f"EVIDENCE_SCHEMA_INVALID_{role_tag}")
    if descriptor.get("role") != expected_role:
        reasons.append(f"EVIDENCE_ROLE_MISMATCH_{role_tag}")
    for field in ("source_kind", "artifact_locator", "fact_key",
                  "producer_contract_id"):
        value = descriptor.get(field)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"EVIDENCE_FIELD_MISSING_{role_tag}")
            break
    if normalize_sha256(descriptor.get("sha256")) is None:
        reasons.append(f"EVIDENCE_DIGEST_INVALID_{role_tag}")
    return reasons


def verify_evidence(descriptor: Mapping[str, Any], *,
                    expected_role: str) -> dict:
    """D1.3: load the referenced artifact, verify its digest and DERIVE
    the role's trade count from the named fact. A digest over an
    in-memory assertion is not evidence. Never raises for a measurement
    problem — returns typed reasons and an unavailable count."""
    import csv as _csv
    import hashlib as _hashlib
    import json as _json
    from pathlib import Path as _Path

    role_tag = expected_role.upper()
    reasons = validate_evidence_descriptor(descriptor,
                                           expected_role=expected_role)
    if reasons:
        return {"verified": False, "derived_trades": None,
                "reason_codes": reasons}
    path = _Path(str(descriptor["artifact_locator"]))
    if not path.is_file():
        return {"verified": False, "derived_trades": None,
                "reason_codes": [f"EVIDENCE_ARTIFACT_MISSING_{role_tag}"]}
    data = path.read_bytes()
    actual = _hashlib.sha256(data).hexdigest()
    if actual != normalize_sha256(descriptor["sha256"]):
        return {"verified": False, "derived_trades": None,
                "reason_codes": [f"EVIDENCE_DIGEST_MISMATCH_{role_tag}"]}
    fact_key = str(descriptor["fact_key"])
    derived: Optional[int] = None
    try:
        if path.suffix == ".json":
            doc = _json.loads(data)
            derived = _coerce_count(doc.get(fact_key))
        else:  # CSV return trace: the fact is the column's final value
            rows = list(_csv.DictReader(
                data.decode("utf-8", "strict").splitlines()))
            values = [row.get(fact_key) for row in rows]
            last = values[-1] if values else None
            if last is not None:
                try:
                    number = float(last)
                except (TypeError, ValueError):
                    number = None
                if number is not None and number >= 0 and \
                        float(number).is_integer():
                    derived = int(number)
    except Exception:
        derived = None
    if derived is None:
        return {"verified": False, "derived_trades": None,
                "reason_codes": [f"EVIDENCE_FACT_MISMATCH_{role_tag}"]}
    return {"verified": True, "derived_trades": derived,
            "reason_codes": []}


def contract_identity_for(calibrated: Mapping[str, Any]) -> str:
    """D4.3: a calibrated contract identity BINDS its payload — schema,
    exact floor, units and evidence digest — so two different floors can
    never share one identity."""
    import hashlib as _hashlib
    import json as _json
    payload = _json.dumps({
        "schema": "agent_multi.activity_floor.calibrated.v1",
        "floor": int(calibrated["floor"]),
        "units": str(calibrated["units"]),
        "evidence_ref": str(calibrated["evidence_ref"]),
    }, sort_keys=True)
    digest = _hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"agent_multi.activity_floor.calibrated.v1+{digest}"
