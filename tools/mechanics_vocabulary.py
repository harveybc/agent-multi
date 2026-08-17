"""Finding 263 (AUD-F1-20260816-263): mechanics vocabulary vs promotion.

``SCREEN_VIABLE_REGION`` conflated the mechanics-admission vocabulary
with scientific/activity viability: a reader of ``handoff_viability ==
"VIABLE"`` inferred a promotion-relevant property the one-epoch
mechanics screen never measured (5/16 active cells, two of seven VIABLE
cells activity-inactive — screen ``0c70ab2ce7804750``).

This module is the typed separation, applied at READ time so sealed
verdict bytes and their digests never change:

- ``load_mechanics_screen_verdict`` is the ONLY sanctioned reader. It
  accepts sealed v1 verdicts and new v2 verdicts, maps the v1 keys to
  the mechanics-prefixed vocabulary in memory, injects
  ``promotion_eligible: None`` and REFUSES any mechanics verdict that
  carries a non-null ``promotion_eligible`` — the screen did not
  measure that quantity, so a value there is manufactured evidence.
- ``promotion_eligibility`` ALWAYS refuses for a mechanics verdict,
  typed ``PROMOTION_ELIGIBILITY_NOT_MEASURED_BY_MECHANICS_SCREEN``.
  There is no argument combination that returns eligibility from
  mechanics facts; consumers that want eligibility must present a
  per-candidate terminal cell record with an activity-eligible
  checkpoint (finding 269 implements that consumer in lts).
- ``evaluate_terminal_disposition`` is the freeze/reinvestigate
  predicate (order 2026-08-16 item 4): FREEZE is only reachable from a
  winning terminal record whose ``activity_status == "active"`` and
  ``promotion_eligible is True``; everything else is a typed
  REINVESTIGATE. Prose cannot freeze a recipe.

Nothing here mutates a verdict file. The shim is in-memory only.
"""
from __future__ import annotations

import json
from pathlib import Path

# Sealed emissions (never rewritten):
SCREEN_VERDICT_SCHEMA_V1 = "agent_multi.p1_difficulty_lr_screen_verdict.v1"
# New emissions (finding 263):
SCREEN_VERDICT_SCHEMA_V2 = "agent_multi.p1_difficulty_lr_screen_verdict.v2"
ACCEPTED_SCREEN_SCHEMAS = (SCREEN_VERDICT_SCHEMA_V1,
                           SCREEN_VERDICT_SCHEMA_V2)

MECHANICS_PURPOSE = "mechanics_and_artifact_custody_only"

DISPOSITION_SCHEMA = "agent_multi.p1lr_terminal_disposition.v1"
CONTRACT_PREDICATE_KEY = "terminal_disposition_predicate"

REFUSAL_NOT_MEASURED = (
    "PROMOTION_ELIGIBILITY_NOT_MEASURED_BY_MECHANICS_SCREEN")


class MechanicsVocabularyError(RuntimeError):
    """Typed refusal. ``code`` is machine-readable; the message names
    the exact defect."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _refuse(code: str, message: str) -> None:
    raise MechanicsVocabularyError(code, message)


def load_mechanics_screen_verdict(source) -> dict:
    """Load a mechanics screen verdict through the vocabulary shim.

    ``source`` is a path or an already-parsed dict. Returns a NEW dict
    carrying the mechanics-prefixed vocabulary; the input bytes/dict are
    never mutated, so sealed digests hold.

    Refusals (fail-closed):
    - unknown schema;
    - ``promotion_eligible`` present and non-null on a mechanics
      verdict, v1 or v2 (the screen never measured it);
    - v2 verdict missing ``purpose`` or ``mechanics_screen_passed``.
    """
    if isinstance(source, (str, Path)):
        raw = json.loads(Path(source).read_text())
    elif isinstance(source, dict):
        raw = source
    else:
        _refuse("SCREEN_VERDICT_UNREADABLE",
                f"unsupported verdict source type {type(source).__name__}")
    schema = raw.get("schema")
    if schema not in ACCEPTED_SCREEN_SCHEMAS:
        _refuse("SCREEN_VERDICT_SCHEMA_UNSUPPORTED",
                f"verdict schema {schema!r} is not one of "
                f"{list(ACCEPTED_SCREEN_SCHEMAS)}")

    if raw.get("promotion_eligible") is not None:
        _refuse(REFUSAL_NOT_MEASURED,
                "a mechanics screen verdict carries promotion_eligible="
                f"{raw.get('promotion_eligible')!r}; the one-epoch "
                "mechanics screen does not measure promotion "
                "eligibility, so any non-null value is manufactured "
                "evidence and the verdict is refused at load")

    out = dict(raw)
    if schema == SCREEN_VERDICT_SCHEMA_V1:
        # In-memory migration of the sealed vocabulary. File unchanged.
        if "viability_matrix" in out:
            out["mechanics_viability_matrix"] = out["viability_matrix"]
        for key in ("viable_cells", "collapsed_cells"):
            entries = out.get(key)
            if isinstance(entries, list):
                out[key] = [
                    ({**e, "mechanics_viability": e["handoff_viability"]}
                     if isinstance(e, dict) and "handoff_viability" in e
                     else e)
                    for e in entries]
        out.setdefault("purpose", MECHANICS_PURPOSE)
        out.setdefault("mechanics_screen_passed",
                       out.get("outcome") == "SCREEN_VIABLE_REGION"
                       and all((out.get("gates") or {}).values()))
        out["vocabulary_migrated_from"] = SCREEN_VERDICT_SCHEMA_V1
    else:
        if out.get("purpose") != MECHANICS_PURPOSE:
            _refuse("SCREEN_VERDICT_PURPOSE_MISSING",
                    "a v2 mechanics verdict must declare purpose="
                    f"{MECHANICS_PURPOSE!r}; got {out.get('purpose')!r}")
        if not isinstance(out.get("mechanics_screen_passed"), bool):
            _refuse("SCREEN_VERDICT_MECHANICS_FLAG_MISSING",
                    "a v2 mechanics verdict must carry a boolean "
                    "mechanics_screen_passed")
    out["promotion_eligible"] = None
    return out


def promotion_eligibility(verdict: dict) -> None:
    """ALWAYS refuses: mechanics facts can never yield promotion
    eligibility. Exists so a consumer that reaches for eligibility in a
    screen verdict fails loudly and typed instead of inferring it from
    ``viable_cells`` (finding 263 adversarial requirement b)."""
    _refuse(REFUSAL_NOT_MEASURED,
            "promotion eligibility was requested from a mechanics "
            "screen verdict (outcome="
            f"{verdict.get('outcome')!r}); mechanics viability is not "
            "activity and confers no eligibility — present a terminal "
            "cell record with an activity-eligible checkpoint to the "
            "promotion consumer instead (finding 269)")


def assert_terminal_disposition_contract(contract: dict) -> dict:
    """Order 2026-08-16 item 4, fail-closed: the NEXT executed decision
    contract must carry the freeze/reinvestigate predicate. Returns the
    predicate block; refuses if absent or malformed."""
    block = contract.get(CONTRACT_PREDICATE_KEY)
    if not isinstance(block, dict):
        _refuse("DECISION_WITHOUT_TERMINAL_DISPOSITION",
                "the contract carries no terminal_disposition_predicate"
                " — a decision run whose freeze/reinvestigate rule is "
                "not a typed contract predicate is refused (order "
                "2026-08-16 item 4)")
    if block.get("schema") != DISPOSITION_SCHEMA:
        _refuse("TERMINAL_DISPOSITION_SCHEMA_UNSUPPORTED",
                f"predicate schema {block.get('schema')!r} != "
                f"{DISPOSITION_SCHEMA!r}")
    if block.get("otherwise") != "REINVESTIGATE":
        _refuse("TERMINAL_DISPOSITION_NOT_FAIL_CLOSED",
                "the predicate's otherwise-branch must be "
                "REINVESTIGATE — FREEZE can never be the default")
    return block


def evaluate_terminal_disposition(contract: dict,
                                  winning_record: dict | None) -> dict:
    """The executable freeze/reinvestigate predicate.

    FREEZE requires ALL of, read from the winning terminal record:
    - ``activity_status == "active"``
    - ``promotion_eligible is True``
    - a non-empty ``best_model_sha256``
    Anything else — including an absent record — is REINVESTIGATE with
    typed reasons. Never raises for a measured outcome; raises only for
    a contract without the predicate.
    """
    block = assert_terminal_disposition_contract(contract)
    reasons: list[str] = []
    if not isinstance(winning_record, dict):
        reasons.append("no_winning_terminal_record")
    else:
        if winning_record.get("activity_status") != "active":
            reasons.append(
                "winning_record_activity_status_"
                f"{winning_record.get('activity_status')}")
        if winning_record.get("promotion_eligible") is not True:
            reasons.append("winning_record_not_promotion_eligible")
        if not winning_record.get("best_model_sha256"):
            reasons.append("winning_record_missing_best_model_sha256")
    disposition = "FREEZE" if not reasons else str(
        block.get("otherwise", "REINVESTIGATE"))
    return {
        "schema": DISPOSITION_SCHEMA,
        "disposition": disposition,
        "reasons": reasons,
        "predicate": block,
    }
