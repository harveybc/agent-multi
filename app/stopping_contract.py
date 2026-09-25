"""Declared-versus-effective L1 stopping contract.

AUD-P1LR-20260815-234. The terminal P1LR decision run (identity
``c0e53cf18b7d60dd``) declared "``max_epochs=2000``, patience 60, floor
40, paired train-monitor/inner-validation stopping" and then ended every
one of its sixteen cells at epoch 80 — 4% of the declared ceiling —
because a SECOND, independently bounded terminator fired: the
activity-ineligible patience (``l1_activity_patience=40`` starting at
epoch 40).

That terminator was not a substitution. It is a disjoint guard on a
state the paired rule cannot terminate at all: ``no_improve`` advances
only on activity-eligible epochs (see
``pipeline_plugins/rl_pipeline_with_validation.py``
``_update_l1_checkpoint_state``), so a policy that never passes the trade
gate can never consume improvement patience and the paired rule can
never fire.  Every terminal record shows ``l1_patience_used == 0`` at
epoch 79.  Without the activity stop those cells would have run to the
ceiling and selected the SAME checkpoint.

The defect is therefore declarative, and it is the kind that costs a
fleet: the contract's prose named one terminator while three could fire,
and the earliest of them was 25x earlier than the declared ceiling.  This
module makes that impossible to repeat.  It derives the rules that CAN
end training from the resolved runtime config, compares them against the
contract's declaration, and REFUSES a decision run whose effective
stopping rule is not the one it declares.

The module is deliberately dependency-free so a launcher can call it
before any GPU, model or dataset work happens.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

SCHEMA = "agent_multi.stopping_contract.v1"

#: Every rule in the epoch loop that can END training, in the order the
#: loop evaluates them.  ``pipeline_plugins/rl_pipeline_with_validation``
#: OWNS this set; ``tests/unit/test_stopping_contract.py`` holds the two
#: together so a fourth terminator cannot be added silently.
TERMINATORS = ("l1_early_stop", "activity_stop", "max_epochs_budget")

#: Stopping defaults the epoch loop falls back to when the config is
#: silent.  Mirrored from ``PipelinePlugin.plugin_params``; the surface
#: test refuses drift.
PIPELINE_STOPPING_DEFAULTS: dict[str, Any] = {
    "max_epochs": 2_000,
    "l1_patience": 60,
    "l1_patience_start_epoch": 40,
    "l1_activity_patience": 40,
    # None means "inherit l1_patience_start_epoch", exactly as the loop
    # resolves it.
    "l1_activity_patience_start_epoch": None,
}

#: Violation codes.  Each names a DIFFERENT way a run can execute a
#: contract nobody approved.
UNDECLARED_TERMINATOR = "UNDECLARED_TERMINATOR"
KNOB_NOT_PROPAGATED = "KNOB_NOT_PROPAGATED"
CEILING_MISMATCH = "CEILING_MISMATCH"
UNDECLARED_PREEMPTION = "UNDECLARED_PREEMPTION"


class StoppingContractViolation(RuntimeError):
    """A run's effective stopping rule differs from its declared one."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = str(code)
        self.detail = str(message)


def _positive_int(value: Any, *, fallback: int) -> int:
    if value is None:
        return int(fallback)
    return int(value)


def effective_stopping_rules(config: Mapping[str, Any]) -> dict:
    """Derive the terminators the epoch loop WILL apply to ``config``.

    ``config`` is the resolved runtime config — the dict the pipeline
    plugin actually receives — not the contract.  The resolution order
    below mirrors ``rl_pipeline_with_validation.PipelinePlugin.
    run_pipeline`` exactly: config value, else plugin default, and the
    activity start epoch inherits the improvement start epoch when it is
    unset.
    """
    defaults = PIPELINE_STOPPING_DEFAULTS
    max_epochs = _positive_int(config.get("max_epochs"),
                               fallback=defaults["max_epochs"])
    patience = _positive_int(config.get("l1_patience"),
                             fallback=defaults["l1_patience"])
    patience_start = max(1, _positive_int(
        config.get("l1_patience_start_epoch"),
        fallback=defaults["l1_patience_start_epoch"]))
    activity = _positive_int(config.get("l1_activity_patience"),
                             fallback=defaults["l1_activity_patience"])
    activity_start_raw = config.get("l1_activity_patience_start_epoch")
    if activity_start_raw is None:
        activity_start_raw = defaults["l1_activity_patience_start_epoch"]
    activity_start = max(1, _positive_int(activity_start_raw,
                                          fallback=patience_start))

    rules = {
        "l1_early_stop": {
            "enabled": patience > 0,
            "patience": patience,
            "start_epoch": patience_start,
            # ``no_improve`` only advances on activity-eligible epochs,
            # so this rule cannot fire against an inactive policy.
            "requires_activity_eligible_epochs": True,
            "earliest_stop_epoch": patience_start + patience,
        },
        "activity_stop": {
            "enabled": activity > 0,
            "patience": activity,
            "start_epoch": activity_start,
            "requires_activity_eligible_epochs": False,
            "earliest_stop_epoch": activity_start + activity,
        },
        "max_epochs_budget": {
            "enabled": True,
            "ceiling_epochs": max_epochs,
            "requires_activity_eligible_epochs": False,
            "earliest_stop_epoch": max_epochs,
        },
    }
    enabled = [name for name in TERMINATORS if rules[name]["enabled"]]
    return {
        "schema": SCHEMA,
        "rules": rules,
        "enabled_terminators": enabled,
        "earliest_effective_stop_epoch": min(
            int(rules[name]["earliest_stop_epoch"]) for name in enabled),
    }


def _declared_knobs(declared: Mapping[str, Any]) -> dict:
    knobs = declared.get("stopping_knobs")
    return dict(knobs) if isinstance(knobs, Mapping) else {}


def _declared_terminators(declared: Mapping[str, Any]) -> set[str]:
    """Terminators the declaration NAMES.

    A terminator is named when the declaration carries its knob, or when
    the human-readable ``effective_stopping_rules.terminators`` list
    carries its name.  Prose alone is never enough: the sentence in
    ``decision_run.stopping`` is documentation, not a declaration.
    """
    named: set[str] = {"max_epochs_budget"}
    knobs = _declared_knobs(declared)
    if "l1_patience" in knobs:
        named.add("l1_early_stop")
    if "l1_activity_patience" in knobs:
        named.add("activity_stop")
    block = declared.get("effective_stopping_rules")
    if isinstance(block, Mapping):
        for name in block.get("terminators") or ():
            named.add(str(name))
    return named


def _declared_ceiling(declared: Mapping[str, Any]) -> int | None:
    ceiling = declared.get("max_global_pass_equivalent_checkpoints")
    return int(ceiling) if isinstance(ceiling, int) else None


def _phase_epochs(declared: Mapping[str, Any]) -> int:
    budget = declared.get("budget_knobs")
    if not isinstance(budget, Mapping):
        return 0
    return int(budget.get("phase1_epochs") or 0)


def assert_declared_stopping_contract(
    *,
    declared: Mapping[str, Any],
    config: Mapping[str, Any],
    label: str = "decision_run",
) -> dict:
    """Refuse a run whose effective stopping rule is not its declared one.

    ``declared`` is the contract block (for P1LR: ``decision_run``).
    ``config`` is the resolved runtime config for ONE cell.  Returns a
    durable evidence block on success; raises
    :class:`StoppingContractViolation` otherwise.

    Four distinct failures are refused:

    ``UNDECLARED_TERMINATOR``
        a rule that can end training is not named by the declaration —
        the run would stop for a reason nobody approved.
    ``KNOB_NOT_PROPAGATED``
        a declared knob never reached the runtime config, so the
        declaration is decorative and editing it changes nothing.
    ``CEILING_MISMATCH``
        phase-1 epochs plus the runtime ``max_epochs`` do not total the
        declared global pass-equivalent ceiling.
    ``UNDECLARED_PREEMPTION``
        some terminator can fire strictly before the declared ceiling
        and the declaration does not state the resulting earliest stop
        epoch.  This is the P1LR failure: stopping at 80 of 2000 was a
        derivable consequence of two declared knobs that no reader of the
        contract ever saw stated.
    """
    effective = effective_stopping_rules(config)
    rules = effective["rules"]
    named = _declared_terminators(declared)
    knobs = _declared_knobs(declared)

    for terminator in effective["enabled_terminators"]:
        if terminator not in named:
            raise StoppingContractViolation(
                UNDECLARED_TERMINATOR,
                f"{label}: {terminator!r} can end training "
                f"(earliest epoch "
                f"{rules[terminator]['earliest_stop_epoch']}) but the "
                f"declaration names only {sorted(named)} — a run must "
                "never stop for a rule its contract does not declare")

    knob_bindings = (
        ("l1_patience", rules["l1_early_stop"]["patience"]),
        ("l1_patience_start_epoch", rules["l1_early_stop"]["start_epoch"]),
        ("l1_activity_patience", rules["activity_stop"]["patience"]),
        ("l1_activity_patience_start_epoch",
         rules["activity_stop"]["start_epoch"]),
    )
    for key, effective_value in knob_bindings:
        if key not in knobs:
            continue
        if int(knobs[key]) != int(effective_value):
            raise StoppingContractViolation(
                KNOB_NOT_PROPAGATED,
                f"{label}: declares {key}={knobs[key]!r} but the "
                f"resolved runtime config will apply "
                f"{effective_value!r} — the declaration is decorative")

    ceiling = _declared_ceiling(declared)
    if ceiling is not None:
        realized = _phase_epochs(declared) + int(
            rules["max_epochs_budget"]["ceiling_epochs"])
        if realized != ceiling:
            raise StoppingContractViolation(
                CEILING_MISMATCH,
                f"{label}: declares a {ceiling} pass-equivalent ceiling "
                f"but phase-1 epochs plus the runtime max_epochs total "
                f"{realized}")

    earliest = int(effective["earliest_effective_stop_epoch"])
    ceiling_epochs = int(rules["max_epochs_budget"]["ceiling_epochs"])
    preempts = earliest < ceiling_epochs
    block = declared.get("effective_stopping_rules")
    declared_earliest = None
    if isinstance(block, Mapping):
        declared_earliest = block.get("earliest_stop_epoch")
    if preempts and declared_earliest is None:
        earliest_rules = sorted(
            name for name in effective["enabled_terminators"]
            if int(rules[name]["earliest_stop_epoch"]) == earliest)
        raise StoppingContractViolation(
            UNDECLARED_PREEMPTION,
            f"{label}: {earliest_rules} can end training at epoch "
            f"{earliest} of a {ceiling_epochs}-epoch ceiling "
            f"({100.0 * earliest / max(1, ceiling_epochs):.1f}% of the "
            "declared budget) and the contract never states it — "
            "declare effective_stopping_rules.earliest_stop_epoch")
    if declared_earliest is not None and int(declared_earliest) != earliest:
        raise StoppingContractViolation(
            UNDECLARED_PREEMPTION,
            f"{label}: declares earliest_stop_epoch="
            f"{int(declared_earliest)} but the resolved rules can end "
            f"training at epoch {earliest}")

    return {
        "schema": SCHEMA,
        "label": str(label),
        "enabled_terminators": list(effective["enabled_terminators"]),
        "earliest_effective_stop_epoch": earliest,
        "declared_ceiling_epochs": ceiling_epochs,
        "preempts_declared_ceiling": bool(preempts),
        "rules": rules,
        "verdict": "STOPPING_CONTRACT_DECLARED",
    }


def stopping_contract_refusal(
    *,
    declared: Mapping[str, Any],
    config: Mapping[str, Any],
    label: str = "decision_run",
) -> tuple[dict | None, dict | None]:
    """Non-raising wrapper for launchers that emit typed refusals.

    Returns ``(evidence, None)`` when the contract holds and
    ``(None, refusal)`` when it does not.
    """
    try:
        return assert_declared_stopping_contract(
            declared=declared, config=config, label=label), None
    except StoppingContractViolation as exc:
        return None, {
            "outcome": "REFUSED_STOPPING_CONTRACT_UNDECLARED",
            "reason": str(exc),
            "code": exc.code,
        }


def describe(rules: Mapping[str, Any]) -> str:
    """One-line human summary of an effective-rules block."""
    parts: Iterable[str] = (
        f"{name}@{rules['rules'][name]['earliest_stop_epoch']}"
        for name in rules["enabled_terminators"])
    return (f"earliest stop epoch "
            f"{rules['earliest_effective_stop_epoch']} via "
            + ", ".join(parts))
