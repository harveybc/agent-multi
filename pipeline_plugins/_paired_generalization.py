"""WP2 — paired_generalization_weekly_v1 (doc 38 §4.2).

ONE reusable comparator imported by pipeline and optimizer. It never
averages lexicographic ORDER KEYS (ordinal transport encodings); it
computes the same common-scale robust weekly utility on each split of a
pair and returns mean - beta*|gap| with typed eligibility, deterministic
tie-breaks and every raw fact. L1 pairs train_monitor with
inner_validation; L2 pairs inner_validation with outer_validation.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

SCHEMA = "agent_multi.paired_generalization_weekly.v1"
METRIC_NAME = "paired_generalization_weekly_v1"
UTILITY_KEY = "robust_weekly_rap_fitness"
FALLBACK_UTILITY_KEY = "mean_weekly_rap"


class PairedComparatorError(ValueError):
    pass


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _split_utility(summary: Mapping[str, Any]) -> tuple[float | None, str]:
    utility = _finite(summary.get(UTILITY_KEY))
    if utility is not None:
        return utility, UTILITY_KEY
    utility = _finite(summary.get(FALLBACK_UTILITY_KEY))
    if utility is not None:
        return utility, FALLBACK_UTILITY_KEY
    return None, "unavailable"


def _split_eligibility(summary: Mapping[str, Any], label: str,
                       min_trades: int) -> list[str]:
    # D2 (order 2026-08-19): the trade predicate is the ONE typed
    # authority — boolean, float, string, missing and zero counts get
    # exactly the L1 dispositions; explicit zero floors refuse typed at
    # the resolver; evidence descriptors are verified and persisted by
    # the caller through the same authority result.
    from . import _activity_authority as _activity_auth
    floor = _activity_auth.validate_floor_value(
        min_trades, source=f"{label}.min_trades") \
        if min_trades is not None \
        else _activity_auth.STRICT_NONZERO_FLOOR
    calibrated = None
    if floor > _activity_auth.STRICT_NONZERO_FLOOR:
        calibrated = {
            "id": "agent_multi.activity_floor.config_declared.v1",
            "floor": floor, "units": "trades",
            "evidence_ref": f"config:{label}.min_trades={floor}"}
    role_activity = _activity_auth.evaluate_role_activity(
        summary.get("trades_total"), role=label, floor=floor,
        calibrated_contract=calibrated)
    reasons = list(role_activity["reason_codes"])
    utility, _source = _split_utility(summary)
    if utility is None:
        reasons.append(f"{label}: no finite common-scale weekly utility")
    lex_key = summary.get("selection_contract")
    if lex_key is not None and summary.get(UTILITY_KEY) is None \
            and summary.get(FALLBACK_UTILITY_KEY) is None:
        reasons.append(
            f"{label}: only a lexicographic order key is present —"
            " ordinal keys cannot enter a paired average")
    return reasons


def paired_generalization_weekly_v1(
    summary_a: Mapping[str, Any],
    summary_b: Mapping[str, Any],
    *,
    beta: float,
    min_trades_a: int = 1,
    min_trades_b: int = 1,
    label_a: str = "split_a",
    label_b: str = "split_b",
    candidate_id: str = "",
) -> dict:
    """Typed paired result. Fail-closed eligibility; ineligible pairs
    carry reasons and no score. Every raw fact stays visible."""
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) \
            or not math.isfinite(float(beta)) or float(beta) < 0:
        raise PairedComparatorError("beta must be a finite number >= 0")
    reasons = (_split_eligibility(summary_a, label_a, min_trades_a)
               + _split_eligibility(summary_b, label_b, min_trades_b))
    result: dict = {
        "schema": SCHEMA,
        "metric": METRIC_NAME,
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "beta": float(beta),
        "labels": [label_a, label_b],
        "candidate_id": candidate_id,
    }
    if reasons:
        result.update({"score_a": None, "score_b": None, "mean": None,
                       "gap": None, "penalty": None, "paired_score": None,
                       "tie_break": None})
        return result

    score_a, source_a = _split_utility(summary_a)
    score_b, source_b = _split_utility(summary_b)
    if source_a != source_b:
        result["eligible"] = False
        result["ineligibility_reasons"] = [
            f"utility sources differ ({source_a} vs {source_b});"
            " scores are not common-scale"]
        result.update({"score_a": None, "score_b": None, "mean": None,
                       "gap": None, "penalty": None,
                       "paired_score": None, "tie_break": None})
        return result

    mean = 0.5 * (score_a + score_b)
    gap = abs(score_a - score_b)
    penalty = float(beta) * gap
    paired = mean - penalty

    def _tb(summary):
        return (
            _finite(summary.get("mean_weekly_return")) or 0.0,
            -(_finite(summary.get("max_drawdown_fraction")) or 0.0),
            -(_finite(summary.get("commission_paid")) or 0.0),
        )

    result.update({
        "utility_source": source_a,
        "score_a": score_a,
        "score_b": score_b,
        "mean": mean,
        "gap": gap,
        "penalty": penalty,
        "paired_score": paired,
        "tie_break": (min(score_a, score_b),) + _tb(summary_a)
        + _tb(summary_b) + (candidate_id,),
        "raw_refs": {
            label_a: {k: summary_a.get(k) for k in
                      ("trades_total", "total_return",
                       "mean_weekly_return", "max_drawdown_fraction",
                       UTILITY_KEY, FALLBACK_UTILITY_KEY)},
            label_b: {k: summary_b.get(k) for k in
                      ("trades_total", "total_return",
                       "mean_weekly_return", "max_drawdown_fraction",
                       UTILITY_KEY, FALLBACK_UTILITY_KEY)},
        },
    })
    return result


class PairedStoppingState:
    """L1 stopping bookkeeping over paired results (doc 38 §5.1).

    Improvement requires an ELIGIBLE paired result whose paired_score
    exceeds the best by min_delta. Ineligible results never touch
    improvement patience (activity patience is a separate ledger owned
    by the caller). Serializable for resume with split identity.
    """

    def __init__(self, *, patience: int, floor: int, min_delta: float,
                 split_identity: str):
        if patience <= 0 or floor < 0:
            raise PairedComparatorError("patience/floor must be sane")
        self.patience = int(patience)
        self.floor = int(floor)
        self.min_delta = float(min_delta)
        self.split_identity = str(split_identity)
        self.best_score: float | None = None
        self.best_checkpoint: int | None = None
        self.best_tie_break: tuple | None = None
        self.waited = 0
        self.checkpoints_seen = 0

    def update(self, paired: Mapping[str, Any],
               checkpoint_index: int) -> dict:
        self.checkpoints_seen += 1
        outcome = {"improved": False, "stop": False,
                   "reason": None}
        if not paired.get("eligible"):
            outcome["reason"] = "ineligible checkpoint; improvement" \
                                " patience untouched"
            return outcome
        score = paired["paired_score"]
        tie = tuple(paired["tie_break"])
        better = (
            self.best_score is None
            or score > self.best_score + self.min_delta
            or (abs(score - self.best_score) <= self.min_delta
                and self.best_tie_break is not None
                and tie > self.best_tie_break))
        if better and (self.best_score is None
                       or score > self.best_score + self.min_delta):
            self.best_score = score
            self.best_checkpoint = checkpoint_index
            self.best_tie_break = tie
            self.waited = 0
            outcome["improved"] = True
            return outcome
        if checkpoint_index >= self.floor:
            self.waited += 1
        if self.waited >= self.patience:
            outcome["stop"] = True
            outcome["reason"] = (
                f"no paired improvement > {self.min_delta} for"
                f" {self.waited} eligible checkpoints past floor"
                f" {self.floor}")
        return outcome

    def to_state(self) -> dict:
        return {
            "schema": "agent_multi.paired_stopping_state.v1",
            "patience": self.patience,
            "floor": self.floor,
            "min_delta": self.min_delta,
            "split_identity": self.split_identity,
            "best_score": self.best_score,
            "best_checkpoint": self.best_checkpoint,
            "best_tie_break": (list(self.best_tie_break)
                               if self.best_tie_break else None),
            "waited": self.waited,
            "checkpoints_seen": self.checkpoints_seen,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any],
                   *, split_identity: str) -> "PairedStoppingState":
        if state.get("split_identity") != split_identity:
            raise PairedComparatorError(
                "resume split identity mismatch: stopping state from a"
                " different split contract cannot continue this run")
        instance = cls(patience=state["patience"], floor=state["floor"],
                       min_delta=state["min_delta"],
                       split_identity=split_identity)
        instance.best_score = state["best_score"]
        instance.best_checkpoint = state["best_checkpoint"]
        instance.best_tie_break = (tuple(state["best_tie_break"])
                                   if state["best_tie_break"] else None)
        instance.waited = state["waited"]
        instance.checkpoints_seen = state["checkpoints_seen"]
        return instance
