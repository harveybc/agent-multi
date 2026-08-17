"""Finding 263 adversarial suite: mechanics vocabulary vs promotion.

Ordered requirements (MUSASHI_RESPONSE 2026-08-16, orders 1 and 3):
(a) a verdict with seven viable cells and ZERO active cells must not
    yield promotion eligibility by any path;
(b) a consumer that derives eligibility from ``viable_cells`` fails;
(c) a mechanics verdict carrying a non-null promotion_eligible is
    rejected at load;
(d) sealed v1 verdicts load through the shim unchanged on disk;
plus the freeze/reinvestigate predicate (order item 4) and the
decision-mode wiring (a v2 contract without the predicate refuses
before any pipeline work).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import mechanics_vocabulary as mv  # noqa: E402
from tools import p1_difficulty_lr_factorial as p1  # noqa: E402
from tests.test_p1_difficulty_lr_factorial import (  # noqa: E402
    CLEAN_SOURCES,
    FakePipeline,
    _viable_screen_gate,
)
from tests import test_p1lr_factorial_v2 as v2t  # noqa: E402


@pytest.fixture(autouse=True)
def pinned_source_identities(monkeypatch):
    monkeypatch.setattr(p1.ladder, "source_identities",
                        lambda: copy.deepcopy(CLEAN_SOURCES))


def _sealed_v1_verdict() -> dict:
    """The exact shape of the sealed screen family (0c70ab2ce7804750 /
    14e7ce8208ac9776): seven VIABLE cells, five active of sixteen —
    and, critically, two viable-but-inactive cells."""
    return {
        "schema": mv.SCREEN_VERDICT_SCHEMA_V1,
        "outcome": "SCREEN_VIABLE_REGION",
        "experiment_identity": "0c70ab2ce7804750",
        "contract_sha256": "c" * 64,
        "records_present": 16,
        "records_expected": 16,
        "gates": {"replica_terminal_loads": True,
                  "records_16_16": True},
        "activity": {"active_cells": 5, "cells_expected": 16,
                     "inactive_cells": 11,
                     "classification": "PARTIAL_ACTIVITY_SURVIVAL"},
        "viability_matrix": {
            "P1E_LR3E5": {"101": "VIABLE", "202": "VIABLE",
                          "303": "VIABLE", "404": "VIABLE"},
            "P1N_LR3E5": {"101": "VIABLE",
                          "202": "BELOW_NORMAL_THRESHOLD",
                          "303": "VIABLE", "404": "VIABLE"},
        },
        "viable_cells": [
            {"seed": 101, "cell": "P1E_LR3E5",
             "handoff_viability": "VIABLE"},
            {"seed": 101, "cell": "P1N_LR3E5",
             "handoff_viability": "VIABLE"},
        ],
        "collapsed_cells": [],
        "performance_claims": "none — collapse/contract screen only",
    }


class TestShimLoad:
    def test_sealed_v1_loads_and_maps_without_mutation(self, tmp_path):
        raw = _sealed_v1_verdict()
        path = tmp_path / "screen_verdict.json"
        path.write_text(json.dumps(raw, sort_keys=True))
        before = path.read_bytes()
        out = mv.load_mechanics_screen_verdict(path)
        # (d) file bytes untouched — sealed digests hold
        assert path.read_bytes() == before
        assert out["mechanics_viability_matrix"] == \
            raw["viability_matrix"]
        assert all(e["mechanics_viability"] == "VIABLE"
                   for e in out["viable_cells"])
        assert out["promotion_eligible"] is None
        assert out["purpose"] == mv.MECHANICS_PURPOSE
        assert out["mechanics_screen_passed"] is True
        assert out["vocabulary_migrated_from"] == \
            mv.SCREEN_VERDICT_SCHEMA_V1

    def test_source_dict_is_never_mutated(self):
        raw = _sealed_v1_verdict()
        snapshot = copy.deepcopy(raw)
        mv.load_mechanics_screen_verdict(raw)
        assert raw == snapshot

    def test_unknown_schema_refused(self):
        with pytest.raises(mv.MechanicsVocabularyError,
                           match="SCREEN_VERDICT_SCHEMA_UNSUPPORTED"):
            mv.load_mechanics_screen_verdict({"schema": "bogus.v9"})

    @pytest.mark.parametrize("value", [True, False, "yes", 1])
    def test_non_null_promotion_eligible_rejected_at_load(self, value):
        # (c) — v1 shape
        raw = _sealed_v1_verdict()
        raw["promotion_eligible"] = value
        with pytest.raises(
                mv.MechanicsVocabularyError,
                match=mv.REFUSAL_NOT_MEASURED):
            mv.load_mechanics_screen_verdict(raw)

    def test_v2_with_non_null_promotion_eligible_rejected(self):
        # (c) — v2 shape: even a well-formed v2 verdict is refused the
        # moment it claims the quantity it cannot measure.
        raw = _sealed_v1_verdict()
        raw["schema"] = mv.SCREEN_VERDICT_SCHEMA_V2
        raw["purpose"] = mv.MECHANICS_PURPOSE
        raw["mechanics_screen_passed"] = True
        raw["promotion_eligible"] = True
        with pytest.raises(mv.MechanicsVocabularyError,
                           match=mv.REFUSAL_NOT_MEASURED):
            mv.load_mechanics_screen_verdict(raw)

    def test_v2_missing_purpose_refused(self):
        raw = _sealed_v1_verdict()
        raw["schema"] = mv.SCREEN_VERDICT_SCHEMA_V2
        raw["mechanics_screen_passed"] = True
        with pytest.raises(mv.MechanicsVocabularyError,
                           match="SCREEN_VERDICT_PURPOSE_MISSING"):
            mv.load_mechanics_screen_verdict(raw)


class TestEligibilityNeverFromMechanics:
    def test_seven_viable_zero_active_yields_no_eligibility(self):
        # (a) the adversarial shape: a fully viable region with zero
        # activity anywhere. Eligibility must be unreachable.
        raw = _sealed_v1_verdict()
        raw["activity"] = {"active_cells": 0, "cells_expected": 16,
                           "inactive_cells": 16,
                           "classification": "NO_ACTIVITY_SURVIVAL"}
        out = mv.load_mechanics_screen_verdict(raw)
        assert out["promotion_eligible"] is None
        with pytest.raises(mv.MechanicsVocabularyError,
                           match=mv.REFUSAL_NOT_MEASURED):
            mv.promotion_eligibility(out)

    def test_deriving_eligibility_from_viable_cells_is_a_failure(self):
        # (b) the exact bad inference this cycle documented: VIABLE is
        # mechanics vocabulary; treating it as eligibility must fail
        # loudly, typed, every time — even for a genuinely active cell.
        out = mv.load_mechanics_screen_verdict(_sealed_v1_verdict())
        assert out["viable_cells"]  # the bait is present
        with pytest.raises(mv.MechanicsVocabularyError) as err:
            mv.promotion_eligibility(out)
        assert err.value.code == mv.REFUSAL_NOT_MEASURED
        assert "finding 269" in str(err.value)


class TestTerminalDispositionPredicate:
    def test_real_v2_contract_carries_the_predicate(self):
        contract = p1.load_contract(p1.CONTRACT_PATH_V2)
        block = mv.assert_terminal_disposition_contract(contract)
        assert block["otherwise"] == "REINVESTIGATE"

    def test_contract_without_predicate_refused(self):
        with pytest.raises(
                mv.MechanicsVocabularyError,
                match="DECISION_WITHOUT_TERMINAL_DISPOSITION"):
            mv.assert_terminal_disposition_contract({})

    def test_freeze_default_is_refused(self):
        block = {"schema": mv.DISPOSITION_SCHEMA, "otherwise": "FREEZE"}
        with pytest.raises(
                mv.MechanicsVocabularyError,
                match="TERMINAL_DISPOSITION_NOT_FAIL_CLOSED"):
            mv.assert_terminal_disposition_contract(
                {mv.CONTRACT_PREDICATE_KEY: block})

    def _contract(self):
        return {mv.CONTRACT_PREDICATE_KEY: {
            "schema": mv.DISPOSITION_SCHEMA,
            "otherwise": "REINVESTIGATE"}}

    def test_active_eligible_winning_record_freezes(self):
        verdict = mv.evaluate_terminal_disposition(self._contract(), {
            "activity_status": "active",
            "promotion_eligible": True,
            "best_model_sha256": "a" * 64})
        assert verdict["disposition"] == "FREEZE"
        assert verdict["reasons"] == []

    @pytest.mark.parametrize("record,expected_reason", [
        (None, "no_winning_terminal_record"),
        ({"activity_status": "inactive", "promotion_eligible": False,
          "best_model_sha256": None},
         "winning_record_activity_status_inactive"),
        ({"activity_status": "active", "promotion_eligible": False,
          "best_model_sha256": "a" * 64},
         "winning_record_not_promotion_eligible"),
        ({"activity_status": "active", "promotion_eligible": True,
          "best_model_sha256": ""},
         "winning_record_missing_best_model_sha256"),
    ])
    def test_everything_else_reinvestigates(self, record,
                                            expected_reason):
        verdict = mv.evaluate_terminal_disposition(
            self._contract(), record)
        assert verdict["disposition"] == "REINVESTIGATE"
        assert expected_reason in verdict["reasons"]


class TestDecisionWiring:
    def test_v2_decision_without_predicate_refuses_before_pipeline(
            self, v2rt):
        del v2rt.contract[mv.CONTRACT_PREDICATE_KEY]
        gate = _viable_screen_gate(v2rt.contract)
        summary = v2t._run_seed(v2rt, mode="decision",
                                screen_gate=gate)
        assert summary["outcome"] == \
            "REFUSED_DECISION_WITHOUT_TERMINAL_DISPOSITION"
        assert not FakePipeline.calls

    def test_gate_with_non_null_promotion_eligible_refuses(self, v2rt):
        gate = _viable_screen_gate(v2rt.contract)
        gate["promotion_eligible"] = True
        summary = v2t._run_seed(v2rt, mode="decision",
                                screen_gate=gate)
        assert summary["outcome"] == "REFUSED_DECISION_UNGATED"
        assert mv.REFUSAL_NOT_MEASURED in summary["reason"]
        assert not FakePipeline.calls


@pytest.fixture()
def v2rt(tmp_path, monkeypatch):
    """The v2 module's runnable-contract fixture, reused through its
    unwrapped function so this module needs no fixture duplication."""
    bindings = p1.load_bindings()
    return v2t.rt.__wrapped__(bindings, tmp_path, monkeypatch)
