"""Mutation tests for the generic L1 factorial aggregator.

Repair spec §7.2: malformed cells, duplicate physical records, contract
drift, tensor mismatch, asset mismatch and absent metrics must all yield
INCONCLUSIVE/refusal and never a promotion outcome. The decision core is
pure, so every mutation runs without models or GPUs; the impure probes
are injected.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import aggregate_l1_factorial as agg  # noqa: E402

NESTED = "examples/config/phase_3_eth_sac_dynamics/splits/" \
         "eth_nested_split_contract_v1.json"
NESTED_SHA = json.loads((REPO / NESTED).read_text())["source_sha256"]
EXP = "feedfacecafe0001"
SEEDS = (101, 202)
CELLS = {
    "L1_N_M10": {"phase1_mode": "normal_realistic",
                 "phase2_lr_multiplier": 1.0},
    "L1_E_M10": {"phase1_mode": "easy_chronological_continuation",
                 "phase2_lr_multiplier": 1.0},
    "L1_N_M03": {"phase1_mode": "normal_realistic",
                 "phase2_lr_multiplier": 0.3},
    "L1_E_M03": {"phase1_mode": "easy_chronological_continuation",
                 "phase2_lr_multiplier": 0.3},
}


def make_contract() -> dict:
    return {
        "schema": "agent_multi.l1_factorial_contract.v3",
        "_contract_sha256": "c" * 64,
        "asset": "ethusdt_4h",
        "nested_split_contract": NESTED,
        "cells": copy.deepcopy(CELLS),
        "anchors": {str(s): {"path": f"/anchors/seed{s}.zip",
                             "sha256": f"{s:04x}" * 16}
                    for s in SEEDS},
    }


def make_record(contract: dict, seed: int, cell: str) -> dict:
    spec = contract["cells"][cell]
    src_hash = "a1" * 32
    return {
        "schema": agg.RECORD_SCHEMA,
        "evidence_class": "decision_run",
        "decision_eligible": True,
        "performance_aggregate_eligible": True,
        "experiment_id": EXP,
        "cell": cell,
        "seed": seed,
        "contract_sha256": contract["_contract_sha256"],
        "phase1_mode": spec["phase1_mode"],
        "phase2_lr_multiplier": spec["phase2_lr_multiplier"],
        "anchor_sha256": contract["anchors"][str(seed)]["sha256"],
        "terminal_model_path": f"/out/seed{seed}/{cell}/model.terminal.zip",
        "started_utc": "2026-08-09T00:00:00+00:00",
        "finished_utc": "2026-08-09T01:00:00+00:00",
        "curriculum": {
            "post_easy": {
                "artifact_sha256": "b2" * 32,
                "phase1_terminal_policy_tensor_sha256": src_hash,
                "phase1_gradient_updates": 500,
            },
        },
        "boundary_transfer_evidence": {
            "policy_hash_matches_source_after_transfer": True,
            "source_policy_tensor_hash": src_hash,
            "target_policy_tensor_hash_after_transfer": src_hash,
            "target_policy_tensor_hash_before_transfer": "d4" * 32,
            "source_artifact_sha256": "b2" * 32,
        },
    }


def make_evidence() -> dict:
    return {"asset": "ethusdt_4h", "data_file_hash": NESTED_SHA}


def healthy_probe(record: dict) -> dict:
    return {
        "loads": True,
        "terminal_policy_tensor_sha256": "e5" * 32,
        "tensor_chain_consistent": True,
        "phase2_updates_occurred": True,
    }


def healthy_rollout(record: dict) -> dict:
    return {
        "trades_total": 7,
        "action_raw_std": 0.21,
        "action_non_hold_rate": 0.4,
        "execution_diagnostics": {"protected_market_entries": 5,
                                  "protected_limit_entries": 0,
                                  "protected_stop_entries": 0},
    }


def inactive_rollout(record: dict) -> dict:
    return {
        "trades_total": 0,
        "action_raw_std": 0.0,
        "action_non_hold_rate": 0.0,
        "execution_diagnostics": {"protected_market_entries": 0,
                                  "protected_limit_entries": 0,
                                  "protected_stop_entries": 0},
    }


def facts(active: bool | None, valid: bool = True) -> dict:
    if not valid:
        return {"valid": False, "active": None,
                "invalid_reasons": ["synthetic invalidity"]}
    return {"valid": True, "active": bool(active), "invalid_reasons": []}


def matrix_from(pattern: dict) -> dict:
    """pattern[mult][seed] = (E_active, N_active)"""
    return {mult: {seed: {"E": facts(e), "N": facts(n)}
                   for seed, (e, n) in seeds.items()}
            for mult, seeds in pattern.items()}


def full_matrix(e_low, n_low, e_high=True, n_high=True) -> dict:
    seeds4 = (101, 202, 303, 404)
    return matrix_from({
        0.3: {s: (e_low, n_low) for s in seeds4},
        1.0: {s: (e_high, n_high) for s in seeds4},
    })


# ---------------------------------------------------------------------------
# §7.2 ordered rules on the pure core
# ---------------------------------------------------------------------------

class TestDecideOutcome:
    def test_easy_contributes(self):
        outcome, why = agg.decide_outcome(
            full_matrix(e_low=True, n_low=False, e_high=True, n_high=False),
            refusals=[])
        assert outcome == "EASY_CONTRIBUTES"
        assert "+4 >= +2" in why or "sum +4" in why or "+4" in why

    def test_lr_only(self):
        outcome, _ = agg.decide_outcome(
            full_matrix(e_low=True, n_low=True, e_high=True, n_high=True),
            refusals=[])
        assert outcome == "LR_ONLY"  # N 4/4, delta sum 0

    def test_easy_harmful(self):
        outcome, _ = agg.decide_outcome(
            full_matrix(e_low=False, n_low=True, e_high=False, n_high=True),
            refusals=[])
        assert outcome == "EASY_HARMFUL"

    def test_easy_harmful_precedes_lr_only(self):
        # N 4/4, E 0/4 satisfies both rule 4 and rule 5; rule 4 wins.
        outcome, _ = agg.decide_outcome(
            full_matrix(e_low=False, n_low=True, e_high=True, n_high=True),
            refusals=[])
        assert outcome == "EASY_HARMFUL"

    def test_interaction_precedes_easy_contributes(self):
        # Low level looks like EASY_CONTRIBUTES (sum +4) but the high
        # level sign-flips (sum -4): rule 2 fires first.
        outcome, why = agg.decide_outcome(
            full_matrix(e_low=True, n_low=False, e_high=False, n_high=True),
            refusals=[])
        assert outcome == "INTERACTION"
        assert "disagree in sign" in why

    def test_interaction_needs_both_sums_nonzero(self):
        # High-level sum 0 -> not INTERACTION; low says EASY_CONTRIBUTES.
        outcome, _ = agg.decide_outcome(
            full_matrix(e_low=True, n_low=False, e_high=True, n_high=True),
            refusals=[])
        assert outcome == "EASY_CONTRIBUTES"

    def test_rule1_invalid_cell_forces_inconclusive(self):
        matrix = full_matrix(e_low=True, n_low=False)
        matrix[0.3][202]["N"] = facts(None, valid=False)
        outcome, why = agg.decide_outcome(matrix, refusals=[])
        assert outcome == "INCONCLUSIVE"
        assert "invalid cell" in why

    def test_rule1_missing_factor_forces_inconclusive(self):
        matrix = full_matrix(e_low=True, n_low=False)
        del matrix[1.0][303]["E"]
        outcome, why = agg.decide_outcome(matrix, refusals=[])
        assert outcome == "INCONCLUSIVE"
        assert "missing cell" in why

    def test_refusals_precede_everything(self):
        outcome, why = agg.decide_outcome(
            full_matrix(e_low=True, n_low=False),
            refusals=["duplicate physical record for seed=101"])
        assert outcome == "INCONCLUSIVE"
        assert "refusals precede" in why

    def test_mixed_pattern_is_inconclusive(self):
        matrix = matrix_from({
            0.3: {101: (True, False), 202: (False, True),
                  303: (True, True), 404: (False, False)},
            1.0: {101: (True, True), 202: (True, True),
                  303: (True, True), 404: (True, True)},
        })
        outcome, why = agg.decide_outcome(matrix, refusals=[])
        assert outcome == "INCONCLUSIVE"
        assert "no rule" in why

    def test_wrong_level_count_is_inconclusive(self):
        matrix = matrix_from({0.3: {101: (True, False)}})
        outcome, why = agg.decide_outcome(matrix, refusals=[])
        assert outcome == "INCONCLUSIVE"
        assert "2 LR levels" in why

    def test_unequal_seed_sets_are_inconclusive(self):
        matrix = matrix_from({
            0.3: {101: (True, False), 202: (True, False)},
            1.0: {101: (True, False), 303: (True, False)},
        })
        outcome, why = agg.decide_outcome(matrix, refusals=[])
        assert outcome == "INCONCLUSIVE"
        assert "seed sets differ" in why


# ---------------------------------------------------------------------------
# §7.1 activity facts: invalid is never inactive
# ---------------------------------------------------------------------------

class TestActivityFacts:
    def test_all_facts_positive_is_active(self):
        f = agg.activity_facts(terminal_probe=healthy_probe({}),
                               rollout_summary=healthy_rollout({}))
        assert f["valid"] is True and f["active"] is True

    def test_zero_activity_is_inactive_not_invalid(self):
        f = agg.activity_facts(terminal_probe=healthy_probe({}),
                               rollout_summary=inactive_rollout({}))
        assert f["valid"] is True and f["active"] is False

    def test_missing_probe_is_invalid_never_inactive(self):
        f = agg.activity_facts(terminal_probe=None,
                               rollout_summary=healthy_rollout({}))
        assert f["valid"] is False and f["active"] is None

    def test_unloadable_terminal_is_invalid(self):
        probe = healthy_probe({})
        probe["loads"] = False
        probe["error"] = "boom"
        f = agg.activity_facts(terminal_probe=probe,
                               rollout_summary=healthy_rollout({}))
        assert f["valid"] is False and f["active"] is None

    def test_tensor_chain_mismatch_is_invalid(self):
        probe = healthy_probe({})
        probe["tensor_chain_consistent"] = False
        f = agg.activity_facts(terminal_probe=probe,
                               rollout_summary=healthy_rollout({}))
        assert f["valid"] is False
        assert any("tensor" in r for r in f["invalid_reasons"])

    def test_zero_phase2_updates_is_invalid(self):
        probe = healthy_probe({})
        probe["phase2_updates_occurred"] = False
        f = agg.activity_facts(terminal_probe=probe,
                               rollout_summary=healthy_rollout({}))
        assert f["valid"] is False

    def test_missing_rollout_is_invalid(self):
        f = agg.activity_facts(terminal_probe=healthy_probe({}),
                               rollout_summary=None)
        assert f["valid"] is False and f["active"] is None

    def test_absent_metric_is_invalid(self):
        summary = healthy_rollout({})
        del summary["action_raw_std"]
        summary["action_raw_std"] = None
        f = agg.activity_facts(terminal_probe=healthy_probe({}),
                               rollout_summary=summary)
        assert f["valid"] is False

    def test_missing_protected_diagnostics_is_invalid(self):
        summary = healthy_rollout({})
        del summary["execution_diagnostics"]
        f = agg.activity_facts(terminal_probe=healthy_probe({}),
                               rollout_summary=summary)
        assert f["valid"] is False

    def test_no_protected_entry_is_inactive(self):
        summary = healthy_rollout({})
        summary["execution_diagnostics"] = {
            "protected_market_entries": 0,
            "protected_limit_entries": 0,
            "protected_stop_entries": 0}
        f = agg.activity_facts(terminal_probe=healthy_probe({}),
                               rollout_summary=summary)
        assert f["valid"] is True and f["active"] is False


# ---------------------------------------------------------------------------
# record binding mutations
# ---------------------------------------------------------------------------

class TestValidateRecordBindings:
    def check(self, record, contract=None, evidence="default", seed=101,
              cell="L1_N_M10"):
        contract = contract or make_contract()
        ev = make_evidence() if evidence == "default" else evidence
        return agg.validate_record_bindings(
            record, contract=contract, seed=seed, cell=cell,
            experiment_id=EXP, evidence=ev)

    def test_faithful_record_has_no_reasons(self):
        contract = make_contract()
        assert self.check(make_record(contract, 101, "L1_N_M10"),
                          contract) == []

    def test_smoke_record_never_aggregates(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["evidence_class"] = "mechanics_smoke"
        record["decision_eligible"] = False
        reasons = self.check(record, contract)
        assert any("decision_run" in r for r in reasons)

    def test_contract_drift_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["contract_sha256"] = "f" * 64
        assert any("contract drift" in r for r in self.check(record,
                                                             contract))

    def test_anchor_sha_mismatch_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["anchor_sha256"] = "9" * 64
        assert any("anchor" in r for r in self.check(record, contract))

    def test_boundary_tensor_mismatch_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["boundary_transfer_evidence"][
            "target_policy_tensor_hash_after_transfer"] = "0" * 64
        assert any("tensor hash" in r for r in self.check(record, contract))

    def test_sham_boundary_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        src = record["boundary_transfer_evidence"][
            "source_policy_tensor_hash"]
        record["boundary_transfer_evidence"][
            "target_policy_tensor_hash_before_transfer"] = src
        assert any("sham boundary" in r for r in self.check(record,
                                                            contract))

    def test_zero_phase1_updates_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["curriculum"]["post_easy"]["phase1_gradient_updates"] = 0
        assert any("gradient updates" in r for r in self.check(record,
                                                               contract))

    def test_asset_mismatch_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        evidence = {"asset": "btcusdt_4h", "data_file_hash": NESTED_SHA}
        assert any("asset mismatch" in r
                   for r in self.check(record, contract, evidence))

    def test_data_hash_mismatch_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        evidence = {"asset": "ethusdt_4h", "data_file_hash": "1" * 64}
        assert any("data mismatch" in r or "data_file_hash" in r
                   for r in self.check(record, contract, evidence))

    def test_missing_evidence_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        assert any("evidence.json missing" in r
                   for r in self.check(record, contract, evidence=None))

    def test_experiment_identity_mismatch_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["experiment_id"] = "0123456789abcdef"
        assert any("identity" in r for r in self.check(record, contract))


# ---------------------------------------------------------------------------
# discovery + end-to-end aggregation with injected probes
# ---------------------------------------------------------------------------

def build_tree(tmp_path: Path, contract: dict, *,
               mutate=None) -> Path:
    root = tmp_path / "out"
    for seed in SEEDS:
        for cell in contract["cells"]:
            rec = make_record(contract, seed, cell)
            if mutate:
                mutate(rec, seed, cell)
            rec_dir = root / EXP / f"seed{seed}" / cell
            (rec_dir / "return_traces").mkdir(parents=True)
            (rec_dir / "l1_cell_record.json").write_text(
                json.dumps(rec, default=str))
            (rec_dir / "return_traces" / "evidence.json").write_text(
                json.dumps(make_evidence()))
            (rec_dir / "results.json").write_text(json.dumps({
                "final_equity": 10_500.0, "mean_weekly_return": 0.001,
                "max_drawdown_pct": 3.2, "sharpe_ratio": 0.4}))
    return root


class TestAggregateEndToEnd:
    def test_healthy_tree_reaches_a_typed_outcome(self, tmp_path):
        contract = make_contract()
        root = build_tree(tmp_path, contract)

        def rollout(record):
            # E cells trade, N cells hold: EASY_CONTRIBUTES pattern.
            if record["phase1_mode"].startswith("easy"):
                return healthy_rollout(record)
            return inactive_rollout(record)

        result = agg.aggregate(root, EXP, contract=contract,
                               probe_fn=healthy_probe, rollout_fn=rollout)
        assert result["outcome"] == "EASY_CONTRIBUTES"
        assert result["refusals"] == []
        key = "seed101/L1_E_M03"
        assert result["raw_metrics_per_seed"][key]["trades_total"] == 7
        assert "units" in result["raw_metrics_per_seed"][key]

    def test_duplicate_physical_record_forces_inconclusive(self, tmp_path):
        contract = make_contract()
        root = build_tree(tmp_path, contract)
        stray = root / EXP / "seed101" / "L1_N_M10_copy"
        stray.mkdir()
        (stray / "l1_cell_record.json").write_text(
            json.dumps(make_record(contract, 101, "L1_N_M10"),
                       default=str))
        result = agg.aggregate(root, EXP, contract=contract,
                               probe_fn=healthy_probe,
                               rollout_fn=healthy_rollout)
        assert result["outcome"] == "INCONCLUSIVE"
        assert any("duplicate" in r or "misplaced" in r
                   for r in result["refusals"])

    def test_missing_record_forces_inconclusive(self, tmp_path):
        contract = make_contract()
        root = build_tree(tmp_path, contract)
        victim = (root / EXP / "seed202" / "L1_E_M03" /
                  "l1_cell_record.json")
        victim.unlink()
        result = agg.aggregate(root, EXP, contract=contract,
                               probe_fn=healthy_probe,
                               rollout_fn=healthy_rollout)
        assert result["outcome"] == "INCONCLUSIVE"
        assert any("missing record" in r for r in result["refusals"])

    def test_single_tensor_mutation_kills_promotion(self, tmp_path):
        contract = make_contract()

        def mutate(rec, seed, cell):
            if seed == 202 and cell == "L1_E_M10":
                rec["boundary_transfer_evidence"][
                    "target_policy_tensor_hash_after_transfer"] = "0" * 64

        root = build_tree(tmp_path, contract, mutate=mutate)
        result = agg.aggregate(root, EXP, contract=contract,
                               probe_fn=healthy_probe,
                               rollout_fn=healthy_rollout)
        assert result["outcome"] == "INCONCLUSIVE"
        assert result["outcome"] not in agg.PROMOTION_OUTCOMES

    def test_absent_results_metrics_are_reported_absent(self, tmp_path):
        contract = make_contract()
        root = build_tree(tmp_path, contract)
        (root / EXP / "seed101" / "L1_N_M10" / "results.json").unlink()
        result = agg.aggregate(root, EXP, contract=contract,
                               probe_fn=healthy_probe,
                               rollout_fn=healthy_rollout)
        raw = result["raw_metrics_per_seed"]["seed101/L1_N_M10"]
        assert "absent" in raw

    def test_write_aggregation_is_append_only(self, tmp_path):
        contract = make_contract()
        root = build_tree(tmp_path, contract)
        result = agg.aggregate(root, EXP, contract=contract,
                               probe_fn=healthy_probe,
                               rollout_fn=healthy_rollout)
        path = agg.write_aggregation(result, root)
        assert path.is_file()
        # Idempotent re-publication of identical content is allowed...
        assert agg.write_aggregation(result, root) == path
        # ...but divergent content is refused.
        divergent = dict(result)
        divergent["outcome_rationale"] = "tampered"
        with pytest.raises(RuntimeError, match="append-only"):
            agg.write_aggregation(divergent, root)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
