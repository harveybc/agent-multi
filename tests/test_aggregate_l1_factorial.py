"""Mutation tests for the generic L1 factorial aggregator (v2).

Order §4 / findings 178-187: malformed cells, duplicate physical
records, contract drift, tensor mismatch, asset mismatch, absent
metrics, terminal replacement, budget drift, system-manifest drift,
dirty executing source and non-finite raw metrics must all yield
INCONCLUSIVE/refusal and never a promotion outcome. Both Musashi
counterexamples are regression tests here. The decision core is pure;
the impure probes (terminal probe, verification rollout, disk facts)
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
NESTED_FILE_SHA = agg.sysid.sha_file(REPO / NESTED)
EXP = "feedfacecafe0001"
MANIFEST_SHA = "ab" * 32
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
CLEAN_SUBJECT = {
    "agent-multi": {"repo_root": "/repo/agent-multi",
                    "commit": "1" * 40, "dirty": False,
                    "dirty_entries": [],
                    "dirty_untracked_digest": None},
    "gym-fx": {"repo_root": "/repo/gym-fx",
               "commit": "2" * 40, "dirty": False,
               "dirty_entries": [],
               "dirty_untracked_digest": None},
}


def make_contract() -> dict:
    return {
        "schema": "agent_multi.l1_factorial_contract.v3",
        "_contract_sha256": "c" * 64,
        "asset": "ETHUSD",
        "env_asset": "ethusdt_4h",
        "nested_split_contract": NESTED,
        "cells": copy.deepcopy(CELLS),
        "decision_budget": {"phase1_epochs": 4,
                            "phase2_max_epochs": 1996},
        "anchors": {str(s): {"path": f"/anchors/seed{s}.zip",
                             "sha256": f"{s:04x}" * 16}
                    for s in SEEDS},
    }


def make_manifest() -> dict:
    return {"schema": agg.sysid.MANIFEST_SCHEMA,
            "_manifest_sha256": MANIFEST_SHA,
            "plugins": {
                "agent_plugin": "sac_agent",
                "pipeline_plugin": "rl_pipeline_with_validation",
                "curriculum_pipeline_plugin":
                    "rl_pipeline_with_solvency_curriculum",
            }}


def make_record(contract: dict, seed: int, cell: str) -> dict:
    spec = contract["cells"][cell]
    src_hash = "a1" * 32
    return {
        "schema": agg.RECORD_SCHEMA,
        "evidence_class": "decision_run",
        "decision_eligible": True,
        "performance_aggregate_eligible": True,
        "experiment_id": EXP,
        "cell_identity": f"{seed:04x}{cell.lower()[:8]:.8}00000000"[:16],
        "cell": cell,
        "seed": seed,
        "contract_sha256": contract["_contract_sha256"],
        "system_manifest_sha256": MANIFEST_SHA,
        "nested_split_contract_sha256": NESTED_FILE_SHA,
        "data_sha256": NESTED_SHA,
        "data_rows": 18085,
        "data_time_bounds": {"first": "2017-09-28T04:00:00",
                             "last": "2025-12-31T20:00:00"},
        "resolved_config_sha256": "e0" * 32,
        "observation_manifest_sha256": "e1" * 32,
        "observation_flattened_shape": [2688],
        "asset": "ETHUSD",
        "env_asset": "ethusdt_4h",
        "metric_schema": "paired_generalization_weekly_v1",
        "initial_cash": 10000.0,
        "cost_contract": {"commission": 0.0002, "slippage": 0.0},
        "phase1_mode": spec["phase1_mode"],
        "phase2_lr_multiplier": spec["phase2_lr_multiplier"],
        "anchor_sha256": contract["anchors"][str(seed)]["sha256"],
        "anchor_policy_tensor_sha256": "a9" * 32,
        "phase1_requested_epochs": 4,
        "phase2_requested_epochs": 1996,
        "phase1_realized_epochs": 4,
        "phase2_realized_epochs": 80,
        "stop_reason": "activity_stop_no_eligible_checkpoint",
        "termination_cause": None,
        "activity_stopped_without_eligible_checkpoint": False,
        "history_len": 80,
        "subject_code_identity": copy.deepcopy(CLEAN_SUBJECT),
        "attempt_dir": f"/src/{EXP}/seed{seed}/{cell}",
        "terminal_model_path":
            f"/src/{EXP}/seed{seed}/{cell}/model.terminal.zip",
        "terminal_model_sha256": "d2" * 32,
        "terminal_policy_tensor_sha256": "d3" * 32,
        "started_utc": "2026-08-09T00:00:00+00:00",
        "finished_utc": "2026-08-09T01:00:00+00:00",
        "curriculum": {
            "post_easy": {
                "artifact":
                    f"/src/{EXP}/seed{seed}/{cell}/model.post_easy.zip",
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


def matching_disk_facts(record: dict) -> dict:
    return {
        "terminal_model_sha256": record.get("terminal_model_sha256"),
        "terminal_policy_tensor_sha256": record.get(
            "terminal_policy_tensor_sha256"),
    }


def healthy_probe(record: dict) -> dict:
    return {
        "loads": True,
        "terminal_policy_tensor_sha256": record.get(
            "terminal_policy_tensor_sha256", "e5" * 32),
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
    return {mult: {seed: {"E": facts(e), "N": facts(n)}
                   for seed, (e, n) in seeds.items()}
            for mult, seeds in pattern.items()}


def full_matrix(e_low, n_low, e_high=True, n_high=True) -> dict:
    seeds4 = (101, 202, 303, 404)
    return matrix_from({
        0.3: {s: (e_low, n_low) for s in seeds4},
        1.0: {s: (e_high, n_high) for s in seeds4},
    })


def build_tree(tmp_path: Path, contract: dict, *, mutate=None) -> Path:
    root = tmp_path / "out"
    for seed in SEEDS:
        for cell in contract["cells"]:
            rec = make_record(contract, seed, cell)
            rec_dir = root / EXP / f"seed{seed}" / cell
            rec["attempt_dir"] = str(rec_dir)
            rec["terminal_model_path"] = str(
                rec_dir / "model.terminal.zip")
            rec["curriculum"]["post_easy"]["artifact"] = str(
                rec_dir / "model.post_easy.zip")
            if mutate:
                mutate(rec, seed, cell)
            (rec_dir / "return_traces").mkdir(parents=True)
            (rec_dir / "l1_cell_record.json").write_text(
                json.dumps(rec, default=str))
            (rec_dir / "return_traces" / "evidence.json").write_text(
                json.dumps(make_evidence()))
            (rec_dir / "results.json").write_text(json.dumps({
                "final_equity": 10_500.0, "mean_weekly_return": 0.001,
                "max_drawdown_pct": 3.2, "sharpe_ratio": 0.4}))
    return root


def run_aggregate(root, contract, *, rollout=None, disk_facts=None):
    return agg.aggregate(
        root, EXP, contract=contract, manifest=make_manifest(),
        probe_fn=healthy_probe,
        rollout_fn=rollout or healthy_rollout,
        disk_facts_fn=disk_facts or matching_disk_facts)


# ---------------------------------------------------------------------------
# §7.2 ordered rules on the pure core
# ---------------------------------------------------------------------------

class TestDecideOutcome:
    def test_easy_contributes(self):
        outcome, why = agg.decide_outcome(
            full_matrix(e_low=True, n_low=False, e_high=True,
                        n_high=False), refusals=[])
        assert outcome == "EASY_CONTRIBUTES"
        assert "+4" in why

    def test_lr_only(self):
        outcome, _ = agg.decide_outcome(
            full_matrix(e_low=True, n_low=True), refusals=[])
        assert outcome == "LR_ONLY"

    def test_easy_harmful_precedes_lr_only(self):
        outcome, _ = agg.decide_outcome(
            full_matrix(e_low=False, n_low=True), refusals=[])
        assert outcome == "EASY_HARMFUL"

    def test_interaction_precedes_easy_contributes(self):
        outcome, why = agg.decide_outcome(
            full_matrix(e_low=True, n_low=False, e_high=False,
                        n_high=True), refusals=[])
        assert outcome == "INTERACTION"
        assert "disagree in sign" in why

    def test_rule1_invalid_cell_forces_inconclusive(self):
        matrix = full_matrix(e_low=True, n_low=False)
        matrix[0.3][202]["N"] = facts(None, valid=False)
        outcome, why = agg.decide_outcome(matrix, refusals=[])
        assert outcome == "INCONCLUSIVE"
        assert "invalid cell" in why

    def test_refusals_precede_everything(self):
        outcome, why = agg.decide_outcome(
            full_matrix(e_low=True, n_low=False),
            refusals=["duplicate physical record"])
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
        f = agg.activity_facts(terminal_probe=probe,
                               rollout_summary=healthy_rollout({}))
        assert f["valid"] is False and f["active"] is None

    def test_missing_rollout_is_invalid(self):
        f = agg.activity_facts(terminal_probe=healthy_probe({}),
                               rollout_summary=None)
        assert f["valid"] is False and f["active"] is None


# ---------------------------------------------------------------------------
# record binding mutations
# ---------------------------------------------------------------------------

class TestValidateRecordBindings:
    def check(self, record, contract=None, evidence="default", seed=101,
              cell="L1_N_M10", manifest="default", disk="match"):
        contract = contract or make_contract()
        ev = make_evidence() if evidence == "default" else evidence
        mf = make_manifest() if manifest == "default" else manifest
        df = matching_disk_facts(record) if disk == "match" else disk
        return agg.validate_record_bindings(
            record, contract=contract, seed=seed, cell=cell,
            experiment_id=EXP, evidence=ev, manifest=mf, disk_facts=df)

    def test_faithful_record_has_no_reasons(self):
        contract = make_contract()
        assert self.check(make_record(contract, 101, "L1_N_M10"),
                          contract) == []

    def test_smoke_record_never_aggregates(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["evidence_class"] = "mechanics_smoke"
        record["decision_eligible"] = False
        assert any("decision_run" in r
                   for r in self.check(record, contract))

    def test_contract_drift_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["contract_sha256"] = "f" * 64
        assert any("contract drift" in r
                   for r in self.check(record, contract))

    def test_musashi_counterexample_mandatory_fields(self):
        # Regression for MUSASHI_L1_FACTORIAL_DELIVERY_REPRO: a record
        # stripped of any mandatory identity field must be refused.
        contract = make_contract()
        for field in agg.MANDATORY_IDENTITY_FIELDS:
            record = make_record(contract, 101, "L1_N_M10")
            record.pop(field, None)
            reasons = self.check(record, contract)
            assert any(field in r for r in reasons), (
                f"missing {field} was accepted")

    def test_system_manifest_drift_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["system_manifest_sha256"] = "9" * 64
        assert any("system-manifest drift" in r
                   for r in self.check(record, contract))

    def test_budget_drift_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["phase2_requested_epochs"] = 500
        assert any("budget drift" in r
                   for r in self.check(record, contract))

    def test_dirty_executing_source_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["subject_code_identity"]["agent-multi"]["dirty"] = True
        record["subject_code_identity"]["agent-multi"][
            "dirty_untracked_digest"] = "f0" * 32
        assert any("dirty executing source" in r
                   for r in self.check(record, contract))

    def test_terminal_replacement_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        disk = {"terminal_model_sha256": "0" * 64,
                "terminal_policy_tensor_sha256":
                    record["terminal_policy_tensor_sha256"]}
        assert any("terminal replacement" in r
                   for r in self.check(record, contract, disk=disk))

    def test_terminal_tensor_swap_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        disk = {"terminal_model_sha256": record["terminal_model_sha256"],
                "terminal_policy_tensor_sha256": "0" * 64}
        assert any("tensor digest does not rehash" in r
                   for r in self.check(record, contract, disk=disk))

    def test_absent_disk_facts_are_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        assert any("disk facts absent" in r
                   for r in self.check(record, contract, disk=None))

    def test_boundary_tensor_mismatch_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        record["boundary_transfer_evidence"][
            "target_policy_tensor_hash_after_transfer"] = "0" * 64
        assert any("tensor hash" in r for r in self.check(record, contract))

    def test_asset_mismatch_is_refused(self):
        contract = make_contract()
        record = make_record(contract, 101, "L1_N_M10")
        evidence = {"asset": "btcusdt_4h", "data_file_hash": NESTED_SHA}
        assert any("asset mismatch" in r
                   for r in self.check(record, contract, evidence))


# ---------------------------------------------------------------------------
# cross-record uniformity (Musashi tampered-code counterexample)
# ---------------------------------------------------------------------------

class TestCrossRecordUniformity:
    def test_uniform_records_pass(self):
        contract = make_contract()
        records = {f"seed{s}/{c}": make_record(contract, s, c)
                   for s in SEEDS for c in contract["cells"]}
        assert agg.cross_record_uniformity(records) == []

    def test_tampered_code_revisions_break_uniformity(self):
        contract = make_contract()
        records = {f"seed{s}/{c}": make_record(contract, s, c)
                   for s in SEEDS for c in contract["cells"]}
        records["seed101/L1_N_M10"]["code_revisions"] = {
            "agent-multi": "adversarial-revision"}
        reasons = agg.cross_record_uniformity(records)
        assert any("code_revisions" in r for r in reasons)

    def test_tampered_subject_identity_breaks_uniformity(self):
        contract = make_contract()
        records = {f"seed{s}/{c}": make_record(contract, s, c)
                   for s in SEEDS for c in contract["cells"]}
        records["seed202/L1_E_M03"]["subject_code_identity"][
            "agent-multi"]["commit"] = "3" * 40
        reasons = agg.cross_record_uniformity(records)
        assert any("subject_code_identity" in r for r in reasons)


# ---------------------------------------------------------------------------
# end-to-end aggregation with injected probes
# ---------------------------------------------------------------------------

class TestAggregateEndToEnd:
    def test_healthy_tree_reaches_a_typed_outcome(self, tmp_path):
        contract = make_contract()
        root = build_tree(tmp_path, contract)

        def rollout(record):
            if record["phase1_mode"].startswith("easy"):
                return healthy_rollout(record)
            return inactive_rollout(record)

        result = run_aggregate(root, contract, rollout=rollout)
        assert result["outcome"] == "EASY_CONTRIBUTES"
        assert result["refusals"] == []
        key = "seed101/L1_E_M03"
        assert result["raw_metrics_per_seed"][key]["trades_total"] == 7
        assert result["raw_metrics_per_seed"][key][
            "total_return"] == pytest.approx(0.05)
        assert "units" in result["raw_metrics_per_seed"][key]
        assert result["subject_execution_revisions"]
        assert "aggregator_revisions" in result

    def test_musashi_counterexample_missing_results_json(self, tmp_path):
        # Regression: removing one results.json must force INCONCLUSIVE.
        contract = make_contract()
        root = build_tree(tmp_path, contract)
        (root / EXP / "seed101" / "L1_N_M10" / "results.json").unlink()
        result = run_aggregate(root, contract)
        assert result["outcome"] == "INCONCLUSIVE"
        assert any("results.json missing" in r
                   for r in result["refusals"])

    def test_musashi_counterexample_tampered_code_revision(self, tmp_path):
        # Regression: one record with tampered code revisions must
        # poison the aggregation into INCONCLUSIVE.
        contract = make_contract()

        def mutate(rec, seed, cell):
            if seed == 101 and cell == "L1_N_M10":
                rec["code_revisions"] = {
                    "agent-multi": "adversarial-revision",
                    "gym-fx": "adversarial-revision"}

        root = build_tree(tmp_path, contract, mutate=mutate)
        result = run_aggregate(root, contract)
        assert result["outcome"] == "INCONCLUSIVE"
        assert any("code_revisions" in r for r in result["refusals"])

    def test_non_finite_raw_metric_forces_inconclusive(self, tmp_path):
        contract = make_contract()
        root = build_tree(tmp_path, contract)
        target = root / EXP / "seed202" / "L1_E_M10" / "results.json"
        target.write_text(json.dumps({
            "final_equity": 10_500.0, "mean_weekly_return": float("nan"),
            "max_drawdown_pct": 3.2, "sharpe_ratio": 0.4}),)
        result = run_aggregate(root, contract)
        assert result["outcome"] == "INCONCLUSIVE"
        assert any("non-finite" in r for r in result["refusals"])

    def test_total_return_uses_bound_initial_cash(self, tmp_path):
        contract = make_contract()

        def mutate(rec, seed, cell):
            rec["initial_cash"] = 21_000.0

        root = build_tree(tmp_path, contract, mutate=mutate)
        result = run_aggregate(root, contract)
        key = "seed101/L1_N_M10"
        assert result["raw_metrics_per_seed"][key][
            "total_return"] == pytest.approx(10_500.0 / 21_000.0 - 1.0)

    def test_duplicate_physical_record_forces_inconclusive(self, tmp_path):
        contract = make_contract()
        root = build_tree(tmp_path, contract)
        stray = root / EXP / "seed101" / "L1_N_M10_copy"
        stray.mkdir()
        rec = make_record(contract, 101, "L1_N_M10")
        (stray / "l1_cell_record.json").write_text(
            json.dumps(rec, default=str))
        result = run_aggregate(root, contract)
        assert result["outcome"] == "INCONCLUSIVE"
        assert any("duplicate" in r or "misplaced" in r
                   for r in result["refusals"])

    def test_terminal_replacement_kills_promotion(self, tmp_path):
        contract = make_contract()
        root = build_tree(tmp_path, contract)

        def swapped_disk(record):
            facts = matching_disk_facts(record)
            if record["seed"] == 202 and record["cell"] == "L1_E_M10":
                facts["terminal_model_sha256"] = "0" * 64
            return facts

        result = run_aggregate(root, contract, disk_facts=swapped_disk)
        assert result["outcome"] == "INCONCLUSIVE"
        assert result["outcome"] not in agg.PROMOTION_OUTCOMES

    def test_cli_exit_semantics(self, tmp_path, monkeypatch, capsys):
        contract = make_contract()
        root = build_tree(tmp_path, contract)
        (root / EXP / "seed101" / "L1_N_M10" / "results.json").unlink()
        monkeypatch.setattr(agg.runner, "load_contract",
                            lambda *a, **k: contract)
        monkeypatch.setattr(agg.runner, "load_system_manifest",
                            lambda *a, **k: make_manifest())
        monkeypatch.setattr(agg, "terminal_disk_facts",
                            matching_disk_facts)
        monkeypatch.setattr(agg, "probe_terminal", healthy_probe)
        monkeypatch.setattr(agg, "verification_rollout", healthy_rollout)
        monkeypatch.setattr(
            sys, "argv",
            ["aggregate", "--experiment-id", EXP,
             "--output-root", str(root)])
        code = agg.main()
        assert code != 0  # INCONCLUSIVE/refusals exit nonzero

    def test_write_aggregation_is_append_only(self, tmp_path):
        contract = make_contract()
        root = build_tree(tmp_path, contract)
        result = run_aggregate(root, contract)
        path = agg.write_aggregation(result, root)
        assert agg.write_aggregation(result, root) == path
        divergent = dict(result)
        divergent["outcome_rationale"] = "tampered"
        with pytest.raises(RuntimeError, match="append-only"):
            agg.write_aggregation(divergent, root)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
