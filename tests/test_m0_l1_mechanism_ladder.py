"""Mechanism-ladder launch gate (finding 220, order WP3 §3.5).

Socket-free proofs that:

* each adjacent arm pair D0->D2->D3->D4 differs by ONE-AND-ONLY-ONE
  declared semantic delta group and NOTHING else (property test over
  the actually-materialized configs);
* the contract loader refuses malformed diagnostics before any model
  construction (wrong schema, unpinned anchor, wrong seed, stray delta
  fields, missing worker assignment, broken baseline chain);
* the M0 executable split windows are exactly the union of the L1
  nested NON-TEST roles on the pinned CSV (sealed 2025 excluded);
* record/claim/heartbeat basics hold (complete-record reuse, single
  writer per arm, WP13 CUDA binding fail-closed, heartbeat v2 facts);
* the D1 evaluator's two activity definitions label fixture facts
  correctly (active-under-both, active-M0-only, invalid-never-
  inactive).
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import m0_l1_mechanism_ladder as ladder  # noqa: E402
from tools.m0_l1_d1_evaluator import (  # noqa: E402
    l1_activity_label,
    m0_activity_label,
)

_SENTINEL = object()

CLEAN_SOURCES = {
    "agent-multi": {"repo_root": "/repo/agent-multi",
                    "commit": "1" * 40, "dirty": False,
                    "dirty_entries": [], "dirty_untracked_digest": None},
    "gym-fx": {"repo_root": "/repo/gym-fx", "commit": "2" * 40,
               "dirty": False, "dirty_entries": [],
               "dirty_untracked_digest": None},
}


def _contract() -> dict:
    return ladder.load_contract()


def _write_contract(tmp_path: Path, contract: dict) -> Path:
    path = tmp_path / "ladder_contract.json"
    clean = {key: value for key, value in contract.items()
             if not key.startswith("_")}
    path.write_text(json.dumps(clean, indent=1, sort_keys=True))
    return path


def _materialized(tmp_path: Path) -> dict:
    contract = _contract()
    out_dir = tmp_path / "out"
    configs = {}
    for arm in ladder.ARM_ORDER:
        config = ladder.materialize_arm_config(contract, arm, out_dir)
        config.pop("_identity")
        configs[arm] = config
    return configs


def _diff(a: dict, b: dict) -> set:
    keys = set(a) | set(b)
    return {key for key in keys
            if a.get(key, _SENTINEL) != b.get(key, _SENTINEL)}


# ---------------------------------------------------------------------------
# (a) one-and-only-one semantic delta between adjacent materializations
# ---------------------------------------------------------------------------

class TestAdjacentDeltaProperty:
    def test_d0_to_d2_changes_exactly_the_boundary_semantics(
            self, tmp_path):
        configs = _materialized(tmp_path)
        d0, d2 = configs["D0_M0_EXACT"], configs["D2_BOUNDARY_ONLY"]
        assert _diff(d0, d2) == {"phase1_handoff_semantics"}
        assert d0["phase1_handoff_semantics"] == "m0_epoch0_eligible_v3"
        assert d2["phase1_handoff_semantics"] == "l1_trained_epoch_v4"

    def test_d2_to_d3_changes_exactly_the_cost_protection_contract(
            self, tmp_path):
        configs = _materialized(tmp_path)
        d2, d3 = configs["D2_BOUNDARY_ONLY"], configs[
            "D3_COST_PROTECTION"]
        # The v3 bindings overwrite mostly value-identical base fields;
        # the ACTUAL config changes are exactly the four bindings the
        # M0 recipe left implicit.
        assert _diff(d2, d3) == {
            "full_spread_rate", "require_protected_entries",
            "min_equity", "financing_treatment"}
        for key in ("full_spread_rate", "require_protected_entries",
                    "min_equity", "financing_treatment"):
            assert key not in d2, (
                f"{key} must be ABSENT from the M0-exact config (the "
                "M0 screen ran the gym-fx defaults)")
        assert d3["full_spread_rate"] == pytest.approx(1e-4)
        assert d3["require_protected_entries"] is True
        assert d3["min_equity"] == pytest.approx(100.0)
        assert d3["financing_treatment"]["charged"] is False

    def test_d2_to_d3_value_identical_bindings_really_are(
            self, tmp_path):
        """Every other v3 cost binding overwrites an IDENTICAL value —
        proof that D3 introduces no hidden second mechanism."""
        configs = _materialized(tmp_path)
        d2 = configs["D2_BOUNDARY_ONLY"]
        contract = _contract()
        delta = contract["arms"]["D3_COST_PROTECTION"]["delta"]
        for key, value in delta.items():
            if key in ("full_spread_rate", "require_protected_entries",
                       "min_equity", "financing_treatment"):
                continue
            assert d2.get(key) == value, (
                f"v3 binding {key} would silently change the value "
                f"{d2.get(key)!r} -> {value!r}; that is a second "
                "mechanism inside the cost delta")

    def test_d3_to_d4_changes_exactly_the_stopper_group(self, tmp_path):
        configs = _materialized(tmp_path)
        d3, d4 = configs["D3_COST_PROTECTION"], configs["D4_FULL_L1"]
        assert _diff(d3, d4) == {
            "l1_patience", "l1_patience_start_epoch",
            "l1_activity_patience", "l1_activity_patience_start_epoch"}
        assert d3["l1_patience"] == 10_000    # M0: no early stopping
        assert d4["l1_patience"] == 60
        # Evidence plumbing is UNIFORM (never an arm delta): every arm
        # lands a typed record when fully inactive.
        assert d3["inactive_terminal_is_typed_result"] is True
        assert d4["inactive_terminal_is_typed_result"] is True

    def test_every_delta_stays_inside_its_declared_group(
            self, tmp_path):
        configs = _materialized(tmp_path)
        contract = _contract()
        groups = contract["delta_groups"]
        pairs = list(zip(ladder.ARM_ORDER, ladder.ARM_ORDER[1:]))
        assert len(pairs) == 3
        for base, treated in pairs:
            group = contract["arms"][treated]["delta_group"]
            allowed = set(groups[group])
            observed = _diff(configs[base], configs[treated])
            assert observed <= allowed, (
                f"{base}->{treated} changed {sorted(observed - allowed)}"
                f" outside the declared {group!r} group")
            assert observed, f"{base}->{treated} changed nothing"

    def test_common_facts_are_bound_identically_in_every_arm(
            self, tmp_path):
        configs = _materialized(tmp_path)
        contract = _contract()
        common = contract["common"]
        for arm, config in configs.items():
            assert config["warm_start_model_sha256"] == \
                common["anchor"]["sha256"], arm
            assert config["eval_seed"] == 101, arm
            assert config["train_seed"] == 101, arm
            assert config["epoch_timesteps"] == 20_000, arm
            assert config["max_epochs"] == 1, arm
            assert config["easy_max_epochs"] == 1, arm
            assert config["learning_rate"] == pytest.approx(3e-5), arm
            assert config["easy_learning_rate"] == pytest.approx(
                1e-4), arm
            assert config["selection_metric"] == \
                "lexicographic_weekly_v1", arm
            assert config["evaluate_test_split"] is False, arm
            assert config["agent_plugin"] == "sac_agent", arm
            assert config["train_end"] == "2023-12-31T23:59:59", arm

    def test_nontest_union_equivalence_holds_on_the_pinned_csv(self):
        facts = ladder.verify_nested_nontest_union(_contract())
        assert facts["m0_train_rows"] == 13_699
        assert facts["m0_train_rows"] == \
            facts["fit_train_plus_inner_validation"]
        assert facts["m0_validation_rows"] == 2_196
        assert facts["m0_validation_rows"] == facts["outer_validation"]
        assert facts["sealed_test_rows_excluded"] == 2_190


# ---------------------------------------------------------------------------
# (b) contract loader refusals
# ---------------------------------------------------------------------------

class TestContractRefusals:
    def _mutated(self, tmp_path, mutate):
        contract = copy.deepcopy(_contract())
        mutate(contract)
        return _write_contract(tmp_path, contract)

    def test_wrong_schema_is_refused(self, tmp_path):
        path = self._mutated(
            tmp_path, lambda c: c.update(schema="agent_multi.other.v1"))
        with pytest.raises(ValueError, match="schema"):
            ladder.load_contract(path)

    def test_missing_anchor_sha_is_refused(self, tmp_path):
        def mutate(c):
            del c["common"]["anchor"]["sha256"]
        path = self._mutated(tmp_path, mutate)
        with pytest.raises(ValueError, match="anchor"):
            ladder.load_contract(path)

    def test_non_hex_anchor_sha_is_refused(self, tmp_path):
        def mutate(c):
            c["common"]["anchor"]["sha256"] = "deadbeef"
        path = self._mutated(tmp_path, mutate)
        with pytest.raises(ValueError, match="sha256"):
            ladder.load_contract(path)

    def test_wrong_seed_is_refused(self, tmp_path):
        path = self._mutated(tmp_path, lambda c: c.update(seed=202))
        with pytest.raises(ValueError, match="seed 101"):
            ladder.load_contract(path)

    def test_d0_with_a_delta_is_refused(self, tmp_path):
        def mutate(c):
            c["arms"]["D0_M0_EXACT"]["delta"] = {"l1_patience": 60}
        path = self._mutated(tmp_path, mutate)
        with pytest.raises(ValueError, match="EMPTY delta"):
            ladder.load_contract(path)

    def test_stray_delta_field_is_refused(self, tmp_path):
        def mutate(c):
            c["arms"]["D2_BOUNDARY_ONLY"]["delta"][
                "learning_rate"] = 1e-3
        path = self._mutated(tmp_path, mutate)
        with pytest.raises(ValueError, match="outside its declared"):
            ladder.load_contract(path)

    def test_missing_assignment_is_refused(self, tmp_path):
        def mutate(c):
            del c["assignments"]["D4_FULL_L1"]["gpu_uuid"]
        path = self._mutated(tmp_path, mutate)
        with pytest.raises(ValueError, match="GPU"):
            ladder.load_contract(path)

    def test_broken_baseline_chain_is_refused(self, tmp_path):
        def mutate(c):
            c["arms"]["D3_COST_PROTECTION"]["baseline"] = "D0_M0_EXACT"
        path = self._mutated(tmp_path, mutate)
        with pytest.raises(ValueError, match="baseline"):
            ladder.load_contract(path)

    def test_unknown_handoff_semantics_is_refused_by_the_pipeline(self):
        from pipeline_plugins.rl_pipeline_with_solvency_curriculum \
            import PipelinePlugin

        pipeline = PipelinePlugin({})
        with pytest.raises(ValueError, match="phase1_handoff_semantics"):
            pipeline._train_easy_phase(
                config={"phase1_handoff_semantics": "bogus"},
                env_plugin=None, agent_plugin=None)

    def test_default_handoff_semantics_is_the_corrected_v4(self):
        from pipeline_plugins.rl_pipeline_with_solvency_curriculum \
            import PipelinePlugin

        assert PipelinePlugin.plugin_params[
            "phase1_handoff_semantics"] == "l1_trained_epoch_v4"

    def test_d3_delta_must_equal_the_frozen_v3_costs(self, tmp_path):
        contract = copy.deepcopy(_contract())
        contract["arms"]["D3_COST_PROTECTION"]["delta"][
            "full_spread_rate"] = 5e-4
        with pytest.raises(RuntimeError, match="v3 manifest cost"):
            ladder.materialize_arm_config(
                contract, "D3_COST_PROTECTION", tmp_path / "out")


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------

class TestIdentities:
    def test_diagnostic_identity_binds_contract_anchor_and_code(self):
        contract = _contract()
        base = ladder.diagnostic_identity(contract,
                                          sources=CLEAN_SOURCES)
        assert len(base) == 16
        int(base, 16)

        moved_code = copy.deepcopy(CLEAN_SOURCES)
        moved_code["agent-multi"]["commit"] = "f" * 40
        assert ladder.diagnostic_identity(
            contract, sources=moved_code) != base

        other_contract = copy.deepcopy(contract)
        other_contract["_contract_sha256"] = "d" * 64
        assert ladder.diagnostic_identity(
            other_contract, sources=CLEAN_SOURCES) != base

        other_anchor = copy.deepcopy(contract)
        other_anchor["common"]["anchor"]["sha256"] = "e" * 64
        assert ladder.diagnostic_identity(
            other_anchor, sources=CLEAN_SOURCES) != base

    def test_arm_identities_differ_and_bind_the_cumulative_delta(self):
        contract = _contract()
        diag = ladder.diagnostic_identity(contract,
                                          sources=CLEAN_SOURCES)
        ids = {arm: ladder.arm_identity(diag, arm, contract)
               for arm in ladder.ARM_ORDER}
        assert len(set(ids.values())) == 4
        mutated = copy.deepcopy(contract)
        mutated["arms"]["D4_FULL_L1"]["delta"]["l1_patience"] = 61
        assert ladder.arm_identity(diag, "D4_FULL_L1", mutated) != \
            ids["D4_FULL_L1"]
        assert ladder.arm_identity(diag, "D2_BOUNDARY_ONLY", mutated) \
            == ids["D2_BOUNDARY_ONLY"]

    def test_cumulative_delta_composes_in_ladder_order(self):
        contract = _contract()
        assert ladder.cumulative_delta(contract, "D0_M0_EXACT") == {}
        d4 = ladder.cumulative_delta(contract, "D4_FULL_L1")
        assert d4["phase1_handoff_semantics"] == "l1_trained_epoch_v4"
        assert d4["require_protected_entries"] is True
        assert d4["l1_patience"] == 60


# ---------------------------------------------------------------------------
# (c) record / claim / heartbeat basics
# ---------------------------------------------------------------------------

class TestRuntimeBasics:
    def _record(self, tmp_path, arm_id) -> Path:
        artifact = tmp_path / "terminal.zip"
        artifact.write_bytes(b"terminal-bytes")
        record = {
            "schema": ladder.RECORD_SCHEMA,
            "arm_identity": arm_id,
            "terminal_model_path": str(artifact),
            "terminal_model_sha256": hashlib.sha256(
                b"terminal-bytes").hexdigest(),
        }
        path = tmp_path / "ladder_arm_record.json"
        path.write_text(json.dumps(record))
        return path

    def test_complete_record_validates(self, tmp_path):
        path = self._record(tmp_path, "a" * 16)
        assert ladder.record_is_complete(path, "a" * 16)

    def test_wrong_arm_identity_fails(self, tmp_path):
        path = self._record(tmp_path, "a" * 16)
        assert not ladder.record_is_complete(path, "b" * 16)

    def test_artifact_hash_mismatch_fails(self, tmp_path):
        path = self._record(tmp_path, "a" * 16)
        (tmp_path / "terminal.zip").write_bytes(b"tampered")
        assert not ladder.record_is_complete(path, "a" * 16)

    def test_missing_artifact_fails(self, tmp_path):
        path = self._record(tmp_path, "a" * 16)
        (tmp_path / "terminal.zip").unlink()
        assert not ladder.record_is_complete(path, "a" * 16)

    def test_exclusive_claim_admits_one_writer(self, tmp_path):
        from tools.l1_fleet_launcher import ExclusiveClaim

        lock = tmp_path / "locks" / "exclusive_claim.D0.lock"
        first = ExclusiveClaim(lock)
        second = ExclusiveClaim(lock)
        assert first.acquire()
        try:
            assert not second.acquire()
            holder = second.holder()
            assert holder.get("pid")
        finally:
            first.release()
        assert second.acquire()
        second.release()

    def test_gpu_binding_wrong_host_refused(self):
        refusal = ladder.check_gpu_binding(
            _contract(), "D0_M0_EXACT", hostname="not-a-worker",
            cuda_env=None, observed_uuids=[])
        assert refusal["outcome"] == "REFUSED_WRONG_HOST"

    def test_gpu_binding_unset_cuda_env_refused(self):
        contract = _contract()
        assigned = contract["assignments"]["D0_M0_EXACT"]["gpu_uuid"]
        refusal = ladder.check_gpu_binding(
            contract, "D0_M0_EXACT", hostname="omega",
            cuda_env=None, observed_uuids=[assigned])
        assert refusal["outcome"] == "REFUSED_GPU_UNBOUND"
        assert "CUDA_VISIBLE_DEVICES" in refusal["reason"]

    def test_gpu_binding_wrong_uuid_refused(self):
        contract = _contract()
        assigned = contract["assignments"]["D0_M0_EXACT"]["gpu_uuid"]
        refusal = ladder.check_gpu_binding(
            contract, "D0_M0_EXACT", hostname="omega",
            cuda_env="GPU-wrong", observed_uuids=[assigned])
        assert refusal["outcome"] == "REFUSED_GPU_UNBOUND"

    def test_gpu_binding_exact_match_passes(self):
        contract = _contract()
        assigned = contract["assignments"]["D0_M0_EXACT"]["gpu_uuid"]
        assert ladder.check_gpu_binding(
            contract, "D0_M0_EXACT", hostname="omega",
            cuda_env=assigned, observed_uuids=[assigned]) is None

    def test_heartbeat_carries_v2_binding_facts(self, tmp_path):
        contract = _contract()
        hb = ladder.ArmHeartbeat(
            tmp_path / "heartbeat.json", contract=contract,
            arm="D2_BOUNDARY_ONLY", diag_id="d" * 16, arm_id="a" * 16)
        hb.beat(terminal_state="RUNNING", progress="training")
        payload = json.loads((tmp_path / "heartbeat.json").read_text())
        assert payload["schema"] == ladder.HEARTBEAT_SCHEMA
        assert payload["arm"] == "D2_BOUNDARY_ONLY"
        assert payload["assigned_gpu_uuid"] == \
            contract["assignments"]["D2_BOUNDARY_ONLY"]["gpu_uuid"]
        assert "cuda_visible_devices" in payload
        assert "observed_gpu_uuids" in payload
        assert "gpu_temperature_c" in payload
        assert "gpu_utilization_pct" in payload
        assert payload["pid"]
        assert payload["pid_start_identity"]
        assert payload["terminal_state"] == "RUNNING"

    def test_run_arm_refuses_wrong_host_and_writes_heartbeat(
            self, tmp_path, monkeypatch):
        contract = copy.deepcopy(_contract())
        contract["output_root"] = str(tmp_path / "out")
        monkeypatch.setattr(ladder.socket, "gethostname",
                            lambda: "not-a-worker")
        outcome = ladder.run_arm("D0_M0_EXACT", contract=contract,
                                 enforce_gpu=True)
        assert outcome["outcome"] == "REFUSED_WRONG_HOST"
        beats = list((tmp_path / "out").rglob("heartbeat.json"))
        assert len(beats) == 1
        payload = json.loads(beats[0].read_text())
        assert payload["terminal_state"] == "REFUSED_WRONG_HOST"

    def test_run_arm_second_claimant_is_already_running(
            self, tmp_path, monkeypatch):
        contract = copy.deepcopy(_contract())
        contract["output_root"] = str(tmp_path / "out")
        monkeypatch.setattr(ladder.socket, "gethostname",
                            lambda: "omega")
        diag = ladder.diagnostic_identity(contract)
        lock = (Path(contract["output_root"]) / diag / "locks" /
                "exclusive_claim.D0_M0_EXACT.lock")
        holder = ladder.ExclusiveClaim(lock)
        assert holder.acquire()
        try:
            outcome = ladder.run_arm("D0_M0_EXACT", contract=contract,
                                     enforce_gpu=False)
        finally:
            holder.release()
        assert outcome["outcome"] == "ALREADY_RUNNING"
        assert outcome["holder"].get("pid")

    def test_run_arm_reuses_a_complete_record(self, tmp_path,
                                              monkeypatch):
        contract = copy.deepcopy(_contract())
        contract["output_root"] = str(tmp_path / "out")
        monkeypatch.setattr(ladder.socket, "gethostname",
                            lambda: "omega")
        diag = ladder.diagnostic_identity(contract)
        arm_id = ladder.arm_identity(diag, "D0_M0_EXACT", contract)
        arm_dir = Path(contract["output_root"]) / diag / "D0_M0_EXACT"
        arm_dir.mkdir(parents=True)
        artifact = arm_dir / "terminal.zip"
        artifact.write_bytes(b"terminal-bytes")
        record = {
            "schema": ladder.RECORD_SCHEMA,
            "arm_identity": arm_id,
            "terminal_model_path": str(artifact),
            "terminal_model_sha256": hashlib.sha256(
                b"terminal-bytes").hexdigest(),
        }
        (arm_dir / "ladder_arm_record.json").write_text(
            json.dumps(record))
        outcome = ladder.run_arm("D0_M0_EXACT", contract=contract,
                                 enforce_gpu=False)
        assert outcome["_reuse"] == "ALREADY_COMPLETE"

    def test_run_arm_refuses_an_invalid_existing_record(
            self, tmp_path, monkeypatch):
        contract = copy.deepcopy(_contract())
        contract["output_root"] = str(tmp_path / "out")
        monkeypatch.setattr(ladder.socket, "gethostname",
                            lambda: "omega")
        diag = ladder.diagnostic_identity(contract)
        arm_dir = Path(contract["output_root"]) / diag / "D0_M0_EXACT"
        arm_dir.mkdir(parents=True)
        (arm_dir / "ladder_arm_record.json").write_text("{corrupt")
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            ladder.run_arm("D0_M0_EXACT", contract=contract,
                           enforce_gpu=False)


# ---------------------------------------------------------------------------
# (d) D1 classifier fixtures
# ---------------------------------------------------------------------------

def _active_probe() -> dict:
    return {"loads": True, "tensor_chain_consistent": True,
            "phase2_updates_occurred": True,
            "terminal_policy_tensor_sha256": "t" * 64}


def _active_rollout() -> dict:
    return {
        "trades_total": 122,
        "action_raw_std": 0.056,
        "action_non_hold_rate": 1.0,
        "execution_diagnostics": {"protected_market_entries": 5,
                                  "protected_limit_entries": 0,
                                  "protected_stop_entries": 0},
    }


class TestD1Classifier:
    def test_active_under_both_definitions(self):
        m0 = m0_activity_label(_active_rollout(), terminal_loads=True,
                               weights_changed_from_anchor=True,
                               updates_positive=True)
        l1 = l1_activity_label(_active_probe(), _active_rollout())
        assert m0["label"] == "active"
        assert l1["label"] == "active"

    def test_active_under_m0_only_when_no_protected_entry(self):
        rollout = _active_rollout()
        rollout["execution_diagnostics"] = {
            "protected_market_entries": 0,
            "protected_limit_entries": 0,
            "protected_stop_entries": 0}
        m0 = m0_activity_label(rollout, terminal_loads=True,
                               weights_changed_from_anchor=True,
                               updates_positive=True)
        l1 = l1_activity_label(_active_probe(), rollout)
        assert m0["label"] == "active"
        assert l1["label"] == "inactive"
        assert "no protected entry" in " ".join(
            l1.get("inactive_signals") or [])

    def test_zero_trades_is_inactive_under_both(self):
        rollout = _active_rollout()
        rollout["trades_total"] = 0
        rollout["action_raw_std"] = 0.0
        rollout["action_non_hold_rate"] = 0.0
        m0 = m0_activity_label(rollout, terminal_loads=True,
                               weights_changed_from_anchor=True,
                               updates_positive=True)
        l1 = l1_activity_label(_active_probe(), rollout)
        assert m0["label"] == "inactive"
        assert l1["label"] == "inactive"

    def test_l1_invalid_is_never_inactive(self):
        rollout = _active_rollout()
        del rollout["action_raw_std"]
        l1 = l1_activity_label(_active_probe(), rollout)
        assert l1["label"] == "invalid"
        assert l1.get("active") is None

    def test_m0_unavailable_fact_is_invalid_not_inactive(self):
        rollout = _active_rollout()
        rollout["trades_total"] = None
        m0 = m0_activity_label(rollout, terminal_loads=True,
                               weights_changed_from_anchor=True,
                               updates_positive=True)
        assert m0["label"] == "invalid"

    def test_unchanged_weights_are_inactive_under_m0(self):
        m0 = m0_activity_label(_active_rollout(), terminal_loads=True,
                               weights_changed_from_anchor=False,
                               updates_positive=True)
        assert m0["label"] == "inactive"


# ---------------------------------------------------------------------------
# dispatch surface
# ---------------------------------------------------------------------------

class TestDispatchSurface:
    def test_dispatch_script_refuses_without_the_launch_word(self):
        script = REPO / "tools/dispatch_mechanism_ladder.sh"
        assert script.exists()
        proc = subprocess.run(["bash", str(script)],
                              capture_output=True, text=True)
        assert proc.returncode == 2
        assert "launch" in proc.stdout

    def test_env_files_carry_the_contract_gpu_bindings(self):
        contract = _contract()
        env_dir = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                   "ladder_env")
        for arm, assignment in contract["assignments"].items():
            env_path = env_dir / f"{arm}.env"
            assert env_path.exists(), f"missing env file for {arm}"
            values = {}
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    values[key] = value
            assert values["LADDER_ARM"] == arm
            assert values["LADDER_HOST"] == assignment["hostname"]
            assert values["CUDA_VISIBLE_DEVICES"] == \
                assignment["gpu_uuid"]
