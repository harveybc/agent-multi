"""P1 difficulty x P1 LR factorial launch gate (order 2026-08-11 WP4).

Socket-free proofs that:

* any two of a seed's four materialized cell configs differ by
  EXACTLY the union of the config fields owned by the factors whose
  levels differ — one-and-only-one intended delta, nothing else
  (property test over the actually-materialized configs, all seeds);
* every held-fixed identity fact (phase-2 LR 3e-5, threshold 0.1 via
  the frozen v3 cost/protection block, l1_trained_epoch_v4 boundary,
  entropy 0.2, one pass-equivalent budget) binds identically in every
  cell, and cross-seed same-cell diffs are exactly the seed + anchor
  bindings;
* cell_order is the contract's cyclic Latin square and the loader
  refuses malformed contracts fail-closed (schema, anchors,
  assignments, non-permutation / non-Latin / non-cyclic orders,
  held-fixed drift, non-executable or oversized screen budgets);
* the runner refuses before any model construction on wrong host,
  unbound/mismatched CUDA (WP13), a refused gpu_readiness launch gate
  (fake probe heartbeat) and an unverifiable per-seed anchor;
* every record carries the finding-223 terminal custody and the
  finding-221 selected-checkpoint viability binding — a fake pipeline
  result missing either is REFUSED, never certified;
* complete records are reused (ALREADY_COMPLETE), invalid existing
  records refuse overwrite, second claimants get ALREADY_RUNNING;
* --screen-verdict evaluates the mechanics_screen gates: 16/16
  presence (refusing on fewer, listing missing), custody and typed
  viability gates, PHASE1_LR_REGION_COLLAPSED when every treatment
  combination collapses at every seed, else SCREEN_VIABLE_REGION with
  the exact viable cells.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import p1_difficulty_lr_factorial as p1  # noqa: E402

_SENTINEL = object()
FAKE_TENSOR_SHA = "f" * 64

CLEAN_SOURCES = {
    "agent-multi": {"repo_root": "/repo/agent-multi",
                    "commit": "1" * 40, "dirty": False,
                    "dirty_entries": [], "dirty_untracked_digest": None},
    "gym-fx": {"repo_root": "/repo/gym-fx", "commit": "2" * 40,
               "dirty": False, "dirty_entries": [],
               "dirty_untracked_digest": None},
}


def _contract() -> dict:
    return p1.load_contract()


@pytest.fixture(scope="module")
def bindings() -> dict:
    return p1.load_bindings()


def _write_contract(tmp_path: Path, contract: dict) -> Path:
    path = tmp_path / "p1lr_contract.json"
    clean = {key: value for key, value in contract.items()
             if not key.startswith("_")}
    path.write_text(json.dumps(clean, indent=1, sort_keys=True))
    return path


def _diff(a: dict, b: dict) -> set:
    keys = set(a) | set(b)
    return {key for key in keys
            if a.get(key, _SENTINEL) != b.get(key, _SENTINEL)}


def _materialized(bindings, tmp_path: Path, seed: int) -> dict:
    contract = _contract()
    out_dir = tmp_path / "out"
    configs = {}
    for cell in p1.CELLS:
        config = p1.materialize_cell_config(
            contract, bindings, seed, cell, out_dir)
        config.pop("_identity")
        configs[cell] = config
    return configs


# ---------------------------------------------------------------------------
# (a) one-and-only-one INTENDED delta between the four cells of a seed
# ---------------------------------------------------------------------------

class TestOneIntendedDeltaProperty:
    def test_pairwise_diff_is_exactly_the_factor_fields_every_seed(
            self, bindings, tmp_path):
        contract = _contract()
        for seed in p1.SEEDS:
            configs = _materialized(bindings, tmp_path, seed)
            cells = list(p1.CELLS)
            for i, cell_a in enumerate(cells):
                for cell_b in cells[i + 1:]:
                    observed = _diff(configs[cell_a], configs[cell_b])
                    intended = p1.intended_delta_fields(
                        contract, cell_a, cell_b)
                    assert observed == intended, (
                        f"seed {seed}: {cell_a} vs {cell_b} changed "
                        f"{sorted(observed)} but the factors own "
                        f"{sorted(intended)}")
                    assert intended, (
                        f"{cell_a} vs {cell_b}: two distinct cells "
                        "must differ in at least one factor")

    def test_lr_only_pair_moves_both_pipeline_lr_knobs(
            self, bindings, tmp_path):
        configs = _materialized(bindings, tmp_path, 101)
        assert _diff(configs["P1N_LR1E4"], configs["P1N_LR3E5"]) == {
            "phase1_learning_rate", "easy_learning_rate"}
        assert configs["P1N_LR1E4"]["phase1_learning_rate"] == \
            pytest.approx(1e-4)
        assert configs["P1N_LR1E4"]["easy_learning_rate"] == \
            pytest.approx(1e-4)
        assert configs["P1N_LR3E5"]["phase1_learning_rate"] == \
            pytest.approx(3e-5)
        assert configs["P1N_LR3E5"]["easy_learning_rate"] == \
            pytest.approx(3e-5)

    def test_dynamics_only_pair_moves_exactly_phase1_mode(
            self, bindings, tmp_path):
        configs = _materialized(bindings, tmp_path, 101)
        assert _diff(configs["P1N_LR1E4"], configs["P1E_LR1E4"]) == {
            "phase1_mode"}
        assert configs["P1N_LR1E4"]["phase1_mode"] == \
            "normal_realistic"
        assert configs["P1E_LR1E4"]["phase1_mode"] == \
            "easy_chronological_continuation"

    def test_double_delta_pair_is_the_union_of_both_factors(
            self, bindings, tmp_path):
        configs = _materialized(bindings, tmp_path, 101)
        assert _diff(configs["P1N_LR1E4"], configs["P1E_LR3E5"]) == {
            "phase1_mode", "phase1_learning_rate",
            "easy_learning_rate"}

    def test_held_fixed_identity_binds_identically_in_every_cell(
            self, bindings, tmp_path):
        contract = _contract()
        configs = _materialized(bindings, tmp_path, 101)
        anchor_sha = contract["anchors"]["101"]["sha256"]
        for cell, config in configs.items():
            # phase-2: the active D0 range point, normal dynamics.
            assert config["learning_rate"] == pytest.approx(3e-5), cell
            # v3 cost/protection block (includes the 0.1 deadband).
            assert config["continuous_action_threshold"] == \
                pytest.approx(0.1), cell
            assert config["full_spread_rate"] == pytest.approx(
                1e-4), cell
            assert config["require_protected_entries"] is True, cell
            assert config["min_equity"] == pytest.approx(100.0), cell
            assert config["financing_treatment"]["charged"] is \
                False, cell
            # corrected v4 boundary, entropy identity.
            assert config["phase1_handoff_semantics"] == \
                "l1_trained_epoch_v4", cell
            assert config["ent_coef"] == pytest.approx(0.2), cell
            # one pass-equivalent screen budget.
            assert config["epoch_timesteps"] == 20_000, cell
            assert config["max_epochs"] == 1, cell
            assert config["easy_max_epochs"] == 1, cell
            assert config["l1_patience"] == 10_000, cell
            assert config["easy_patience"] == 10_000, cell
            # anchored start, sealed test untouched, plugin identity.
            assert config["warm_start_model_sha256"] == anchor_sha, cell
            assert config["evaluate_test_split"] is False, cell
            assert config["agent_plugin"] == "sac_agent", cell
            assert config["eval_seed"] == 101, cell
            assert config["train_seed"] == 101, cell
            assert config["inactive_terminal_is_typed_result"] is \
                True, cell
            assert config["train_end"] == "2023-12-31T23:59:59", cell

    def test_cross_seed_same_cell_diff_is_exactly_seed_and_anchor(
            self, bindings, tmp_path):
        a = _materialized(bindings, tmp_path, 101)
        b = _materialized(bindings, tmp_path, 202)
        for cell in p1.CELLS:
            assert _diff(a[cell], b[cell]) == {
                "eval_seed", "train_seed", "ga_seed",
                "warm_start_model", "warm_start_model_sha256"}, cell


# ---------------------------------------------------------------------------
# (b) contract loading: Latin square + fail-closed refusals
# ---------------------------------------------------------------------------

class TestContractValidation:
    def test_real_contract_loads_and_order_is_the_cyclic_latin_square(
            self):
        contract = _contract()
        rows = {seed: list(contract["cell_order"][str(seed)])
                for seed in p1.SEEDS}
        base = rows[101]
        # every row is a permutation; every cell occupies every
        # within-seed position exactly once across seeds; rows are the
        # declared cyclic rotations.
        for seed, row in rows.items():
            assert sorted(row) == sorted(p1.CELLS), seed
        for position in range(4):
            assert len({rows[seed][position]
                        for seed in p1.SEEDS}) == 4, position
        for offset, seed in enumerate(p1.SEEDS):
            assert rows[seed] == [base[(i + offset) % 4]
                                  for i in range(4)], seed

    def _mutated(self, tmp_path, mutate):
        contract = copy.deepcopy(_contract())
        mutate(contract)
        return _write_contract(tmp_path, contract)

    def test_wrong_schema_is_refused(self, tmp_path):
        path = self._mutated(
            tmp_path, lambda c: c.update(schema="agent_multi.other.v1"))
        with pytest.raises(ValueError, match="schema"):
            p1.load_contract(path)

    def test_missing_anchor_sha_is_refused(self, tmp_path):
        def mutate(c):
            del c["anchors"]["202"]["sha256"]
        with pytest.raises(ValueError, match="sha256"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_non_hex_anchor_sha_is_refused(self, tmp_path):
        def mutate(c):
            c["anchors"]["303"]["sha256"] = "deadbeef"
        with pytest.raises(ValueError, match="sha256"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_missing_gpu_assignment_is_refused(self, tmp_path):
        def mutate(c):
            del c["assignments"]["404"]["gpu_uuid"]
        with pytest.raises(ValueError, match="GPU"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_duplicate_factor_combination_is_refused(self, tmp_path):
        def mutate(c):
            c["cells"]["P1N_LR3E5"]["phase1_learning_rate"] = 0.0001
        with pytest.raises(ValueError, match="factorial"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_non_permutation_cell_order_row_is_refused(self, tmp_path):
        def mutate(c):
            c["cell_order"]["101"][0] = "P1N_LR3E5"
        with pytest.raises(ValueError, match="permutation"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_non_latin_column_is_refused(self, tmp_path):
        def mutate(c):
            row = c["cell_order"]["202"]
            row[0], row[1] = row[1], row[0]
        with pytest.raises(ValueError, match="Latin"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_non_cyclic_row_assignment_is_refused(self, tmp_path):
        def mutate(c):
            c["cell_order"]["202"], c["cell_order"]["303"] = (
                c["cell_order"]["303"], c["cell_order"]["202"])
        with pytest.raises(ValueError, match="CYCLIC"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_moved_phase2_lr_is_refused(self, tmp_path):
        def mutate(c):
            c["held_fixed"]["phase2_learning_rate"] = 0.0001
        with pytest.raises(ValueError, match="phase2_learning_rate"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_moved_handoff_semantics_is_refused(self, tmp_path):
        def mutate(c):
            c["held_fixed"]["phase1_handoff_semantics"] = \
                "m0_epoch0_eligible_v3"
        with pytest.raises(ValueError, match="l1_trained_epoch_v4"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_missing_budget_knobs_are_refused(self, tmp_path):
        def mutate(c):
            del c["mechanics_screen"]["budget_knobs"]
        with pytest.raises(ValueError, match="budget_knobs"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_more_than_one_pass_equivalent_is_refused(self, tmp_path):
        def mutate(c):
            c["mechanics_screen"]["budget_knobs"]["phase1_epochs"] = 4
        with pytest.raises(ValueError, match="pass-equivalent"):
            p1.load_contract(self._mutated(tmp_path, mutate))

    def test_materialize_refuses_a_moved_threshold(self, bindings,
                                                   tmp_path):
        contract = copy.deepcopy(_contract())
        contract["held_fixed"]["phase2_action_threshold"] = 0.2
        with pytest.raises(RuntimeError,
                           match="phase2_action_threshold"):
            p1.materialize_cell_config(contract, bindings, 101,
                                       "P1N_LR1E4", tmp_path / "out")

    def test_materialize_refuses_a_non_ladder_pass_budget(
            self, bindings, tmp_path):
        contract = copy.deepcopy(_contract())
        contract["mechanics_screen"]["budget_knobs"][
            "epoch_timesteps"] = 10_000
        with pytest.raises(RuntimeError, match="pass budget"):
            p1.materialize_cell_config(contract, bindings, 101,
                                       "P1N_LR1E4", tmp_path / "out")

    def test_materialize_refuses_an_entropy_override(self, bindings,
                                                     tmp_path):
        contract = copy.deepcopy(_contract())
        contract["held_fixed"]["entropy"]["value"] = 0.3
        with pytest.raises(RuntimeError, match="ent_coef"):
            p1.materialize_cell_config(contract, bindings, 101,
                                       "P1N_LR1E4", tmp_path / "out")

    def test_viability_labels_mirror_the_pipeline_enum(self):
        from pipeline_plugins import (
            rl_pipeline_with_solvency_curriculum as curriculum)

        assert p1.HANDOFF_VIABILITY_VALUES == \
            curriculum.HANDOFF_VIABILITY_VALUES
        assert set(p1.COLLAPSE_LABELS) == {
            curriculum.VIABILITY_CONSTANT_POLICY,
            curriculum.VIABILITY_BELOW_NORMAL_THRESHOLD}


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------

class TestIdentities:
    def test_experiment_identity_binds_contract_anchors_and_code(
            self, bindings):
        contract = _contract()
        base = p1.experiment_identity(contract, bindings,
                                      sources=CLEAN_SOURCES)
        assert len(base) == 16
        int(base, 16)

        moved_code = copy.deepcopy(CLEAN_SOURCES)
        moved_code["agent-multi"]["commit"] = "f" * 40
        assert p1.experiment_identity(
            contract, bindings, sources=moved_code) != base

        other_contract = copy.deepcopy(contract)
        other_contract["_contract_sha256"] = "d" * 64
        assert p1.experiment_identity(
            other_contract, bindings, sources=CLEAN_SOURCES) != base

        other_anchor = copy.deepcopy(contract)
        other_anchor["anchors"]["202"]["sha256"] = "e" * 64
        assert p1.experiment_identity(
            other_anchor, bindings, sources=CLEAN_SOURCES) != base

    def test_the_sixteen_cell_identities_are_distinct(self, bindings):
        contract = _contract()
        exp = p1.experiment_identity(contract, bindings,
                                     sources=CLEAN_SOURCES)
        ids = {p1.cell_identity(exp, seed, cell, contract)
               for seed in p1.SEEDS for cell in p1.CELLS}
        assert len(ids) == 16


# ---------------------------------------------------------------------------
# fake pipeline / probe fixtures (socket-free runtime proofs)
# ---------------------------------------------------------------------------

def _fake_evidence(epoch: int, label: str) -> dict:
    return {
        "schema": "agent_multi.solvency_curriculum.handoff_viability.v1",
        "epoch": epoch,
        "handoff_viability": (label if epoch > 0 else "UNAVAILABLE"),
        "trained_treatment": epoch > 0,
        "viable_as_trained_treatment": bool(
            epoch > 0 and label == "VIABLE"),
        "any_action_crosses_phase2_threshold": label == "VIABLE",
        "probe_trades_total": 5 if label == "VIABLE" else 0,
    }


def _fake_result(config: dict, *, label: str = "VIABLE",
                 with_terminal: bool = True,
                 with_selected: bool = True,
                 with_checkpoint_evidence: bool = True) -> dict:
    out_dir = Path(config["save_model"]).parent
    terminal = out_dir / "model.terminal.zip"
    if with_terminal:
        terminal.parent.mkdir(parents=True, exist_ok=True)
        terminal.write_bytes(b"terminal-bytes-" + label.encode())
    history_rows = []
    for epoch in (0, 1):
        row = {"epoch": epoch,
               "checkpoint_source": ("warm_start_baseline" if epoch == 0
                                     else "easy_training_epoch")}
        if with_checkpoint_evidence:
            row["handoff_viability_evidence"] = _fake_evidence(
                epoch, label)
        history_rows.append(row)
    post_easy = {
        "schema": "agent_multi.solvency_curriculum.post_easy.v4",
        "selection_basis": "paired_comparator_best_trained_epoch",
        "best_easy_epoch": 1,
        "phase1_gradient_updates": 20_000,
        "artifact_sha256": "a" * 64,
        "phase1_terminal_policy_tensor_sha256": "b" * 64,
        "phase1_mode": config.get("phase1_mode"),
        "history": history_rows,
    }
    if with_selected:
        post_easy["selected_handoff_viability"] = {
            "schema": ("agent_multi.solvency_curriculum."
                       "selected_handoff_viability.v1"),
            "best_easy_epoch": 1,
            "handoff_viability": label,
            "trained_treatment": True,
            "anchor_passthrough": False,
            "selection_is_diagnostic_fallback": False,
            "selected_as_viable_handoff": label == "VIABLE",
        }
    return {
        "terminal_model_path": (str(terminal) if with_terminal
                                else None),
        "best_model_path": None,
        "history": [{"epoch": 1, "gradient_updates_total": 40_000}],
        "stop_reason": "max_epochs_budget",
        "termination_cause": None,
        "warm_start_transfer_evidence": {
            "optimizer_state_transferred": False,
            "replay_size_at_boundary": 0,
            "replay_transitions_transferred": 0,
        },
        "curriculum": {"post_easy": post_easy},
    }


class FakePipeline:
    """Socket-free stand-in for the solvency-curriculum pipeline."""

    calls: list = []

    def __init__(self, config: dict, *, label: str = "VIABLE",
                 **result_kwargs):
        self.config = config
        self.label = label
        self.result_kwargs = result_kwargs

    def run_pipeline(self, *, config, env_plugin, agent_plugin,
                     mode) -> dict:
        assert mode == "train"
        type(self).calls.append(dict(config))
        return _fake_result(config, label=self.label,
                            **self.result_kwargs)


def _fake_gate_heartbeat(observed: list,
                         classification: str = "GPU_READY") -> dict:
    return {
        "classification": classification,
        "hostname": "omega",
        "generated_utc": "2026-08-11T00:00:00+00:00",
        "running_kernel": {"release": "7.0.0-29-generic"},
        "driver": {"driver_version": "580.82.09",
                   "nvidia_smi_ok": True, "error": None},
        "gpus": {"observed_uuids": list(observed),
                 "expected_uuids": list(observed)},
        "framework": {"status": "TORCH_CUDA_OK"},
        "disk": {"classification": "HOST_DISK_NOT_EVALUATED",
                 "available_bytes": None, "output_fs": None,
                 "required_bytes": None},
    }


def _fake_dispatch_binding(uuid: str) -> dict:
    return {
        "schema": "agent_multi.gpu_dispatch_binding.v1",
        "hostname": "omega",
        "running_kernel": "7.0.0-29-generic",
        "driver_version": "580.82.09",
        "assigned_gpu_uuid": uuid,
        "observed_gpu_uuids": [uuid],
        "cuda_visible_devices": uuid,
        "framework_build": {"status": "TORCH_CUDA_OK"},
    }


@pytest.fixture()
def runtime(bindings, tmp_path, monkeypatch):
    """A runnable contract copy: tmp output root, tmp hash-bound
    anchors, host/GPU environment pinned to the seed-101 assignment."""
    contract = copy.deepcopy(_contract())
    contract["output_root"] = str(tmp_path / "out")
    for seed in p1.SEEDS:
        anchor = tmp_path / f"anchor_seed{seed}.zip"
        anchor.write_bytes(f"anchor-bytes-{seed}".encode())
        contract["anchors"][str(seed)] = {
            "path": str(anchor),
            "sha256": p1._sha_file(anchor),
        }
    assigned = contract["assignments"]["101"]["gpu_uuid"]
    monkeypatch.setattr(p1.socket, "gethostname", lambda: "omega")
    monkeypatch.setattr(p1, "visible_gpu_uuids", lambda: [assigned])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", assigned)
    FakePipeline.calls = []
    return SimpleNamespace(
        contract=contract, bindings=bindings, assigned=assigned,
        agent_loader=lambda name: SimpleNamespace(name=name),
        tensor_sha_fn=lambda path: FAKE_TENSOR_SHA,
        gate_heartbeat=_fake_gate_heartbeat([assigned]),
        dispatch_binding_fn=_fake_dispatch_binding,
    )


def _run_seed(runtime, **overrides):
    kwargs = dict(
        contract=runtime.contract, bindings=runtime.bindings,
        enforce_gpu=True, pipeline_factory=FakePipeline,
        agent_loader=runtime.agent_loader,
        tensor_sha_fn=runtime.tensor_sha_fn,
        gate_heartbeat=runtime.gate_heartbeat,
        dispatch_binding_fn=runtime.dispatch_binding_fn,
    )
    kwargs.update(overrides)
    return p1.run_seed(101, **kwargs)


# ---------------------------------------------------------------------------
# (c) the seed batch: order, anchoring, custody, viability binding
# ---------------------------------------------------------------------------

class TestSeedBatch:
    def test_batch_runs_the_contract_cell_order_from_the_anchor(
            self, runtime):
        summary = _run_seed(runtime)
        assert summary["outcome"] == "SEED_COMPLETE"
        expected_order = runtime.contract["cell_order"]["101"]
        assert summary["cell_order"] == expected_order
        # the pipeline saw the four cells IN ORDER, each warm-started
        # from the exact per-seed anchor, never a preceding terminal.
        anchor = runtime.contract["anchors"]["101"]
        assert [c["phase1_mode"] for c in FakePipeline.calls] == [
            runtime.contract["cells"][cell]["phase1_dynamics"]
            for cell in expected_order]
        for config in FakePipeline.calls:
            assert config["warm_start_model"] == anchor["path"]
            assert config["warm_start_model_sha256"] == anchor["sha256"]

    def test_records_carry_custody_viability_and_gpu_binding(
            self, runtime):
        summary = _run_seed(runtime)
        exp_id = summary["experiment_identity"]
        out_root = Path(runtime.contract["output_root"])
        for cell in p1.CELLS:
            record = json.loads(
                (out_root / exp_id / "seed101" / cell /
                 "cell_record.json").read_text())
            assert record["schema"] == p1.RECORD_SCHEMA
            assert record["evidence_class"] == "mechanics_screen"
            assert record["decision_eligible"] is False
            # finding 223: terminal custody, hash-bound + load-proven.
            assert record["terminal_model_path"]
            assert Path(record["terminal_model_path"]).is_file()
            assert record["terminal_model_sha256"] == p1._sha_file(
                Path(record["terminal_model_path"]))
            assert record["terminal_policy_tensor_sha256"] == \
                FAKE_TENSOR_SHA
            # finding 221: the selected-checkpoint viability binding.
            viability = record["handoff_viability"]
            assert viability["selected_label"] == "VIABLE"
            assert viability["selected"]["trained_treatment"] is True
            assert viability["selected_is_collapse"] is False
            assert [row["epoch"] for row in
                    viability["per_checkpoint"]] == [0, 1]
            # WP13/198 + probe: assigned/bound/observed + dispatch
            # binding + gate payload.
            assert record["gpu_binding"]["assigned_gpu_uuid"] == \
                runtime.assigned
            assert record["gpu_binding"]["cuda_visible_devices"] == \
                runtime.assigned
            assert record["gpu_dispatch_binding"][
                "assigned_gpu_uuid"] == runtime.assigned
            assert record["gpu_launch_gate"]["outcome"] == "GATE_PASS"
            assert record["boundary_transfer_evidence"][
                "optimizer_state_transferred"] is False
            assert record["factors"] == \
                runtime.contract["cells"][cell]

    def test_missing_terminal_artifact_is_refused_not_certified(
            self, runtime):
        def factory(config):
            return FakePipeline(config, with_terminal=False)
        summary = _run_seed(runtime, pipeline_factory=factory)
        assert summary["outcome"] == "SEED_FAILED"
        for cell, facts in summary["cells"].items():
            assert facts["outcome"] == "CELL_FAILED", cell
            assert "finding 223" in facts["error"], cell
        exp_id = summary["experiment_identity"]
        out_root = Path(runtime.contract["output_root"])
        assert not list((out_root / exp_id).rglob("cell_record.json"))

    def test_missing_selected_viability_is_refused(self, runtime):
        def factory(config):
            return FakePipeline(config, with_selected=False)
        summary = _run_seed(runtime, pipeline_factory=factory)
        assert summary["outcome"] == "SEED_FAILED"
        for facts in summary["cells"].values():
            assert facts["outcome"] == "CELL_FAILED"
            assert "selected_handoff_viability" in facts["error"]

    def test_missing_per_checkpoint_evidence_is_refused(self, runtime):
        def factory(config):
            return FakePipeline(config, with_checkpoint_evidence=False)
        summary = _run_seed(runtime, pipeline_factory=factory)
        assert summary["outcome"] == "SEED_FAILED"
        for facts in summary["cells"].values():
            assert "handoff_viability_evidence" in facts["error"]

    def test_anchor_sha_mismatch_refuses_the_batch(self, runtime):
        Path(runtime.contract["anchors"]["101"]["path"]).write_bytes(
            b"tampered")
        summary = _run_seed(runtime)
        assert summary["outcome"] == "REFUSED_ANCHOR_UNVERIFIED"
        assert "anchor hash mismatch" in summary["reason"]
        assert not FakePipeline.calls

    def test_missing_anchor_refuses_the_batch(self, runtime):
        Path(runtime.contract["anchors"]["101"]["path"]).unlink()
        summary = _run_seed(runtime)
        assert summary["outcome"] == "REFUSED_ANCHOR_UNVERIFIED"
        assert not FakePipeline.calls


# ---------------------------------------------------------------------------
# (d) GPU gate + WP13 binding refusals (fake probe, framework-free)
# ---------------------------------------------------------------------------

class TestGpuGates:
    def test_wrong_host_refused_and_heartbeat_written(self, runtime,
                                                      monkeypatch):
        monkeypatch.setattr(p1.socket, "gethostname",
                            lambda: "not-a-worker")
        summary = _run_seed(runtime)
        assert summary["outcome"] == "REFUSED_WRONG_HOST"
        assert not FakePipeline.calls
        beats = list(Path(runtime.contract["output_root"]).rglob(
            "runner_heartbeat.json"))
        assert len(beats) == 1
        assert json.loads(beats[0].read_text())[
            "terminal_state"] == "REFUSED_WRONG_HOST"

    def test_unset_cuda_env_refused(self, runtime, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        summary = _run_seed(runtime)
        assert summary["outcome"] == "REFUSED_GPU_UNBOUND"
        assert "CUDA_VISIBLE_DEVICES" in summary["reason"]
        assert not FakePipeline.calls

    def test_wrong_cuda_env_refused(self, runtime, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-wrong")
        summary = _run_seed(runtime)
        assert summary["outcome"] == "REFUSED_GPU_UNBOUND"
        assert not FakePipeline.calls

    def test_probe_gate_refusal_blocks_before_any_pipeline(
            self, runtime):
        heartbeat = _fake_gate_heartbeat(
            ["GPU-other"], classification="GPU_UUID_MISMATCH")
        summary = _run_seed(runtime, gate_heartbeat=heartbeat)
        assert summary["outcome"] == "REFUSED_GPU_UNBOUND"
        assert "launch gate refused" in summary["reason"]
        assert not FakePipeline.calls

    def test_ready_probe_but_assigned_uuid_absent_refuses(
            self, runtime):
        heartbeat = _fake_gate_heartbeat(["GPU-other"])
        payload, refusal = p1.gpu_launch_gate(runtime.assigned,
                                              heartbeat=heartbeat)
        assert refusal["outcome"] == "REFUSED_GPU_UNBOUND"
        assert payload["outcome"] == "REFUSED_GPU_UNBOUND"
        assert "GPU_UUID_MISMATCH" in payload["blocking"]

    def test_no_gpu_check_skips_gate_but_not_the_host_pin(
            self, runtime, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        summary = _run_seed(runtime, enforce_gpu=False)
        assert summary["outcome"] == "SEED_COMPLETE"
        monkeypatch.setattr(p1.socket, "gethostname",
                            lambda: "not-a-worker")
        FakePipeline.calls = []
        contract = copy.deepcopy(runtime.contract)
        contract["output_root"] = str(
            Path(runtime.contract["output_root"]).parent / "out2")
        summary = _run_seed(runtime, contract=contract,
                            enforce_gpu=False)
        assert summary["outcome"] == "REFUSED_WRONG_HOST"


# ---------------------------------------------------------------------------
# (e) reuse / no-overwrite / single-writer
# ---------------------------------------------------------------------------

class TestRecordCustody:
    def test_complete_records_are_reused_not_retrained(self, runtime):
        first = _run_seed(runtime)
        assert first["outcome"] == "SEED_COMPLETE"
        trained_calls = len(FakePipeline.calls)
        assert trained_calls == 4
        second = _run_seed(runtime)
        assert second["outcome"] == "ALREADY_COMPLETE"
        assert len(FakePipeline.calls) == trained_calls
        for facts in second["cells"].values():
            assert facts["outcome"] == "ALREADY_COMPLETE"

    def test_tampered_terminal_invalidates_reuse(self, runtime):
        first = _run_seed(runtime)
        exp_id = first["experiment_identity"]
        cell = runtime.contract["cell_order"]["101"][0]
        record_path = (Path(runtime.contract["output_root"]) / exp_id /
                       "seed101" / cell / "cell_record.json")
        record = json.loads(record_path.read_text())
        Path(record["terminal_model_path"]).write_bytes(b"tampered")
        assert not p1.record_is_complete(
            record_path, record["cell_identity"])
        summary = _run_seed(runtime)
        assert summary["outcome"] == "SEED_FAILED"
        assert "refusing to overwrite" in \
            summary["cells"][cell]["error"]

    def test_invalid_existing_record_refuses_overwrite(self, runtime):
        exp_id = p1.experiment_identity(runtime.contract,
                                        runtime.bindings)
        cell_dir = (Path(runtime.contract["output_root"]) / exp_id /
                    "seed101" / "P1N_LR1E4")
        cell_dir.mkdir(parents=True)
        (cell_dir / "cell_record.json").write_text("{corrupt")
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            p1.run_cell(101, "P1N_LR1E4", contract=runtime.contract,
                        bindings=runtime.bindings, exp_id=exp_id,
                        sources_before=p1.ladder.source_identities(),
                        gpu_dispatch_binding=None, gpu_gate=None,
                        pipeline_factory=FakePipeline,
                        agent_loader=runtime.agent_loader,
                        tensor_sha_fn=runtime.tensor_sha_fn)

    def test_second_claimant_is_already_running(self, runtime):
        exp_id = p1.experiment_identity(runtime.contract,
                                        runtime.bindings)
        lock = (Path(runtime.contract["output_root"]) / exp_id /
                "locks" / "exclusive_claim.seed101.P1E_LR1E4.lock")
        holder = p1.ExclusiveClaim(lock)
        assert holder.acquire()
        try:
            outcome = p1.run_cell(
                101, "P1E_LR1E4", contract=runtime.contract,
                bindings=runtime.bindings, exp_id=exp_id,
                sources_before=p1.ladder.source_identities(),
                gpu_dispatch_binding=None, gpu_gate=None,
                pipeline_factory=FakePipeline,
                agent_loader=runtime.agent_loader,
                tensor_sha_fn=runtime.tensor_sha_fn)
        finally:
            holder.release()
        assert outcome["outcome"] == "ALREADY_RUNNING"
        assert outcome["holder"].get("pid")


# ---------------------------------------------------------------------------
# (f) --screen-verdict gates and typed outcomes
# ---------------------------------------------------------------------------

def _verdict_record(contract, seed: int, cell: str, label: str,
                    tmp_path: Path, *, exp_id: str = "e" * 16,
                    trained: bool = True) -> dict:
    return {
        "schema": p1.RECORD_SCHEMA,
        "experiment_identity": exp_id,
        "cell_identity": f"{seed}{cell}"[:16].ljust(16, "0"),
        "seed": seed,
        "cell": cell,
        "factors": dict(contract["cells"][cell]),
        "contract_sha256": contract["_contract_sha256"],
        "terminal_model_path": str(
            tmp_path / f"absent-{seed}-{cell}.zip"),
        "terminal_model_sha256": "c" * 64,
        "handoff_viability": {
            "selected": {"handoff_viability": label,
                         "trained_treatment": trained},
            "selected_label": label,
            "selected_is_collapse": label in p1.COLLAPSE_LABELS,
        },
    }


def _all_records(contract, tmp_path, label_fn) -> dict:
    return {(seed, cell): _verdict_record(
                contract, seed, cell, label_fn(seed, cell), tmp_path)
            for seed in p1.SEEDS for cell in p1.CELLS}


class TestScreenVerdict:
    def test_all_viable_region(self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        payload, code = p1.screen_verdict(contract, records=records)
        assert payload["outcome"] == "SCREEN_VIABLE_REGION"
        assert code == 0
        assert len(payload["viable_cells"]) == 16
        assert payload["collapsed_cells"] == []
        assert "none" in payload["performance_claims"]

    def test_total_collapse_is_the_typed_region_collapse(
            self, tmp_path):
        contract = _contract()
        labels = {"P1N_LR1E4": "CONSTANT_POLICY",
                  "P1N_LR3E5": "BELOW_NORMAL_THRESHOLD",
                  "P1E_LR1E4": "CONSTANT_POLICY",
                  "P1E_LR3E5": "BELOW_NORMAL_THRESHOLD"}
        records = _all_records(contract, tmp_path,
                               lambda s, c: labels[c])
        payload, code = p1.screen_verdict(contract, records=records)
        assert payload["outcome"] == "PHASE1_LR_REGION_COLLAPSED"
        assert code == 0
        assert len(payload["collapsed_cells"]) == 16
        assert payload["viable_cells"] == []
        assert "stop" in payload["next_step"]

    def test_one_surviving_cell_names_the_viable_region(
            self, tmp_path):
        contract = _contract()

        def label(seed, cell):
            if seed == 303 and cell == "P1E_LR3E5":
                return "NO_TRADE"       # not CONSTANT, not BELOW
            return "CONSTANT_POLICY"
        records = _all_records(contract, tmp_path, label)
        payload, code = p1.screen_verdict(contract, records=records)
        assert payload["outcome"] == "SCREEN_VIABLE_REGION"
        assert payload["viable_cells"] == [
            {"seed": 303, "cell": "P1E_LR3E5",
             "handoff_viability": "NO_TRADE"}]
        assert len(payload["collapsed_cells"]) == 15
        assert payload["viability_matrix"]["P1E_LR3E5"]["303"] == \
            "NO_TRADE"

    def test_fifteen_records_refuse_listing_the_missing(
            self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        del records[(202, "P1E_LR1E4")]
        payload, code = p1.screen_verdict(contract, records=records)
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert code == 4
        assert payload["missing_records"] == [
            {"seed": 202, "cell": "P1E_LR1E4"}]

    def test_missing_terminal_custody_refuses(self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        records[(404, "P1N_LR1E4")]["terminal_model_sha256"] = ""
        payload, code = p1.screen_verdict(contract, records=records)
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert code == 4
        assert payload["custody_failures"][0]["seed"] == 404
        assert "finding 223" in \
            payload["custody_failures"][0]["error"]

    def test_unavailable_viability_refuses(self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        records[(101, "P1N_LR3E5")]["handoff_viability"]["selected"][
            "handoff_viability"] = "UNAVAILABLE"
        payload, code = p1.screen_verdict(contract, records=records)
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert any(f["seed"] == 101
                   for f in payload["viability_failures"])

    def test_untrained_selected_checkpoint_refuses(self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        records[(101, "P1N_LR1E4")]["handoff_viability"]["selected"][
            "trained_treatment"] = False
        payload, code = p1.screen_verdict(contract, records=records)
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert "trained treatment" in \
            payload["viability_failures"][0]["error"]

    def test_identity_fragmentation_refuses(self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        records[(303, "P1N_LR1E4")]["experiment_identity"] = "f" * 16
        payload, code = p1.screen_verdict(contract, records=records)
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert any("fragmentation" in reason
                   for reason in payload["reasons"])

    def test_disk_discovery_reads_the_sixteen_records(self, tmp_path):
        contract = _contract()
        exp_id = "a" * 16
        root = tmp_path / "root"
        for seed in p1.SEEDS:
            for cell in p1.CELLS:
                record = _verdict_record(contract, seed, cell,
                                         "VIABLE", tmp_path,
                                         exp_id=exp_id)
                cell_dir = root / exp_id / f"seed{seed}" / cell
                cell_dir.mkdir(parents=True)
                (cell_dir / "cell_record.json").write_text(
                    json.dumps(record))
        payload, code = p1.screen_verdict(contract, records_root=root)
        assert payload["outcome"] == "SCREEN_VIABLE_REGION"
        assert payload["experiment_identity"] == exp_id

    def test_ambiguous_experiment_dirs_require_an_explicit_id(
            self, tmp_path):
        root = tmp_path / "root"
        (root / ("a" * 16)).mkdir(parents=True)
        (root / ("b" * 16)).mkdir()
        payload, code = p1.screen_verdict(_contract(),
                                          records_root=root)
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert code == 4
        assert "--experiment-id" in payload["reasons"][0]


# ---------------------------------------------------------------------------
# dispatch surface
# ---------------------------------------------------------------------------

class TestDispatchSurface:
    def test_env_files_carry_the_contract_gpu_bindings(self):
        contract = _contract()
        env_dir = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                   "p1lr_env")
        for seed in p1.SEEDS:
            env_path = env_dir / f"seed{seed}.env"
            assert env_path.exists(), f"missing env file for {seed}"
            values = {}
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    values[key] = value
            assignment = contract["assignments"][str(seed)]
            assert values["P1LR_SEED"] == str(seed)
            assert values["P1LR_HOST"] == assignment["hostname"]
            assert values["CUDA_VISIBLE_DEVICES"] == \
                assignment["gpu_uuid"]

    def test_systemd_unit_matches_the_runner_exit_contract(self):
        unit = REPO / "examples/systemd/p1lr-screen@.service"
        assert unit.exists()
        text = unit.read_text()
        assert "p1_difficulty_lr_factorial.py --seed %i" in text
        assert "SuccessExitStatus=3" in text
        assert "RestartPreventExitStatus=4" in text
        assert "p1lr_env/seed%i.env" in text
