"""P1 difficulty x P1 LR factorial launch gate (order 2026-08-11 WP4;
corrections 224/225/226, order 2026-08-11 §3-§5).

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
* finding 224: the EXECUTABLE config binds the typed nested split
  contract (path + verified sha), nested_split_mode=l1 and the paired
  selection metric with NO legacy split window surviving; the
  mandatory refusal matrix fires BEFORE training — wrong role path,
  wrong role count, wrong role sha, outer used as inner, missing
  context flag, context counted in score, paired-metric drift, any
  sealed-test materialization;
* cell_order is the contract's cyclic Latin square and the loader
  refuses malformed contracts fail-closed;
* the runner refuses before any model construction on wrong host,
  unbound/mismatched CUDA (WP13), a refused gpu_readiness launch gate
  and an unverifiable per-seed anchor;
* every record carries the finding-223 terminal custody, the
  finding-221 viability binding AND the finding-224 nested facts
  (contract sha, manifest sha, per-role CSV shas, scored/context
  counts, exact score dates);
* finding 225: --screen-verdict REFUSES without the typed replica
  proof; zero, 15, 17, duplicate, swapped, foreign, hash-altered and
  loads=false proofs all refuse; replica_terminal_loads is a BOOLEAN;
  per-checkpoint handoff facts and nested split identity are
  revalidated at aggregation time;
* finding 226: --mode decision runs under a distinct identity and
  output root, starts from the ORIGINAL anchors, requires the passing
  screen gate, restores the best checkpoint, performs ONE final
  outer-validation evaluation, and --decision-verdict emits the
  document-38 outcomes with per-seed paired effects and per-cell
  weekly metrics with units.
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
            # Finding 224: the nested contract IS the split authority.
            nested_spec = contract["nested_split_contract"]
            assert config["nested_split_contract"] == str(
                REPO / nested_spec["path"]), cell
            assert config["nested_split_mode"] == "l1", cell
            assert config["selection_metric"] == \
                "paired_generalization_weekly_v1", cell
            for legacy in ("train_start", "train_end",
                           "validation_start", "validation_end",
                           "test_start", "test_end", "train_years",
                           "val_years", "test_years"):
                assert config.get(legacy) is None, (cell, legacy)

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
                 with_checkpoint_evidence: bool = True,
                 with_best: bool = False) -> dict:
    out_dir = Path(config["save_model"]).parent
    terminal = out_dir / "model.terminal.zip"
    if with_terminal:
        terminal.parent.mkdir(parents=True, exist_ok=True)
        terminal.write_bytes(b"terminal-bytes-" + label.encode())
    best = out_dir / "model.zip"
    if with_best:
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"best-bytes-" + label.encode())
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
        "best_model_path": str(best) if with_best else None,
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


def _fake_nested_roles_fn(contract: dict):
    """Socket-free stand-in for materialize_nested_roles: echoes the
    contract's pinned role facts (the real function verifies a fresh
    materialization against these same pins)."""
    spec = contract["nested_split_contract"]
    pins = spec["role_facts"]

    def fn(contract_arg, bindings_arg, out_dir):
        split_dir = Path(out_dir) / "nested_splits"
        roles = {}
        for role, pin in pins.items():
            entry = dict(pin)
            if pin["status"] == "MATERIALIZED":
                entry["csv"] = str(split_dir / f"{role}.csv")
            roles[role] = entry
        return {
            "binding": {
                "path": str(REPO / spec["path"]),
                "contract_relative_path": spec["path"],
                "sha256": spec["sha256"],
                "mode": spec["mode"],
                "context_bars": spec["context_bars"],
                "role_facts_pinned": pins,
            },
            "split_dir": str(split_dir),
            "manifest_path": str(split_dir /
                                 "nested_split_manifest.json"),
            "manifest_sha256": "d" * 64,
            "roles": roles,
        }
    return fn


def _fake_outer_eval(*, config, agent, best_model_path, nested_roles,
                     seed, mean_weekly_rap: float = 0.01,
                     trades: int = 7) -> dict:
    role = nested_roles["roles"]["outer_validation"]
    return {
        "role": "outer_validation",
        "purpose": "final truth ONLY — one evaluation after selection",
        "csv_sha256": role["csv_sha256"],
        "scored_rows": role["scored_rows"],
        "context_rows_forced_hold": role["context_rows"],
        "context_excluded_from_metrics": True,
        "score_start": role["score_start"],
        "score_end": role["score_end"],
        "best_model_path": str(best_model_path),
        "best_model_sha256": p1._sha_file(Path(best_model_path)),
        "scored_steps": role["scored_rows"],
        "metrics": {
            "metric_schema": "trading.weekly.v1",
            "total_return": 0.05,
            "mean_weekly_return": 0.001,
            "annualized_return": 0.05,
            "annual_return": 0.052,
            "mean_weekly_rap": mean_weekly_rap,
            "annual_rap": 52 * mean_weekly_rap,
            "max_drawdown_fraction": 0.1,
            "evaluation_weeks": 52,
        },
        "weekly_return_vector": [0.001] * 52,
        "trades_total": trades,
        "activity": {"traded": trades > 0, "trades_total": trades},
        "units": dict(p1.DECISION_METRIC_UNITS),
    }


@pytest.fixture()
def runtime(bindings, tmp_path, monkeypatch):
    """A runnable contract copy: tmp output roots (both modes), tmp
    hash-bound anchors, host/GPU environment pinned to the seed-101
    assignment, and a fake nested-roles verifier echoing the pins."""
    contract = copy.deepcopy(_contract())
    contract["output_root"] = str(tmp_path / "out")
    contract["decision_run"]["output_root"] = str(
        tmp_path / "out_decision")
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
        nested_roles_fn=_fake_nested_roles_fn(contract),
    )


def _run_seed(runtime, **overrides):
    kwargs = dict(
        contract=runtime.contract, bindings=runtime.bindings,
        enforce_gpu=True, pipeline_factory=FakePipeline,
        agent_loader=runtime.agent_loader,
        tensor_sha_fn=runtime.tensor_sha_fn,
        gate_heartbeat=runtime.gate_heartbeat,
        dispatch_binding_fn=runtime.dispatch_binding_fn,
        nested_roles_fn=runtime.nested_roles_fn,
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
            # finding 224: nested facts bound in EVERY record.
            spec = runtime.contract["nested_split_contract"]
            assert record["nested_split_contract_sha256"] == \
                spec["sha256"]
            assert record["nested_split_manifest_sha256"]
            assert record["selection_metric"] == \
                "paired_generalization_weekly_v1"
            for role, pin in spec["role_facts"].items():
                got = record["nested_role_facts"][role]
                for key in p1.NESTED_ROLE_FACT_KEYS:
                    assert got[key] == pin[key], (role, key)
            assert record["nested_role_facts"]["sealed_test"][
                "status"] == "SEALED"

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
                        tensor_sha_fn=runtime.tensor_sha_fn,
                        nested_roles_fn=runtime.nested_roles_fn)

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
                tensor_sha_fn=runtime.tensor_sha_fn,
                nested_roles_fn=runtime.nested_roles_fn)
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
    spec = contract["nested_split_contract"]
    return {
        "schema": p1.RECORD_SCHEMA,
        "experiment_identity": exp_id,
        "cell_identity": f"{seed}{cell}"[:16].ljust(16, "0"),
        "seed": seed,
        "cell": cell,
        "factors": dict(contract["cells"][cell]),
        "contract_sha256": contract["_contract_sha256"],
        "terminal_model_path": str(
            tmp_path / "root" / exp_id / f"seed{seed}" / cell /
            "attempt-01" / "model.terminal.zip"),
        "terminal_model_sha256": "c" * 64,
        # Finding 224: nested facts every record must bind.
        "selection_metric": "paired_generalization_weekly_v1",
        "nested_split_contract_path": spec["path"],
        "nested_split_contract_sha256": spec["sha256"],
        "nested_split_mode": "l1",
        "nested_split_manifest_sha256": "d" * 64,
        "nested_role_facts": {
            role: {key: spec["role_facts"][role].get(key)
                   for key in p1.NESTED_ROLE_FACT_KEYS}
            for role in spec["role_facts"]},
        "handoff_viability": {
            "selected": {"handoff_viability": label,
                         "trained_treatment": trained},
            "selected_label": label,
            "selected_is_collapse": label in p1.COLLAPSE_LABELS,
            # Finding 225: per-checkpoint facts revalidated at
            # aggregation time.
            "per_checkpoint": [
                {"epoch": 0, "handoff_viability": "UNAVAILABLE",
                 "trained_treatment": False},
                {"epoch": 1, "handoff_viability": label,
                 "trained_treatment": trained},
            ],
        },
    }


def _all_records(contract, tmp_path, label_fn) -> dict:
    return {(seed, cell): _verdict_record(
                contract, seed, cell, label_fn(seed, cell), tmp_path)
            for seed in p1.SEEDS for cell in p1.CELLS}


def _proof_for(contract, records) -> dict:
    """A valid typed 16-entry replica proof for these records."""
    identities = {rec["experiment_identity"]
                  for rec in records.values()}
    exp_id = sorted(identities)[0]
    return {
        "schema": p1.REPLICA_PROOF_SCHEMA,
        "experiment_identity": exp_id,
        "contract_sha256": contract["_contract_sha256"],
        "replica_host": "dragon",
        "collection_tree_digest": "f" * 64,
        "proofs": [
            {
                "experiment_identity": rec["experiment_identity"],
                "contract_sha256": contract["_contract_sha256"],
                "seed": seed,
                "cell": cell,
                "cell_identity": rec["cell_identity"],
                "terminal_relative_path":
                    p1.expected_terminal_relative(rec),
                "terminal_model_sha256": rec["terminal_model_sha256"],
                "loads": True,
            }
            for (seed, cell), rec in sorted(records.items())
        ],
    }


class TestScreenVerdict:
    def test_all_viable_region(self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "SCREEN_VIABLE_REGION"
        assert code == 0
        assert len(payload["viable_cells"]) == 16
        assert payload["collapsed_cells"] == []
        assert "none" in payload["performance_claims"]
        assert payload["gates"]["replica_terminal_loads"] is True

    def test_total_collapse_is_the_typed_region_collapse(
            self, tmp_path):
        contract = _contract()
        labels = {"P1N_LR1E4": "CONSTANT_POLICY",
                  "P1N_LR3E5": "BELOW_NORMAL_THRESHOLD",
                  "P1E_LR1E4": "CONSTANT_POLICY",
                  "P1E_LR3E5": "BELOW_NORMAL_THRESHOLD"}
        records = _all_records(contract, tmp_path,
                               lambda s, c: labels[c])
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
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
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
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
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert code == 4
        assert payload["missing_records"] == [
            {"seed": 202, "cell": "P1E_LR1E4"}]

    def test_missing_terminal_custody_refuses(self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        records[(404, "P1N_LR1E4")]["terminal_model_sha256"] = ""
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
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
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert any(f["seed"] == 101
                   for f in payload["viability_failures"])

    def test_untrained_selected_checkpoint_refuses(self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        records[(101, "P1N_LR1E4")]["handoff_viability"]["selected"][
            "trained_treatment"] = False
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert "trained treatment" in \
            payload["viability_failures"][0]["error"]

    def test_identity_fragmentation_refuses(self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        records[(303, "P1N_LR1E4")]["experiment_identity"] = "f" * 16
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert any("fragmentation" in reason
                   for reason in payload["reasons"])

    def test_missing_per_checkpoint_facts_refuse_at_aggregation(
            self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        records[(202, "P1N_LR3E5")]["handoff_viability"].pop(
            "per_checkpoint")
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert payload["gates"][
            "per_checkpoint_facts_revalidated"] is False
        assert any("per-checkpoint" in f["error"]
                   for f in payload["checkpoint_failures"])

    def test_nested_identity_drift_refuses_at_aggregation(
            self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        records[(303, "P1E_LR1E4")]["nested_split_contract_sha256"] = \
            "0" * 64
        records[(404, "P1E_LR3E5")]["nested_role_facts"][
            "inner_validation"]["scored_rows"] = 2196
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert payload["gates"][
            "nested_split_identity_revalidated"] is False
        errors = " ".join(f["error"]
                          for f in payload["nested_identity_failures"])
        assert "nested_split_contract_sha256" in errors
        assert "inner_validation" in errors

    def test_legacy_selection_metric_in_a_record_refuses(
            self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        records[(101, "P1E_LR1E4")]["selection_metric"] = \
            "lexicographic_weekly_v1"
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert any("paired" in f["error"]
                   for f in payload["nested_identity_failures"])

    def test_disk_discovery_reads_the_sixteen_records(self, tmp_path):
        contract = _contract()
        exp_id = "a" * 16
        root = tmp_path / "root"
        records = {}
        for seed in p1.SEEDS:
            for cell in p1.CELLS:
                record = _verdict_record(contract, seed, cell,
                                         "VIABLE", tmp_path,
                                         exp_id=exp_id)
                records[(seed, cell)] = record
                cell_dir = root / exp_id / f"seed{seed}" / cell
                cell_dir.mkdir(parents=True)
                (cell_dir / "cell_record.json").write_text(
                    json.dumps(record))
        payload, code = p1.screen_verdict(
            contract, records_root=root,
            replica_proof=_proof_for(contract, records))
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
# (g) finding 225: the replica proof is a REAL gate
# ---------------------------------------------------------------------------

class TestReplicaProofGate:
    def _viable(self, tmp_path):
        contract = _contract()
        records = _all_records(contract, tmp_path,
                               lambda s, c: "VIABLE")
        return contract, records

    def test_no_proof_refuses_and_gate_is_boolean_false(
            self, tmp_path):
        contract, records = self._viable(tmp_path)
        payload, code = p1.screen_verdict(contract, records=records)
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert code == 4
        assert payload["gates"]["replica_terminal_loads"] is False
        assert isinstance(payload["gates"]["replica_terminal_loads"],
                          bool)
        assert any("replica proof required" in r
                   for r in payload["reasons"])

    def test_valid_sixteen_proof_passes(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        ok, refusals, facts = p1.validate_replica_proof(
            proof, contract=contract, records=records)
        assert (ok, refusals) == (True, [])
        assert facts["entries_bound"] == 16

    def _refused(self, contract, records, proof):
        payload, code = p1.screen_verdict(contract, records=records,
                                          replica_proof=proof)
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert code == 4
        assert payload["gates"]["replica_terminal_loads"] is False
        return payload

    def test_zero_proof_entries_refuse(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        proof["proofs"] = []
        payload = self._refused(contract, records, proof)
        assert sum("NO entry" in r for r in
                   payload["replica_proof_refusals"]) == 16

    def test_fifteen_proof_entries_refuse(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        proof["proofs"] = proof["proofs"][:15]
        payload = self._refused(contract, records, proof)
        assert any("NO entry" in r
                   for r in payload["replica_proof_refusals"])

    def test_seventeen_proof_entries_refuse(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        proof["proofs"].append(dict(proof["proofs"][0]))
        payload = self._refused(contract, records, proof)
        assert any("duplicate" in r
                   for r in payload["replica_proof_refusals"])

    def test_duplicate_entry_refuses(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        proof["proofs"][1] = dict(proof["proofs"][0])
        payload = self._refused(contract, records, proof)
        refusals = " ".join(payload["replica_proof_refusals"])
        assert "duplicate" in refusals and "NO entry" in refusals

    def test_swapped_entries_refuse(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        a, b = proof["proofs"][0], proof["proofs"][1]
        a["terminal_relative_path"], b["terminal_relative_path"] = (
            b["terminal_relative_path"], a["terminal_relative_path"])
        payload = self._refused(contract, records, proof)
        assert any("relative path" in r
                   for r in payload["replica_proof_refusals"])

    def test_foreign_entry_refuses(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        proof["proofs"][0]["seed"] = 999
        payload = self._refused(contract, records, proof)
        refusals = " ".join(payload["replica_proof_refusals"])
        assert "foreign" in refusals

    def test_hash_altered_entry_refuses(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        proof["proofs"][5]["terminal_model_sha256"] = "0" * 64
        payload = self._refused(contract, records, proof)
        assert any("hash-altered or swapped" in r
                   for r in payload["replica_proof_refusals"])

    def test_loads_false_refuses(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        proof["proofs"][3]["loads"] = False
        payload = self._refused(contract, records, proof)
        assert any("did not load" in r
                   for r in payload["replica_proof_refusals"])

    def test_loads_as_text_refuses(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        proof["proofs"][3]["loads"] = "verified externally"
        payload = self._refused(contract, records, proof)
        assert any("did not load" in r
                   for r in payload["replica_proof_refusals"])

    def test_wrong_schema_or_identity_refuses(self, tmp_path):
        contract, records = self._viable(tmp_path)
        proof = _proof_for(contract, records)
        proof["schema"] = "agent_multi.other.v1"
        proof["experiment_identity"] = "f" * 16
        payload = self._refused(contract, records, proof)
        refusals = " ".join(payload["replica_proof_refusals"])
        assert "schema" in refusals and "identity" in refusals


# ---------------------------------------------------------------------------
# (h) finding 224: the mandatory pre-training refusal matrix
# ---------------------------------------------------------------------------

class TestNestedBindingRefusals:
    def _pins(self):
        return copy.deepcopy(
            _contract()["nested_split_contract"]["role_facts"])

    def _manifest_roles(self):
        contract = _contract()
        roles = {}
        for role, pin in contract["nested_split_contract"][
                "role_facts"].items():
            entry = dict(pin)
            entry.pop("score_start", None)
            entry["score_start"] = pin["score_start"]
            entry["score_end"] = pin["score_end"]
            if pin["status"] == "MATERIALIZED":
                entry["csv"] = f"/tmp/{role}.csv"
            else:
                entry.pop("csv_sha256")
                entry.pop("scored_rows")
                entry.pop("context_rows")
            roles[role] = {k: v for k, v in entry.items()
                           if v is not None or k == "status"}
        return roles

    def test_wrong_role_path_refuses(self, bindings, tmp_path):
        contract = copy.deepcopy(_contract())
        contract["nested_split_contract"]["path"] = \
            "examples/config/does_not_exist.json"
        with pytest.raises(RuntimeError, match="wrong role path"):
            p1.verify_nested_split_binding(contract, bindings)

    def test_wrong_role_sha_refuses(self, bindings):
        contract = copy.deepcopy(_contract())
        contract["nested_split_contract"]["sha256"] = "0" * 64
        with pytest.raises(RuntimeError, match="wrong role sha"):
            p1.verify_nested_split_binding(contract, bindings)

    def test_wrong_role_count_refuses(self):
        roles = self._manifest_roles()
        roles["train_monitor"]["scored_rows"] = 2189
        refusals = p1.verify_role_facts(roles, self._pins())
        assert any("train_monitor: scored_rows" in r for r in refusals)

    def test_context_counted_in_score_refuses(self):
        # A materializer that counted the 256 context rows as scored
        # rows drifts BOTH counts — the pins catch either direction.
        roles = self._manifest_roles()
        roles["inner_validation"]["scored_rows"] = 2190 + 256
        roles["inner_validation"]["context_rows"] = 0
        refusals = p1.verify_role_facts(roles, self._pins())
        assert any("inner_validation: scored_rows" in r
                   for r in refusals)
        assert any("inner_validation: context_rows" in r
                   for r in refusals)

    def test_missing_context_flag_refuses(self):
        roles = self._manifest_roles()
        roles["outer_validation"]["context_rows"] = None
        refusals = p1.verify_role_facts(roles, self._pins())
        assert any("outer_validation: context_rows" in r
                   for r in refusals)

    def test_wrong_role_csv_sha_refuses(self):
        roles = self._manifest_roles()
        roles["fit_train"]["csv_sha256"] = "0" * 64
        refusals = p1.verify_role_facts(roles, self._pins())
        assert any("fit_train: csv_sha256" in r for r in refusals)

    def test_sealed_test_materialization_refuses(self):
        roles = self._manifest_roles()
        roles["sealed_test"] = {
            "status": "MATERIALIZED", "csv": "/tmp/sealed_test.csv",
            "csv_sha256": "1" * 64, "scored_rows": 2190,
            "context_rows": 256}
        refusals = p1.verify_role_facts(roles, self._pins())
        assert any("sealed 2025 may never be materialized" in r
                   for r in refusals)
        assert any("csv_sha256" in r and "sealed" in r.lower()
                   for r in refusals)

    def test_outer_used_as_inner_refuses(self):
        contract = _contract()
        nested = json.loads(
            (REPO / contract["nested_split_contract"]["path"])
            .read_text())
        # swap the inner/outer role windows: the 2024 outer year now
        # occupies the inner selection slot.
        nested["roles"]["inner_validation"], \
            nested["roles"]["outer_validation"] = (
                nested["roles"]["outer_validation"],
                nested["roles"]["inner_validation"])
        refusals = p1.verify_role_semantics(
            nested, contract["nested_split_contract"]["role_facts"])
        assert refusals
        assert any("inner" in r for r in refusals)

    def test_paired_metric_drift_refuses_at_load(self, tmp_path):
        contract = copy.deepcopy(_contract())
        contract["selection_metric"] = "lexicographic_weekly_v1"
        path = _write_contract(tmp_path, contract)
        with pytest.raises(ValueError, match="lexicographic"):
            p1.load_contract(path)

    def test_materialize_refuses_before_training_on_bad_nested_pin(
            self, bindings, tmp_path):
        contract = copy.deepcopy(_contract())
        contract["nested_split_contract"]["sha256"] = "0" * 64
        with pytest.raises(RuntimeError, match="wrong role sha"):
            p1.materialize_cell_config(contract, bindings, 101,
                                       "P1N_LR1E4", tmp_path / "out")

    def test_runner_refuses_before_training_via_nested_roles_fn(
            self, runtime):
        def refusing_fn(contract, bindings, out_dir):
            raise RuntimeError(
                "nested role facts refused before training: test")
        summary = _run_seed(runtime, nested_roles_fn=refusing_fn)
        assert summary["outcome"] == "SEED_FAILED"
        for facts in summary["cells"].values():
            assert "refused before training" in facts["error"]
        # the pipeline never ran: refusal precedes model construction
        assert not FakePipeline.calls

    def test_experiment_identity_binds_the_nested_contract(
            self, bindings):
        contract = _contract()
        base = p1.experiment_identity(contract, bindings,
                                      sources=CLEAN_SOURCES)
        moved = copy.deepcopy(contract)
        moved["nested_split_contract"]["sha256"] = "9" * 64
        assert p1.experiment_identity(
            moved, bindings, sources=CLEAN_SOURCES) != base


# ---------------------------------------------------------------------------
# (i) finding 226: decision mode + decision verdict
# ---------------------------------------------------------------------------

def _viable_screen_gate(contract) -> dict:
    return {
        "schema": p1.VERDICT_SCHEMA,
        "outcome": "SCREEN_VIABLE_REGION",
        "contract_sha256": contract["_contract_sha256"],
        "gates": {"replica_terminal_loads": True},
    }


class TestDecisionMode:
    def test_decision_identity_and_root_are_distinct(self, bindings):
        contract = _contract()
        screen = p1.experiment_identity(contract, bindings,
                                        sources=CLEAN_SOURCES,
                                        mode="screen")
        decision = p1.experiment_identity(contract, bindings,
                                          sources=CLEAN_SOURCES,
                                          mode="decision")
        assert screen != decision
        assert p1.output_root_for_mode(contract, "screen") != \
            p1.output_root_for_mode(contract, "decision")

    def test_decision_config_carries_document38_stopping(
            self, bindings, tmp_path):
        contract = _contract()
        config = p1.materialize_cell_config(
            contract, bindings, 101, "P1N_LR1E4", tmp_path / "out",
            mode="decision")
        config.pop("_identity")
        assert config["max_epochs"] == 1996
        assert config["easy_max_epochs"] == 4
        assert config["l1_patience"] == 60
        assert config["l1_patience_start_epoch"] == 40
        assert config["total_max_passes"] == 2000
        assert config["learning_rate"] == pytest.approx(3e-5)
        assert config["selection_metric"] == \
            "paired_generalization_weekly_v1"
        assert config["nested_split_contract"]
        assert config["evaluate_test_split"] is False
        # decision cells start from the ORIGINAL per-seed anchor
        anchor = contract["anchors"]["101"]
        assert config["warm_start_model"] == str(
            Path(anchor["path"]).expanduser())
        assert config["warm_start_model_sha256"] == anchor["sha256"]

    def test_anchor_under_an_output_root_is_refused(self, bindings,
                                                    tmp_path):
        contract = copy.deepcopy(_contract())
        contract["output_root"] = str(tmp_path / "out")
        fake_terminal = (tmp_path / "out" / "old-screen" /
                         "model.terminal.zip")
        fake_terminal.parent.mkdir(parents=True)
        fake_terminal.write_bytes(b"screen-terminal")
        contract["anchors"]["101"] = {
            "path": str(fake_terminal),
            "sha256": p1._sha_file(fake_terminal)}
        with pytest.raises(RuntimeError, match="never anchor"):
            p1.materialize_cell_config(contract, bindings, 101,
                                       "P1N_LR1E4",
                                       tmp_path / "cell",
                                       mode="decision")

    def test_decision_without_screen_gate_is_refused(self, runtime):
        summary = _run_seed(runtime, mode="decision")
        assert summary["outcome"] == "REFUSED_DECISION_UNGATED"
        assert not FakePipeline.calls

    def test_decision_with_failed_gate_is_refused(self, runtime):
        gate = _viable_screen_gate(runtime.contract)
        gate["gates"]["replica_terminal_loads"] = (
            "EXTERNAL_COLLECTOR_REQUIRED")
        summary = _run_seed(runtime, mode="decision",
                            screen_gate=gate)
        assert summary["outcome"] == "REFUSED_DECISION_UNGATED"
        assert "replica" in summary["reason"]
        assert not FakePipeline.calls

    def test_decision_seed_runs_and_records_outer_truth(self,
                                                        runtime):
        def factory(config):
            return FakePipeline(config, with_best=True)
        summary = _run_seed(
            runtime, mode="decision",
            screen_gate=_viable_screen_gate(runtime.contract),
            pipeline_factory=factory,
            outer_eval_fn=_fake_outer_eval)
        assert summary["outcome"] == "SEED_COMPLETE"
        assert summary["mode"] == "decision"
        exp_id = summary["experiment_identity"]
        out_root = Path(runtime.contract["decision_run"][
            "output_root"])
        for cell in p1.CELLS:
            record = json.loads(
                (out_root / exp_id / "seed101" / cell /
                 "cell_record.json").read_text())
            assert record["mode"] == "decision"
            assert record["evidence_class"] == "decision_run"
            assert record["decision_eligible"] is True
            outer = record["outer_validation_final"]
            assert outer["role"] == "outer_validation"
            assert outer["context_excluded_from_metrics"] is True
            assert outer["metrics"]["mean_weekly_rap"] == \
                pytest.approx(0.01)
            assert len(outer["weekly_return_vector"]) == 52
            assert record["best_model_path"]
        # every decision cell warm-started from the ORIGINAL anchor
        anchor = runtime.contract["anchors"]["101"]
        for config in FakePipeline.calls:
            assert config["warm_start_model"] == anchor["path"]

    def test_decision_without_best_checkpoint_is_refused(self,
                                                         runtime):
        summary = _run_seed(
            runtime, mode="decision",
            screen_gate=_viable_screen_gate(runtime.contract),
            outer_eval_fn=_fake_outer_eval)
        assert summary["outcome"] == "SEED_FAILED"
        for facts in summary["cells"].values():
            assert "best checkpoint" in facts["error"]


def _decision_records(contract, tmp_path, rap_fn, trades_fn=None):
    trades_fn = trades_fn or (lambda s, c: 7)
    records = {}
    for seed in p1.SEEDS:
        for cell in p1.CELLS:
            record = _verdict_record(contract, seed, cell, "VIABLE",
                                     tmp_path)
            record["mode"] = "decision"
            record["evidence_class"] = "decision_run"
            rap = rap_fn(seed, cell)
            trades = trades_fn(seed, cell)
            record["outer_validation_final"] = {
                "role": "outer_validation",
                "metrics": {
                    "mean_weekly_rap": rap,
                    "mean_weekly_return": rap,
                    "annualized_return": 52 * rap,
                    "annual_return": 52 * rap,
                    "annual_rap": 52 * rap,
                    "max_drawdown_fraction": 0.1,
                    "evaluation_weeks": 52,
                },
                "weekly_return_vector": [rap] * 52,
                "trades_total": trades,
                "activity": {"traded": trades > 0,
                             "trades_total": trades},
            }
            records[(seed, cell)] = record
    return records


class TestDecisionVerdict:
    def test_lr_main_effect(self, tmp_path):
        contract = _contract()

        def rap(seed, cell):
            return 0.02 if cell.endswith("LR3E5") else 0.01
        records = _decision_records(contract, tmp_path, rap)
        payload, code = p1.decision_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "PHASE1_LR_MAIN_EFFECT"
        assert code == 0
        effects = payload["per_seed_paired_effects"]
        for seed in p1.SEEDS:
            assert effects[str(seed)]["phase1_lr_effect"] == \
                pytest.approx(0.01)
            assert effects[str(seed)][
                "lr_x_difficulty_interaction"] == pytest.approx(0.0)
        cell = payload["per_cell_metrics"]["seed101/P1N_LR3E5"]
        assert cell["mean_weekly_rap"] == pytest.approx(0.02)
        assert len(cell["weekly_return_vector"]) == 52
        assert "fraction per week" in cell["units_and_horizons"][
            "mean_weekly_rap"]

    def test_difficulty_main_effect(self, tmp_path):
        contract = _contract()

        def rap(seed, cell):
            return 0.03 if cell.startswith("P1E") else 0.01
        records = _decision_records(contract, tmp_path, rap)
        payload, code = p1.decision_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "PHASE1_DIFFICULTY_MAIN_EFFECT"
        assert code == 0

    def test_interaction(self, tmp_path):
        contract = _contract()

        def rap(seed, cell):
            return 0.04 if cell == "P1E_LR3E5" else 0.01
        records = _decision_records(contract, tmp_path, rap)
        payload, code = p1.decision_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == \
            "PHASE1_LR_DIFFICULTY_INTERACTION"

    def test_no_material_effect(self, tmp_path):
        contract = _contract()

        def rap(seed, cell):
            # sign flips across seeds: nothing is sign-consistent
            return 0.01 if (seed + len(cell)) % 2 else -0.01
        records = _decision_records(
            contract, tmp_path,
            lambda s, c: 0.01 if s in (101, 303) else 0.01)
        payload, code = p1.decision_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "NO_MATERIAL_EFFECT"
        assert code == 0

    def test_total_activity_collapse(self, tmp_path):
        contract = _contract()
        records = _decision_records(contract, tmp_path,
                                    lambda s, c: 0.0,
                                    trades_fn=lambda s, c: 0)
        payload, code = p1.decision_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "TOTAL_ACTIVITY_COLLAPSE"
        assert code == 0

    def test_screen_record_never_aggregates_as_decision(
            self, tmp_path):
        contract = _contract()
        records = _decision_records(contract, tmp_path,
                                    lambda s, c: 0.01)
        records[(101, "P1N_LR1E4")]["mode"] = "screen"
        records[(101, "P1N_LR1E4")]["evidence_class"] = \
            "mechanics_screen"
        payload, code = p1.decision_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "INCONCLUSIVE"
        assert code == 4
        assert any("not a decision_run record" in r
                   for r in payload["reasons"])

    def test_missing_outer_truth_is_inconclusive(self, tmp_path):
        contract = _contract()
        records = _decision_records(contract, tmp_path,
                                    lambda s, c: 0.01)
        records[(202, "P1E_LR3E5")].pop("outer_validation_final")
        payload, code = p1.decision_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "INCONCLUSIVE"
        assert any("outer" in r for r in payload["reasons"])

    def test_no_replica_proof_is_inconclusive(self, tmp_path):
        contract = _contract()
        records = _decision_records(contract, tmp_path,
                                    lambda s, c: 0.01)
        payload, code = p1.decision_verdict(contract, records=records)
        assert payload["outcome"] == "INCONCLUSIVE"
        assert payload["gates"]["replica_terminal_loads"] is False

    def test_outcomes_enum_matches_the_contract(self):
        contract = _contract()
        assert tuple(contract["decision_run"]["decision_outcomes"]) \
            == p1.DECISION_OUTCOMES


# ---------------------------------------------------------------------------
# (j) the no-training materialization preflight (acceptance boundary)
# ---------------------------------------------------------------------------

class TestPreflight:
    def test_preflight_proves_roles_sealing_and_identities(
            self, bindings):
        contract = _contract()
        payload, code = p1.preflight(contract, bindings)
        assert payload["outcome"] == "PREFLIGHT_PASS", \
            payload["refusals"]
        assert code == 0
        assert payload["training_used"] is False
        # exact nested role counts/hashes
        pins = contract["nested_split_contract"]["role_facts"]
        for role, pin in pins.items():
            got = payload["nested_role_facts"][role]
            for key in p1.NESTED_ROLE_FACT_KEYS:
                assert got[key] == pin[key], (role, key)
        # sealed-test absence
        assert payload["sealed_test_state"] == "SEALED"
        assert payload["sealed_test_csv_absent"] is True
        # paired selection everywhere, both modes
        for mode in p1.MODES:
            facts = payload["modes"][mode]
            assert facts["selection_metrics_materialized"] == [
                "paired_generalization_weekly_v1"]
            assert facts["evaluate_test_split_values"] == [False]
            assert len(set(facts["cell_identities"].values())) == 16
        assert payload["modes"]["screen"]["experiment_identity"] != \
            payload["modes"]["decision"]["experiment_identity"]


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
