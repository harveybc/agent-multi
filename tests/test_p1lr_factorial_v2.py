"""Corrected-observation P1LR v2 launch gate (order 2026-08-15 §6/§7;
AUD-P1LR-20260815-235).

The 12 required tests of order §7, one class per requirement:

  Req1   a missing or drifted observation declaration refuses BEFORE
         any GPU use;
  Req2   a 2,724-input legacy anchor refuses BEFORE model
         construction;
  Req3   each seed's four cells begin from ONE identical zero-update
         policy tensor;
  Req4   different seeds have DISTINCT genesis tensors;
  Req5   no cell inherits another cell's trained weights, replay or
         optimizer;
  Req6   phase-1 easy AND normal treatment reaches the actual
         training environment (the executing pipeline config);
  Req7   both phases see 2,660 inputs and the same ordered feature
         contract;
  Req8   zero live units / missing liveness evidence / an unmeasured
         actor is NON-PROMOTABLE;
  Req9   a constant selected policy is NON-PROMOTABLE even with a
         favorable scalar metric;
  Req10  the selected-policy tensor identity must DIFFER from the
         genesis;
  Req11  the test (sealed 2025) and context-prefix contracts remain
         unchanged from v1;
  Req12  all 16 terminal artifacts load on the replica preserving
         identity (socket-free, injected transports — the
         tests/test_p1lr_collect.py pattern).

Per the order's last paragraph, the live-unit FRACTION is recorded but
NO invented minimum (such as 256/256) gates promotion — asserted
explicitly in Req8. Additional guards: v1 replayability (identities,
roots and record schema unchanged and disjoint from v2), and the
liveness-classification enum mirror.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import p1_difficulty_lr_factorial as p1  # noqa: E402
from tools import p1lr_collect as p1c  # noqa: E402
from tests.test_p1_difficulty_lr_factorial import (  # noqa: E402
    CLEAN_SOURCES,
    FakePipeline,
    _diff,
    _fake_dispatch_binding,
    _fake_gate_heartbeat,
    _fake_nested_roles_fn,
    _fake_outer_eval,
    _fake_result,
    _inactive_at,
    _proof_for,
    _verdict_record,
    _viable_screen_gate,
)
from tests.test_p1lr_collect import (  # noqa: E402
    _echo_verify,
    _local_fetch,
    _noop_replicate,
)

EXPECTED_DIM = 2660
LEGACY_DIM = 2724


def _v2_contract() -> dict:
    return p1.load_contract(p1.CONTRACT_PATH_V2)


def _v1_contract() -> dict:
    return p1.load_contract(p1.CONTRACT_PATH)


@pytest.fixture(scope="module")
def bindings() -> dict:
    return p1.load_bindings()


@pytest.fixture(autouse=True)
def pinned_source_identities(monkeypatch):
    """Hermetic code identity (see the v1 module's fixture rationale:
    record-custody proofs must not depend on the developer tree)."""
    monkeypatch.setattr(p1.ladder, "source_identities",
                        lambda: copy.deepcopy(CLEAN_SOURCES))


def _fake_tensor(path) -> str:
    """Deterministic fake policy-tensor digest: a pure function of the
    artifact BYTES, so identical bytes share one tensor identity and
    different artifacts differ — the same property the real digest
    has."""
    return hashlib.sha256(
        b"tensor:" + Path(path).read_bytes()).hexdigest()


def _write_contract(tmp_path: Path, contract: dict) -> Path:
    path = tmp_path / "p1lr_v2_contract.json"
    clean = {key: value for key, value in contract.items()
             if not key.startswith("_")}
    path.write_text(json.dumps(clean, indent=1, sort_keys=True))
    return path


# ---------------------------------------------------------------------------
# v2 fake pipeline: v1 result + typed liveness history
# ---------------------------------------------------------------------------

def _liveness_row(epoch: int, *, classification="ACTOR_ALIVE",
                  live=256, units=256, constant=False,
                  std=0.02) -> dict:
    return {
        "schema": "agent_multi.actor_liveness.v1",
        "epoch": epoch,
        "phase": "normal_realistic",
        "split": "train_epoch+validation_epoch",
        "measured": classification != "ACTOR_UNMEASURED",
        "classification": classification,
        "live_unit_count": live,
        "first_layer_units": units,
        "live_unit_fraction": live / units,
        "varying_unit_count": live,
        "preactivation_mean": 0.1,
        "action_raw_std": 0.0 if constant else std,
        "constant_policy": constant,
    }


def _attach_liveness(result: dict, liveness: str) -> dict:
    if liveness == "missing":
        result.pop("actor_liveness_history", None)
        return result
    if liveness == "alive":
        rows = [_liveness_row(1), _liveness_row(2)]
    elif liveness == "dead":
        rows = [_liveness_row(1),
                _liveness_row(2,
                              classification="ACTOR_FIRST_LAYER_DEAD",
                              live=0)]
    elif liveness == "degraded_low_fraction":
        # 3/256 live — far below ANY plausible invented minimum, yet
        # NOT a hard failure (order §7: the fraction is recorded,
        # never gated).
        rows = [_liveness_row(1),
                _liveness_row(
                    2, classification="ACTOR_FIRST_LAYER_DEGRADED",
                    live=3)]
    elif liveness == "constant":
        rows = [_liveness_row(1),
                _liveness_row(2,
                              classification="ACTOR_CONSTANT_POLICY",
                              constant=True, live=200)]
    elif liveness == "unmeasured":
        rows = [_liveness_row(1),
                _liveness_row(2, classification="ACTOR_UNMEASURED")]
    elif liveness == "untyped":
        rows = [_liveness_row(1)]
        rows.append({**_liveness_row(2), "classification": "BOGUS"})
    else:  # pragma: no cover - test bug
        raise AssertionError(liveness)
    result["actor_liveness_history"] = rows
    result["actor_liveness"] = rows[-1]
    return result


def _v2_factory(liveness: str = "alive", **result_kwargs):
    """A FakePipeline factory whose results carry the v2 liveness
    evidence (or deliberately lack it)."""

    def factory(config):
        pipe = FakePipeline(config, **result_kwargs)
        original = pipe.run_pipeline

        def run_pipeline(**kwargs):
            result = original(**kwargs)
            _attach_liveness(result, liveness)
            result["observation_contract"] = {
                "schema": "agent_multi.observation_contract."
                          "application.v1",
                "declared": True,
                "source": "config.observation_contract",
                "applied": {"include_price_window":
                            {"from": True, "to": False}},
            }
            return result

        pipe.run_pipeline = run_pipeline
        return pipe

    return factory


def _v2_inactive_factory(*positions: int, liveness: str = "alive"):
    base = _inactive_at(*positions)

    def factory(config):
        pipe = base(config)
        original = pipe.run_pipeline

        def run_pipeline(**kwargs):
            result = original(**kwargs)
            _attach_liveness(result, liveness)
            return result

        pipe.run_pipeline = run_pipeline
        return pipe

    return factory


@pytest.fixture()
def rt(bindings, tmp_path, monkeypatch):
    """A runnable v2 contract copy: tmp output roots, tmp hash-bound
    FAKE genesis artifacts (container sha real, tensor pin from
    ``_fake_tensor``), seed-101 host/GPU environment, fake nested
    verifier, injected dimension/tensor functions."""
    contract = copy.deepcopy(_v2_contract())
    contract["output_root"] = str(tmp_path / "out")
    contract["decision_run"]["output_root"] = str(
        tmp_path / "out_decision")
    genesis_dir = tmp_path / "genesis"
    genesis_dir.mkdir()
    for seed in p1.SEEDS:
        artifact = genesis_dir / f"zero_update_genesis_seed{seed}.zip"
        artifact.write_bytes(f"fake-genesis-{seed}".encode())
        contract["genesis"]["seeds"][str(seed)] = {
            "path": str(artifact),
            "container_sha256": p1._sha_file(artifact),
            "policy_tensor_sha256": _fake_tensor(artifact),
        }
    assigned = contract["assignments"]["101"]["gpu_uuid"]
    monkeypatch.setattr(p1.socket, "gethostname", lambda: "omega")
    monkeypatch.setattr(p1, "visible_gpu_uuids", lambda: [assigned])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", assigned)
    FakePipeline.calls = []
    return SimpleNamespace(
        contract=contract, bindings=bindings, assigned=assigned,
        genesis_dir=genesis_dir,
        agent_loader=lambda name: SimpleNamespace(name=name),
        tensor_sha_fn=_fake_tensor,
        genesis_tensor_sha_fn=_fake_tensor,
        observation_dim_fn=lambda path: EXPECTED_DIM,
        gate_heartbeat=_fake_gate_heartbeat([assigned]),
        dispatch_binding_fn=_fake_dispatch_binding,
        nested_roles_fn=_fake_nested_roles_fn(contract),
    )


def _run_seed(rt, **overrides):
    kwargs = dict(
        contract=rt.contract, bindings=rt.bindings,
        enforce_gpu=True, pipeline_factory=_v2_factory(),
        agent_loader=rt.agent_loader,
        tensor_sha_fn=rt.tensor_sha_fn,
        genesis_tensor_sha_fn=rt.genesis_tensor_sha_fn,
        observation_dim_fn=rt.observation_dim_fn,
        gate_heartbeat=rt.gate_heartbeat,
        dispatch_binding_fn=rt.dispatch_binding_fn,
        nested_roles_fn=rt.nested_roles_fn,
    )
    kwargs.update(overrides)
    return p1.run_seed(101, **kwargs)


def _records_of(rt, summary, mode: str = "screen") -> dict:
    root = p1.output_root_for_mode(rt.contract, mode)
    exp_dir = root / summary["experiment_identity"] / "seed101"
    return {cell: json.loads(
        (exp_dir / cell / "cell_record.json").read_text())
        for cell in p1.CELLS
        if (exp_dir / cell / "cell_record.json").is_file()}


def _genesis_pin(rt, seed: int = 101) -> dict:
    return rt.contract["genesis"]["seeds"][str(seed)]


# ---------------------------------------------------------------------------
# Req 1 — missing/drifted observation declaration refuses before GPU
# ---------------------------------------------------------------------------

class TestReq1ObservationDeclarationRefusal:
    def test_missing_observation_contract_refuses_at_load(
            self, tmp_path):
        contract = copy.deepcopy(_v2_contract())
        del contract["observation_contract"]
        with pytest.raises(ValueError,
                           match="UNDECLARED_OBSERVATION_CONTRACT"):
            p1.load_contract(_write_contract(tmp_path, contract))

    def test_price_window_reintroduction_refuses_at_load(
            self, tmp_path):
        contract = copy.deepcopy(_v2_contract())
        contract["observation_contract"]["include_price_window"] = True
        with pytest.raises(ValueError,
                           match="include_price_window=false"):
            p1.load_contract(_write_contract(tmp_path, contract))

    def test_optin_only_declaration_refuses_at_load(self, tmp_path):
        contract = copy.deepcopy(_v2_contract())
        contract["observation_contract"][
            "require_feature_aware_preprocessor"] = False
        with pytest.raises(ValueError, match="fail-closed"):
            p1.load_contract(_write_contract(tmp_path, contract))

    def test_wrong_expected_dimension_refuses_at_load(self, tmp_path):
        contract = copy.deepcopy(_v2_contract())
        contract["expected_observation"]["expected_dimension"] = \
            LEGACY_DIM
        contract["expected_observation"]["feature_count"] = 85
        with pytest.raises(ValueError, match="drifted|inconsistent"):
            p1.load_contract(_write_contract(tmp_path, contract))

    def test_feature_set_drift_refuses_before_any_pipeline(
            self, rt, tmp_path):
        rt.contract["observation_contract"][
            "feature_columns_sha256"] = "a" * 64
        with pytest.raises(RuntimeError,
                           match="OBSERVATION_CONTRACT_DRIFT"):
            p1.materialize_cell_config(
                rt.contract, rt.bindings, 101, "P1N_LR1E4",
                tmp_path / "cell")
        assert not FakePipeline.calls

    def test_seed_run_with_drifted_declaration_never_trains(self, rt):
        rt.contract["observation_contract"][
            "feature_columns_sha256"] = "a" * 64
        summary = _run_seed(rt)
        assert summary["outcome"] == "SEED_FAILED"
        for facts in summary["cells"].values():
            assert facts["outcome"] == "CELL_FAILED"
            assert "OBSERVATION_CONTRACT_DRIFT" in facts["error"]
        assert not FakePipeline.calls


# ---------------------------------------------------------------------------
# Req 2 — a 2,724-input legacy anchor refuses before model construction
# ---------------------------------------------------------------------------

def _legacy_policy_zip(path: Path, width: int = LEGACY_DIM) -> None:
    """A real policy container whose actor first layer has ``width``
    input columns — the shape of the dead 2,724-input anchors."""
    import io
    import zipfile

    import torch

    buffer = io.BytesIO()
    torch.save({"actor.latent_pi.0.weight": torch.zeros((8, width))},
               buffer)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("policy.pth", buffer.getvalue())


class TestReq2LegacyAnchorRefusal:
    def test_v2_contract_with_anchors_block_refuses_at_load(
            self, tmp_path):
        contract = copy.deepcopy(_v2_contract())
        contract["anchors"] = _v1_contract()["anchors"]
        with pytest.raises(ValueError, match="anchors"):
            p1.load_contract(_write_contract(tmp_path, contract))

    def test_2724_width_artifact_refuses_before_model_construction(
            self, rt):
        artifact = rt.genesis_dir / "legacy_anchor_2724.zip"
        _legacy_policy_zip(artifact)
        rt.contract["genesis"]["seeds"]["101"] = {
            "path": str(artifact),
            "container_sha256": p1._sha_file(artifact),
            "policy_tensor_sha256": _fake_tensor(artifact),
        }
        # The REAL width reader proves the refusal on a real container.
        with pytest.raises(RuntimeError, match="2724-input"):
            p1._verify_initialization(rt.contract, 101,
                                      genesis_tensor_sha_fn=_fake_tensor)
        summary = _run_seed(
            rt, observation_dim_fn=p1.policy_observation_dim)
        assert summary["outcome"] == "REFUSED_GENESIS_UNVERIFIED"
        assert "2724-input" in summary["reason"]
        assert not FakePipeline.calls

    def test_real_width_reader_accepts_the_corrected_width(
            self, tmp_path):
        artifact = tmp_path / "corrected_width.zip"
        _legacy_policy_zip(artifact, width=EXPECTED_DIM)
        assert p1.policy_observation_dim(artifact) == EXPECTED_DIM


# ---------------------------------------------------------------------------
# Req 3 / Req 4 — genesis tensor identity within and across seeds
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_genesis(bindings, tmp_path_factory):
    """REAL zero-update genesis constructions (CPU, SB3) through the
    shipping tool: seed 101 twice into two roots + seed 202 once."""
    from tools import p1lr_genesis_artifacts as gen

    contract = _v2_contract()
    root_a = tmp_path_factory.mktemp("genesis_a")
    root_b = tmp_path_factory.mktemp("genesis_b")
    return {
        "101a": gen.build_seed_genesis(contract, bindings, 101, root_a),
        "101b": gen.build_seed_genesis(contract, bindings, 101, root_b),
        "202": gen.build_seed_genesis(contract, bindings, 202, root_a),
    }


class TestReq3SameGenesisTensorWithinSeed:
    def test_all_four_cells_bind_the_one_persisted_genesis(
            self, rt, tmp_path):
        pin = _genesis_pin(rt)
        identities = set()
        for cell in p1.CELLS:
            config = p1.materialize_cell_config(
                rt.contract, rt.bindings, 101, cell,
                tmp_path / "m" / cell)
            identity = config.pop("_identity")
            assert config["warm_start_model"] == pin["path"], cell
            assert config["warm_start_model_sha256"] == \
                pin["container_sha256"], cell
            assert identity["initialization"]["kind"] == \
                "zero_update_genesis", cell
            identities.add(
                identity["initialization"]["policy_tensor_sha256"])
        assert identities == {pin["policy_tensor_sha256"]}

    def test_every_executed_cell_started_from_the_genesis(self, rt):
        summary = _run_seed(rt)
        assert summary["outcome"] == "SEED_COMPLETE"
        pin = _genesis_pin(rt)
        assert len(FakePipeline.calls) == 4
        for config in FakePipeline.calls:
            assert config["warm_start_model"] == pin["path"]
            assert config["warm_start_model_sha256"] == \
                pin["container_sha256"]
        records = _records_of(rt, summary)
        for cell, record in records.items():
            assert record["genesis_policy_tensor_sha256"] == \
                pin["policy_tensor_sha256"], cell
            assert record["genesis_container_sha256"] == \
                pin["container_sha256"], cell
            assert record["genesis_artifact_type"] == \
                "zero_update_genesis", cell

    def test_real_construction_reproduces_one_tensor_identity(
            self, real_genesis):
        a, b = real_genesis["101a"], real_genesis["101b"]
        assert a["policy_tensor_sha256"] == b["policy_tensor_sha256"]
        assert a["observation_dim"] == EXPECTED_DIM
        assert a["construction_deterministic"] is True
        assert a["identity_preserved_after_save_load"] is True
        assert a["zero_update_proof"]["gradient_updates"] == 0
        assert a["zero_update_proof"][
            "replay_transitions_written"] == 0
        assert a["artifact_type"] == "zero_update_genesis"
        assert a["never_a_trained_champion_or_handoff"] is True

    def test_tampered_genesis_tensor_pin_refuses_the_seed(self, rt):
        rt.contract["genesis"]["seeds"]["101"][
            "policy_tensor_sha256"] = "9" * 64
        summary = _run_seed(rt)
        assert summary["outcome"] == "REFUSED_GENESIS_UNVERIFIED"
        assert not FakePipeline.calls


class TestReq4DistinctGenesisTensorsAcrossSeeds:
    def test_contract_pins_are_pairwise_distinct(self):
        contract = _v2_contract()
        tensors = [contract["genesis"]["seeds"][str(seed)][
            "policy_tensor_sha256"] for seed in p1.SEEDS]
        containers = [contract["genesis"]["seeds"][str(seed)][
            "container_sha256"] for seed in p1.SEEDS]
        assert len(set(tensors)) == 4
        assert len(set(containers)) == 4

    def test_identical_tensor_pins_across_seeds_refuse_at_load(
            self, tmp_path):
        contract = copy.deepcopy(_v2_contract())
        contract["genesis"]["seeds"]["202"]["policy_tensor_sha256"] = \
            contract["genesis"]["seeds"]["101"]["policy_tensor_sha256"]
        with pytest.raises(ValueError, match="DISTINCT"):
            p1.load_contract(_write_contract(tmp_path, contract))

    def test_real_seeds_construct_distinct_tensors(self, real_genesis):
        assert real_genesis["101a"]["policy_tensor_sha256"] != \
            real_genesis["202"]["policy_tensor_sha256"]
        assert real_genesis["202"]["observation_dim"] == EXPECTED_DIM


# ---------------------------------------------------------------------------
# Req 5 — no cell inherits another cell's weights/replay/optimizer
# ---------------------------------------------------------------------------

class TestReq5NoCrossCellInheritance:
    def test_every_cell_starts_from_the_genesis_never_a_terminal(
            self, rt):
        summary = _run_seed(rt)
        assert summary["outcome"] == "SEED_COMPLETE"
        pin = _genesis_pin(rt)
        terminals = set()
        records = _records_of(rt, summary)
        for config in FakePipeline.calls:
            # the warm start is the genesis — never any produced
            # artifact of an earlier cell
            assert config["warm_start_model"] == pin["path"]
            assert config["warm_start_model"] not in terminals
        for record in records.values():
            terminals.add(record["terminal_model_path"])
            evidence = record["boundary_transfer_evidence"]
            assert evidence["optimizer_state_transferred"] is False
            assert evidence["replay_size_at_boundary"] == 0
            assert evidence["replay_transitions_transferred"] == 0
        # four distinct attempt dirs, four distinct terminals
        assert len(terminals) == 4

    def test_genesis_under_an_output_root_is_refused(self, rt,
                                                     tmp_path):
        fake_terminal = (Path(rt.contract["output_root"]) /
                         "old-screen" / "model.terminal.zip")
        fake_terminal.parent.mkdir(parents=True)
        fake_terminal.write_bytes(b"screen-terminal")
        rt.contract["genesis"]["seeds"]["101"] = {
            "path": str(fake_terminal),
            "container_sha256": p1._sha_file(fake_terminal),
            "policy_tensor_sha256": _fake_tensor(fake_terminal),
        }
        with pytest.raises(RuntimeError, match="never anchor"):
            p1.materialize_cell_config(rt.contract, rt.bindings, 101,
                                       "P1N_LR1E4", tmp_path / "cell")


# ---------------------------------------------------------------------------
# Req 6 — the treatment reaches the actual training environment
# ---------------------------------------------------------------------------

class TestReq6TreatmentReachesTraining:
    def test_easy_and_normal_phase1_reach_the_executing_config(
            self, rt):
        summary = _run_seed(rt)
        assert summary["outcome"] == "SEED_COMPLETE"
        order = rt.contract["cell_order"]["101"]
        assert [c["phase1_mode"] for c in FakePipeline.calls] == [
            rt.contract["cells"][cell]["phase1_dynamics"]
            for cell in order]
        modes = {c["phase1_mode"] for c in FakePipeline.calls}
        assert modes == {"normal_realistic",
                         "easy_chronological_continuation"}
        for config, cell in zip(FakePipeline.calls, order):
            expected_lr = float(
                rt.contract["cells"][cell]["phase1_learning_rate"])
            assert config["phase1_learning_rate"] == \
                pytest.approx(expected_lr), cell
            assert config["easy_learning_rate"] == \
                pytest.approx(expected_lr), cell
        records = _records_of(rt, summary)
        for cell, record in records.items():
            assert record["phase1_mode"] == \
                rt.contract["cells"][cell]["phase1_dynamics"]

    def test_factor_deltas_are_exactly_the_factor_fields(
            self, rt, tmp_path):
        configs = {}
        for cell in p1.CELLS:
            config = p1.materialize_cell_config(
                rt.contract, rt.bindings, 101, cell,
                tmp_path / "d")
            config.pop("_identity")
            configs[cell] = config
        cells = list(p1.CELLS)
        for i, cell_a in enumerate(cells):
            for cell_b in cells[i + 1:]:
                assert _diff(configs[cell_a], configs[cell_b]) == \
                    p1.intended_delta_fields(rt.contract, cell_a,
                                             cell_b)


# ---------------------------------------------------------------------------
# Req 7 — both phases see 2,660 inputs, one ordered feature contract
# ---------------------------------------------------------------------------

class TestReq7BothPhases2660:
    def test_materialized_config_validates_to_2660_feature_aware(
            self, rt, tmp_path):
        from pipeline_plugins._observation_contract import (
            apply_observation_contract,
            feature_columns_sha256,
            validate_observation_contract,
        )
        config = p1.materialize_cell_config(
            rt.contract, rt.bindings, 101, "P1E_LR1E4",
            tmp_path / "cell")
        identity = config.pop("_identity")
        # The inline declaration the pipeline applies in BOTH phases
        # (phase 2 derives its config from the same dict).
        assert config["observation_contract"][
            "include_price_window"] is False
        bound, application = apply_observation_contract(dict(config))
        assert application["declared"] is True
        validation = validate_observation_contract(bound)
        assert validation["outcome"] == \
            "FEATURE_AWARE_OBSERVATION_CONTRACT"
        assert validation["feature_column_count"] == 83
        assert validation["include_price_window"] is False
        derived = (int(bound["window_size"])
                   * validation["feature_column_count"] + 4)
        assert derived == EXPECTED_DIM
        # one ordered feature contract, pinned
        assert feature_columns_sha256(config["feature_columns"]) == \
            rt.contract["observation_contract"][
                "feature_columns_sha256"]
        assert identity["observation"][
            "derived_observation_dimension"] == EXPECTED_DIM

    def test_both_phase_treatments_share_the_observation_identity(
            self, rt, tmp_path):
        shas = set()
        for cell in ("P1N_LR1E4", "P1E_LR1E4"):
            config = p1.materialize_cell_config(
                rt.contract, rt.bindings, 101, cell,
                tmp_path / "p" / cell)
            identity = config.pop("_identity")
            assert config["observation_contract"] == \
                rt.contract["observation_contract"]
            shas.add(identity["observation"][
                "observation_contract_sha256"])
        assert len(shas) == 1

    def test_records_bind_the_observation_identity(self, rt):
        summary = _run_seed(rt)
        records = _records_of(rt, summary)
        expected_sha = p1.observation_contract_sha256(rt.contract)
        for cell, record in records.items():
            assert record["observation_contract_sha256"] == \
                expected_sha, cell
            assert record["expected_observation"][
                "expected_dimension"] == EXPECTED_DIM, cell


# ---------------------------------------------------------------------------
# Req 8 — unmeasured / missing-evidence / zero-live is non-promotable
# ---------------------------------------------------------------------------

class TestReq8UnmeasuredOrDeadIsNonPromotable:
    def _one_cell(self, rt, liveness):
        summary = _run_seed(rt,
                            pipeline_factory=_v2_factory(liveness))
        assert summary["outcome"] == "SEED_COMPLETE"
        return _records_of(rt, summary)

    def test_missing_liveness_evidence_is_non_promotable(self, rt):
        records = self._one_cell(rt, "missing")
        for cell, record in records.items():
            assert record["promotion_eligible"] is False, cell
            assert record["promotion_ineligibility_causes"] == [
                "ACTOR_UNMEASURED"], cell
            assert record["actor_liveness_binding"][
                "measured"] is False, cell
            # still a landed, custody-complete record
            assert record["terminal_model_sha256"], cell

    def test_unmeasured_actor_is_non_promotable(self, rt):
        records = self._one_cell(rt, "unmeasured")
        for record in records.values():
            assert record["promotion_eligible"] is False
            assert "ACTOR_UNMEASURED" in \
                record["promotion_ineligibility_causes"]

    def test_zero_live_units_is_non_promotable(self, rt):
        records = self._one_cell(rt, "dead")
        for record in records.values():
            assert record["promotion_eligible"] is False
            assert "ZERO_LIVE_UNITS" in \
                record["promotion_ineligibility_causes"]
            assert record["selected_policy_gates"][
                "live_unit_count"] == 0

    def test_low_live_fraction_is_recorded_but_NOT_gated(self, rt):
        """Order §7 last paragraph: no invented minimum (256/256 or
        any other) may gate promotion — 3/256 live units with varying
        actions stays promotable; the fraction is recorded."""
        records = self._one_cell(rt, "degraded_low_fraction")
        for record in records.values():
            gates = record["selected_policy_gates"]
            assert gates["live_unit_count"] == 3
            assert gates["live_unit_fraction"] == \
                pytest.approx(3 / 256)
            assert gates["live_unit_fraction_is_gate"] is False
            assert gates["non_promotable_causes"] == []
            assert record["promotion_eligible"] is True

    def test_untyped_liveness_row_is_a_harness_failure(self, rt):
        summary = _run_seed(rt,
                            pipeline_factory=_v2_factory("untyped"))
        assert summary["outcome"] == "SEED_FAILED"

    def test_screen_names_non_promotable_cells_outside_the_viable_set(
            self, rt):
        summary = _run_seed(rt, pipeline_factory=_v2_factory("dead"))
        disk_records = _records_of(rt, summary)
        records = {(101, cell): record
                   for cell, record in disk_records.items()}
        # complete the 16-cell picture with promotable synthetic cells
        for seed in (202, 303, 404):
            for cell in p1.CELLS:
                records[(seed, cell)] = _v2_verdict_record(
                    rt.contract, seed, cell, "VIABLE",
                    Path(rt.contract["output_root"]),
                    exp_id=summary["experiment_identity"])
        payload, code = p1.screen_verdict(
            rt.contract, records=records,
            replica_proof=_proof_for(rt.contract, records))
        assert payload["outcome"] == "SCREEN_VIABLE_REGION"
        assert code == 0
        named = {(entry["seed"], entry["cell"])
                 for entry in payload["non_promotable_cells"]}
        assert named == {(101, cell) for cell in p1.CELLS}
        viable = {(entry["seed"], entry["cell"])
                  for entry in payload["viable_cells"]}
        assert not (named & viable)

    def test_liveness_enum_mirror_matches_the_measurement_module(
            self):
        from pipeline_plugins import _actor_liveness
        assert tuple(p1.ACTOR_LIVENESS_CLASSIFICATIONS) == \
            tuple(_actor_liveness.CLASSIFICATIONS)


# ---------------------------------------------------------------------------
# Req 9 — a constant selected policy is non-promotable, metrics be damned
# ---------------------------------------------------------------------------

def _v2_verdict_record(contract, seed, cell, label, tmp_path, *,
                       exp_id="e" * 16, activity_status="active",
                       liveness="alive",
                       selected_equals_genesis=False) -> dict:
    """A synthetic v2 cell record: the v1 verdict-record shape plus the
    mandatory v2 provenance, with gates DERIVED by the real gate
    function."""
    record = _verdict_record(contract, seed, cell, label,
                             Path(tmp_path), exp_id=exp_id,
                             activity_status=activity_status)
    record["schema"] = p1.RECORD_SCHEMA_V2
    # full finding-223 custody (assert_cell_record_custody demands the
    # terminal tensor digest and a load proof bound to it)
    record["terminal_policy_tensor_sha256"] = hashlib.sha256(
        f"terminal-tensor-{seed}-{cell}".encode()).hexdigest()
    record["terminal_load_proof"] = {
        "schema": p1.TERMINAL_LOAD_PROOF_SCHEMA,
        "path": record["terminal_model_path"],
        "sha256": record["terminal_model_sha256"],
        "policy_tensor_sha256":
            record["terminal_policy_tensor_sha256"],
        "loaded": True,
    }
    genesis_pin = contract["genesis"]["seeds"][str(seed)]
    result = {}
    _attach_liveness(result, liveness)
    liveness_binding = p1.bind_actor_liveness(result)
    selected_sha = (genesis_pin["policy_tensor_sha256"]
                    if selected_equals_genesis
                    else hashlib.sha256(
                        f"selected-{seed}-{cell}".encode()).hexdigest())
    svg = p1.bind_selected_vs_genesis(
        init_facts={"policy_tensor_sha256":
                    genesis_pin["policy_tensor_sha256"]},
        selected_path=f"selected-{seed}-{cell}.zip",
        selected_role=("terminal" if activity_status == "inactive"
                       else "best_checkpoint"),
        selected_tensor_sha=selected_sha)
    gates = p1.selected_policy_gates(liveness_binding, svg,
                                     activity_status)
    record.update({
        "genesis_container_sha256": genesis_pin["container_sha256"],
        "genesis_policy_tensor_sha256":
            genesis_pin["policy_tensor_sha256"],
        "genesis_artifact_type": "zero_update_genesis",
        "observation_contract_sha256":
            p1.observation_contract_sha256(contract),
        "expected_observation": dict(contract["expected_observation"]),
        "actor_liveness_binding": liveness_binding,
        "selected_vs_genesis": svg,
        "selected_policy_gates": gates,
        "promotion_ineligibility_causes":
            gates["non_promotable_causes"],
    })
    if activity_status != "inactive":
        record["promotion_eligible"] = bool(gates["promotable"])
    return record


def _v2_decision_records(contract, tmp_path, rap_fn, *,
                         liveness_fn=None,
                         equals_genesis=frozenset()) -> dict:
    liveness_fn = liveness_fn or (lambda seed, cell: "alive")
    records = {}
    for seed in p1.SEEDS:
        for cell in p1.CELLS:
            record = _v2_verdict_record(
                contract, seed, cell, "VIABLE", tmp_path,
                liveness=liveness_fn(seed, cell),
                selected_equals_genesis=(seed, cell) in equals_genesis)
            record["mode"] = "decision"
            record["evidence_class"] = "decision_run"
            rap = rap_fn(seed, cell)
            record["outer_validation_artifact_role"] = \
                "best_checkpoint"
            record["outer_validation_final"] = {
                "role": "outer_validation",
                "evaluated_artifact_role": "best_checkpoint",
                "evaluated_model_path": record["best_model_path"],
                "best_model_path": record["best_model_path"],
                "best_model_sha256": record["best_model_sha256"],
                "promotion_eligible": True,
                "diagnostic_only": False,
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
                "trades_total": 9,
                "activity": {"traded": True, "trades_total": 9},
            }
            records[(seed, cell)] = record
    return records


class TestReq9ConstantSelectedPolicyIsNonPromotable:
    def test_constant_policy_record_is_non_promotable(self, rt):
        summary = _run_seed(rt,
                            pipeline_factory=_v2_factory("constant"))
        assert summary["outcome"] == "SEED_COMPLETE"
        for record in _records_of(rt, summary).values():
            assert record["promotion_eligible"] is False
            assert "CONSTANT_SELECTED_POLICY" in \
                record["promotion_ineligibility_causes"]
            assert record["selected_policy_gates"][
                "constant_policy"] is True

    def test_favorable_scalar_metric_cannot_buy_promotion(
            self, rt, tmp_path):
        """The constant cell posts the BEST mean weekly RAP of the
        whole factorial — and is still excluded from the paired
        utilities, so its seed's effects are typed unavailable and
        the decision is withheld."""
        contract = rt.contract

        def rap(seed, cell):
            if (seed, cell) == (101, "P1N_LR3E5"):
                return 0.99  # spectacular — and meaningless
            return 0.01

        records = _v2_decision_records(
            contract, tmp_path, rap,
            liveness_fn=lambda seed, cell: (
                "constant" if (seed, cell) == (101, "P1N_LR3E5")
                else "alive"))
        payload, code = p1.decision_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "INCONCLUSIVE"
        assert code == 4
        named = {(entry["seed"], entry["cell"]): entry["causes"]
                 for entry in payload["non_promotable_cells"]}
        assert named == {(101, "P1N_LR3E5"):
                         ["CONSTANT_SELECTED_POLICY"]}
        cell = payload["per_cell_metrics"]["seed101/P1N_LR3E5"]
        assert cell["performance_eligible"] is False
        assert cell["mean_weekly_rap"] == pytest.approx(0.99)
        assert payload["per_seed_paired_effects"]["101"][
            "available"] is False
        assert "non-promotable" in payload["outcome_rationale"]

    def test_record_custody_refuses_promoting_a_constant_selection(
            self, rt, tmp_path):
        record = _v2_verdict_record(rt.contract, 101, "P1N_LR1E4",
                                    "VIABLE", tmp_path,
                                    liveness="constant")
        record["promotion_eligible"] = True  # forged
        with pytest.raises(RuntimeError, match="NEVER promotable"):
            p1.assert_cell_record_custody(record)


# ---------------------------------------------------------------------------
# Req 10 — the selected policy tensor must differ from the genesis
# ---------------------------------------------------------------------------

class TestReq10SelectedMustDifferFromGenesis:
    def test_selection_that_is_the_genesis_is_non_promotable(self, rt):
        pin = _genesis_pin(rt)

        def tensor_from_genesis(path):
            # every artifact hashes to the GENESIS tensor: selection
            # never left the initialization
            return pin["policy_tensor_sha256"]

        summary = _run_seed(rt, tensor_sha_fn=tensor_from_genesis)
        assert summary["outcome"] == "SEED_COMPLETE"
        for record in _records_of(rt, summary).values():
            svg = record["selected_vs_genesis"]
            assert svg["selected_equals_genesis"] is True
            assert record["promotion_eligible"] is False
            assert "SELECTED_EQUALS_GENESIS" in \
                record["promotion_ineligibility_causes"]

    def test_a_learned_selection_differs_and_is_promotable(self, rt):
        summary = _run_seed(rt)
        for record in _records_of(rt, summary).values():
            svg = record["selected_vs_genesis"]
            assert svg["selected_equals_genesis"] is False
            assert svg["genesis_policy_tensor_sha256"] == \
                _genesis_pin(rt)["policy_tensor_sha256"]
            assert svg["selected_policy_tensor_sha256"] != \
                svg["genesis_policy_tensor_sha256"]
            assert record["promotion_eligible"] is True

    def test_decision_verdict_excludes_a_genesis_equal_selection(
            self, rt, tmp_path):
        records = _v2_decision_records(
            rt.contract, tmp_path, lambda s, c: 0.01,
            equals_genesis={(202, "P1E_LR1E4")})
        payload, _code = p1.decision_verdict(
            rt.contract, records=records,
            replica_proof=_proof_for(rt.contract, records))
        named = {(entry["seed"], entry["cell"]): entry["causes"]
                 for entry in payload["non_promotable_cells"]}
        assert named == {(202, "P1E_LR1E4"):
                         ["SELECTED_EQUALS_GENESIS"]}
        assert payload["per_seed_paired_effects"]["202"][
            "available"] is False


# ---------------------------------------------------------------------------
# Req 11 — test and context-prefix contracts unchanged
# ---------------------------------------------------------------------------

class TestReq11NestedAndContextContractsUnchanged:
    def test_v2_nested_split_contract_equals_v1_byte_for_byte(self):
        v1 = _v1_contract()["nested_split_contract"]
        v2 = _v2_contract()["nested_split_contract"]
        assert v2 == v1

    def test_context_prefix_and_sealed_test_pins(self):
        nested = _v2_contract()["nested_split_contract"]
        assert nested["context_bars"] == 256
        assert nested["role_facts"]["sealed_test"]["status"] == \
            "SEALED"
        assert nested["role_facts"]["sealed_test"]["csv_sha256"] is None
        assert nested["role_facts"]["fit_train"]["scored_rows"] == 11509
        assert nested["role_facts"]["train_monitor"][
            "scored_rows"] == 2190
        assert nested["role_facts"]["inner_validation"][
            "scored_rows"] == 2190
        assert nested["role_facts"]["outer_validation"][
            "scored_rows"] == 2196
        for role in ("train_monitor", "inner_validation",
                     "outer_validation"):
            assert nested["role_facts"][role]["context_rows"] == 256

    def test_v2_binding_verifies_against_the_same_ladder_pins(
            self, bindings):
        contract = _v2_contract()
        binding = p1.verify_nested_split_binding(contract, bindings)
        assert binding["sha256"] == \
            contract["nested_split_contract"]["sha256"]
        assert binding["context_bars"] == 256
        assert contract["selection_metric"] == \
            _v1_contract()["selection_metric"]


# ---------------------------------------------------------------------------
# Req 12 — all 16 terminals load on the replica preserving identity
# ---------------------------------------------------------------------------

def _v2_collect_record(contract, seed, cell, terminal: Path,
                       exp_id: str) -> dict:
    spec = contract["nested_split_contract"]
    record = _v2_verdict_record(contract, seed, cell, "VIABLE",
                                terminal.parent, exp_id=exp_id)
    record.update({
        "mode": "screen",
        "evidence_class": "mechanics_screen",
        "cell_identity": hashlib.sha256(
            f"{seed}:{cell}".encode()).hexdigest()[:16],
        "terminal_model_path": str(terminal),
        "terminal_model_sha256": hashlib.sha256(
            terminal.read_bytes()).hexdigest(),
        "nested_split_manifest_sha256": "d" * 64,
        "nested_role_facts": {
            role: {key: spec["role_facts"][role].get(key)
                   for key in p1.NESTED_ROLE_FACT_KEYS}
            for role in spec["role_facts"]},
    })
    return record


class TestReq12SixteenReplicaTerminalLoads:
    @pytest.fixture()
    def v2world(self, tmp_path):
        exp_id = "feedfacefeedface"
        contract = copy.deepcopy(_v2_contract())
        contract["output_root"] = str(tmp_path / "out")
        source_root = tmp_path / "out" / exp_id
        for seed in p1.SEEDS:
            for cell in p1.CELLS:
                attempt = (source_root / f"seed{seed}" / cell /
                           "attempt-01")
                attempt.mkdir(parents=True)
                terminal = attempt / "model.terminal.zip"
                terminal.write_bytes(
                    f"v2-terminal-{seed}-{cell}".encode())
                record = _v2_collect_record(contract, seed, cell,
                                            terminal, exp_id)
                (attempt.parent / p1c.RECORD_NAME).write_text(
                    json.dumps(record))
        return contract, source_root, exp_id

    def test_sixteen_v2_terminals_seal_replicate_and_pass_the_verdict(
            self, v2world, tmp_path):
        contract, source_root, exp_id = v2world
        manifest = p1c.collect(
            contract=contract, experiment_identity=exp_id,
            collection_root=tmp_path / "collection",
            fetch_fn=_local_fetch, replica_host="dragon",
            replicate_fn=_noop_replicate,
            replica_verify_fn=_echo_verify)
        assert manifest["outcome"] == "COLLECTION_SEALED", \
            manifest["refusals"]
        assert len(manifest["terminals"]) == 16
        proof = json.loads(
            Path(manifest["replica_proof_file"]).read_text())
        assert proof["schema"] == p1.REPLICA_PROOF_SCHEMA
        assert len(proof["proofs"]) == 16
        for entry in proof["proofs"]:
            assert entry["loads"] is True
            assert entry["experiment_identity"] == exp_id
            assert entry["contract_sha256"] == \
                contract["_contract_sha256"]

        sealed_root = tmp_path / "collection" / "sealed" / exp_id
        records = {}
        for seed in p1.SEEDS:
            for cell in p1.CELLS:
                records[(seed, cell)] = json.loads(
                    (sealed_root / f"seed{seed}" / cell /
                     p1c.RECORD_NAME).read_text())
        ok, refusals, _facts = p1.validate_replica_proof(
            proof, contract=contract, records=records)
        assert (ok, refusals) == (True, [])
        payload, code = p1.screen_verdict(contract, records=records,
                                          replica_proof=proof)
        assert payload["outcome"] == "SCREEN_VIABLE_REGION"
        assert payload["gates"]["replica_terminal_loads"] is True
        assert payload["gates"]["v2_selected_policy_evidence"] is True
        assert code == 0
        # identity preserved: every proof entry echoes the exact
        # terminal sha + relative path the record binds
        by_key = {(entry["seed"], entry["cell"]): entry
                  for entry in proof["proofs"]}
        for key, record in records.items():
            entry = by_key[key]
            assert entry["terminal_model_sha256"] == \
                record["terminal_model_sha256"]
            assert entry["terminal_relative_path"] == \
                p1.expected_terminal_relative(record)

    def test_a_v1_schema_record_cannot_enter_a_v2_collection(
            self, v2world, tmp_path):
        contract, source_root, exp_id = v2world
        rec_path = (source_root / "seed303" / "P1N_LR3E5" /
                    p1c.RECORD_NAME)
        record = json.loads(rec_path.read_text())
        record["schema"] = p1.RECORD_SCHEMA
        rec_path.write_text(json.dumps(record))
        manifest = p1c.collect(
            contract=contract, experiment_identity=exp_id,
            collection_root=tmp_path / "collection",
            fetch_fn=_local_fetch, replica_host="dragon",
            replicate_fn=_noop_replicate,
            replica_verify_fn=_echo_verify)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("wrong schema" in r for r in manifest["refusals"])

    def test_a_stripped_v2_record_refuses_at_the_verdict(
            self, v2world, tmp_path):
        contract, source_root, exp_id = v2world
        records = {}
        for seed in p1.SEEDS:
            for cell in p1.CELLS:
                records[(seed, cell)] = json.loads(
                    (source_root / f"seed{seed}" / cell /
                     p1c.RECORD_NAME).read_text())
        del records[(404, "P1E_LR3E5")]["selected_vs_genesis"]
        payload, code = p1.screen_verdict(
            contract, records=records,
            replica_proof=_proof_for(contract, records))
        assert payload["outcome"] == "SCREEN_REFUSED"
        assert code == 4
        assert payload["gates"]["v2_selected_policy_evidence"] is False
        assert any("finding 235" in r for r in payload["reasons"])


# ---------------------------------------------------------------------------
# v1/v2 separation: replayability and disjoint identities/roots
# ---------------------------------------------------------------------------

class TestV1V2Separation:
    def test_v2_identities_never_collide_with_v1(self, bindings):
        v1 = _v1_contract()
        v2 = _v2_contract()
        for mode in p1.MODES:
            id_v1 = p1.experiment_identity(v1, bindings,
                                           sources=CLEAN_SOURCES,
                                           mode=mode)
            id_v2 = p1.experiment_identity(v2, bindings,
                                           sources=CLEAN_SOURCES,
                                           mode=mode)
            assert id_v1 != id_v2
        assert v2["output_root"] != v1["output_root"]
        assert v2["decision_run"]["output_root"] != \
            v1["decision_run"]["output_root"]
        assert "20260815_v2" in v2["output_root"]
        assert v2["replica"]["collection_root"] not in (
            v1.get("replica", {}).get("collection_root"),
            v1["output_root"], v1["decision_run"]["output_root"])

    def test_v1_contract_still_loads_with_v1_semantics(self):
        v1 = _v1_contract()
        assert p1.contract_version(v1) == 1
        assert p1.record_schema_for(v1) == p1.RECORD_SCHEMA
        init = p1.initialization_binding(v1, 101)
        assert init["kind"] == "anchor"
        assert init["container_sha256"] == \
            v1["anchors"]["101"]["sha256"]

    def test_v2_seed_gpu_assignments_and_latin_square_match_v1(self):
        v1 = _v1_contract()
        v2 = _v2_contract()
        assert v2["assignments"] == v1["assignments"]
        assert v2["cell_order"] == v1["cell_order"]
        assert v2["cells"] == v1["cells"]
        assert v2["factors"] == v1["factors"]

    def test_v2_decision_mode_runs_under_the_shipped_stopping_contract(
            self, rt):
        summary = _run_seed(
            rt, mode="decision",
            screen_gate=_viable_screen_gate(rt.contract),
            outer_eval_fn=_fake_outer_eval)
        assert summary["outcome"] == "SEED_COMPLETE"
        records = _records_of(rt, summary, mode="decision")
        for record in records.values():
            assert record["mode"] == "decision"
            assert record["outer_validation_final"][
                "evaluated_artifact_role"] == "best_checkpoint"
            assert record["promotion_eligible"] is True

    def test_inactive_v2_cell_is_still_a_measured_outcome(self, rt):
        order = rt.contract["cell_order"]["101"]
        summary = _run_seed(
            rt, pipeline_factory=_v2_inactive_factory(1))
        assert summary["outcome"] == "SEED_COMPLETE_WITH_INACTIVE"
        assert summary["inactive_cells"] == [order[1]]
        record = _records_of(rt, summary)[order[1]]
        assert record["activity_status"] == "inactive"
        assert record["promotion_eligible"] is False
        assert record["selected_vs_genesis"][
            "selected_artifact_role"] == "terminal"
        assert record["genesis_policy_tensor_sha256"] == \
            _genesis_pin(rt)["policy_tensor_sha256"]

    def test_v2_record_reuse_requires_the_v2_evidence(self, rt):
        summary = _run_seed(rt)
        assert summary["outcome"] == "SEED_COMPLETE"
        cell = rt.contract["cell_order"]["101"][0]
        root = p1.output_root_for_mode(rt.contract, "screen")
        record_path = (root / summary["experiment_identity"] /
                       "seed101" / cell / "cell_record.json")
        record = json.loads(record_path.read_text())
        del record["actor_liveness_binding"]
        record_path.write_text(json.dumps(record))
        again = _run_seed(rt)
        assert again["outcome"] == "SEED_FAILED"
        assert "refusing to overwrite" in \
            again["cells"][cell]["error"]
