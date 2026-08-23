"""P1 curriculum driver v2 tests + 307-312 mandatory counterexamples."""
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load("l1_curriculum_experiment",
                 REPO / "tools" / "l1_curriculum_experiment.py")


@pytest.fixture(scope="module")
def bundle_mod():
    import sys
    sys.path.insert(0, str(REPO))
    from pipeline_plugins import _checkpoint_bundle
    return _checkpoint_bundle


def _trace(path, raws):
    path.write_text("timestamp,action_raw\n" + "\n".join(
        f"2026-01-0{i%9+1},{v}" for i, v in enumerate(raws)))
    return path


def _bundle(tmp_path, *, epoch=2, crossing_raws=(0.5, -0.5, 0.5),
            state=None):
    model = tmp_path / "best_model.zip"
    with zipfile.ZipFile(model, "w") as zf:
        zf.writestr("policy.pth", b"weights")
    trace = _trace(tmp_path / "selected_inner_validation_trace.csv",
                   crossing_raws)
    import hashlib
    doc = {"schema": "agent_multi.selected_checkpoint_bundle.v1",
           "epoch": epoch,
           "model": {"path": str(model),
                     "sha256": hashlib.sha256(
                         model.read_bytes()).hexdigest()},
           "named_state_sha256": state or {"policy.w": "a" * 64},
           "replay": {"path": str(tmp_path / "r.pkl"),
                      "sha256": "b" * 64, "size": 100},
           "traces": {"inner_validation": {
               "path": str(trace),
               "sha256": hashlib.sha256(
                   trace.read_bytes()).hexdigest()}},
           "_manifest_path": str(tmp_path / "m.json")}
    return doc


class TestNormalCrossings:
    def test_zigzag_counts_each_side_change(self, tool, tmp_path):
        p = _trace(tmp_path / "t.csv", [0.5, -0.5, 0.5, -0.5])
        assert tool.count_normal_crossings(p, 0.1) == 4

    def test_subthreshold_never_counts(self, tool, tmp_path):
        p = _trace(tmp_path / "t.csv", [0.05, -0.05, 0.08])
        assert tool.count_normal_crossings(p, 0.1) == 0


class TestHandoffBundleAuthority:
    """Finding 307: handoff consumes ONLY the selected bundle."""

    def test_valid_bundle_authorizes(self, tool, tmp_path):
        b = _bundle(tmp_path)
        out = tool.verify_handoff({"accepted": True}, b, 0.1)
        assert out["bundle_epoch"] == 2
        assert out["validation_crossings"] == 3

    def test_drifted_snapshot_trace_refuses(self, tool, tmp_path):
        """Counterexample: selected checkpoint at epoch 3 with a
        terminal-epoch trace substituted — the sha binding refuses."""
        b = _bundle(tmp_path, epoch=3)
        _trace(Path(b["traces"]["inner_validation"]["path"]),
               [0.9, -0.9] * 10)  # overwritten by a later epoch
        with pytest.raises(tool.CurriculumError, match="drifted"):
            tool.verify_handoff({"accepted": True}, b, 0.1)

    def test_insufficient_crossings_refuse(self, tool, tmp_path):
        b = _bundle(tmp_path, crossing_raws=(0.5, 0.6, 0.7))
        with pytest.raises(tool.CurriculumError, match="crossings"):
            tool.verify_handoff({"accepted": True}, b, 0.1)

    def test_unaccepted_easy_refuses_with_activity_note(self, tool,
                                                        tmp_path):
        with pytest.raises(tool.CurriculumError,
                           match="never rejects easy"):
            tool.verify_handoff({"accepted": False},
                                _bundle(tmp_path), 0.1)


class TestContinuityV2:
    def _normal(self, *, mode, epoch=2, loaded=0, exact=True,
                tensors=148):
        return {"replay_disposition": {
            "mode": mode, "bundle_epoch": epoch,
            "loaded_transitions": loaded,
            "state_verification": {"exact": exact,
                                   "tensors_verified": tensors}}}

    def _handoff(self, epoch=2):
        return {"bundle_epoch": epoch}

    def test_enw_exact(self, tool):
        out = tool.verify_continuity(
            self._normal(mode="fresh"), self._handoff(), "EN-W")
        assert out["named_state_verified_exact"] is True

    def test_epoch_mismatch_refuses(self, tool):
        """Counterexample: selected model and replay from different
        epochs — the binding refuses."""
        with pytest.raises(tool.CurriculumError, match="different epoch"):
            tool.verify_continuity(
                self._normal(mode="selected_epoch_full_continuity",
                             epoch=10, loaded=500),
                self._handoff(epoch=3), "EN-F")

    def test_unverified_state_refuses(self, tool):
        with pytest.raises(tool.CurriculumError, match="309"):
            tool.verify_continuity(
                self._normal(mode="fresh", exact=False),
                self._handoff(), "EN-W")

    def test_wrong_replay_semantics_refuse(self, tool):
        with pytest.raises(tool.CurriculumError, match="declared"):
            tool.verify_continuity(
                self._normal(mode="selected_epoch_full_continuity",
                             loaded=500),
                self._handoff(), "EN-W")

    def test_enf_empty_buffer_refuses(self, tool):
        with pytest.raises(tool.CurriculumError, match="empty"):
            tool.verify_continuity(
                self._normal(mode="selected_epoch_full_continuity",
                             loaded=0), self._handoff(), "EN-F")


class TestExactStateHashing:
    """Finding 309 counterexamples at the digest level."""

    def test_equal_l1_different_tensors_have_different_digests(
            self, bundle_mod):
        import torch
        a = torch.tensor([2.0, 0.0])
        b = torch.tensor([1.0, 1.0])
        assert float(a.abs().sum()) == float(b.abs().sum())
        assert (bundle_mod._tensor_digest("w", a)
                != bundle_mod._tensor_digest("w", b))

    def test_actor_exact_but_optimizer_changed_refuses(self,
                                                       bundle_mod,
                                                       monkeypatch):
        manifest = {"named_state_sha256": {
            "policy.actor.w": "a" * 64,
            "critic.optimizer.state.0.exp_avg": "b" * 64}}
        monkeypatch.setattr(
            bundle_mod, "named_state_hashes",
            lambda m: {"policy.actor.w": "a" * 64,
                       "critic.optimizer.state.0.exp_avg": "X" * 64})
        with pytest.raises(bundle_mod.BundleError, match="changed"):
            bundle_mod.verify_loaded_model(object(), manifest)

    def test_missing_optimizer_state_refuses(self, bundle_mod,
                                             monkeypatch):
        manifest = {"named_state_sha256": {
            "policy.actor.w": "a" * 64,
            "actor.optimizer.state.0.exp_avg": "b" * 64}}
        monkeypatch.setattr(
            bundle_mod, "named_state_hashes",
            lambda m: {"policy.actor.w": "a" * 64})
        with pytest.raises(bundle_mod.BundleError, match="missing"):
            bundle_mod.verify_loaded_model(object(), manifest)

    def test_dtype_and_shape_frame_the_digest(self, bundle_mod):
        import torch
        a = torch.tensor([1.0, 2.0], dtype=torch.float32)
        b = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        assert (bundle_mod._tensor_digest("w", a)
                != bundle_mod._tensor_digest("w", b))


class TestArmIdentity:
    """Finding 312 counterexamples."""

    _BASE = {"feature_columns": ["a", "b", "c"], "learning_rate": 3e-4,
             "continuous_action_threshold": 0.0, "seed": 101,
             "epoch_timesteps": 512, "solvency_mode": "x",
             "output_dir": "/x/N"}

    def test_allowlisted_factors_keep_identity(self, tool):
        records = []
        for arm, solvency in (("N", "normal_realistic"),
                              ("EN-W", "normal_realistic")):
            cfg = dict(self._BASE, solvency_mode=solvency,
                       output_dir=f"/x/{arm}",
                       warm_start_bundle=f"/x/{arm}/m.json")
            records.append({"contracts": tool.arm_contracts(cfg, arm)})
        tool.verify_arm_identity(records)

    @pytest.mark.parametrize("mutation", [
        {"feature_columns": ["b", "a", "c"]},   # reordered features
        {"continuous_action_threshold": 0.1},   # changed action map
        {"learning_rate": 1e-4},
        {"epoch_timesteps": 1024},
    ])
    def test_undeclared_difference_refuses(self, tool, mutation):
        r1 = {"contracts": tool.arm_contracts(dict(self._BASE), "N")}
        r2 = {"contracts": tool.arm_contracts(
            dict(self._BASE, **mutation), "EN-W")}
        with pytest.raises(tool.CurriculumError, match="identity"):
            tool.verify_arm_identity([r1, r2])


class TestDataContractRefusals:
    """Finding 311 counterexamples."""

    def test_driver_has_no_day_flags(self, tool):
        src = (REPO / "tools" / "l1_curriculum_experiment.py"
               ).read_text()
        assert "--train-days" not in src.replace(
            "day splits refuse structurally", "")
        assert "--nested-contract" in src

    def test_runner_refuses_days_with_nested(self, tmp_path, capsys):
        runner = _load("wp4_cpu_smoke",
                       REPO / "tools" / "wp4_cpu_smoke.py")
        rc = runner.main([
            "--device", "cpu", "--l1-patience", "2",
            "--l1-patience-start-epoch", "0",
            "--nested-contract", str(tmp_path / "c.json"),
            "--train-days", "120"])
        assert rc == 2
        assert "REFUSED_DAY_SPLITS_WITH_NESTED" in (
            capsys.readouterr().out)

    def test_sealed_materialized_refuses_outer_endpoint(self, tool,
                                                        tmp_path):
        normal_dir = tmp_path / "normal"
        (normal_dir / "nested_splits").mkdir(parents=True)
        launch = tmp_path / "normal_report.launch_manifest.json"
        launch.write_text(json.dumps({"effective_config": {}}))
        (normal_dir / "nested_splits" /
         "nested_split_manifest.json").write_text(json.dumps({
            "roles": {"outer_validation": {"status": "MATERIALIZED",
                                           "csv": "x", "rows": 1},
                      "sealed_test": {"status": "MATERIALIZED"}}}))
        with pytest.raises(tool.CurriculumError, match="sealed"):
            tool.outer_endpoint(normal_dir, {})

    def test_outer_runs_only_after_terminal_phase(self, tool):
        """Counterexample: outer metrics cannot reach checkpoint or
        stopping — structurally, the driver calls outer_endpoint only
        after the normal phase subprocess exited and its report was
        loaded."""
        src = (REPO / "tools" / "l1_curriculum_experiment.py"
               ).read_text()
        i_report = src.index('normal_report = json.loads')
        i_outer = src.index('record["outer_endpoint"]')
        assert i_report < i_outer
        assert "evaluated_after_phase_terminal" in src


class TestDeclarations:
    def test_predeclared_rule_and_flat_mlp_identity(self, tool):
        src = (REPO / "tools" / "l1_curriculum_experiment.py"
               ).read_text()
        assert "PREDECLARED DIRECTION RULE" in src
        assert "never merged" in src.lower()
        assert "flat_mlp" in src


class TestTreatmentDivergence:
    def test_identical_maps_flag_inert_treatment(self, tool):
        m = {"named_state_sha256": {"a": "1", "b": "2"}}
        out = tool.treatment_divergence(m, m)
        assert out["easy_treatment_diverged"] is False
        assert out["identical_tensors"] == 2

    def test_any_divergence_marks_active_treatment(self, tool):
        a = {"named_state_sha256": {"a": "1", "b": "2"}}
        b = {"named_state_sha256": {"a": "1", "b": "X"}}
        assert tool.treatment_divergence(a, b)[
            "easy_treatment_diverged"] is True
