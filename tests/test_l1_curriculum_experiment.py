"""P1 curriculum driver contract tests (orders 2026-08-23)."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "l1_curriculum_experiment",
        REPO / "tools" / "l1_curriculum_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load()


def _trace(tmp_path, raws):
    p = tmp_path / "validation_epoch_return_trace.csv"
    p.write_text("timestamp,action_raw\n" + "\n".join(
        f"2026-01-0{i%9+1},{v}" for i, v in enumerate(raws)))
    return p


class TestNormalCrossings:
    def test_zigzag_counts_each_side_change(self, tool, tmp_path):
        p = _trace(tmp_path, [0.5, -0.5, 0.5, -0.5])
        assert tool.count_normal_crossings(p, 0.1) == 4

    def test_subthreshold_movement_never_counts(self, tool, tmp_path):
        p = _trace(tmp_path, [0.05, -0.05, 0.08, -0.02])
        assert tool.count_normal_crossings(p, 0.1) == 0

    def test_monotone_single_entry_counts_once(self, tool, tmp_path):
        p = _trace(tmp_path, [0.5, 0.6, 0.7, 0.8])
        assert tool.count_normal_crossings(p, 0.1) == 1


class TestHandoffGate:
    def test_unaccepted_easy_refuses_with_activity_note(self, tool,
                                                        tmp_path):
        with pytest.raises(tool.CurriculumError,
                           match="never rejects easy"):
            tool.verify_handoff({"accepted": False}, tmp_path, 0.0)

    def test_missing_checkpoint_refuses(self, tool, tmp_path):
        with pytest.raises(tool.CurriculumError, match="eligible"):
            tool.verify_handoff({"accepted": True,
                                 "selected_checkpoint": None},
                                tmp_path, 0.0)

    def test_insufficient_crossings_refuse(self, tool, tmp_path):
        model = tmp_path / "best_model.zip"
        import zipfile
        with zipfile.ZipFile(model, "w") as zf:
            zf.writestr("policy.pth", b"x")
        (tmp_path / "traces").mkdir()
        (tmp_path / "traces" /
         "validation_epoch_return_trace.csv").write_text(
            "timestamp,action_raw\n2026-01-01,0.5\n2026-01-02,0.6\n")
        with pytest.raises(tool.CurriculumError, match="crossings"):
            tool.verify_handoff(
                {"accepted": True,
                 "selected_checkpoint": str(model)}, tmp_path, 0.1)


class TestContinuity:
    def _reports(self, *, easy_l1=100.0, normal_l1=100.0, mode="fresh",
                 loaded=0):
        easy = {"history": [
            {"epoch": 1, "checkpoint_improved": True,
             "policy_actor_l1_after": easy_l1}]}
        normal = {"history": [
            {"epoch": 1, "policy_actor_l1_before": normal_l1}],
            "replay_disposition": {"mode": mode,
                                   "loaded_transitions": loaded}}
        handoff = {"artifact_sha256": "a" * 64}
        return easy, normal, handoff

    def test_exact_continuity_passes(self, tool):
        e, n, h = self._reports()
        out = tool.verify_continuity(e, n, h, "EN-W")
        assert out["actor_l1_continuity"]["identical"] is True

    def test_tensor_drift_refuses(self, tool):
        e, n, h = self._reports(normal_l1=100.001)
        with pytest.raises(tool.CurriculumError, match="continuity"):
            tool.verify_continuity(e, n, h, "EN-W")

    def test_enw_with_carried_replay_refuses(self, tool):
        e, n, h = self._reports(mode="full_continuity", loaded=500)
        with pytest.raises(tool.CurriculumError, match="declared"):
            tool.verify_continuity(e, n, h, "EN-W")

    def test_enf_with_fresh_replay_refuses(self, tool):
        e, n, h = self._reports(mode="fresh", loaded=0)
        with pytest.raises(tool.CurriculumError, match="declared"):
            tool.verify_continuity(e, n, h, "EN-F")

    def test_enf_empty_carried_buffer_refuses(self, tool):
        e, n, h = self._reports(mode="full_continuity", loaded=0)
        with pytest.raises(tool.CurriculumError, match="empty replay"):
            tool.verify_continuity(e, n, h, "EN-F")

    def test_baseline_row_without_l1_is_skipped(self, tool):
        e, n, h = self._reports()
        n["history"].insert(0, {"baseline": True})
        out = tool.verify_continuity(e, n, h, "EN-W")
        assert out["actor_l1_continuity"]["identical"] is True


class TestDeclarations:
    def test_predeclared_rule_and_flat_mlp_identity(self, tool):
        src = (REPO / "tools" / "l1_curriculum_experiment.py"
               ).read_text()
        assert "PREDECLARED DIRECTION RULE" in src
        assert "never merged" in src.lower() or "NEVER merged" in src
        assert "flat_mlp" in src
        assert "feature_extractor_plugin" in src  # the NEVER-set note
