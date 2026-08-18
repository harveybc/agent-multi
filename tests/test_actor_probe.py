"""WP-C focused and adversarial tests for tools/p1lr_actor_probe.py.

The order's eight preconditions for committing the probe. These pin the
five that are testable without a GPU checkpoint (refusals, validation,
custody and exit semantics); the observation-binding and delta-splitting
paths are exercised against a real checkpoint in the acceptance packet.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "p1lr_actor_probe", REPO / "tools" / "p1lr_actor_probe.py")
probe_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe_mod)


class TestSealedRefusal:
    @pytest.mark.parametrize("path", [
        "/data/sealed_test_2025/model.zip",
        "/data/f9379f596e80fda4/sealed/model.zip",
        "/runs/2025/model.zip",
    ])
    def test_sealed_artifact_refused_on_any_path_component(self, path):
        with pytest.raises(probe_mod.ProbeRefusal,
                           match="REFUSED_SEALED_ARTIFACT"):
            probe_mod.assert_not_sealed(Path(path))

    def test_ordinary_path_allowed(self, tmp_path):
        probe_mod.assert_not_sealed(tmp_path / "model.zip")


class TestOutputRefusal:
    def test_output_inside_campaign_identity_refused(self):
        with pytest.raises(probe_mod.ProbeRefusal,
                           match="REFUSED_OUTPUT_INSIDE_IDENTITY"):
            probe_mod.assert_output_outside_identity(
                Path("/x/f9379f596e80fda4/probe.json"))

    def test_diagnostic_root_allowed(self, tmp_path):
        probe_mod.assert_output_outside_identity(tmp_path / "probe.json")


class TestCountValidation:
    @pytest.mark.parametrize("bad", [0, 1, -5, "x", None,
                                     probe_mod.MAX_DRAWS + 1])
    def test_invalid_draws_refused(self, bad):
        with pytest.raises(probe_mod.ProbeRefusal):
            probe_mod._validate_count("draws", bad, probe_mod.MAX_DRAWS)

    def test_valid_counts_accepted(self):
        assert probe_mod._validate_count("grid", 41,
                                         probe_mod.MAX_GRID) == 41


class TestNonFiniteRefusal:
    @pytest.mark.parametrize("values", [[float("nan")], [float("inf")],
                                        [0.1, float("-inf")]])
    def test_nonfinite_probe_output_refused(self, values):
        with pytest.raises(probe_mod.ProbeRefusal,
                           match="REFUSED_NONFINITE_OUTPUT"):
            probe_mod._stats("test", values)

    def test_finite_output_summarised(self):
        stats = probe_mod._stats("test", [0.1, 0.2, 0.3])
        assert stats["count"] == 3 and stats["unique_count"] == 3


class TestObservationLoading:
    def test_wrong_shape_refused(self, tmp_path):
        path = tmp_path / "obs.csv"
        path.write_text("1,2,3\n4,5,6\n")
        with pytest.raises(probe_mod.ProbeRefusal,
                           match="REFUSED_OBSERVATION_SHAPE"):
            probe_mod.load_observations(path, dim=7)

    def test_correct_shape_loads(self, tmp_path):
        path = tmp_path / "obs.csv"
        path.write_text("1,2,3\n4,5,6\n")
        arr = probe_mod.load_observations(path, dim=3)
        assert arr.shape == (2, 3)

    def test_sealed_observations_refused(self, tmp_path):
        sealed = tmp_path / "sealed_test_2025"
        sealed.mkdir()
        path = sealed / "obs.csv"
        path.write_text("1,2,3\n")
        with pytest.raises(probe_mod.ProbeRefusal,
                           match="REFUSED_SEALED_ARTIFACT"):
            probe_mod.load_observations(path, dim=3)


class TestExitSemantics:
    def test_no_model_probed_returns_non_zero(self, tmp_path, capsys):
        code = probe_mod.main([
            "--model", str(tmp_path / "absent.zip"),
            "--out", str(tmp_path / "probe.json")])
        assert code == 1
        printed = json.loads(capsys.readouterr().out)
        assert printed["outcome"] == "NO_MODEL_PROBED"
        assert printed["probed"] == 0 and printed["refused"] == 1

    def test_invalid_grid_refuses_before_any_load(self, tmp_path,
                                                  capsys):
        code = probe_mod.main([
            "--model", str(tmp_path / "absent.zip"),
            "--out", str(tmp_path / "probe.json"), "--grid", "1"])
        assert code == 2
        assert "REFUSED_INVALID_GRID" in capsys.readouterr().out

    def test_output_digest_is_published(self, tmp_path, capsys):
        probe_mod.main(["--model", str(tmp_path / "absent.zip"),
                        "--out", str(tmp_path / "probe.json")])
        printed = json.loads(capsys.readouterr().out)
        assert len(printed["digest"]) == 64
