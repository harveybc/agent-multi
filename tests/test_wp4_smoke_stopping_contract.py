"""Regression: smoke stopping semantics are never derived from budgets.

MUSASHI_CORRECTION_SMOKE_PATIENCE_WAS_UNAUTHORIZED_2026_08_21: the tool
once computed ``l1_patience = max(2, max_epochs // 5)`` with start epoch
0 — an unauthorized scientific parameter invented from a runtime budget.
These tests pin the corrected contract: both values are explicit,
required, and invariant under max_epochs.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "wp4_cpu_smoke", REPO / "tools" / "wp4_cpu_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def _args(tool, **over):
    ns = argparse.Namespace(
        device="cpu", epoch_timesteps=512, max_epochs=3, seed=101,
        output_dir=Path("/tmp/x"), report=None, preflight=False,
        l1_patience=60, l1_patience_start_epoch=40,
        selection_metric="episodic_activity_economic_v1",
        plateau_lr_json=None)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


class TestStoppingContractNotDerived:
    def test_max_epochs_changes_neither_patience_field(self, tool):
        features = ["f1", "f2"]
        for budget in (3, 10, 50, 2000):
            cfg = tool.build_config(_args(tool, max_epochs=budget),
                                    features)
            assert cfg["l1_patience"] == 60, budget
            assert cfg["l1_patience_start_epoch"] == 40, budget

    def test_explicit_values_pass_through_verbatim(self, tool):
        cfg = tool.build_config(
            _args(tool, l1_patience=7, l1_patience_start_epoch=2),
            ["f1"])
        assert cfg["l1_patience"] == 7
        assert cfg["l1_patience_start_epoch"] == 2

    def test_no_budget_derivation_expression_remains(self, tool):
        src = (REPO / "tools" / "wp4_cpu_smoke.py").read_text()
        assert "max_epochs // 5" not in src

    def test_missing_patience_flags_refuse(self, tool, capsys):
        with pytest.raises(SystemExit) as exc:
            tool.main(["--device", "cpu", "--max-epochs", "3"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "--l1-patience" in err

    def test_missing_start_epoch_refuses(self, tool, capsys):
        with pytest.raises(SystemExit) as exc:
            tool.main(["--device", "cpu", "--l1-patience", "60"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "--l1-patience-start-epoch" in err


class TestGpuUuidProvenance:
    """Replication defect 2026-08-21: seed 404 ran on the 5090 (torch
    name + CVD mask agree) but the report attributed the 5070 Ti UUID
    from host index 0. The UUID must come from the mask when the mask
    names one GPU-UUID; ambiguous host enumeration reports None."""

    def test_mask_uuid_is_authoritative(self, tool, monkeypatch):
        import types
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda i: "NVIDIA GeForce RTX 5090"))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setenv(
            "CUDA_VISIBLE_DEVICES",
            "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8")
        facts = tool.resolve_device("cuda")
        assert facts["gpu_uuid"] == (
            "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8")
        assert facts["gpu_uuid_provenance"] == "cuda_visible_devices_mask"

    def test_multi_gpu_without_mask_is_unresolved_not_wrong(
            self, tool, monkeypatch):
        import types
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda i: "X"))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        fake_result = types.SimpleNamespace(
            stdout="GPU-aaa\nGPU-bbb\n")
        monkeypatch.setattr(tool.subprocess, "run",
                            lambda *a, **k: fake_result)
        facts = tool.resolve_device("cuda")
        assert facts["gpu_uuid"] is None
        assert facts["gpu_uuid_provenance"] == (
            "ambiguous_multi_gpu_unresolved")


class TestPlrScreenLabels:
    """AUD-F1-20260821-PLR-02/PLR-04 regressions."""

    def test_no_long_horizon_label_exists(self, tool):
        src = (REPO / "tools" / "wp4_cpu_smoke.py").read_text()
        assert "long_horizon_contract" not in src
        assert "BOUNDED_120_40_40_DAY_SCHEDULER_SCREEN" in src

    def test_holdout_is_named_diagnostic_not_test(self, tool):
        src = (REPO / "tools" / "wp4_cpu_smoke.py").read_text()
        assert "internal_test_split" not in src
        assert '"diagnostic_holdout"' in src
        assert "not_the_sealed_2025_test" in src

    def test_facts_from_maps_test_trace_to_diagnostic_holdout(
            self, tool, tmp_path):
        trace = tmp_path / "test_return_trace.csv"
        trace.write_text(
            "timestamp,split,action_raw,closed_trades_cumulative\n"
            "2018-03-07 04:00:00,test,0.1,0\n"
            "2018-04-16 00:00:00,test,0.2,1\n")
        facts = tool.facts_from([], tmp_path)
        assert "diagnostic_holdout" in facts["traces"]
        assert "test" not in facts["traces"]
        assert facts["traces"]["diagnostic_holdout"]["file"].endswith(
            "test_return_trace.csv")
