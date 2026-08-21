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
