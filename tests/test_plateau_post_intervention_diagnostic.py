"""Adversarial tests for the WP1 post-intervention diagnostic (2026-08-22)."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "plateau_post_intervention_diagnostic",
        REPO / "tools" / "plateau_post_intervention_diagnostic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load()


def _hist(composites, *, reduced_at=(), extra=None):
    rows = []
    for i, c in enumerate(composites, start=1):
        row = {"epoch": i, "composite": c,
               "val_total_return": 0.01, "val_max_drawdown_fraction":
               0.02, "val_trades": 10, "train_tail_trades": 5,
               "val_action_raw_std": 0.1,
               "val_action_non_hold_rate": 0.5,
               "policy_actor_delta": 1.0, "policy_critic_delta": 2.0,
               "plateau_lr": ({"reduced": True, "old_lr": 3e-4,
                               "new_lr": 1.5e-4}
                              if i in reduced_at else None)}
        if extra:
            row.update(extra.get(i, {}))
        rows.append(row)
    return rows


def _write(path, hist, *, seed=101, data_sha="d" * 64):
    path.write_text(json.dumps({
        "history": hist, "budgets": {"seed": seed},
        "data_sha256": data_sha}))


def _pair(tmp_path, fixed_c, plateau_c, reduced_at=(5,), **kw):
    f = tmp_path / "seed101_fixed_report.json"
    p = tmp_path / "seed101_plateau_report.json"
    _write(f, _hist(fixed_c), **kw)
    _write(p, _hist(plateau_c, reduced_at=reduced_at), **kw)
    return f, p


class TestDiagnosis:
    def test_matches_hand_computation(self, tool, tmp_path):
        fixed = [0.1, 0.2, 0.3, 0.3, 0.3, 0.25, 0.20, 0.15]
        plateau = [0.1, 0.2, 0.3, 0.3, 0.3, 0.28, 0.10, 0.05]
        f, p = _pair(tmp_path, fixed, plateau)
        d = tool.diagnose_seed(f, p, 101)
        assert d["first_reduction_epoch"] == 5
        assert d["aligned_window_epochs"] == [6, 8]
        assert d["prefix_identical"] is True
        assert d["best_post_delta"] == pytest.approx(0.28 - 0.25)
        assert d["terminal_delta"] == pytest.approx(0.05 - 0.15)
        # trapezoid over deltas [0.03, -0.10, -0.10]
        assert d["auc_delta"] == pytest.approx(
            (0.03 + -0.10) / 2 + (-0.10 + -0.10) / 2)
        assert d["label"] == "POST_HOC_EXPLORATORY"

    def test_changed_prefix_refuses(self, tool, tmp_path):
        f, p = _pair(tmp_path,
                     [0.1, 0.2, 0.3, 0.3, 0.3, 0.2],
                     [0.1, 0.2, 0.31, 0.3, 0.3, 0.2])
        with pytest.raises(tool.DiagnosticError, match="prefix"):
            tool.diagnose_seed(f, p, 101)

    def test_off_by_one_epoch_numbering_refuses(self, tool, tmp_path):
        f, p = _pair(tmp_path, [0.1] * 6, [0.1] * 6)
        doc = json.loads(p.read_text())
        doc["history"][3]["epoch"] = 5  # gap: ...3,5,5,6
        p.write_text(json.dumps(doc))
        with pytest.raises(tool.DiagnosticError, match="off-by-one"):
            tool.diagnose_seed(f, p, 101)

    def test_unequal_history_lengths_align_explicitly(self, tool,
                                                     tmp_path):
        f, p = _pair(tmp_path, [0.1] * 10, [0.1] * 7)
        d = tool.diagnose_seed(f, p, 101)
        assert d["aligned_window_epochs"] == [6, 7]
        assert d["unaligned_tail_epochs"] == {"fixed": 3, "plateau": 0}

    def test_nan_monitor_fact_refuses(self, tool, tmp_path):
        plateau = [0.1, 0.1, 0.1, 0.1, 0.1, float("nan"), 0.1]
        f, p = _pair(tmp_path, [0.1] * 7, plateau)
        with pytest.raises(tool.DiagnosticError, match="finite"):
            tool.diagnose_seed(f, p, 101)

    def test_missing_optional_fact_is_unavailable_not_zero(self, tool,
                                                           tmp_path):
        f = tmp_path / "seed101_fixed_report.json"
        p = tmp_path / "seed101_plateau_report.json"
        hf = _hist([0.1] * 7)
        hp = _hist([0.1] * 7, reduced_at=(5,))
        for row in hf + hp:
            del row["val_action_raw_std"]
        _write(f, hf); _write(p, hp)
        d = tool.diagnose_seed(f, p, 101)
        assert d["terminal_fact_deltas"]["val_action_raw_std"] == (
            "unavailable")
        assert d["terminal_fact_deltas"]["val_trades"] == 0

    def test_mismatched_seed_identity_refuses(self, tool, tmp_path):
        f = tmp_path / "seed101_fixed_report.json"
        _write(f, _hist([0.1] * 3), seed=202)
        with pytest.raises(tool.DiagnosticError, match="mismatched"):
            tool._load_history(f, 101)

    def test_mismatched_data_sha_refuses(self, tool, tmp_path):
        f, p = _pair(tmp_path, [0.1] * 6, [0.1] * 6)
        doc = json.loads(p.read_text())
        doc["data_sha256"] = "e" * 64
        p.write_text(json.dumps(doc))
        with pytest.raises(tool.DiagnosticError, match="data_sha256"):
            tool.diagnose_seed(f, p, 101)

    def test_reduction_in_fixed_arm_refuses(self, tool, tmp_path):
        f = tmp_path / "seed101_fixed_report.json"
        p = tmp_path / "seed101_plateau_report.json"
        _write(f, _hist([0.1] * 6, reduced_at=(4,)))
        _write(p, _hist([0.1] * 6, reduced_at=(5,)))
        with pytest.raises(tool.DiagnosticError, match="fixed arm"):
            tool.diagnose_seed(f, p, 101)

    def test_no_reduction_is_typed_not_fabricated(self, tool, tmp_path):
        f, p = _pair(tmp_path, [0.1] * 6, [0.1] * 6, reduced_at=())
        d = tool.diagnose_seed(f, p, 101)
        assert "no LR reduction" in d["intervention"]
        assert "best_post_delta" not in d

    def test_no_promotion_authority_in_output(self, tool, tmp_path):
        for s in (101, 202, 303, 404):
            _write(tmp_path / f"seed{s}_fixed_report.json",
                   _hist([0.1] * 7), seed=s)
            _write(tmp_path / f"seed{s}_plateau_report.json",
                   _hist([0.1] * 7, reduced_at=(5,)), seed=s)
        out = tmp_path / "diag.json"
        assert tool.main(["--screen-dir", str(tmp_path),
                          "--out-json", str(out)]) == 0
        doc = json.loads(out.read_text())
        assert doc["label"] == "POST_HOC_EXPLORATORY"
        assert "NONE" in doc["authority"]
        assert "INCONCLUSIVE" in doc["authority"]
