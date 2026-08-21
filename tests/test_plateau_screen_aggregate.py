"""Tests for the predeclared bounded-screen aggregator (PLR orders 2-3)."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "plateau_screen_aggregate",
        REPO / "tools" / "plateau_screen_aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load()


def _report(path, best_composite, val_ret=0.01, epochs=10,
            data_sha="d" * 64, reduced_at=()):
    history = []
    for e in range(1, epochs + 1):
        history.append({
            "epoch": e,
            "composite": best_composite if e == epochs // 2 + 1
            else best_composite - 0.05,
            "l1_checkpoint_eligible": True,
            "checkpoint_improved": e in (1, epochs // 2 + 1),
            "val_total_return": val_ret,
            "val_trades": 10, "train_tail_trades": 5,
            "val_max_drawdown_fraction": 0.02,
            "observed_learning_rates": {"actor": 3e-4},
            "plateau_lr": ({"reduced": True, "old_lr": 3e-4,
                            "new_lr": 1.5e-4}
                           if e in reduced_at else None),
        })
    path.write_text(json.dumps({
        "history": history, "stop_reason": "l1_early_stop",
        "epochs_run": epochs, "elapsed_seconds": 100.0,
        "data_sha256": data_sha}))


def _screen(tmp_path, deltas):
    for seed, d in zip((101, 202, 303, 404), deltas):
        _report(tmp_path / f"seed{seed}_fixed_report.json", 0.10)
        _report(tmp_path / f"seed{seed}_plateau_report.json", 0.10 + d,
                reduced_at=(5,))
    return tmp_path


class TestPredeclaredRule:
    def test_signal_for(self, tool, tmp_path, capsys):
        d = _screen(tmp_path, [0.01, 0.02, 0.03, -0.01])
        out = tmp_path / "agg.json"
        assert tool.main(["--screen-dir", str(d),
                          "--out-json", str(out)]) == 0
        r = json.loads(out.read_text())
        assert r["outcome"] == "SHORT_SCREEN_SIGNAL_FOR_PLATEAU"
        assert r["dispersion"]["positive_seeds"] == 3

    def test_signal_against(self, tool, tmp_path):
        d = _screen(tmp_path, [-0.01, -0.02, -0.03, 0.01])
        out = tmp_path / "agg.json"
        assert tool.main(["--screen-dir", str(d),
                          "--out-json", str(out)]) == 0
        assert json.loads(out.read_text())["outcome"] == (
            "SHORT_SCREEN_SIGNAL_AGAINST")

    def test_inconclusive_split(self, tool, tmp_path):
        d = _screen(tmp_path, [0.01, 0.02, -0.01, -0.02])
        out = tmp_path / "agg.json"
        assert tool.main(["--screen-dir", str(d),
                          "--out-json", str(out)]) == 0
        assert json.loads(out.read_text())["outcome"] == "INCONCLUSIVE"

    def test_incomplete_screen_refuses(self, tool, tmp_path, capsys):
        _report(tmp_path / "seed101_fixed_report.json", 0.1)
        assert tool.main(["--screen-dir", str(tmp_path),
                          "--out-json",
                          str(tmp_path / "agg.json")]) == 2
        assert "REFUSED_INCOMPLETE_SCREEN" in capsys.readouterr().out

    def test_mismatched_data_hash_refuses(self, tool, tmp_path):
        _screen(tmp_path, [0.01, 0.01, 0.01, 0.01])
        _report(tmp_path / "seed101_plateau_report.json", 0.11,
                data_sha="e" * 64)
        with pytest.raises(tool.ScreenAggregationError,
                           match="different data hashes"):
            tool.main(["--screen-dir", str(tmp_path),
                       "--out-json", str(tmp_path / "agg.json")])

    def test_wall_clock_is_descriptive_only(self, tool, tmp_path):
        d = _screen(tmp_path, [0.01, 0.02, 0.03, 0.04])
        out = tmp_path / "agg.json"
        tool.main(["--screen-dir", str(d), "--out-json", str(out)])
        r = json.loads(out.read_text())
        for seed in ("101", "202", "303", "404"):
            arm = r["pairs"][seed]["fixed"]
            assert "elapsed_seconds" in arm["descriptive_only"]
            assert arm["descriptive_only"][
                "excluded_from_causal_conclusion"] is True
            assert "elapsed" not in json.dumps(
                r["primary_deltas_by_seed"])

    def test_no_eligible_checkpoint_refuses(self, tool, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"history": [
            {"epoch": 1, "composite": 0.0,
             "l1_checkpoint_eligible": False,
             "checkpoint_improved": False}],
            "data_sha256": "d" * 64}))
        with pytest.raises(tool.ScreenAggregationError,
                           match="typed refusal"):
            tool.arm_facts(p)
