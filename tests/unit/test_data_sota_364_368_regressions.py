"""Frozen counterexamples for DATA-SOTA-364..368 (correction order
2026-08-27). 364 executable-barrier fixtures live in
test_pretrain_objectives_wp1.py (wick/gap/collision/parity); this
module pins 365 (monitor isolation) and 366 (per-horizon support).
PRE: docs/audits/evidence/DATA_SOTA_364_368_REPRODUCTIONS_PRE.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.pretrain_objective_screen import (  # noqa: E402
    evaluate_manifest, target_degeneracy)
from tests.unit.test_branch_pretraining import (  # noqa: E402
    SOURCE_CONFIG, contract_with, synthetic_csv)


def minimal_manifest(probe_std=1.0, probe_norms=None,
                     monitor_extra=None):
    record = {"epoch": 0,
              "train": {"reconstruction": 1.0,
                        "weighted_total": 1.0},
              "monitor_fit_tail": {"reconstruction": 1.0,
                                   **(monitor_extra or {})},
              "mechanics_probe": {
                  "probe": "train_tail_frozen", "windows": 32,
                  "representation_std": probe_std,
                  "gradient_diagnostics": {
                      "norms": probe_norms or {"reconstruction": 0.5},
                      "cosine:a|b": 0.1}},
              "gradient_diagnostics": {"norms": {"reconstruction": 0.5}},
              "seconds": 1.0}
    return {"progress": {"alpha": {"losses": [record]}},
            "partitions": {
                "train": {"target_range": {"last_target_row": 10},
                          "first_step": 1, "last_step": 9},
                "calibration": {"first_step": 20, "last_step": 29,
                                "target_range": {"last_target_row": 30}},
                "monitor": {"first_step": 40, "last_step": 49}}}


class TestDataSota365MonitorIsolation:
    def test_monitor_mutation_cannot_change_eligibility(self):
        """The PRE counterexample: collapse/gradient gates read
        monitor-probe facts. POST: a wild monitor mutation leaves the
        mechanics verdict IDENTICAL."""
        baseline = evaluate_manifest(minimal_manifest())
        mutated = evaluate_manifest(minimal_manifest(
            monitor_extra={"representation_std": 0.0,
                           "reconstruction": float("1e9"),
                           "contrastive_diagnostics":
                               {"projection_std": 0.0}}))
        assert baseline["rejections"] == mutated["rejections"] == []

    def test_probe_mutation_does_change_eligibility(self):
        collapsed = evaluate_manifest(minimal_manifest(probe_std=0.0))
        assert any("collapse" in r for r in collapsed["rejections"])
        dead = evaluate_manifest(minimal_manifest(
            probe_norms={"reconstruction": 0.0}))
        assert any("ZERO encoder gradient" in r
                   for r in dead["rejections"])

    def test_gate_source_never_reads_the_monitor(self):
        import inspect
        source = inspect.getsource(evaluate_manifest)
        # no CODE access to the monitor block (the word may appear in
        # a comment explaining exactly this prohibition)
        assert 'record["monitor_fit_tail"]' not in source
        assert '.get("monitor_fit_tail"' not in source
        assert 'record["mechanics_probe"]' in source

    def test_runner_probe_is_train_tail_not_monitor(self):
        source = (REPO / "tools/pretrain_branches.py").read_text()
        assert "probe_idx = train_idx[-batch_size:]" in source
        assert "probe = monitor_windows[:batch_size]" not in source


class TestDataSota366PerHorizonSupport:
    @pytest.fixture(scope="class")
    def degenerate_case(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("h366")
        csv = root / "synthetic.csv"
        synthetic_csv(csv, hours=600)
        source = root / "source_config.json"
        source.write_text(json.dumps(SOURCE_CONFIG))
        contract = contract_with()
        contract["observation_pipeline"]["source_config"] = str(source)
        # barriers so wide NOTHING ever hits: classes 0 and 1 absent at
        # EVERY horizon — must refuse per horizon, never by aggregate
        contract["objectives"]["barrier_hit"] = {
            "weight": 1.0, "horizons": [2, 4],
            "barrier_scale": {
                "estimator": "trailing_realized_vol_close_to_close",
                "lookback": 16, "epsilon": 10.0},
            "upper_mult": 50.0, "lower_mult": 50.0,
            "same_bar_collision": "conservative_adverse_first",
            "class_weights_from": "calibration_only",
            "ohlc_columns": {"open": "OPEN", "high": "HIGH",
                             "low": "LOW", "close": "CLOSE"}}
        return contract, csv

    def test_deficient_horizons_refuse_individually(self, degenerate_case):
        contract, csv = degenerate_case
        report = target_degeneracy(contract, csv)
        assert report["support_partition"] == "calibration_only"
        named = " ".join(report["rejections"])
        assert "horizon h2" in named and "horizon h4" in named
        assert "DATA-SOTA-366" in named
        per_h = report["barrier_per_horizon"]
        assert set(per_h) == {"h2", "h4"}
        for h in per_h.values():
            assert "counts" in h and "fractions" in h

    def test_predeclared_rule_is_calibration_sized(self, degenerate_case):
        contract, csv = degenerate_case
        report = target_degeneracy(contract, csv)
        assert "max(10, ceil(0.01*n_calibration))" in \
            report["predeclared_min_support_per_class_per_horizon"]
