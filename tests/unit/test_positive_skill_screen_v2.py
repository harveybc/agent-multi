"""Adversarial tests for the v2 screen driver (Musashi correction 1,
2026-09-03) on top of the runtime-CORE tests. Covers the
screen-specific slices of the permanent order's required list plus
the science-equality guarantee for the device-capable trainer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def home_tmp():
    """A NON-/tmp run root: decide_halving/status build production
    RunDirectories, which refuse volatile /tmp (invariant 9)."""
    import shutil
    import uuid
    root = Path.home() / ".cache" / "screen_v2_tests" / uuid.uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "screen_v2", REPO / "tools" / "positive_skill_screen_v2.py")
screen_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen_v2)

from agent_plugins.experiment_runtime import (  # noqa: E402
    RunDirectory, RuntimePreflightError)


def _mini_run(tmp_path, n_units=3):
    run = RunDirectory(tmp_path / "round1",
                       allow_volatile_for_tests=True)
    units = [screen_v2._identity("returns_momentum", 32, 16, 300,
                                 101 + i, 1, "cell")
             for i in range(n_units)]
    ledger = screen_v2._ledger_for(
        units, {"code": "c" * 64, "config": "d" * 64},
        wall_ceiling_s=600.0, unit_timeout_s=30.0,
        extras={"phase": "round1", "invalid_cells": [],
                "family_meta": {}, "predeclaration": "x"})
    run.write_ledger(ledger)
    return run, [screen_v2.unit_id(u) for u in units]


class TestHalvingDecisions:

    def test_later_round_refuses_without_persisted_decision(
            self, tmp_path):
        (tmp_path / "round1").mkdir()
        with pytest.raises(RuntimePreflightError,
                           match="halving decision"):
            screen_v2.materialize_phase(
                tmp_path, "round2", tmp_path / "nope",
                unit_timeout_s=30.0, wall_ceiling_s=600.0,
                max_windows=10, stride=4)

    def test_survivors_refuse_without_round3_decision(self, tmp_path):
        with pytest.raises(RuntimePreflightError, match="round3"):
            screen_v2.materialize_phase(
                tmp_path, "survivors", tmp_path / "nope",
                unit_timeout_s=30.0, wall_ceiling_s=600.0,
                max_windows=10, stride=4)

    def test_persisted_decision_never_overwritten(self, home_tmp):
        run, uids = _mini_run(home_tmp)
        for i, uid in enumerate(uids):
            run.claim(uid, expected_digests={})
            run.release(uid, "COMPLETED", result={
                "family": "returns_momentum", "window": 32,
                "latent": 16 + i, "budget": 300,
                "calibration_r2": 0.1 * i, "monitor_r2": 0.0})
        first = screen_v2.decide_halving(home_tmp, "round1")
        again = screen_v2.decide_halving(home_tmp, "round1")
        assert first["advance"] == again["advance"]
        # a conflicting decision on disk refuses
        path = run.root / "decisions" / "halving.json"
        conflicting = json.loads(path.read_text())
        conflicting["advance"] = ["returns_momentum|w32|d999"]
        path.write_text(json.dumps(conflicting))
        with pytest.raises(RuntimePreflightError, match="disagrees"):
            screen_v2.decide_halving(home_tmp, "round1")

    def test_decision_refuses_on_incomplete_phase(self, home_tmp):
        run, uids = _mini_run(home_tmp)
        run.claim(uids[0], expected_digests={})
        run.release(uids[0], "COMPLETED", result={
            "family": "returns_momentum", "window": 32,
            "latent": 16, "budget": 300, "calibration_r2": 0.5,
            "monitor_r2": 0.0})
        with pytest.raises(RuntimePreflightError,
                           match="not COMPLETED"):
            screen_v2.decide_halving(home_tmp, "round1")


class TestWorkerContract:

    def test_worker_refuses_unknown_unit(self, tmp_path):
        _run, _uids = _mini_run(tmp_path)

        class Args:
            run_root = str(tmp_path)
            phase = "round1"
            unit = "not-a-unit"
            pretrain_dir = str(tmp_path)
            timeout = 5.0
            volatile_ok = True
        with pytest.raises(RuntimePreflightError,
                           match="not in ledger"):
            screen_v2.worker_main(Args())

    def test_concurrent_claim_refused(self, tmp_path):
        run, uids = _mini_run(tmp_path)
        run.claim(uids[0], expected_digests={})
        from agent_plugins.experiment_runtime import UnitClaimError
        with pytest.raises(UnitClaimError):
            run.claim(uids[0], expected_digests={})

    def test_completed_unit_never_reruns(self, tmp_path):
        run, uids = _mini_run(tmp_path)
        run.claim(uids[0], expected_digests={})
        run.release(uids[0], "COMPLETED",
                    result={"family": "returns_momentum",
                            "window": 32, "latent": 16,
                            "budget": 300, "calibration_r2": 0.0,
                            "monitor_r2": 0.0})
        from agent_plugins.experiment_runtime import UnitClaimError
        with pytest.raises(UnitClaimError, match="not claimable"):
            run.claim(uids[0], expected_digests={})

    def test_digest_drift_refuses_resume(self, tmp_path):
        run, uids = _mini_run(tmp_path)
        with pytest.raises(RuntimePreflightError, match="drift"):
            run.claim(uids[0],
                      expected_digests={"code": "e" * 64})


class TestAggregationBoundaries:

    def test_aggregate_refuses_missing_units(self, tmp_path):
        from agent_plugins.experiment_runtime import aggregate
        run, uids = _mini_run(tmp_path)
        with pytest.raises(RuntimePreflightError, match="COMPLETED"):
            aggregate(run, uids)

    def test_aggregate_refuses_foreign_units(self, tmp_path):
        from agent_plugins.experiment_runtime import aggregate
        run, uids = _mini_run(tmp_path)
        with pytest.raises(RuntimePreflightError, match="foreign"):
            aggregate(run, uids + ["deadbeef"])


class TestScienceEquality:

    def test_v2_train_cell_matches_retired_cpu(self):
        """The device-capable trainer must reproduce the retired CPU
        implementation EXACTLY (same seeds, same batches, same
        checkpoints) — the CUDA benchmark measures real science."""
        import numpy as np
        import torch
        science = screen_v2._science()
        rng = np.random.default_rng(7)
        windows = rng.normal(size=(120, 16, 3)).astype("float32")
        target = rng.normal(size=120)
        fit_i = np.arange(0, 70)
        cal_i = np.arange(70, 95)
        mon_i = np.arange(95, 120)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16 * 3, 8)

            def forward(self, x):
                return self.linear(x.reshape(x.shape[0], -1))

        torch.manual_seed(0)
        m1 = Tiny()
        torch.manual_seed(0)
        m2 = Tiny()
        r_retired = science.train_cell(
            m1, 8, windows, target, fit_i, cal_i, mon_i, 60, 42)
        r_v2 = screen_v2.train_cell_device(
            m2, 8, windows, target, fit_i, cal_i, mon_i, 60, 42,
            device="cpu")
        assert r_retired[0] == pytest.approx(r_v2[0], abs=1e-12)
        assert r_retired[1] == pytest.approx(r_v2[1], abs=1e-12)
        assert r_retired[2] == r_v2[2]


class TestStatusMachineReadable:

    def test_status_without_process_attachment(self, home_tmp,
                                               capsys):
        _run, _uids = _mini_run(home_tmp)

        class Args:
            run_root = str(home_tmp)
        screen_v2.status_main(Args())
        out = json.loads(capsys.readouterr().out)
        assert out["phases"]["round1"]["counts"]["PENDING"] == 3
        assert out["phases"]["fusion"]["state"] == "NOT_MATERIALIZED"
