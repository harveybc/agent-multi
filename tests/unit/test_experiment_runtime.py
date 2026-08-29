"""Required adversarial tests for the observable/resumable experiment
runtime (PERMANENT order @95e088da). Every listed requirement is one
test."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import (  # noqa: E402
    RunDirectory, RuntimePreflightError, UnitClaimError, aggregate,
    atomic_write_json, preflight_or_refuse, run_one_unit, sha_obj,
    unit_id)


def make_run(tmp_path, n_units=3, timeout=30.0):
    run = RunDirectory(tmp_path / "run",
                   allow_volatile_for_tests=True)
    units = []
    for k in range(n_units):
        identity = {"experiment": "t", "family": "fam", "window": 32,
                    "latent": 16, "budget": 10, "seed": k,
                    "origin": 0, "treatment": "trained"}
        units.append({"unit_id": unit_id(identity),
                      "identity": identity})
    run.write_ledger({
        "units": units,
        "digests": {"code": "c" * 64, "data": "d" * 64,
                    "config": "f" * 64},
        "campaign_wall_ceiling_s": 3600,
        "unit_timeout_s": timeout})
    return run, [u["unit_id"] for u in units]


DIGESTS = {"code": "c" * 64, "data": "d" * 64, "config": "f" * 64}


def ok_executor(identity, log_path):
    log_path.write_text("unit ran\n")
    return {"score": identity["seed"] * 1.0}


def test_run_root_under_tmp_refuses(tmp_path):
    with pytest.raises(RuntimePreflightError, match="/tmp"):
        RunDirectory(Path("/tmp/claude-anything/run"))


def test_preflight_refuses_without_ledger(tmp_path):
    run = RunDirectory(tmp_path / "bare",
                   allow_volatile_for_tests=True)
    with pytest.raises(RuntimePreflightError, match="ledger"):
        preflight_or_refuse(run, 3600, None)


def test_crash_after_completed_unit_preserves_and_reuses(tmp_path):
    run, uids = make_run(tmp_path)
    run_one_unit(run, uids[0], ok_executor,
                 expected_digests=DIGESTS, timeout_s=30)
    # simulated crash: new RunDirectory over the same root (restart)
    run2 = RunDirectory(run.root,
                    allow_volatile_for_tests=True)
    assert run2.unit_state(uids[0])["state"] == "COMPLETED"
    assert run2.result(uids[0])["score"] == 0.0
    # exact resume never reruns completed units
    with pytest.raises(UnitClaimError):
        run2.claim(uids[0], expected_digests=DIGESTS)


def test_crash_during_unit_marks_only_that_unit(tmp_path):
    run, uids = make_run(tmp_path)

    def boom(identity, log_path):
        raise RuntimeError("mid-unit crash")
    with pytest.raises(RuntimeError):
        run_one_unit(run, uids[1], boom,
                     expected_digests=DIGESTS, timeout_s=30)
    states = run.states()
    assert states[uids[1]]["state"] == "FAILED"
    assert states[uids[0]]["state"] == "PENDING"
    assert states[uids[2]]["state"] == "PENDING"
    # a failed unit gets a FRESH attempt (claimable again)
    state = run.claim(uids[1], expected_digests=DIGESTS)
    assert state["attempt"] == 2


def test_changed_digests_refuse_resume(tmp_path):
    run, uids = make_run(tmp_path)
    with pytest.raises(RuntimePreflightError, match="drift"):
        run.claim(uids[0], expected_digests={**DIGESTS,
                                             "code": "0" * 64})


def test_stale_heartbeat_and_dead_process_watchdog(tmp_path):
    run, uids = make_run(tmp_path)
    state = run.claim(uids[0], expected_digests=DIGESTS)
    state["pid"] = 999999999  # dead
    state["timeout_s"] = 30
    atomic_write_json(run._state_path(uids[0]), state)
    run.heartbeat(current_unit=uids[0])
    status = json.loads((run.root / "status.json").read_text())
    status["timestamp"] -= 10_000  # stale
    atomic_write_json(run.root / "status.json", status)
    alerts = run.watchdog()
    kinds = {a["type"] for a in alerts}
    assert "stale_heartbeat" in kinds
    assert "dead_process" in kinds
    # evidence preserved: unit terminal, state file intact
    assert run.unit_state(uids[0])["state"] == "TIMED_OUT"


def test_concurrent_workers_cannot_claim_same_unit(tmp_path):
    run, uids = make_run(tmp_path)
    run.claim(uids[0], expected_digests=DIGESTS)
    with pytest.raises(UnitClaimError, match="claimed"):
        run.claim(uids[0], expected_digests=DIGESTS)


def test_duplicate_result_identical_digest_idempotent(tmp_path):
    run, uids = make_run(tmp_path)
    run.claim(uids[0], expected_digests=DIGESTS)
    run.release(uids[0], "COMPLETED", result={"score": 1.0})
    # re-release with the SAME payload: idempotent
    run.release(uids[0], "COMPLETED", result={"score": 1.0})
    assert run.result(uids[0])["score"] == 1.0


def test_conflicting_duplicate_result_refuses(tmp_path):
    run, uids = make_run(tmp_path)
    run.claim(uids[0], expected_digests=DIGESTS)
    run.release(uids[0], "COMPLETED", result={"score": 1.0})
    with pytest.raises(RuntimePreflightError, match="conflicting"):
        run.release(uids[0], "COMPLETED", result={"score": 2.0})


def test_aggregation_refuses_missing_duplicated_foreign(tmp_path):
    run, uids = make_run(tmp_path)
    run_one_unit(run, uids[0], ok_executor,
                 expected_digests=DIGESTS, timeout_s=30)
    with pytest.raises(RuntimePreflightError, match="not COMPLETED"):
        aggregate(run, uids)  # others missing
    run_one_unit(run, uids[1], ok_executor,
                 expected_digests=DIGESTS, timeout_s=30)
    run_one_unit(run, uids[2], ok_executor,
                 expected_digests=DIGESTS, timeout_s=30)
    with pytest.raises(RuntimePreflightError, match="duplicated"):
        aggregate(run, uids + [uids[0]])
    with pytest.raises(RuntimePreflightError, match="foreign"):
        aggregate(run, uids + ["ff" * 10])
    results = aggregate(run, uids)
    assert len(results) == 3


def test_eta_updates_after_each_comparable_unit(tmp_path):
    run, uids = make_run(tmp_path)
    status0 = run.heartbeat(current_unit=None)
    assert status0["eta"] is None
    run_one_unit(run, uids[0], ok_executor,
                 expected_digests=DIGESTS, timeout_s=30)
    status1 = run.heartbeat(current_unit=uids[1])
    assert status1["eta"] is not None
    assert status1["eta"]["remaining_units"] == 2
    run_one_unit(run, uids[1], ok_executor,
                 expected_digests=DIGESTS, timeout_s=30)
    status2 = run.heartbeat(current_unit=uids[2])
    assert status2["eta"]["remaining_units"] == 1
    for key in ("median_unit_s", "p90_unit_s", "eta_pessimistic_s"):
        assert key in status2["eta"]


def test_fsync_failure_never_reports_completion(tmp_path, monkeypatch):
    run, uids = make_run(tmp_path)
    run.claim(uids[0], expected_digests=DIGESTS)
    real_fsync = os.fsync

    def failing_fsync(fd):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        run.release(uids[0], "COMPLETED", result={"score": 1.0})
    monkeypatch.setattr(os, "fsync", real_fsync)
    # no durable result, unit NOT completed
    assert run.result(uids[0]) is None
    assert run.unit_state(uids[0])["state"] == "RUNNING"


def test_sigterm_produces_durable_interrupted(tmp_path):
    run, uids = make_run(tmp_path)

    def executor_that_gets_termed(identity, log_path):
        os.kill(os.getpid(), signal_mod.SIGTERM)
        time.sleep(5)
        return {"score": 0}
    import signal as signal_mod
    with pytest.raises(SystemExit):
        run_one_unit(run, uids[0], executor_that_gets_termed,
                     expected_digests=DIGESTS, timeout_s=30)
    assert run.unit_state(uids[0])["state"] == "INTERRUPTED"
    # a fresh attempt is claimable
    state = run.claim(uids[0], expected_digests=DIGESTS)
    assert state["attempt"] == 2


def test_per_unit_log_is_durable(tmp_path):
    run, uids = make_run(tmp_path)
    run_one_unit(run, uids[0], ok_executor,
                 expected_digests=DIGESTS, timeout_s=30)
    logs = list((run.root / "units").glob(
        f"{uids[0]}.attempt1.log"))
    assert logs and logs[0].read_text() == "unit ran\n"
