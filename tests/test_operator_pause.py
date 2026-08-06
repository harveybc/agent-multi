"""Operator pause verification (AUD-F1-20260805-115).

The pause must stop and VERIFY every owned worker process group, stay
sticky against the tick loop's restart logic, and report a surviving
worker as a failed pause — ``systemctl inactive`` with live workers was
the audited failure mode.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from app.campaign_supervisor import CampaignSupervisor, _pid_matches


def _write_fleet(tmp_path: Path) -> Path:
    doin_root = tmp_path / "doin"
    config_dir = doin_root / "examples/trading/smoke"
    config_dir.mkdir(parents=True)
    worker_config = {
        "port": 18470,
        "data_dir": str(tmp_path / "worker-data"),
        "domains": [{
            "domain_id": "pause-test-domain",
            "optimization_plugin": "trading_asset",
            "optimization_config": {
                "shared_population": True,
                "shared_population_size": 4,
                "ga_seed": 1,
                "population_size": 4,
            },
        }],
    }
    (config_dir / "omega_node.json").write_text(
        json.dumps(worker_config))
    plan = {
        "schema_version": "agent_multi.doin_campaign_plan.v1",
        "plan_id": "pause-test-plan",
        "participants": [{
            "node_id": "omega",
            "supervisor_url": "http://127.0.0.1:18795",
            "workers": ["omega"],
        }],
        "jobs": [{
            "ordinal": 0,
            "job_id": "pause-test-job",
            "domain_id": "pause-test-domain",
            "higher_is_better": True,
            "worker_configs": {
                "omega": "examples/trading/smoke/omega_node.json",
            },
        }],
    }
    (tmp_path / "campaign_plan.json").write_text(json.dumps(plan))
    profile = {
        "schema_version": "agent_multi.doin_campaign_profile.v1",
        "node_id": "omega",
        "plan_file": "campaign_plan.json",
        "state_dir": str(tmp_path / "state"),
        "listen_port": 18795,
        "stop_timeout_seconds": 5,
        "workers": {"omega": {
            "doin_node_root": str(doin_root),
            "python": "/usr/bin/python3",
        }},
    }
    profile_path = tmp_path / "omega_profile.json"
    profile_path.write_text(json.dumps(profile))
    return profile_path


@pytest.fixture()
def supervisor(tmp_path):
    sup = CampaignSupervisor(_write_fleet(tmp_path))
    yield sup
    if sup._lock_handle:
        sup._lock_handle.close()


def _spawn_fake_worker() -> subprocess.Popen:
    return subprocess.Popen(
        ["/bin/sleep", "600"], start_new_session=True)


def _register(sup: CampaignSupervisor, process: subprocess.Popen) -> None:
    worker = sup._worker_state("omega")
    worker["pid"] = process.pid
    worker["pid_start_ticks"] = __import__(
        "app.campaign_supervisor", fromlist=["_pid_start_ticks"]
    )._pid_start_ticks(process.pid)
    worker["owns_process_group"] = True


def test_pause_stops_and_verifies_worker_group(supervisor):
    process = _spawn_fake_worker()
    _register(supervisor, process)
    report = supervisor.request_pause()
    try:
        assert report["paused"] is True
        entry = report["workers"]["omega"]
        assert entry["process_gone"] is True
        assert supervisor.state["phase"] == "paused"
        assert not _pid_matches(
            process.pid,
            supervisor._worker_state("omega").get("pid_start_ticks"))
        persisted = json.loads(
            (Path(supervisor.state_dir) / "state.json").read_text())
        assert persisted["phase"] == "paused"
        assert persisted["pause_report"]["paused"] is True
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def test_pause_is_sticky_against_tick(supervisor):
    report = supervisor.request_pause()
    assert report["paused"] is True
    supervisor.tick()
    assert supervisor.state["phase"] == "paused"
    worker = supervisor._worker_state("omega")
    assert not _pid_matches(
        worker.get("pid"), worker.get("pid_start_ticks"))


def test_sigkill_escalation_for_term_immune_worker(supervisor, tmp_path):
    """A worker that ignores SIGTERM must be visibly escalated to
    SIGKILL and still verified gone."""
    script = tmp_path / "stubborn.py"
    script.write_text(
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(600)\n")
    process = subprocess.Popen(
        ["/usr/bin/python3", str(script)], start_new_session=True)
    time.sleep(0.5)
    _register(supervisor, process)
    report = supervisor.request_pause()
    try:
        assert report["workers"]["omega"]["process_gone"] is True
        assert report["paused"] is True
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def test_surviving_worker_is_a_failed_pause(supervisor, monkeypatch):
    process = _spawn_fake_worker()
    _register(supervisor, process)
    monkeypatch.setattr(
        supervisor, "_stop_worker",
        lambda *args, **kwargs: False)  # simulate a stop that failed
    try:
        report = supervisor.request_pause()
        assert report["paused"] is False
        assert report["workers"]["omega"]["process_gone"] is False
        codes = [a.get("code") for a in
                 supervisor.state.get("alerts", [])]
        assert "operator_pause_incomplete" in codes
    finally:
        os.killpg(process.pid, signal.SIGKILL)


def test_gpu_verification_unavailable_fails_pause(supervisor, monkeypatch):
    """AUD-F1-20260805-121: missing nvidia-smi evidence is a FAILED
    pause, never success."""
    import app.campaign_supervisor as sup_mod

    def broken_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(sup_mod.subprocess, "run", broken_run)
    report = supervisor.request_pause()
    assert report["paused"] is False
    assert "gpu verification unavailable" in report["failure_reason"]


def test_resume_requires_paused_state(supervisor):
    report = supervisor.request_resume("0" * 64)
    assert report["resumed"] is False
    assert "paused state" in report["reason"]


def test_resume_rejects_wrong_binding_hash(supervisor):
    pause = supervisor.request_pause()
    assert pause["paused"] is True
    report = supervisor.request_resume("f" * 64)
    assert report["resumed"] is False
    assert "not authorized" in report["reason"]
    assert supervisor.state["phase"] == "paused"


def test_resume_refuses_profile_drift(supervisor):
    pause = supervisor.request_pause()
    assert pause["paused"] is True
    profile = Path(supervisor.profile_path)
    data = json.loads(profile.read_text())
    data["poll_seconds"] = 99          # drift the profile on disk
    profile.write_text(json.dumps(data))
    report = supervisor.request_resume(pause["binding_hash"])
    assert report["resumed"] is False
    assert report["reason"] == "campaign identity drift"
    assert "profile_sha256" in report["drift"]


def test_resume_roundtrip_and_idempotency(supervisor):
    pause = supervisor.request_pause()
    assert pause["paused"] is True
    assert pause["pause_binding"]["plan_hash"] == supervisor.plan_hash
    report = supervisor.request_resume(pause["binding_hash"])
    assert report["resumed"] is True
    assert supervisor.state["phase"] == "starting"
    again = supervisor.request_resume(pause["binding_hash"])
    assert again["resumed"] is True    # idempotent repeat, no restart
    events = [
        row for row in supervisor.history.campaigns()
    ]
    persisted = json.loads(
        (Path(supervisor.state_dir) / "state.json").read_text())
    assert persisted["resume_report"]["resumed"] is True


def test_resume_refused_over_unverified_pause(supervisor, monkeypatch):
    monkeypatch.setattr(
        supervisor, "_stop_worker", lambda *a, **k: False)
    process = _spawn_fake_worker()
    _register(supervisor, process)
    try:
        pause = supervisor.request_pause()
        assert pause["paused"] is False
        report = supervisor.request_resume(pause["binding_hash"])
        assert report["resumed"] is False
        assert "unverified" in report["reason"]
    finally:
        os.killpg(process.pid, signal.SIGKILL)
