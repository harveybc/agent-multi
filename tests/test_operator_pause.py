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




def _bind_identity(sup: CampaignSupervisor) -> None:
    """Give the supervisor a COMPLETE campaign identity so the pause
    binding is resumable (finding 128 requires completeness)."""
    sup.state["coordination"] = {
        "domain_id": "pause-test-domain",
        "domain_semantic_hash": "sem-1",
        "canonical_lineage": {
            "genesis_hash": "genesis-abc",
            "population_fingerprint": "popfp-xyz",
        },
        "component_versions": sup._component_versions(),
    }
    worker = sup._worker_state("omega")
    worker["tip_hash"] = "tip-1"
    worker["chain_height"] = 5
    worker["api_url"] = "http://127.0.0.1:1"


def _observe(sup: CampaignSupervisor, *, genesis="genesis-abc",
             popfp="popfp-xyz", domain="pause-test-domain",
             tip="tip-1", height=5, observed_at=None) -> None:
    from app.campaign_supervisor import _utc_now
    worker = sup._worker_state("omega")
    worker.update({
        "last_seen": observed_at or _utc_now(),
        "status": "running",
        "bootstrap_evidence": {"genesis_hash": genesis,
                               "population_fingerprint": popfp},
        "shared_population": {"domain_id": domain},
        "tip_hash": tip, "chain_height": height,
        "api_url": "http://127.0.0.1:1"})


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
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    assert pause["paused"] is True
    report = supervisor.request_resume("f" * 64)
    assert report["resumed"] is False
    assert "not authorized" in report["reason"]
    assert supervisor.state["phase"] == "paused"


def test_resume_refuses_profile_drift(supervisor):
    _bind_identity(supervisor)
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


def test_incomplete_binding_is_not_resumable(supervisor):
    """AUD-F1-20260806-128: a pause whose bound identity is incomplete
    must refuse resume outright — absence is never wildcard equality."""
    pause = supervisor.request_pause()   # fixture has NO identity
    assert pause["paused"] is True
    report = supervisor.request_resume(pause["binding_hash"])
    assert report["resumed"] is False
    assert "INCOMPLETE" in report["reason"]
    assert supervisor.state["phase"] == "paused"
    codes = [a.get("code") for a in supervisor.state.get("alerts", [])]
    assert "resume_refused_incomplete_binding" in codes


def test_empty_lineage_never_proves_rejoin(supervisor):
    """Musashi reproducer `empty_lineage_rejoin` as regression: even if
    an incomplete binding slipped into a pending resume, a worker with
    no chain evidence must never yield rejoin_proven=true."""
    from app.campaign_supervisor import _utc_now
    supervisor.state["resume_pending"] = {
        "binding_hash": "x" * 64,
        "accepted_at": "2000-01-01T00:00:00+00:00",
        "binding": {"domain_id": None, "genesis_hash": None,
                    "population_fingerprint": None},
    }
    supervisor.state["resume_report"] = {"binding_hash": "x" * 64}
    worker = supervisor._worker_state("omega")
    worker.update({"status": "running", "bootstrap_evidence": {},
                   "shared_population": {}, "last_seen": _utc_now()})
    result = supervisor.verify_rejoin() or {}
    assert result.get("rejoin_proven") is not True
    assert result.get("resumed") is not True
    assert result.get("rejoin_contradictions")
    assert supervisor.state["phase"] == "paused"


def test_resume_acceptance_is_not_resumption(supervisor):
    """AUD-F1-20260806-122: acceptance must not claim success before the
    workers prove they rejoined the bound chain."""
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    assert pause["paused"] is True
    assert pause["pause_binding"]["plan_hash"] == supervisor.plan_hash
    report = supervisor.request_resume(pause["binding_hash"])
    assert report["resume_accepted"] is True
    assert report["resumed"] is False
    assert report["rejoin_proven"] is False
    assert supervisor.state["phase"] == "starting"
    assert supervisor.state.get("resume_pending")
    persisted = json.loads(
        (Path(supervisor.state_dir) / "state.json").read_text())
    assert persisted["resume_report"]["resumed"] is False


def test_rejoin_proof_requires_matching_lineage(supervisor):
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    supervisor.request_resume(pause["binding_hash"])
    _observe(supervisor)                 # same tip: ancestry trivial
    report = supervisor.verify_rejoin()
    assert report["rejoin_proven"] is True
    assert report["resumed"] is True
    assert report["rejoin_proof"]["genesis_hash"] == "genesis-abc"
    assert not supervisor.state.get("resume_pending")


def test_rejoin_on_foreign_chain_is_refuted_and_repaused(supervisor):
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    supervisor.request_resume(pause["binding_hash"])
    _observe(supervisor, genesis="genesis-OTHER")   # foreign chain
    report = supervisor.verify_rejoin()
    assert report["rejoin_proven"] is False
    assert report["resumed"] is False
    assert any("genesis_hash" in c
               for c in report["rejoin_contradictions"])
    assert supervisor.state["phase"] == "paused"
    codes = [a.get("code") for a in supervisor.state.get("alerts", [])]
    assert "resume_lineage_mismatch" in codes


def test_missing_lineage_keeps_resume_pending(supervisor):
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    supervisor.request_resume(pause["binding_hash"])
    worker = supervisor._worker_state("omega")
    worker["status"] = "running"
    worker["bootstrap_evidence"] = {}        # no evidence yet
    report = supervisor.verify_rejoin()
    assert report["rejoin_proven"] is not True
    assert report["resumed"] is not True
    assert "rejoin_pending_reason" in report
    assert supervisor.state.get("resume_pending")


def test_profile_drift_blocks_worker_launch(supervisor):
    supervisor.state["profile_drift_block"] = {
        "configured": "/other/profile.json",
        "loaded": str(supervisor.profile_path), "since": "now"}
    job = supervisor.plan["jobs"][0]
    supervisor._start_or_adopt_worker(job, "omega", {"path": "/nope"})
    worker = supervisor._worker_state("omega")
    assert worker.get("launch_ready") is False
    assert "profile drift" in worker.get("launch_reason", "")
    assert worker.get("pid") in (None, 0)
    codes = [a.get("code") for a in supervisor.state.get("alerts", [])]
    assert "worker_launch_blocked" in codes


def test_gpu_probe_nonzero_exit_fails_pause(supervisor, monkeypatch):
    """AUD-F1-20260806-124: nonzero exit with EMPTY stdout must not read
    as GPU-clear."""
    import app.campaign_supervisor as sup_mod

    class _Probe:
        returncode = 9
        stdout = ""
        stderr = "NVML init failed"

    monkeypatch.setattr(sup_mod.subprocess, "run",
                        lambda *a, **k: _Probe())
    report = supervisor.request_pause()
    assert report["paused"] is False
    assert "gpu verification unavailable" in report["failure_reason"]
    assert "exit 9" in report["gpu_probe"]["error"]


def test_resume_refused_over_unverified_pause(supervisor, monkeypatch):
    _bind_identity(supervisor)
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


def test_drift_check_ignores_unmanaged_process(supervisor, monkeypatch):
    """A manually launched supervisor must not be blocked by the systemd
    unit's ExecStart; only the unit's own MainPID is compared."""
    import app.campaign_supervisor as sup_mod

    class _Probe:
        returncode = 0
        stdout = ("ExecStart={ path=/x ; argv[]=/x -m app.campaign_supervisor"
                  " --profile /other/profile.json ; }\nMainPID=999999\n")
        stderr = ""

    monkeypatch.setattr(sup_mod.subprocess, "run",
                        lambda *a, **k: _Probe())
    supervisor.check_profile_drift()
    assert not supervisor.state.get("profile_drift_block")


def test_drift_block_set_for_managed_process(supervisor, monkeypatch):
    import os
    import app.campaign_supervisor as sup_mod

    class _Probe:
        returncode = 0
        stdout = ("ExecStart={ argv[]=/x -m app.campaign_supervisor"
                  " --profile /other/profile.json ; }\n"
                  f"MainPID={os.getpid()}\n")
        stderr = ""

    monkeypatch.setattr(sup_mod.subprocess, "run",
                        lambda *a, **k: _Probe())
    supervisor.check_profile_drift()
    block = supervisor.state.get("profile_drift_block")
    assert block and block["configured"] == "/other/profile.json"
    assert supervisor._launch_blocked_reason() is not None



def test_component_revision_drift_refuses_resume(supervisor,
                                                 monkeypatch):
    """AUD-F1-20260806-135: a changed component revision is campaign
    identity drift, not a resumable state."""
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    assert pause["paused"] is True
    monkeypatch.setattr(
        supervisor, "_component_versions",
        lambda: {"agent-multi": "DIFFERENT", "gym-fx": "x"})
    report = supervisor.request_resume(pause["binding_hash"])
    assert report["resumed"] is False
    assert report["reason"] == "campaign identity drift"
    assert "component_versions" in report["drift"]


def test_inexact_rejoin_foreign_tip_is_refuted(supervisor, monkeypatch):
    """Musashi reproducer `inexact_rejoin`: same genesis and gen-0
    population but an unrelated tip must NOT prove rejoin."""
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    supervisor.request_resume(pause["binding_hash"])
    _observe(supervisor, tip="foreign-tip-with-no-ancestor-proof",
             height=9)

    import app.campaign_supervisor as sup_mod
    monkeypatch.setattr(
        sup_mod, "_http_json",
        lambda url, timeout: {"hash": "some-other-branch-block"})
    report = supervisor.verify_rejoin()
    assert report["rejoin_proven"] is not True
    assert any("tip ancestry" in c
               for c in report.get("rejoin_contradictions", []))
    assert supervisor.state["phase"] == "paused"


def test_descendant_tip_proves_rejoin(supervisor, monkeypatch):
    """A chain that ADVANCED past the bound tip still proves descent
    when the bound tip remains at its index."""
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    supervisor.request_resume(pause["binding_hash"])
    _observe(supervisor, tip="new-tip-after-more-blocks", height=9)

    import app.campaign_supervisor as sup_mod
    monkeypatch.setattr(
        sup_mod, "_http_json",
        lambda url, timeout: {"hash": "tip-1"})   # bound tip still there
    report = supervisor.verify_rejoin()
    assert report["rejoin_proven"] is True
    proof = report["observed_lineage"]["omega"]["tip_ancestry"]
    assert proof["mode"] == "descends_from_bound_tip"


def test_rollback_below_bound_tip_is_contradiction(supervisor):
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    supervisor.request_resume(pause["binding_hash"])
    _observe(supervisor, tip="short-chain-tip", height=2)  # rolled back
    report = supervisor.verify_rejoin()
    assert report["rejoin_proven"] is not True
    assert any("rolled back or replaced" in c
               for c in report.get("rejoin_contradictions", []))



def test_stale_cached_observation_never_proves_rejoin(supervisor):
    """AUD-F1-20260806-150: an observation older than the accepted
    resume is a CACHE, not proof."""
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    accepted = supervisor.request_resume(pause["binding_hash"])
    assert accepted["resume_accepted"] is True
    _observe(supervisor, observed_at="2000-01-01T00:00:00+00:00")
    report = supervisor.verify_rejoin()
    assert report.get("rejoin_proven") is not True
    assert "STALE" in report["rejoin_pending_reason"]
    assert supervisor.state.get("resume_pending")


def test_missing_observation_timestamp_keeps_pending(supervisor):
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    supervisor.request_resume(pause["binding_hash"])
    _observe(supervisor)
    supervisor._worker_state("omega")["last_seen"] = None
    report = supervisor.verify_rejoin()
    assert report.get("rejoin_proven") is not True
    assert "no observation timestamp" in report["rejoin_pending_reason"]


def test_rejoin_deadline_expiry_returns_to_paused_and_alerts(
        supervisor, monkeypatch):
    """At expiry the supervisor must settle into a stable paused state
    and alert once — not stay pending forever."""
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    accepted = supervisor.request_resume(pause["binding_hash"])
    assert accepted["rejoin_deadline_at"]
    pending = supervisor.state["resume_pending"]
    pending["deadline_at"] = "2000-01-01T00:00:00+00:00"   # expired
    supervisor._worker_state("omega")["status"] = "starting"
    report = supervisor.verify_rejoin()
    assert report["rejoin_timed_out"] is True
    assert report["resumed"] is False
    assert supervisor.state["phase"] == "paused"
    assert not supervisor.state.get("resume_pending")
    codes = [a.get("code") for a in supervisor.state.get("alerts", [])]
    assert "resume_deadline_expired" in codes


def test_fresh_observation_after_acceptance_can_prove(supervisor,
                                                      monkeypatch):
    _bind_identity(supervisor)
    pause = supervisor.request_pause()
    supervisor.request_resume(pause["binding_hash"])
    _observe(supervisor)                       # fresh by construction
    report = supervisor.verify_rejoin()
    assert report["rejoin_proven"] is True
