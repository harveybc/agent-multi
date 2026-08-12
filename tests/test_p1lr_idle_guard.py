"""Socket-free tests for the P1LR idle guard (order 2026-08-11 §7.8).

An assigned GPU idle >15 minutes while a P1LR cell is pending emits ONE
deduplicated incident and triggers BOUNDED service recovery. Process
facts, GPU telemetry, unit existence, restart calls and ledger emissions
are all injected fakes; heartbeat/record mtimes are real files under a
tmp output root with controlled ``os.utime``. The run root is only ever
read.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import p1lr_idle_guard as guard  # noqa: E402

NOW = datetime(2026, 8, 12, 6, 0, 0, tzinfo=timezone.utc)
IDENTITY = "cd823e2b5c753497"
HOSTS = {101: "omega", 202: "dragon", 303: "gamma", 404: "gamma"}
ORDER = {
    101: ["P1N_LR1E4", "P1N_LR3E5", "P1E_LR1E4", "P1E_LR3E5"],
    202: ["P1N_LR3E5", "P1E_LR1E4", "P1E_LR3E5", "P1N_LR1E4"],
    303: ["P1E_LR1E4", "P1E_LR3E5", "P1N_LR1E4", "P1N_LR3E5"],
    404: ["P1E_LR3E5", "P1N_LR1E4", "P1N_LR3E5", "P1E_LR1E4"],
}


def _contract(tmp_path):
    return {
        "schema": "agent_multi.p1_difficulty_lr_factorial.v1",
        "experiment": "p1_difficulty_lr_factorial_20260811_v1",
        "cells": {name: {} for name in ORDER[101]},
        "seeds": [101, 202, 303, 404],
        "assignments": {str(s): {"hostname": HOSTS[s],
                                 "gpu_uuid": f"GPU-{s}"}
                        for s in HOSTS},
        "cell_order": {str(s): ORDER[s] for s in HOSTS},
        "output_root": str(tmp_path / "out"),
    }


def _write(tmp_path, seed, relpath, payload, *, age_seconds, now=NOW):
    path = (tmp_path / "out" / IDENTITY / f"seed{seed}" / relpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    stamp = (now - timedelta(seconds=age_seconds)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def _heartbeat(tmp_path, seed, *, age_seconds, cell=None,
               terminal_state="RUNNING", now=NOW):
    cell = cell or ORDER[seed][0]
    return _write(tmp_path, seed, f"{cell}/heartbeat.json",
                  {"terminal_state": terminal_state, "cell": cell},
                  age_seconds=age_seconds, now=now)


def _record(tmp_path, seed, cell, *, age_seconds=3600, now=NOW):
    return _write(tmp_path, seed, f"{cell}/cell_record.json",
                  {"schema": "agent_multi.p1_difficulty_lr_cell_record.v1",
                   "seed": seed, "cell": cell},
                  age_seconds=age_seconds, now=now)


class FakeEmitter:
    def __init__(self, ok=True):
        self.ok = ok
        self.observed = []
        self.recovered = []

    def observe(self, event_code, severity, summary, payload,
                affected_object="-"):
        self.observed.append({
            "event_code": event_code, "severity": severity,
            "summary": summary, "payload": payload,
            "affected_object": affected_object})
        return self.ok

    def recover(self, event_code, evidence, affected_object="-"):
        self.recovered.append({
            "event_code": event_code, "evidence": evidence,
            "affected_object": affected_object})
        return self.ok


def _poll(tmp_path, *, state=None, now=NOW, emitter=None,
          local_hostname="omega", process_alive=False, utilization=0,
          temperature=45, unit_exists=True, restart_calls=None, **kw):
    emitter = emitter if emitter is not None else FakeEmitter()
    restart_calls = restart_calls if restart_calls is not None else []

    def restart_fn(unit):
        restart_calls.append(unit)
        return {"ok": True, "returncode": 0, "stderr": ""}

    report = guard.poll(
        contract=_contract(tmp_path), identity=IDENTITY,
        state=state or guard.default_state(), now=now,
        local_hostname=local_hostname, emitter=emitter,
        process_alive_fn=lambda seed: process_alive,
        gpu_telemetry_fn=lambda uuid: {
            "gpu_utilization_pct": utilization,
            "gpu_temperature_c": temperature},
        unit_exists_fn=lambda unit: unit_exists,
        restart_fn=restart_fn, **kw)
    return report, emitter, restart_calls


# ── idle + pending → ONE incident + ONE bounded restart ──

def test_idle_pending_emits_one_incident_and_one_bounded_restart(tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=1200)
    report, emitter, restarts = _poll(tmp_path)
    entry = report["seeds"]["101"]
    assert entry["idle"] is True
    assert entry["pending"] is True
    assert entry["observed_idle_seconds"] == 1200.0
    assert "incident_emitted" in entry["actions"]
    assert "restart_attempted" in entry["actions"]
    assert len(emitter.observed) == 1
    incident = emitter.observed[0]
    assert incident["event_code"] == "p1lr_gpu_idle_pending.seed101"
    assert incident["affected_object"] == f"{IDENTITY}/seed101"
    assert "idle" in incident["summary"]
    assert incident["payload"]["records_landed"] == 0
    assert incident["payload"]["cells_total"] == 4
    assert incident["payload"]["gpu_utilization_pct"] == 0
    assert restarts == ["p1lr-screen@101.service"]
    assert report["state"]["seeds"]["101"]["idle_active"] is True
    assert len(report["state"]["seeds"]["101"]["restarts"]) == 1


# ── active facts → nothing ──

def test_fresh_heartbeat_progress_means_no_incident(tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=60)
    report, emitter, restarts = _poll(tmp_path)
    assert report["seeds"]["101"]["idle"] is False
    assert emitter.observed == []
    assert restarts == []


def test_live_runner_process_means_no_incident(tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=1200)
    report, emitter, restarts = _poll(tmp_path, process_alive=True)
    assert report["seeds"]["101"]["idle"] is False
    assert emitter.observed == [] and restarts == []


def test_busy_gpu_never_reads_idle(tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=1200)
    report, emitter, restarts = _poll(tmp_path, utilization=95)
    assert report["seeds"]["101"]["idle"] is False
    assert emitter.observed == [] and restarts == []


def test_unknown_gpu_utilization_never_reads_idle(tmp_path):
    """nvidia-smi unreadable -> unknown facts, never an invented alert."""
    _heartbeat(tmp_path, 101, age_seconds=1200)
    report, emitter, restarts = _poll(tmp_path, utilization=None)
    entry = report["seeds"]["101"]
    assert entry["idle"] is False
    assert entry["gpu_utilization_unknown"] is True
    assert emitter.observed == [] and restarts == []


def test_completed_seed_never_alerts(tmp_path):
    for cell in ORDER[101]:
        _record(tmp_path, 101, cell)
    report, emitter, restarts = _poll(tmp_path)
    entry = report["seeds"]["101"]
    assert entry["pending"] is False
    assert entry["idle"] is False
    assert emitter.observed == [] and restarts == []


# ── dedup across repeated polls; backoff and restart cap ──

def test_dedup_across_repeated_polls(tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=1200)
    emitter = FakeEmitter()
    restarts = []
    report, _, _ = _poll(tmp_path, emitter=emitter, restart_calls=restarts)
    report, _, _ = _poll(tmp_path, emitter=emitter, restart_calls=restarts,
                         state=report["state"],
                         now=NOW + timedelta(seconds=300))
    assert len(emitter.observed) == 1  # deduplicated, not re-emitted
    assert restarts == ["p1lr-screen@101.service"]  # backoff holds
    assert "restart_waiting_backoff" in report["seeds"]["101"]["actions"]
    assert report["seeds"]["101"]["next_restart_due_in_seconds"] == 600.0


def test_restart_cap_and_backoff_respected(tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=1200)
    emitter = FakeEmitter()
    restarts = []
    state = guard.default_state()
    # t0: restart 1 (immediate). t0+1000 >= 900 backoff: restart 2.
    # t0+3000 (2000 since last >= 1800): restart 3 = cap. Then nothing.
    times = [NOW, NOW + timedelta(seconds=1000),
             NOW + timedelta(seconds=3000),
             NOW + timedelta(seconds=10000),
             NOW + timedelta(seconds=20000)]
    reports = []
    for now in times:
        report, _, _ = _poll(tmp_path, emitter=emitter,
                             restart_calls=restarts, state=state, now=now)
        state = report["state"]
        reports.append(report)
    assert len(restarts) == 3  # the cap, never an unbounded loop
    codes = [call["event_code"] for call in emitter.observed]
    assert codes == ["p1lr_gpu_idle_pending.seed101",
                     "p1lr_idle_restart_cap.seed101"]  # each exactly once
    cap = emitter.observed[1]
    assert cap["payload"]["max_restarts"] == 3
    assert "manual intervention" in cap["summary"]
    assert "restart_cap_reached" in reports[-1]["seeds"]["101"]["actions"]


# ── recovery notice on progress resumption ──

def test_recovery_notice_on_progress_resumption(tmp_path):
    hb = _heartbeat(tmp_path, 101, age_seconds=1200)
    emitter = FakeEmitter()
    restarts = []
    report, _, _ = _poll(tmp_path, emitter=emitter, restart_calls=restarts)
    assert len(emitter.observed) == 1
    # the runner resumes: fresh heartbeat progress
    later = NOW + timedelta(seconds=600)
    stamp = later.timestamp()
    os.utime(hb, (stamp, stamp))
    report, _, _ = _poll(tmp_path, emitter=emitter, restart_calls=restarts,
                         state=report["state"], now=later)
    assert [r["event_code"] for r in emitter.recovered] == \
        ["p1lr_gpu_idle_pending.seed101"]
    assert "recovery_emitted" in report["seeds"]["101"]["actions"]
    seed_state = report["state"]["seeds"]["101"]
    assert seed_state["idle_active"] is False
    assert seed_state["restarts"] == []  # bounded budget resets on recovery
    # a NEW stall later is a NEW deduplicated cycle
    stale = later + timedelta(seconds=2000)
    report, _, _ = _poll(tmp_path, emitter=emitter, restart_calls=restarts,
                         state=report["state"], now=stale)
    assert len(emitter.observed) == 2


def test_recovery_also_clears_cap_incident(tmp_path):
    hb = _heartbeat(tmp_path, 101, age_seconds=1200)
    emitter = FakeEmitter()
    state = guard.default_state()
    times = [NOW, NOW + timedelta(seconds=1000),
             NOW + timedelta(seconds=3000),
             NOW + timedelta(seconds=10000)]
    for now in times:
        report, _, _ = _poll(tmp_path, emitter=emitter, state=state,
                             now=now)
        state = report["state"]
    assert state["seeds"]["101"]["cap_emitted"] is True
    later = NOW + timedelta(seconds=20000)
    os.utime(hb, (later.timestamp(), later.timestamp()))
    report, _, _ = _poll(tmp_path, emitter=emitter, state=state, now=later)
    recovered_codes = [r["event_code"] for r in emitter.recovered]
    assert recovered_codes == ["p1lr_gpu_idle_pending.seed101",
                               "p1lr_idle_restart_cap.seed101"]


# ── unit existence and typed refusals bound the recovery ──

def test_missing_unit_emits_exact_relaunch_command_without_restart(
        tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=1200)
    report, emitter, restarts = _poll(tmp_path, unit_exists=False)
    assert restarts == []  # never restarts a unit that does not exist
    entry = report["seeds"]["101"]
    assert "restart_withheld_unit_missing" in entry["actions"]
    command = guard.relaunch_command(101)
    assert "systemctl --user enable --now p1lr-screen@101.service" in \
        command
    assert entry["relaunch_command"] == command
    incident = emitter.observed[0]
    assert incident["payload"]["remediation"] == command
    assert command in incident["summary"]


def test_typed_refusal_withholds_restart(tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=1200,
               terminal_state="REFUSED_WRONG_HOST")
    report, emitter, restarts = _poll(tmp_path)
    assert restarts == []  # a configuration refusal is not transient
    entry = report["seeds"]["101"]
    assert "restart_withheld_typed_refusal" in entry["actions"]
    assert len(emitter.observed) == 1
    remediation = emitter.observed[0]["payload"]["remediation"]
    assert "REFUSED_WRONG_HOST" in remediation
    assert "reset-failed" in remediation


# ── emission-failure retry, host scoping, no-artifact bounding ──

def test_failed_emission_keeps_state_so_next_poll_retries(tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=1200)
    failing = FakeEmitter(ok=False)
    report, _, _ = _poll(tmp_path, emitter=failing)
    assert report["state"]["seeds"]["101"]["idle_active"] is False
    assert "incident_emission_failed_will_retry" in \
        report["seeds"]["101"]["actions"]
    working = FakeEmitter()
    report, _, _ = _poll(tmp_path, emitter=working,
                         state=report["state"],
                         now=NOW + timedelta(seconds=60))
    assert len(working.observed) == 1  # the retry emits exactly once
    assert report["state"]["seeds"]["101"]["idle_active"] is True


def test_only_locally_assigned_seeds_are_guarded(tmp_path):
    for seed in HOSTS:
        _heartbeat(tmp_path, seed, age_seconds=1200)
    report, emitter, restarts = _poll(tmp_path, local_hostname="gamma")
    assert set(report["seeds"]) == {"303", "404"}  # never remote seeds
    assert {c["affected_object"] for c in emitter.observed} == \
        {f"{IDENTITY}/seed303", f"{IDENTITY}/seed404"}
    assert sorted(restarts) == ["p1lr-screen@303.service",
                                "p1lr-screen@404.service"]


def test_seed_without_any_artifact_is_bounded_by_first_observation(
        tmp_path):
    """No heartbeat has EVER landed: idleness is bounded by the guard's
    own first pending observation — never an invented age, and never an
    instant alert."""
    emitter = FakeEmitter()
    report, _, restarts = _poll(tmp_path, emitter=emitter)
    entry = report["seeds"]["101"]
    assert entry["idle"] is False  # first observation only starts the clock
    assert entry["observed_idle_seconds"] == 0.0
    assert "first pending observation" in entry["idle_basis"]
    assert emitter.observed == [] and restarts == []
    report, _, restarts = _poll(tmp_path, emitter=emitter,
                                state=report["state"],
                                now=NOW + timedelta(seconds=1200))
    entry = report["seeds"]["101"]
    assert entry["idle"] is True
    assert entry["observed_idle_seconds"] == 1200.0
    assert len(emitter.observed) == 1
