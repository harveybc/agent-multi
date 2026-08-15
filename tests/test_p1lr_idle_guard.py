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
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import experiment_transition_queue as etq  # noqa: E402
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


DECISION_IDENTITY = "1434685bfdf52911"
ROOT_DIR = {"screen": "out", "decision": "out_decision"}


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
        "output_root": str(tmp_path / ROOT_DIR["screen"]),
        # Finding 226/233: the decision run has its OWN durable root.
        "decision_run": {
            "output_root": str(tmp_path / ROOT_DIR["decision"]),
            "max_global_pass_equivalent_checkpoints": 2000},
    }


def _write(tmp_path, seed, relpath, payload, *, age_seconds, now=NOW,
           mode="screen", identity=None):
    identity = identity or (DECISION_IDENTITY if mode == "decision"
                            else IDENTITY)
    path = (tmp_path / ROOT_DIR[mode] / identity / f"seed{seed}" / relpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    stamp = (now - timedelta(seconds=age_seconds)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def _heartbeat(tmp_path, seed, *, age_seconds, cell=None,
               terminal_state="RUNNING", now=NOW, mode="screen",
               identity=None):
    cell = cell or ORDER[seed][0]
    return _write(tmp_path, seed, f"{cell}/heartbeat.json",
                  {"terminal_state": terminal_state, "cell": cell},
                  age_seconds=age_seconds, now=now, mode=mode,
                  identity=identity)


def _record(tmp_path, seed, cell, *, age_seconds=3600, now=NOW,
            mode="screen", identity=None):
    return _write(tmp_path, seed, f"{cell}/cell_record.json",
                  {"schema": "agent_multi.p1_difficulty_lr_cell_record.v1",
                   "seed": seed, "cell": cell, "mode": mode},
                  age_seconds=age_seconds, now=now, mode=mode,
                  identity=identity)


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
          temperature=45, unit_exists=True, restart_calls=None,
          identity=IDENTITY, **kw):
    emitter = emitter if emitter is not None else FakeEmitter()
    restart_calls = restart_calls if restart_calls is not None else []

    def restart_fn(unit):
        restart_calls.append(unit)
        return {"ok": True, "returncode": 0, "stderr": ""}

    report = guard.poll(
        contract=_contract(tmp_path), identity=identity,
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


def test_completed_seed_is_never_stalled_and_never_restarted(tmp_path):
    """Order 2026-08-15 §3 bullet 1: a terminal seed with no pending
    cells is NOT a stalled seed — no stall incident, no restart."""
    for cell in ORDER[101]:
        _record(tmp_path, 101, cell)
    report, emitter, restarts = _poll(tmp_path)
    entry = report["seeds"]["101"]
    assert entry["pending"] is False
    assert entry["stalled"] is False
    assert entry["terminal_complete"] is True
    assert restarts == []
    assert [c["event_code"] for c in emitter.observed] == []
    assert "restart_withheld_terminal_complete" in entry["actions"]


def test_completed_seed_without_successor_is_never_idle_false(tmp_path):
    """The observed defect: ``process_alive: false`` + terminal 4/4 + no
    next executable job rendered ``idle: false`` merely because the
    previous experiment had completed. It must render
    ``completed_untransitioned`` (§3 bullets 1-2)."""
    for cell in ORDER[101]:
        _record(tmp_path, 101, cell)
    report, emitter, restarts = _poll(tmp_path, process_alive=False)
    entry = report["seeds"]["101"]
    assert entry["process_alive"] is False
    assert entry["records_landed"] == {"value": 4, "of": 4,
                                       "unit": "cell_records"}
    assert entry["idle"] is True                      # never False again
    assert entry["idle_class"] == "completed_untransitioned"
    assert entry["stalled"] is False
    assert "fleet idle time" in entry["idle_class_reason"]
    assert entry["transition_state"] == "completed_untransitioned"
    assert report["transition"]["value"] == "completed_untransitioned"
    assert report["terminal_seeds_local"] == [101]


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


# ── finding 233: ONE validated mode binds root, unit and report ──

def _decision_poll(tmp_path, **kw):
    kw.setdefault("identity", DECISION_IDENTITY)
    kw.setdefault("mode", "decision")
    return _poll(tmp_path, **kw)


def test_decision_mode_binds_root_unit_and_report_to_the_decision_run(
        tmp_path):
    """The guard reads the DECISION root, restarts the DECISION unit and
    says so in its report — the screen root is not consulted at all."""
    _heartbeat(tmp_path, 101, age_seconds=1200, mode="decision")
    report, emitter, restarts = _decision_poll(tmp_path)

    assert report["mode"] == "decision"
    assert report["mode_basis"] == "explicit_validated_parameter"
    assert report["output_root"] == str(tmp_path / "out_decision")
    assert report["unit_template"] == "p1lr-decision@{seed}.service"
    assert report["cells_total"] == 16
    entry = report["seeds"]["101"]
    assert entry["mode"] == "decision"
    assert entry["output_root"] == str(tmp_path / "out_decision")
    assert entry["seed_dir"] == str(
        tmp_path / "out_decision" / DECISION_IDENTITY / "seed101")
    assert entry["unit"] == "p1lr-decision@101.service"
    assert entry["idle"] is True
    assert restarts == ["p1lr-decision@101.service"]
    # the incident names the decision root, never the screen root
    payload = emitter.observed[0]["payload"]
    assert payload["mode"] == "decision"
    assert payload["output_root"] == str(tmp_path / "out_decision")
    assert "decision" in emitter.observed[0]["summary"]


def test_screen_progress_is_never_decision_progress(tmp_path):
    """Fresh heartbeats under the SCREEN root must not make a decision
    guard believe the decision run is progressing (and vice versa)."""
    _heartbeat(tmp_path, 101, age_seconds=5, mode="screen")
    report, _, _ = _decision_poll(tmp_path)
    entry = report["seeds"]["101"]
    assert entry["pending"] is True
    # no decision artifact exists at all -> bounded by first observation
    assert entry["observed_idle_seconds"] == 0.0
    assert "first pending observation" in entry["idle_basis"]

    # …and the mirror: decision progress is not screen progress
    _heartbeat(tmp_path, 101, age_seconds=5, mode="decision")
    screen_report, _, _ = _poll(tmp_path)
    assert screen_report["mode"] == "screen"
    assert screen_report["output_root"] == str(tmp_path / "out")


def test_decision_records_complete_the_decision_seed_only(tmp_path):
    for cell in ORDER[101]:
        _record(tmp_path, 101, cell, mode="decision")
    report, emitter, restarts = _decision_poll(tmp_path)
    assert report["seeds"]["101"]["pending"] is False
    assert emitter.observed == [] and restarts == []
    # the same records do NOT complete the screen seed
    screen_report, _, _ = _poll(tmp_path)
    assert screen_report["seeds"]["101"]["pending"] is True


def test_relaunch_command_is_mode_specific_and_never_defaults_to_screen():
    screen = guard.relaunch_command(101, "screen")
    decision = guard.relaunch_command(101, "decision")
    assert "p1lr-screen@.service" in screen
    assert "enable --now p1lr-screen@101.service" in screen
    assert "p1lr-decision@.service" in decision
    assert "enable --now p1lr-decision@101.service" in decision
    assert "screen@" not in decision
    # the default stays screen for the historical single-mode callers
    assert guard.relaunch_command(101) == screen


def test_unknown_guard_mode_is_a_typed_refusal(tmp_path):
    with pytest.raises(guard.P1lrModeRefusal) as excinfo:
        guard.relaunch_command(101, "decisive")
    assert excinfo.value.code == "P1LR_MODE_INVALID"
    with pytest.raises(guard.P1lrModeRefusal) as excinfo:
        _poll(tmp_path, mode="decisive")
    assert excinfo.value.code == "P1LR_MODE_INVALID"


def test_missing_decision_unit_carries_the_decision_relaunch_command(
        tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=1200, mode="decision")
    report, emitter, restarts = _decision_poll(tmp_path, unit_exists=False)
    assert restarts == []
    entry = report["seeds"]["101"]
    assert entry["relaunch_command"] == \
        guard.relaunch_command(101, "decision")
    assert "p1lr-decision@101.service" in \
        emitter.observed[0]["payload"]["remediation"]


# ── the CLI refuses a cross-mode identity instead of reporting zeros ──

def _run_cli(tmp_path, *args, contract=None):
    contract = contract if contract is not None else _contract(tmp_path)
    cpath = tmp_path / "contract.json"
    cpath.write_text(json.dumps(contract))
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools/p1lr_idle_guard.py"),
         "--contract", str(cpath), "--dry-run", *args],
        capture_output=True, text=True, timeout=120, cwd=str(REPO))
    return proc


def test_cli_refuses_a_decision_identity_under_the_screen_root(tmp_path):
    (tmp_path / "out_decision" / DECISION_IDENTITY).mkdir(parents=True)
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    proc = _run_cli(tmp_path, "--mode", "screen",
                    "--identity", DECISION_IDENTITY)
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["error_code"] == "P1LR_IDENTITY_MODE_MISMATCH"
    assert payload["identity_mode"] == "decision"
    assert "seeds" not in payload          # no zero-count report at all


def test_cli_refuses_decision_mode_without_a_decision_root(tmp_path):
    contract = _contract(tmp_path)
    contract.pop("decision_run")
    proc = _run_cli(tmp_path, "--mode", "decision", contract=contract)
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["error_code"] == "P1LR_DECISION_ROOT_MISSING"


def test_cli_report_binds_to_the_decision_root_and_is_writable(tmp_path):
    _heartbeat(tmp_path, 101, age_seconds=30, mode="decision")
    out = tmp_path / "reports" / "last_report.json"
    proc = _run_cli(tmp_path, "--mode", "decision",
                    "--identity", DECISION_IDENTITY,
                    "--report-out", str(out))
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    assert payload["mode"] == "decision"
    assert payload["output_root"] == str(tmp_path / "out_decision")
    assert payload["dry_run"] is True
    assert payload["identity_presence"]["identity_under_mode_root"] is True
    assert payload["identity_presence"][
        "identity_under_other_mode_root"] is False


# ── the shipped durability surface: units, timer and the gate wrapper ──

SYSTEMD = REPO / "examples/systemd"
GATE_SCHEMA = "agent_multi.p1_difficulty_lr_screen_verdict.v1"


def _unit(name):
    return (SYSTEMD / name).read_text()


def _directives(name):
    """The unit's EFFECTIVE text: comments carry no systemd semantics."""
    return "\n".join(line for line in _unit(name).splitlines()
                     if line.strip() and not line.strip().startswith("#"))


def test_decision_unit_always_runs_decision_mode_and_never_screen():
    text = _directives("p1lr-decision@.service")
    exec_start = text.split("ExecStart=", 1)[1]
    assert "--mode decision" in exec_start
    assert "--mode screen" not in text
    assert "p1lr-screen" not in text
    # no operator EXTRA_ARGS can append a second --mode and downgrade it
    assert "EXTRA_ARGS" not in text
    # the screen gate is PINNED and passed to the runner
    assert "Environment=P1LR_SCREEN_GATE=" in text
    assert "--screen-gate ${P1LR_SCREEN_GATE}" in exec_start
    # …and VERIFIED before the runner starts
    assert "ExecStartPre=" in text
    assert "p1lr_decision_gate_check.sh ${P1LR_SCREEN_GATE}" in text
    # a configuration refusal (exit 4) is never retried
    assert "RestartPreventExitStatus=4" in text
    assert "Restart=on-failure" in text
    assert "SuccessExitStatus=3" in text
    # The long decision run is uniformly bounded on every host.  In
    # particular, Gamma runs two workers and must not be able to consume the
    # old 21.8 GiB advertised replay allocation per process.
    assert "MemoryHigh=5G" in text
    assert "MemoryMax=6G" in text
    assert "MemorySwapMax=1G" in text
    assert "WantedBy=default.target" in text


def test_decision_unit_pin_matches_the_guard_relaunch_command():
    text = _unit("p1lr-decision@.service")
    assert guard.MODE_UNIT_FILES["decision"] == \
        "examples/systemd/p1lr-decision@.service"
    assert (SYSTEMD / "p1lr-decision@.service").is_file()
    assert "p1lr-decision@101.service" in \
        guard.relaunch_command(101, "decision")
    assert "Description=P1 difficulty x P1 LR factorial seed %i" in text


def test_guard_timer_is_fifteen_minutes_and_persistent():
    timer = _unit("p1lr-idle-guard.timer")
    assert "OnUnitActiveSec=15min" in timer
    assert "OnBootSec=5min" in timer
    assert "Persistent=true" in timer
    assert "Unit=p1lr-idle-guard.service" in timer
    assert "WantedBy=timers.target" in timer
    assert str(int(guard.DEFAULT_IDLE_AFTER_SECONDS // 60)) == "15"


def test_guard_service_binds_the_decision_root_by_default():
    service = _unit("p1lr-idle-guard.service")
    assert "Type=oneshot" in service
    assert "Environment=P1LR_GUARD_MODE=decision" in service
    assert "--mode ${P1LR_GUARD_MODE}" in service
    assert "--report-out ${P1LR_GUARD_REPORT}" in service
    # the reviewed per-host override file comes AFTER the default, so it
    # wins when present and changes nothing when absent
    assert service.index("Environment=P1LR_GUARD_MODE") < \
        service.index("EnvironmentFile=-")


def _gate_check(tmp_path, gate_payload, *, contract_text=None,
                write_gate=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    gate = tmp_path / "screen_verdict.json"
    if write_gate:
        gate.write_text(json.dumps(gate_payload))
    contract = tmp_path / "contract.json"
    contract.write_text(contract_text if contract_text is not None
                        else json.dumps({"schema": "x"}))
    return subprocess.run(
        ["bash", str(SYSTEMD / "p1lr_decision_gate_check.sh"),
         str(gate), str(contract)],
        capture_output=True, text=True, timeout=120)


def _viable_gate(contract_text):
    import hashlib
    return {
        "schema": GATE_SCHEMA,
        "outcome": "SCREEN_VIABLE_REGION",
        "gates": {"replica_terminal_loads": True},
        "contract_sha256": hashlib.sha256(
            contract_text.encode()).hexdigest(),
    }


def test_gate_wrapper_accepts_only_a_verified_viable_gate(tmp_path):
    text = json.dumps({"schema": "contract"})
    proc = _gate_check(tmp_path, _viable_gate(text), contract_text=text)
    assert proc.returncode == 0, proc.stderr
    assert "VERIFIED" in proc.stdout


def test_gate_wrapper_refuses_missing_wrong_and_foreign_gates(tmp_path):
    text = json.dumps({"schema": "contract"})

    missing = _gate_check(tmp_path / "a", {}, contract_text=text,
                          write_gate=False)
    assert missing.returncode == 4

    not_viable = dict(_viable_gate(text),
                      outcome="PHASE1_LR_REGION_COLLAPSED")
    proc = _gate_check(tmp_path / "b", not_viable, contract_text=text)
    assert proc.returncode == 4
    assert "REFUSED_SCREEN_NOT_VIABLE" in proc.stdout

    no_replica = dict(_viable_gate(text),
                      gates={"replica_terminal_loads": False})
    proc = _gate_check(tmp_path / "c", no_replica, contract_text=text)
    assert proc.returncode == 4
    assert "REFUSED_REPLICA_PROOF_MISSING" in proc.stdout

    foreign = dict(_viable_gate(text), contract_sha256="0" * 64)
    proc = _gate_check(tmp_path / "d", foreign, contract_text=text)
    assert proc.returncode == 4
    assert "REFUSED_SCREEN_GATE_FOREIGN" in proc.stdout

    bad_schema = dict(_viable_gate(text), schema="something.else")
    proc = _gate_check(tmp_path / "e", bad_schema, contract_text=text)
    assert proc.returncode == 4
    assert "REFUSED_SCREEN_GATE_SCHEMA" in proc.stdout


def test_install_script_installs_but_never_enables():
    script = (SYSTEMD / "install_p1lr_decision_and_guard.sh").read_text()
    for line in script.splitlines():
        stripped = line.strip()
        if "systemctl --user enable" in stripped or \
                "systemctl --user start" in stripped:
            assert stripped.startswith("echo "), \
                f"install script must only PRINT enable commands: {line}"
    assert "systemctl --user daemon-reload" in script
    assert "p1lr-idle-guard.timer" in script
    assert "p1lr-decision@" in script
    assert os.access(SYSTEMD / "install_p1lr_decision_and_guard.sh", os.X_OK)
    assert os.access(SYSTEMD / "p1lr_decision_gate_check.sh", os.X_OK)


def test_decision_runtime_pin_separates_canonical_and_experiment_checkouts():
    script_path = SYSTEMD / "pin_p1lr_decision_runtime.sh"
    script = script_path.read_text()

    assert "worktree add --detach" in script
    assert "status --porcelain" in script
    assert "p1lr-decision@.service.d" in script
    assert "WorkingDirectory=$RUNTIME_DIR" in script
    assert "EnvironmentFile=" in script
    assert "p1lr_decision_gate_check.sh" in script
    assert "--preflight" in script
    assert "PREFLIGHT_PASS" in script
    assert os.access(script_path, os.X_OK)


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


# ── order 2026-08-15 §3: terminal-to-next-job transition ───────────────

def _terminal_seed(tmp_path, seed=101, mode="screen", identity=None):
    for cell in ORDER[seed]:
        _record(tmp_path, seed, cell, mode=mode, identity=identity)


def _enrolled(tmp_path, *, now=NOW, identity=IDENTITY, mode="screen",
              budget=3600.0):
    """Enrol the terminal transition exactly as the fleet-level observer
    (multifront_status) would, then hand back the queue dir."""
    queue_dir = tmp_path / "transition-queue"
    etq.ensure_terminal_record(
        queue_dir, experiment="p1_difficulty_lr_factorial_20260811_v1",
        mode=mode, identity=identity, records_landed=16, cells_total=16,
        terminal_utc=now.isoformat(), now=now,
        transition_budget_seconds=budget)
    return queue_dir


def test_dispatched_successor_is_the_only_thing_that_clears_idleness(
        tmp_path):
    """A durable, DISPATCHED successor — and nothing else — makes a
    terminal seed non-idle (§3 bullets 2 and 4)."""
    _terminal_seed(tmp_path)
    queue_dir = _enrolled(tmp_path)
    tid = etq.transition_id("p1_difficulty_lr_factorial_20260811_v1",
                            "screen", IDENTITY)

    report, _, _ = _poll(tmp_path, transition_queue_dir=queue_dir)
    assert report["seeds"]["101"]["idle"] is True
    assert report["seeds"]["101"]["idle_class"] == "completed_untransitioned"

    record = etq.load_record(etq.record_path(queue_dir, tid))
    record = etq.approve_successor(record, job_id="l2-en-v1",
                                   approved_by="owner", chain_id="chainA",
                                   now=NOW)
    record = etq.set_materialization(record, "materialized", now=NOW)
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id="chainA", now=NOW)
    record = etq.confirm_dispatch(record, claim_id="c1", now=NOW)
    etq.save_record(queue_dir, record, now=NOW)

    report, emitter, restarts = _poll(tmp_path,
                                      transition_queue_dir=queue_dir)
    entry = report["seeds"]["101"]
    assert entry["idle"] is False
    assert entry["idle_class"] == "completed_transitioned"
    assert entry["next_job_id"] == "l2-en-v1"
    assert emitter.observed == [] and restarts == []


def test_undispatched_successor_past_budget_emits_one_incident_and_closes(
        tmp_path):
    """§3 bullet 6: ONE deduplicated incident, and recovery closes the
    SAME event code — never a message flood."""
    _terminal_seed(tmp_path)
    queue_dir = _enrolled(tmp_path, budget=3600.0)
    tid = etq.transition_id("p1_difficulty_lr_factorial_20260811_v1",
                            "screen", IDENTITY)
    record = etq.load_record(etq.record_path(queue_dir, tid))
    record = etq.approve_successor(record, job_id="l2-en-v1",
                                   approved_by="owner", chain_id="chainA",
                                   now=NOW)
    etq.save_record(queue_dir, record, now=NOW)

    emitter = FakeEmitter()
    over = NOW + timedelta(seconds=5400)          # 1.5x the budget
    for offset in (0, 600, 1200):                 # three polls, one alert
        _poll(tmp_path, emitter=emitter, transition_queue_dir=queue_dir,
              now=over + timedelta(seconds=offset))
    codes = [c["event_code"] for c in emitter.observed]
    assert codes == [f"experiment_transition_undispatched.{tid}"]
    payload = emitter.observed[0]["payload"]
    assert payload["next_job"]["id"] == "l2-en-v1"
    assert payload["over_budget_seconds"] == 1800.0
    assert emitter.observed[0]["affected_object"] == (
        f"p1_difficulty_lr_factorial_20260811_v1/{IDENTITY}")

    # …the successor is dispatched: the SAME code is recovered, once.
    record = etq.load_record(etq.record_path(queue_dir, tid))
    record = etq.set_materialization(record, "materialized", now=over)
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id="chainA", now=over)
    record = etq.confirm_dispatch(record, claim_id="c1", now=over)
    etq.save_record(queue_dir, record, now=over)
    later = over + timedelta(seconds=1800)
    report, _, _ = _poll(tmp_path, emitter=emitter,
                         transition_queue_dir=queue_dir, now=later)
    assert [r["event_code"] for r in emitter.recovered] == \
        [f"experiment_transition_undispatched.{tid}"]
    _poll(tmp_path, emitter=emitter, transition_queue_dir=queue_dir,
          now=later + timedelta(seconds=900))
    assert len(emitter.recovered) == 1            # no recovery flood
    assert report["seeds"]["101"]["idle"] is False


def test_transition_incident_is_not_emitted_before_the_budget_expires(
        tmp_path):
    _terminal_seed(tmp_path)
    queue_dir = _enrolled(tmp_path, budget=3600.0)
    emitter = FakeEmitter()
    report, _, _ = _poll(tmp_path, emitter=emitter,
                         transition_queue_dir=queue_dir,
                         now=NOW + timedelta(seconds=1800))
    assert emitter.observed == []
    assert report["seeds"]["101"]["idle_class"] == \
        "completed_untransitioned"
    assert report["transition"]["over_budget"] is False


def test_guard_survives_reboot_by_reading_durable_records_only(tmp_path):
    """§3 bullet 4: after a reboot every heartbeat mtime is old, no
    process exists and the guard's own dedup state file is empty — the
    transition verdict still comes out right, from the records alone."""
    _terminal_seed(tmp_path)
    queue_dir = _enrolled(tmp_path)
    tid = etq.transition_id("p1_difficulty_lr_factorial_20260811_v1",
                            "screen", IDENTITY)
    record = etq.load_record(etq.record_path(queue_dir, tid))
    record = etq.approve_successor(record, job_id="l2-en-v1",
                                   approved_by="owner", chain_id="chainA",
                                   now=NOW)
    record = etq.set_materialization(record, "materialized", now=NOW)
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id="chainA", now=NOW)
    record = etq.confirm_dispatch(record, claim_id="c1", now=NOW)
    etq.save_record(queue_dir, record, now=NOW)

    after_reboot = NOW + timedelta(hours=9)
    report, emitter, restarts = _poll(
        tmp_path, state=guard.default_state(), now=after_reboot,
        process_alive=False, transition_queue_dir=queue_dir)
    entry = report["seeds"]["101"]
    assert entry["idle_class"] == "completed_transitioned"
    assert entry["idle"] is False
    assert report["transition"]["basis"] == "durable_record"
    assert report["transition"]["next_job_id"] == "l2-en-v1"
    assert emitter.observed == [] and restarts == []


def test_no_transition_queue_configured_still_surfaces_untransitioned(
        tmp_path):
    """Without a durable queue nothing can PROVE a dispatch, so the
    honest answer is completed_untransitioned — never healthy silence."""
    _terminal_seed(tmp_path)
    report, emitter, restarts = _poll(tmp_path, transition_queue_dir=None)
    entry = report["seeds"]["101"]
    assert entry["idle"] is True
    assert entry["idle_class"] == "completed_untransitioned"
    assert report["transition"]["basis"] == "transition_queue_not_configured"
    assert report["transition_action"]["action"] == "no_durable_record"
    assert emitter.observed == [] and restarts == []
