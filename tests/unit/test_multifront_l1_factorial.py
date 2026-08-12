"""Finding 204 (order 2026-08-10 §6/WP3): Front-1 status must describe the
work that is ACTUALLY running — the L1 matched factorial — as a first-class
source, with the paused DOIN campaign rendered strictly as history.

All worker-host facts are injected through a fake reader: no test ever
opens an ssh connection or touches the live run root.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tools.multifront_status as mfs  # noqa: E402

NOW = datetime(2026, 8, 10, 7, 0, 0, tzinfo=timezone.utc)
IDENTITY = "2de49ea9225e2baf"
HOSTS = {101: "omega", 202: "dragon", 303: "gamma", 404: "gamma"}
CELLS = {
    "L1_N_M10": ("normal_realistic", 1.0),
    "L1_E_M10": ("easy_chronological_continuation", 1.0),
    "L1_N_M03": ("normal_realistic", 0.3),
    "L1_E_M03": ("easy_chronological_continuation", 0.3),
}


class FakeReader:
    """Injected worker-host reader: dict-backed, never ssh."""

    local_hostname = "omega"

    def __init__(self, files=None, mtimes=None, restarts=None,
                 latest=None, unreachable=(), units_loaded=None):
        self.files = dict(files or {})
        self.mtimes = dict(mtimes or {})
        self.restarts = dict(restarts or {})
        self.latest = dict(latest or {})
        self.unreachable = set(unreachable)
        # None = "unit load state unknown"; False = no unit on that host
        # (the DIRECT nohup case); True = a loaded systemd unit.
        self.units_loaded = units_loaded
        self.errors = {}

    def _gone(self, host):
        if host in self.unreachable:
            self.errors[host] = "ssh transport failure (exit 255)"
            return True
        return False

    def read_text(self, host, path):
        return None if self._gone(host) else self.files.get((host, path))

    def read_tail(self, host, path, max_bytes=131072):
        return None if self._gone(host) else self.files.get((host, path))

    def mtime(self, host, path):
        return None if self._gone(host) else self.mtimes.get((host, path))

    def nrestarts(self, host, unit):
        if self._gone(host):
            return None
        return self.restarts.get((host, unit), 0)

    def unit_loaded(self, host, unit):
        if self._gone(host):
            return None
        if isinstance(self.units_loaded, dict):
            return self.units_loaded.get((host, unit))
        return self.units_loaded

    def latest_heartbeat(self, host, output_root):
        return None if self._gone(host) else self.latest.get(host)


def _contract(tmp_path):
    return {
        "schema": "agent_multi.l1_factorial_contract.v3",
        "experiment": "l1_matched_factorial",
        "asset": "ETHUSD",
        "stopping": {"l1_activity_patience": 40, "l1_patience": 60,
                     "max_epochs": 2000},
        "cells": {name: {"phase1_mode": mode, "phase2_lr_multiplier": lr}
                  for name, (mode, lr) in CELLS.items()},
        "seeds": [101, 202, 303, 404],
        "assignments": {str(s): {"hostname": HOSTS[s],
                                 "gpu_uuid": f"GPU-{s}"}
                        for s in HOSTS},
        "output_root": str(tmp_path / "out"),
    }


def _log(epoch=34, streak=0, trades=(0, 0, 0)):
    return (
        f"[epoch  {epoch}/1996] L1 no-activity {streak}/40  L2 -/4  "
        "paired_generalization_weekly_v1 composite=-1000000.0000 "
        "raw=+0.0000 trade_gate=FAIL best=-inf (checkpoint ineligible)\n"
        f"            TRAIN trades=  {trades[0]:2d} win%= 0.00 sharpe=+nan "
        "profit=+0.00% bal=10000.00\n"
        f"            TRAIN_TAIL trades=  {trades[1]:2d} win%= 0.00 "
        "sharpe=+nan profit=+0.00% bal=10000.00\n"
        f"            VAL   trades=  {trades[2]:2d} win%= 0.00 sharpe=+nan "
        "profit=+0.00% bal=10000.00\n"
    )


def _fixture(tmp_path, *, now=NOW, epochs=None, streaks=None,
             hb_cells=None, attempts=None, progress=None,
             log_age_seconds=30):
    contract = _contract(tmp_path)
    cpath = tmp_path / "contract.json"
    cpath.write_text(json.dumps(contract))
    root = contract["output_root"]
    files, mtimes, latest = {}, {}, {}
    for seed, host in HOSTS.items():
        heartbeat = {
            "schema": "agent_multi.l1_launcher_heartbeat.v2",
            "seed": seed, "cell": (hb_cells or {}).get(seed, "L1_N_M10"),
            "terminal_state": "RUNNING",
            "pid": 1000 + seed, "pid_start_identity": str(seed * 7),
            "assigned_gpu_uuid": f"GPU-{seed}",
            "cuda_visible_devices": f"GPU-{seed}",
            "observed_gpu_uuids": [f"GPU-{seed}"],
            "progress": (progress or {}).get(seed, "0/4 cells"),
            "updated_utc": (now - timedelta(seconds=45)).isoformat(),
        }
        if (attempts or {}).get(seed):
            heartbeat["attempt"] = attempts[seed]
        hb_path = f"{root}/{IDENTITY}/seed{seed}/launcher_heartbeat.json"
        files[(host, hb_path)] = json.dumps(heartbeat)
        latest.setdefault(host, hb_path)
        log_path = f"{root}/logs/seed{seed}.log"
        files[(host, log_path)] = _log(
            epoch=(epochs or {}).get(seed, 34),
            streak=(streaks or {}).get(seed, 0))
        mtimes[(host, log_path)] = (
            now - timedelta(seconds=log_age_seconds)).timestamp()
    reader = FakeReader(files=files, mtimes=mtimes, latest=latest)
    return cpath, reader


def _root(tmp_path):
    return str(tmp_path / "out")


def _paused_supervisor(monkeypatch, plan_jobs=None):
    status = {
        "plan_id": "eth-anchored-solvency-v2", "plan_hash": "b" * 64,
        "job_id": "job-0", "phase": "paused",
        "updated_at": NOW.isoformat(),
        "workers": {"w0": {
            "shared_population": {"generation": 7, "evaluated": 152,
                                  "pop_size": 480},
            "candidate": {"stage": 1, "total_stages": 3},
            "candidate_eta": {}, "best_performance": 0.1234,
        }},
    }
    network = {"plan_hash": "b" * 64, "plan_jobs": plan_jobs or [],
               "participants": {}}
    monkeypatch.setattr(
        mfs, "_get_url",
        lambda url, timeout: (status if url.endswith("/api/status")
                              else network))


def _collect(tmp_path, cpath, reader, **kw):
    kw.setdefault("l1_identity", IDENTITY)
    kw.setdefault("l1_state_dir", tmp_path / "state")
    kw.setdefault("l1_now_fn", lambda: NOW)
    return mfs.collect(
        snapshot_path=tmp_path / "m.json",
        watchdog_path=tmp_path / "m2.json",
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://mock", timeout=0.1,
        l0_heartbeat_path=tmp_path / "no-hb.json",
        l0_db_path=tmp_path / "no-l0.sqlite",
        l1_contract_path=cpath, l1_reader=reader,
        l1_local_hostname="omega", **kw)


# ── THE contradiction (finding 204): paused supervisor + active workers ──

def test_paused_supervisor_with_four_active_workers_renders_active_front1(
        tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path)
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader)
    f1 = packet["fronts"]["f1_optimization"]
    active = f1["active_l1_factorial"]
    assert active["state"] == "active"
    assert active["source"] == "l1_factorial"
    assert active["workers_running_fresh"]["value"] == 4
    history = f1["doin_campaign_history"]
    assert history["phase"] == "paused"
    assert "HISTORY" in history["note"]
    # the paused campaign never replaces the active factorial
    fields = {entry["field"] for entry in packet["unavailable"]}
    assert "f1_optimization" not in fields
    assert "f1_optimization.active_l1_factorial" not in fields


def test_worker_facts_are_reported_per_seed(tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path)
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader)
    workers = packet["fronts"]["f1_optimization"][
        "active_l1_factorial"]["workers"]
    assert set(workers) == {"101", "202", "303", "404"}
    w = workers["202"]
    assert w["host"] == "dragon"
    assert w["unit"] == "l1-factorial@202.service"
    assert w["terminal_state"] == "RUNNING"
    assert w["assigned_gpu_uuid"] == "GPU-202"
    assert w["cuda_visible_devices"] == "GPU-202"
    assert w["pid"] == 1202 and w["pid_start_identity"] == "1414"
    assert w["cell"] == "L1_N_M10"
    assert w["cell_factors"] == {"phase1_mode": "normal_realistic",
                                 "phase2_lr_multiplier": 1.0}
    assert w["epoch"] == {"value": 34, "of": 1996, "unit": "epochs",
                          "horizon": "cell"}
    assert w["activity_patience"]["value"] == 0
    assert w["activity_patience"]["of"] == 40
    assert w["activity_patience"]["declared_patience"] == 40
    assert w["trades"]["TRAIN"]["trades"] == 0
    assert w["trades"]["TRAIN_TAIL"]["trades"] == 0
    assert w["trades"]["VAL"]["trades"] == 0
    assert w["restart_count"]["value"] == 0
    assert w["last_progress_utc"] is not None
    assert w["progress"] == "0/4 cells"


def test_active_factorial_leads_the_executable_queue(tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path)
    _paused_supervisor(monkeypatch, plan_jobs=[
        {"job_id": "job-ok", "status": "running"}])
    packet = _collect(tmp_path, cpath, reader)
    queue = packet["queue"]
    assert queue[0]["id"] == f"l1-matched-factorial-{IDENTITY}"
    assert queue[0]["state"] == "running"
    assert mfs._valid_sha256(queue[0]["hashes"]["config_sha256"])
    # supervisor jobs may coexist but never displace the factorial
    assert any(item["id"] == "job-ok" for item in queue)


# ── tolerance: unreachable host degrades to a typed unavailable worker ──

def test_unreachable_host_yields_typed_unavailable_worker(
        tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path)
    reader.unreachable = {"dragon"}
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader)
    active = packet["fronts"]["f1_optimization"]["active_l1_factorial"]
    assert active["state"] == "active"  # three fresh workers remain
    assert active["workers_running_fresh"]["value"] == 3
    w = active["workers"]["202"]
    assert w["terminal_state"] == "unavailable"
    assert "host unreachable" in w["unavailable_reason"]
    fields = {entry["field"] for entry in packet["unavailable"]}
    assert "f1_optimization.active_l1_factorial.workers.202" in fields


# ── ETA: observed durations only, formula + sample size, else missing fact ──

def test_eta_unavailable_first_then_derived_with_formula(
        tmp_path, monkeypatch):
    _paused_supervisor(monkeypatch)
    state_dir = tmp_path / "state"
    times = [NOW, NOW + timedelta(seconds=600), NOW + timedelta(seconds=1200)]
    packets = []
    for observation, now in zip((30, 31, 32), times):
        cpath, reader = _fixture(tmp_path, now=now,
                                 epochs={s: observation for s in HOSTS})
        packets.append(_collect(
            tmp_path, cpath, reader, l1_state_dir=state_dir,
            l1_now_fn=lambda now=now: now))
    first = packets[0]["fronts"]["f1_optimization"][
        "active_l1_factorial"]["workers"]["101"]["current_cell_eta"]
    assert first["value"] == "unavailable"
    assert "epoch log lines carry no timestamps" in first["missing"]
    final = packets[-1]["fronts"]["f1_optimization"][
        "active_l1_factorial"]["workers"]["101"]["current_cell_eta"]
    assert final["sample_size"] == {"value": 2, "unit": "observation_pairs"}
    assert final["seconds_per_epoch"]["median"] == 600.0
    assert final["remaining_epochs"] == 1996 - 32
    assert final["eta_seconds"]["value"] == 600.0 * (1996 - 32)
    assert "median(delta_seconds/delta_epochs" in final["formula"]
    assert "uncertainty" in final


def test_experiment_eta_requires_observed_durations(tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path)
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader)
    experiment_eta = packet["fronts"]["f1_optimization"][
        "active_l1_factorial"]["experiment_eta"]
    assert experiment_eta["value"] == "unavailable"
    assert "completed cell records" in experiment_eta["missing"]


def test_experiment_eta_is_max_worker_path_never_serial_sum(
        tmp_path, monkeypatch):
    """AUD-F1-20260810-213: four workers run their cells CONCURRENTLY, so
    the experiment ETA is the longest single-worker remaining path. The
    predecessor test enforced the defect (mean duration * ALL remaining
    cells, which serializes parallel work: 63000s here instead of the
    18000s critical path); it is REPLACED by this parallel contract."""
    cpath, reader = _fixture(tmp_path)
    root = json.loads(cpath.read_text())["output_root"]
    for cell, hours in (("L1_N_M10", 1), ("L1_E_M10", 2)):
        record = {
            "schema": "agent_multi.l1_factorial_cell_record.v2",
            "seed": 101, "cell": cell,
            "phase1_mode": CELLS[cell][0],
            "phase2_lr_multiplier": CELLS[cell][1],
            "stop_reason": "max_epochs_budget", "history_len": 2,
            "decision_eligible": False,
            "started_utc": (NOW - timedelta(hours=hours + 1)).isoformat(),
            "finished_utc": (NOW - timedelta(hours=1)).isoformat(),
        }
        reader.files[("omega",
                      f"{root}/{IDENTITY}/seed101/{cell}/"
                      "l1_cell_record.json")] = json.dumps(record)
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader)
    active = packet["fronts"]["f1_optimization"]["active_l1_factorial"]
    assert active["records_landed"] == {
        "value": 2, "of": 16, "unit": "cell_records",
        "horizon": "experiment"}
    w = active["workers"]["101"]
    assert w["records_landed"]["value"] == 2
    assert w["landed_cells"]["L1_N_M10"]["stop_reason"] == "max_epochs_budget"
    assert w["landed_cells"]["L1_N_M10"]["duration_seconds"] == 3600.0
    # observed durations 3600s and 7200s -> mean 5400s. seed101 has 2 cells
    # left (1 active + 1 queued) = 10800s; seeds 202/303/404 have 4 left
    # (1 active + 3 queued) = 21600s. The critical path is 21600s — the
    # serial formula would have claimed 5400s * 14 = 75600s.
    experiment_eta = active["experiment_eta"]
    assert experiment_eta["eta_seconds"]["value"] == 21600.0
    assert experiment_eta["eta_seconds"]["value"] != 5400.0 * 14  # not serial
    assert experiment_eta["eta_seconds"]["low"] == 3600.0 * 4
    assert experiment_eta["eta_seconds"]["high"] == 7200.0 * 4
    assert experiment_eta["critical_path_seed"] in {"202", "303", "404"}
    per_worker = experiment_eta["per_worker_paths"]
    assert per_worker["101"]["remaining_cells"] == 2
    assert per_worker["101"]["queued_cells"] == 1
    assert per_worker["101"]["path_seconds"]["value"] == 10800.0
    assert per_worker["202"]["queued_cells"] == 3
    assert per_worker["202"]["path_seconds"]["value"] == 21600.0
    assert "mean_completed_cell_duration" in \
        per_worker["202"]["active_cell_eta_source"]
    assert experiment_eta["sample_size"] == {"value": 2,
                                             "unit": "completed_cells"}
    assert "max over workers" in experiment_eta["formula"]
    assert "never the serial sum" in experiment_eta["formula"]
    assert "uncertainty" in experiment_eta
    # the current-cell ETA remains a SEPARATE observable with its own
    # sample count (epoch observation pairs, none here) — finding 213
    assert w["current_cell_eta"]["value"] == "unavailable"


def test_experiment_eta_degrades_when_a_worker_is_unknown(
        tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path)
    reader.unreachable = {"dragon"}
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader)
    experiment_eta = packet["fronts"]["f1_optimization"][
        "active_l1_factorial"]["experiment_eta"]
    assert experiment_eta["value"] == "unavailable"
    assert "202" in experiment_eta["missing"]
    assert "critical-path" in experiment_eta["missing"]


# ── zero-trade monitoring at the declared patience boundary ──

def test_boundary_emits_exactly_one_bounded_deduped_alert(
        tmp_path, monkeypatch):
    _paused_supervisor(monkeypatch)
    calls = []

    def emitter(**kw):
        calls.append(kw)
        return True

    for _ in range(2):  # a second collection must dedup, not re-emit
        cpath, reader = _fixture(tmp_path, streaks={101: 40})
        packet = _collect(tmp_path, cpath, reader,
                          l1_alert_emitter=emitter)
    assert len(calls) == 1
    call = calls[0]
    assert call["front"] == "front1"
    assert call["source"] == "multifront_status"
    assert call["affected_object"] == f"{IDENTITY}/seed101/L1_N_M10"
    assert "terminal inactivity" in call["summary"]
    assert call["payload"]["no_activity"] == 40
    assert call["payload"]["declared_patience"] == 40
    alerts = packet["fronts"]["f1_optimization"][
        "active_l1_factorial"]["zero_trade_alerts"]
    assert alerts == [{"seed": 101, "cell": "L1_N_M10",
                       "condition": alerts[0]["condition"],
                       "emitted": False, "deduped": True}]
    marker = (tmp_path / "state" / "alerts" /
              f"l1_zero_trade.{IDENTITY}.seed101.L1_N_M10.json")
    assert marker.exists()


def test_below_boundary_emits_nothing(tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path, streaks={101: 39})
    _paused_supervisor(monkeypatch)
    calls = []
    packet = _collect(tmp_path, cpath, reader,
                      l1_alert_emitter=lambda **kw: calls.append(kw) or True)
    assert calls == []
    active = packet["fronts"]["f1_optimization"]["active_l1_factorial"]
    assert "zero_trade_alerts" not in active


def test_terminal_activity_stop_record_also_alerts(tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path)
    root = json.loads(cpath.read_text())["output_root"]
    record = {
        "schema": "agent_multi.l1_factorial_cell_record.v2",
        "seed": 303, "cell": "L1_N_M10",
        "stop_reason": "l1_activity_patience",
        "activity_stopped_without_eligible_checkpoint": True,
        "started_utc": (NOW - timedelta(hours=3)).isoformat(),
        "finished_utc": (NOW - timedelta(hours=1)).isoformat(),
    }
    reader.files[("gamma",
                  f"{root}/{IDENTITY}/seed303/L1_N_M10/"
                  "l1_cell_record.json")] = json.dumps(record)
    _paused_supervisor(monkeypatch)
    calls = []
    packet = _collect(tmp_path, cpath, reader,
                      l1_alert_emitter=lambda **kw: calls.append(kw) or True)
    assert len(calls) == 1
    assert calls[0]["affected_object"] == f"{IDENTITY}/seed303/L1_N_M10"
    alerts = packet["fronts"]["f1_optimization"][
        "active_l1_factorial"]["zero_trade_alerts"]
    assert alerts[0]["emitted"] is True
    assert "activity stop" in alerts[0]["condition"]


def test_never_emits_without_dedup_capability(tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path, streaks={101: 40})
    _paused_supervisor(monkeypatch)
    calls = []
    packet = _collect(tmp_path, cpath, reader, l1_state_dir=None,
                      l1_alert_emitter=lambda **kw: calls.append(kw) or True)
    assert calls == []  # unbounded emission is refused, not risked
    alerts = packet["fronts"]["f1_optimization"][
        "active_l1_factorial"]["zero_trade_alerts"]
    assert "no state dir" in alerts[0]["skipped"]


# ── finding 212: telemetry binds to (identity, seed, cell, attempt) ──

def test_stale_cell_unbound_log_never_reads_as_current_telemetry(
        tmp_path, monkeypatch):
    """AUD-F1-20260810-212 exact defect: a ~12.7h-old global seed log plus
    a FRESH heartbeat on a DIFFERENT cell reported the old cell's epochs
    (34) as current telemetry and displayed the previous cell's attempt
    path. Facts must render typed unavailability with the source age."""
    stale_age = int(12.7 * 3600)
    previous_attempt = (f"{_root(tmp_path)}/{IDENTITY}/seed101/L1_N_M10/"
                        "attempt-2abdaf3f2972bb94-01")
    cpath, reader = _fixture(
        tmp_path,
        hb_cells={seed: "L1_E_M03" for seed in HOSTS},
        attempts={101: previous_attempt},
        log_age_seconds=stale_age)
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader)
    active = packet["fronts"]["f1_optimization"]["active_l1_factorial"]
    assert active["state"] == "active"  # heartbeats are fresh
    w = active["workers"]["101"]
    # epoch/patience/trade facts: typed unavailable with the source age,
    # never the stale numbers
    assert w["epoch"]["value"] == "unavailable"
    assert w["epoch"]["source_age_seconds"] >= 12 * 3600
    assert "stale" in w["epoch"]["reason"]
    assert w["activity_patience"]["value"] == "unavailable"
    assert w["trades"]["value"] == "unavailable"
    assert w["telemetry_binding"]["bound"] is False
    dumped = json.dumps(w)
    assert '"value": 34' not in dumped  # the stale epoch never surfaces
    # the previous cell's attempt path is withheld, with a typed reason
    assert w["attempt"] is None
    assert w["attempt_withheld"]["bound_cell"] == "L1_N_M10"
    assert previous_attempt not in dumped
    # stale facts feed no ETA sample
    assert w["current_cell_eta"]["value"] == "unavailable"


def test_current_attempt_path_is_shown_when_bound(tmp_path, monkeypatch):
    current_attempt = (f"{_root(tmp_path)}/{IDENTITY}/seed101/L1_N_M10/"
                       "attempt-2abdaf3f2972bb94-02")
    cpath, reader = _fixture(tmp_path, attempts={101: current_attempt})
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader)
    w = packet["fronts"]["f1_optimization"][
        "active_l1_factorial"]["workers"]["101"]
    assert w["attempt"] == current_attempt
    assert "attempt_withheld" not in w
    # fresh co-temporal facts stay bound and current
    assert w["telemetry_binding"]["bound"] is True
    assert w["telemetry_binding"]["binds"]["cell"] == "L1_N_M10"
    assert w["telemetry_binding"]["binds"]["attempt"] == current_attempt
    assert w["telemetry_binding"]["source_age_seconds"] is not None
    assert w["epoch"] == {"value": 34, "of": 1996, "unit": "epochs",
                          "horizon": "cell"}


def test_structured_heartbeat_progress_binds_without_any_log(
        tmp_path, monkeypatch):
    """Forward-compat (finding 212): a launcher that publishes structured
    progress inside its heartbeat is bound by construction — even when the
    global log is stale."""
    cpath, reader = _fixture(
        tmp_path,
        progress={101: {"cell": "L1_N_M10", "epoch": 55, "epoch_max": 1996,
                        "no_activity": 3, "no_activity_of": 40}},
        log_age_seconds=13 * 3600)
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader)
    workers = packet["fronts"]["f1_optimization"][
        "active_l1_factorial"]["workers"]
    assert workers["101"]["telemetry_binding"]["source"] == \
        "heartbeat_progress"
    assert workers["101"]["epoch"]["value"] == 55
    assert workers["101"]["activity_patience"]["value"] == 3
    # the string-progress workers still degrade on their stale logs
    assert workers["202"]["epoch"]["value"] == "unavailable"


# ── identity discovery and history separation ──

def test_identity_discovered_from_latest_heartbeat(tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path)
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader, l1_identity=None)
    active = packet["fronts"]["f1_optimization"]["active_l1_factorial"]
    assert active["identity"] == IDENTITY
    assert active["identity_basis"].startswith(
        "discovered_latest_heartbeat_mtime")


# ═══ Order 2026-08-11 §7.7: the RUNNING P1LR factorial screen front ═══

P1LR_IDENTITY = "cd823e2b5c753497"
P1LR_ORDER = {
    101: ["P1N_LR1E4", "P1N_LR3E5", "P1E_LR1E4", "P1E_LR3E5"],
    202: ["P1N_LR3E5", "P1E_LR1E4", "P1E_LR3E5", "P1N_LR1E4"],
    303: ["P1E_LR1E4", "P1E_LR3E5", "P1N_LR1E4", "P1N_LR3E5"],
    404: ["P1E_LR3E5", "P1N_LR1E4", "P1N_LR3E5", "P1E_LR1E4"],
}
P1LR_CELLS = {
    "P1N_LR1E4": ("normal_realistic", 1e-4),
    "P1N_LR3E5": ("normal_realistic", 3e-5),
    "P1E_LR1E4": ("easy_chronological_continuation", 1e-4),
    "P1E_LR3E5": ("easy_chronological_continuation", 3e-5),
}


P1LR_DECISION_IDENTITY = "1434685bfdf52911"


def _p1lr_contract(tmp_path):
    return {
        "schema": "agent_multi.p1_difficulty_lr_factorial.v1",
        "experiment": "p1_difficulty_lr_factorial_20260811_v1",
        "asset": "ETHUSD_4h",
        "cells": {name: {"phase1_dynamics": mode,
                         "phase1_learning_rate": lr}
                  for name, (mode, lr) in P1LR_CELLS.items()},
        "seeds": [101, 202, 303, 404],
        "assignments": {str(s): {"hostname": HOSTS[s],
                                 "gpu_uuid": f"GPU-{s}"}
                        for s in HOSTS},
        "cell_order": {str(s): P1LR_ORDER[s] for s in HOSTS},
        "output_root": str(tmp_path / "p1out"),
        # Finding 226/233: the decision run has its OWN root.
        "decision_run": {"output_root": str(tmp_path / "p1out_decision"),
                         "max_global_pass_equivalent_checkpoints": 2000},
    }


def _p1lr_fixture(tmp_path, *, now=NOW, hb_age_seconds=45,
                  terminal_states=None, stages=None, cells=None,
                  gpu=None, lock_elapsed=None, records=None,
                  mode="screen", identity=None, units_loaded=None,
                  record_mode=None):
    contract = _p1lr_contract(tmp_path)
    cpath = tmp_path / "p1lr_contract.json"
    cpath.write_text(json.dumps(contract))
    root = (contract["decision_run"]["output_root"] if mode == "decision"
            else contract["output_root"])
    ident = identity or P1LR_IDENTITY
    files = {}
    for seed, host in HOSTS.items():
        cell = (cells or {}).get(seed, P1LR_ORDER[seed][0])
        temperature, utilization = (gpu or {}).get(seed, (61, 97))
        heartbeat = {
            "schema": "agent_multi.p1_difficulty_lr_heartbeat.v1",
            "seed": seed, "cell": cell,
            "experiment_identity": ident,
            "cell_identity": f"cid{seed}",
            "pid": 2000 + seed, "pid_start_identity": str(seed * 11),
            "hostname": host,
            "assigned_gpu_uuid": f"GPU-{seed}",
            "cuda_visible_devices": f"GPU-{seed}",
            "observed_gpu_uuids": [f"GPU-{seed}"],
            "gpu_temperature_c": temperature,
            "gpu_utilization_pct": utilization,
            "terminal_state": (terminal_states or {}).get(seed, "RUNNING"),
            "progress": (stages or {}).get(seed, "training"),
            "attempt": (f"{root}/{ident}/seed{seed}/{cell}/"
                        f"attempt-cid{seed}-01"),
            "updated_utc": (
                now - timedelta(seconds=hb_age_seconds)).isoformat(),
        }
        files[(host, f"{root}/{ident}/seed{seed}/{cell}/"
                     "heartbeat.json")] = json.dumps(heartbeat)
        if lock_elapsed is not None:
            files[(host, f"{root}/{ident}/locks/"
                         f"exclusive_claim.seed{seed}.{cell}.lock")] = \
                json.dumps({
                    "pid": 2000 + seed,
                    "acquired_utc": (
                        now - timedelta(seconds=lock_elapsed)).isoformat(),
                })
    for (seed, cell), payload in (records or {}).items():
        record = {"schema": "agent_multi.p1_difficulty_lr_cell_record.v1",
                  "seed": seed, "cell": cell, **payload}
        if record_mode is not None:
            record.setdefault("mode", record_mode)
            record.setdefault(
                "evidence_class",
                "decision_run" if record_mode == "decision"
                else "mechanics_screen")
        files[(HOSTS[seed], f"{root}/{ident}/seed{seed}/{cell}/"
                            "cell_record.json")] = json.dumps(record)
    return cpath, FakeReader(files=files, units_loaded=units_loaded)


def _root_of(cpath, key="output_root"):
    contract = json.loads(Path(cpath).read_text())
    if key == "decision":
        return contract["decision_run"]["output_root"]
    return contract[key]


def _collect_p1lr(tmp_path, cpath, reader, **kw):
    kw.setdefault("p1lr_identity", P1LR_IDENTITY)
    kw.setdefault("p1lr_now_fn", lambda: NOW)
    return mfs.collect(
        snapshot_path=tmp_path / "m.json",
        watchdog_path=tmp_path / "m2.json",
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://127.0.0.1:1", timeout=0.1,
        l0_heartbeat_path=tmp_path / "no-hb.json",
        l0_db_path=tmp_path / "no-l0.sqlite",
        p1lr_contract_path=cpath, p1lr_reader=reader,
        p1lr_local_hostname="omega", **kw)


def test_p1lr_running_screen_renders_current_cell_and_checkpoint(tmp_path):
    cpath, reader = _p1lr_fixture(tmp_path)
    packet = _collect_p1lr(tmp_path, cpath, reader)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    assert block["source"] == "p1lr_factorial"
    assert block["state"] == "active"
    assert block["mode"] == "screen"
    assert block["identity"] == P1LR_IDENTITY
    assert block["identity_basis"] == "explicit_parameter"
    assert block["workers_running_fresh"]["value"] == 4
    w = block["workers"]["202"]
    assert w["host"] == "dragon"
    assert w["unit"] == "p1lr-screen@202.service"
    assert w["terminal_state"] == "RUNNING"
    assert w["heartbeat_fresh"] is True
    assert w["heartbeat_age_seconds"] == 45.0
    # current seed/cell/checkpoint claims come from the FRESH heartbeat
    assert w["current_cell"] == "P1N_LR3E5"
    assert w["current_cell_factors"] == {
        "phase1_dynamics": "normal_realistic",
        "phase1_learning_rate": 3e-05}
    assert w["checkpoint"]["stage"] == "training"
    assert w["checkpoint"]["source_age_seconds"] == 45.0
    assert w["attempt"].endswith("attempt-cid202-01")
    # GPU utilization + temperature from the runner's nvidia-smi sample
    assert w["gpu"]["utilization_pct"]["value"] == 97
    assert w["gpu"]["temperature_c"]["value"] == 61
    assert w["gpu"]["source_age_seconds"] == 45.0
    assert w["records_landed"] == {
        "value": 0, "of": 4, "unit": "cell_records", "horizon": "seed",
        "mode": "screen", "output_root": _root_of(cpath)}
    assert block["records_landed"]["value"] == 0
    assert block["records_landed"]["of"] == 16
    assert "fleet_note" in block["records_landed"]
    # the RUNNING screen leads the executable queue
    queue = packet["queue"]
    assert queue[0]["id"] == f"p1lr-factorial-{P1LR_IDENTITY}"
    assert queue[0]["state"] == "running"
    assert mfs._valid_sha256(queue[0]["hashes"]["config_sha256"])


def test_p1lr_remote_latency_cannot_create_negative_heartbeat_age(tmp_path):
    cpath, reader = _p1lr_fixture(tmp_path, hb_age_seconds=-10.0)
    observations = iter([NOW] + [NOW + timedelta(seconds=12)] * 8)

    packet = _collect_p1lr(
        tmp_path,
        cpath,
        reader,
        p1lr_now_fn=lambda: next(observations),
    )
    workers = packet["fronts"]["f1_optimization"][
        "active_p1lr_factorial"]["workers"]

    for worker in workers.values():
        assert worker["heartbeat_age_seconds"] == 2.0
        assert worker["heartbeat_clock_ahead_seconds"] == 0.0


def test_p1lr_stale_heartbeat_is_typed_staleness_never_current_claims(
        tmp_path):
    cpath, reader = _p1lr_fixture(tmp_path, hb_age_seconds=2 * 3600)
    packet = _collect_p1lr(tmp_path, cpath, reader)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    assert block["state"] == "inactive_or_unknown"
    assert block["workers_running_fresh"]["value"] == 0
    w = block["workers"]["101"]
    assert w["heartbeat_fresh"] is False
    assert w["current_cell"]["value"] == "unavailable"
    assert "stale" in w["current_cell"]["reason"]
    assert w["current_cell"]["heartbeat_age_seconds"] >= 7200.0
    assert w["current_cell"]["last_known"]["cell"] == "P1N_LR1E4"
    assert w["checkpoint"]["value"] == "unavailable"
    assert w["checkpoint"]["last_known"]["stage"] == "training"
    # stale GPU sample is history, not a current reading
    assert w["gpu"]["value"] == "unavailable"
    assert "history" in w["gpu"]["reason"]
    assert w["current_cell_eta"]["value"] == "unavailable"
    # an inactive screen never enters the executable queue
    assert not any(str(item["id"]).startswith("p1lr-factorial")
                   for item in packet["queue"])


def test_p1lr_records_counted_per_seed_and_fleet(tmp_path):
    records = {
        (101, "P1N_LR1E4"): {"elapsed_seconds": 3600.0,
                             "stop_reason": "budget_complete",
                             "terminal_model_sha256": "a" * 64,
                             "finished_utc": NOW.isoformat()},
        (101, "P1N_LR3E5"): {
            "started_utc": (NOW - timedelta(hours=3)).isoformat(),
            "finished_utc": (NOW - timedelta(hours=1)).isoformat()},
        (303, "P1E_LR1E4"): {"elapsed_seconds": 1800.0},
    }
    cpath, reader = _p1lr_fixture(
        tmp_path, cells={101: "P1E_LR1E4", 303: "P1E_LR3E5"},
        records=records)
    packet = _collect_p1lr(tmp_path, cpath, reader)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    w101 = block["workers"]["101"]
    assert w101["records_landed"] == {
        "value": 2, "of": 4, "unit": "cell_records", "horizon": "seed",
        "mode": "screen", "output_root": _root_of(cpath)}
    assert w101["landed_cells"]["P1N_LR1E4"]["stop_reason"] == \
        "budget_complete"
    assert w101["landed_cells"]["P1N_LR1E4"]["duration_seconds"] == 3600.0
    assert w101["landed_cells"]["P1N_LR3E5"]["duration_seconds"] == 7200.0
    assert block["workers"]["303"]["records_landed"]["value"] == 1
    assert block["records_landed"]["value"] == 3
    assert block["records_landed"]["of"] == 16
    assert "LOCAL output root" in block["records_landed"]["fleet_note"]
    assert "N/4" in block["records_landed"]["fleet_note"]


def test_p1lr_current_cell_eta_and_critical_path_reuse_finding_213(
        tmp_path):
    """Finding 213 reuse: the current-cell ETA derives from observed
    completed-cell durations minus the exclusive-claim elapsed time, and
    the experiment ETA stays the MAXIMUM per-worker remaining path."""
    records = {
        (101, "P1N_LR1E4"): {"elapsed_seconds": 3600.0},
        (101, "P1N_LR3E5"): {"elapsed_seconds": 7200.0},
    }
    cpath, reader = _p1lr_fixture(
        tmp_path, cells={101: "P1E_LR1E4"}, lock_elapsed=1800,
        records=records)
    packet = _collect_p1lr(tmp_path, cpath, reader)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    eta = block["workers"]["101"]["current_cell_eta"]
    # durations 3600/7200 -> mean 5400; elapsed 1800
    assert eta["eta_seconds"] == {"value": 3600.0, "low": 1800.0,
                                  "high": 5400.0, "unit": "seconds"}
    assert eta["elapsed_seconds"] == 1800.0
    assert eta["sample_size"] == {"value": 2, "unit": "completed_cells"}
    assert "exclusive claim" in eta["formula"]
    assert "uncertainty" in eta
    experiment_eta = block["experiment_eta"]
    # seed101: 2 remaining (active 3600 + 1 queued * 5400) = 9000s;
    # the other seeds: 4 remaining (active 3600 + 3 * 5400) = 19800s —
    # the critical path, never the serial sum across workers.
    per_worker = experiment_eta["per_worker_paths"]
    assert per_worker["101"]["path_seconds"]["value"] == 9000.0
    assert per_worker["202"]["path_seconds"]["value"] == 19800.0
    assert per_worker["202"]["active_cell_eta_source"] == \
        "current_cell_duration_eta"
    assert experiment_eta["eta_seconds"]["value"] == 19800.0
    assert experiment_eta["critical_path_seed"] in {"202", "303", "404"}
    assert "max over workers" in experiment_eta["formula"]
    assert experiment_eta["sample_size"] == {"value": 2,
                                             "unit": "completed_cells"}


def test_p1lr_eta_unavailable_without_observed_durations(tmp_path):
    cpath, reader = _p1lr_fixture(tmp_path, lock_elapsed=600)
    packet = _collect_p1lr(tmp_path, cpath, reader)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    eta = block["workers"]["101"]["current_cell_eta"]
    assert eta["value"] == "unavailable"
    assert "fewer than 2 completed cell records" in eta["missing"]
    assert block["experiment_eta"]["value"] == "unavailable"


def test_p1lr_unreachable_host_degrades_typed(tmp_path):
    cpath, reader = _p1lr_fixture(tmp_path, lock_elapsed=600)
    reader.unreachable = {"dragon"}
    packet = _collect_p1lr(tmp_path, cpath, reader)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    assert block["state"] == "active"  # three fresh workers remain
    w = block["workers"]["202"]
    assert w["terminal_state"] == "unavailable"
    assert "host unreachable" in w["unavailable_reason"]
    fields = {entry["field"] for entry in packet["unavailable"]}
    assert ("f1_optimization.active_p1lr_factorial.workers.202"
            in fields)
    experiment_eta = block["experiment_eta"]
    assert experiment_eta["value"] == "unavailable"
    assert "202" in experiment_eta["missing"]


def test_p1lr_identity_discovered_from_local_root(tmp_path):
    cpath, reader = _p1lr_fixture(tmp_path)
    root = Path(json.loads(cpath.read_text())["output_root"])
    hb_dir = root / P1LR_IDENTITY / "seed101" / "P1N_LR1E4"
    hb_dir.mkdir(parents=True)
    (hb_dir / "heartbeat.json").write_text("{}")
    packet = _collect_p1lr(tmp_path, cpath, reader, p1lr_identity=None)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    assert block["identity"] == P1LR_IDENTITY
    assert block["identity_basis"] == \
        "discovered_latest_heartbeat_mtime(local,screen_root)"


# ═══ Finding 233: ONE validated mode derives root/unit/heartbeat/cells ═══


def test_mode_derivation_table_screen_and_decision(tmp_path):
    """mode → (output root, unit, expected heartbeat mode, total cells).
    Every mode-dependent fact has exactly one source: the mode."""
    contract = _p1lr_contract(tmp_path)
    screen = mfs.p1lr_mode_binding(contract, "screen")
    decision = mfs.p1lr_mode_binding(contract, "decision")

    assert screen["output_root"] == contract["output_root"]
    assert screen["unit_template"] == "p1lr-screen@{seed}.service"
    assert screen["unit_example"] == "p1lr-screen@101.service"
    assert screen["heartbeat_mode_expected"] == "screen"
    assert screen["record_mode_expected"] == "screen"
    assert screen["evidence_class_expected"] == "mechanics_screen"
    assert screen["decision_eligible_expected"] is False
    assert screen["total_cells"] == 16
    assert screen["other_mode"] == "decision"
    assert screen["other_mode_output_root"] == \
        contract["decision_run"]["output_root"]

    assert decision["output_root"] == contract["decision_run"]["output_root"]
    assert decision["output_root"] != screen["output_root"]
    assert decision["unit_template"] == "p1lr-decision@{seed}.service"
    assert decision["unit_example"] == "p1lr-decision@101.service"
    assert decision["heartbeat_mode_expected"] == "decision"
    assert decision["record_mode_expected"] == "decision"
    assert decision["evidence_class_expected"] == "decision_run"
    assert decision["decision_eligible_expected"] is True
    assert decision["total_cells"] == 16
    assert decision["other_mode"] == "screen"
    assert decision["other_mode_output_root"] == contract["output_root"]


def test_unknown_mode_is_a_typed_refusal(tmp_path):
    contract = _p1lr_contract(tmp_path)
    for bad in ("Screen", "decide", "", None):
        with pytest.raises(mfs.P1lrModeRefusal) as excinfo:
            mfs.p1lr_mode_binding(contract, bad)
        assert excinfo.value.code == "P1LR_MODE_INVALID"


def test_decision_mode_without_a_decision_root_refuses(tmp_path):
    contract = _p1lr_contract(tmp_path)
    contract.pop("decision_run")
    with pytest.raises(mfs.P1lrModeRefusal) as excinfo:
        mfs.p1lr_mode_binding(contract, "decision")
    assert excinfo.value.code == "P1LR_DECISION_ROOT_MISSING"
    # …and it never silently falls back to the screen root
    assert "never fall back" in excinfo.value.reason


def test_decision_root_equal_to_screen_root_refuses(tmp_path):
    contract = _p1lr_contract(tmp_path)
    contract["decision_run"]["output_root"] = contract["output_root"]
    with pytest.raises(mfs.P1lrModeRefusal) as excinfo:
        mfs.p1lr_mode_binding(contract, "decision")
    assert excinfo.value.code == "P1LR_MODE_ROOTS_COLLIDE"


# ── the auditor's exact acceptance fixture (finding 233) ──

def _p1lr_direct_decision_fixture(tmp_path, **kw):
    """THE auditor's live shape: four DIRECT (nohup, NON-systemd)
    decision workers — one per seed, omega/dragon/gamma/gamma — writing
    under decision_run.output_root, with NO p1lr-decision@ unit loaded
    on any host."""
    kw.setdefault("mode", "decision")
    kw.setdefault("identity", P1LR_DECISION_IDENTITY)
    kw.setdefault("units_loaded", False)   # no systemd unit anywhere
    return _p1lr_fixture(tmp_path, **kw)


def test_acceptance_four_direct_decision_workers_render_4_of_4(tmp_path):
    records = {(101, "P1N_LR1E4"): {"elapsed_seconds": 3600.0,
                                    "mode": "decision",
                                    "evidence_class": "decision_run"}}
    cpath, reader = _p1lr_direct_decision_fixture(tmp_path, records=records)
    packet = _collect_p1lr(tmp_path, cpath, reader, p1lr_mode="decision",
                           p1lr_identity=P1LR_DECISION_IDENTITY)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]

    assert block["state"] == "active"
    assert block["mode"] == "decision"
    assert block["mode_basis"] == "explicit_validated_parameter"
    assert block["identity"] == P1LR_DECISION_IDENTITY
    assert block["output_root"] == _root_of(cpath, "decision")
    assert block["heartbeat_mode_expected"] == "decision"
    assert block["evidence_class_expected"] == "decision_run"
    assert block["unit_template"] == "p1lr-decision@{seed}.service"
    # 4/4 FRESH workers — never the 0/4 the screen root rendered
    assert block["workers_running_fresh"] == {
        "value": 4, "of": 4, "unit": "workers", "horizon": "instant"}

    # per-host record semantics: each seed counts only ITS cells, and
    # the fleet total is 16 bound to the DECISION root
    assert block["workers"]["101"]["records_landed"]["value"] == 1
    assert block["workers"]["101"]["host"] == "omega"
    assert block["workers"]["202"]["records_landed"]["value"] == 0
    assert block["workers"]["303"]["host"] == "gamma"
    assert block["workers"]["404"]["host"] == "gamma"
    assert block["records_landed"]["value"] == 1
    assert block["records_landed"]["of"] == 16
    assert block["records_landed"]["mode"] == "decision"
    assert block["records_landed"]["output_root"] == _root_of(cpath,
                                                              "decision")
    assert "LOCAL output root" in block["records_landed"]["fleet_note"]
    assert block["workers"]["101"]["landed_cells"]["P1N_LR1E4"]["mode"] == \
        "decision"

    # the DIRECT launch is named as such: no unit, so no restart count
    for seed in ("101", "202", "303", "404"):
        worker = block["workers"][seed]
        assert worker["mode"] == "decision"
        assert worker["unit"] == f"p1lr-decision@{seed}.service"
        assert worker["unit_loaded"] is False
        assert worker["restart_count"]["value"] == "unavailable"
        assert worker["launch_durability"]["value"] == "no_unit_loaded"
        assert f"p1lr-decision@{seed}.service" in \
            worker["launch_durability"]["remediation"]

    queue = packet["queue"]
    assert queue[0]["id"] == f"p1lr-factorial-{P1LR_DECISION_IDENTITY}"
    assert queue[0]["mode"] == "decision"
    assert queue[0]["output_root"] == _root_of(cpath, "decision")


def test_acceptance_same_fixture_under_the_screen_root_refuses(tmp_path):
    """The SAME four direct decision workers, read as the screen: a
    typed refusal with NO counts — never the false 0/4, 0/16 that the
    auditor's live invocation produced over four busy GPUs."""
    cpath, reader = _p1lr_direct_decision_fixture(tmp_path)
    # the decision identity exists only under the decision root
    decision_root = Path(_root_of(cpath, "decision"))
    (decision_root / P1LR_DECISION_IDENTITY).mkdir(parents=True)
    Path(_root_of(cpath)).mkdir(parents=True, exist_ok=True)

    packet = _collect_p1lr(tmp_path, cpath, reader, p1lr_mode="screen",
                           p1lr_identity=P1LR_DECISION_IDENTITY)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]

    assert block["state"] == "refused"
    assert block["error_code"] == "P1LR_IDENTITY_MODE_MISMATCH"
    assert block["identity_mode"] == "decision"
    assert block["output_root"] == _root_of(cpath)
    assert block["other_mode_output_root"] == _root_of(cpath, "decision")
    assert block["corrective_command"] == (
        "tools/multifront_status.py --p1lr-mode decision "
        f"--p1lr-identity {P1LR_DECISION_IDENTITY}")
    # NO fabricated counts of any kind
    assert "workers" not in block
    assert "workers_running_fresh" not in block
    assert "records_landed" not in block
    assert "false idle picture" in block["refusal_contract"]
    # the refusal is surfaced, and never enters the executable queue
    fields = {entry["field"] for entry in packet["unavailable"]}
    assert "f1_optimization.active_p1lr_factorial" in fields
    assert not any(str(item["id"]).startswith("p1lr-factorial")
                   for item in packet["queue"])


def test_screen_identity_under_the_decision_root_refuses(tmp_path):
    """The mirror direction refuses identically."""
    cpath, reader = _p1lr_fixture(tmp_path)
    screen_root = Path(_root_of(cpath))
    (screen_root / P1LR_IDENTITY).mkdir(parents=True)
    Path(_root_of(cpath, "decision")).mkdir(parents=True, exist_ok=True)

    packet = _collect_p1lr(tmp_path, cpath, reader, p1lr_mode="decision",
                           p1lr_identity=P1LR_IDENTITY)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    assert block["state"] == "refused"
    assert block["error_code"] == "P1LR_IDENTITY_MODE_MISMATCH"
    assert block["identity_mode"] == "screen"
    assert block["corrective_command"].endswith(
        f"--p1lr-mode screen --p1lr-identity {P1LR_IDENTITY}")
    assert "records_landed" not in block


def test_invalid_mode_refuses_at_the_collector_without_counts(tmp_path):
    cpath, reader = _p1lr_fixture(tmp_path)
    packet = _collect_p1lr(tmp_path, cpath, reader, p1lr_mode="scren")
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    assert block["state"] == "refused"
    assert block["error_code"] == "P1LR_MODE_INVALID"
    assert block["known_modes"] == ["screen", "decision"]
    assert "records_landed" not in block


def test_no_readable_fact_renders_typed_unavailable_never_zero(tmp_path):
    """Finding 233's core: with nothing readable there is nothing to
    count. 0/16 and 0/4 are claims, and a claim needs a fact."""
    cpath, _ = _p1lr_fixture(tmp_path)
    empty = FakeReader(files={})
    packet = _collect_p1lr(tmp_path, cpath, empty)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    assert block["state"] == "unavailable"
    assert block["records_landed"]["value"] == "unavailable"
    assert block["records_landed"]["of"] == 16
    assert block["workers_running_fresh"]["value"] == "unavailable"
    assert "no worker fact readable" in block["records_landed"]["reason"]
    assert block["output_root"] in block["records_landed"]["reason"]


def test_a_record_of_the_other_mode_is_rejected_not_counted(tmp_path):
    """A screen record sitting under the decision root is contamination,
    not decision evidence."""
    records = {(101, "P1N_LR1E4"): {"elapsed_seconds": 10.0,
                                    "mode": "screen",
                                    "evidence_class": "mechanics_screen"},
               (101, "P1N_LR3E5"): {"elapsed_seconds": 20.0,
                                    "mode": "decision",
                                    "evidence_class": "decision_run"}}
    cpath, reader = _p1lr_direct_decision_fixture(tmp_path, records=records)
    packet = _collect_p1lr(tmp_path, cpath, reader, p1lr_mode="decision",
                           p1lr_identity=P1LR_DECISION_IDENTITY)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    assert block["workers"]["101"]["records_landed"]["value"] == 1
    assert block["records_landed"]["value"] == 1
    rejected = block["records_rejected_mode_mismatch"]
    assert len(rejected) == 1
    assert rejected[0]["cell"] == "P1N_LR1E4"
    assert rejected[0]["record_mode"] == "screen"
    assert rejected[0]["expected_mode"] == "decision"


def test_fresh_activity_under_the_other_root_is_announced(tmp_path):
    """Reading the screen while a decision run writes fresh heartbeats
    locally must SAY so — the silent version is the observed defect."""
    cpath, reader = _p1lr_fixture(tmp_path)
    decision_root = Path(_root_of(cpath, "decision"))
    hb = (decision_root / P1LR_DECISION_IDENTITY / "seed101" / "P1N_LR1E4"
          / "heartbeat.json")
    hb.parent.mkdir(parents=True)
    hb.write_text("{}")
    now = datetime.now(timezone.utc)
    packet = _collect_p1lr(tmp_path, cpath, reader, p1lr_now_fn=lambda: now)
    block = packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    advisory = block["other_mode_activity"]
    assert advisory["other_mode"] == "decision"
    assert advisory["other_mode_identity"] == P1LR_DECISION_IDENTITY
    assert advisory["other_mode_output_root"] == _root_of(cpath, "decision")
    assert advisory["corrective_command"].startswith(
        "tools/multifront_status.py --p1lr-mode decision")


def test_heartbeat_declaring_another_mode_is_flagged(tmp_path):
    """Future-proofing: the heartbeat schema carries no mode field
    today, but if one appears it must agree with the root it sits in."""
    contract = _p1lr_contract(tmp_path)
    binding = mfs.p1lr_mode_binding(contract, "decision")
    assert binding["heartbeat_mode_expected"] == "decision"
    cpath, reader = _p1lr_direct_decision_fixture(tmp_path)
    root = _root_of(cpath, "decision")
    key = (HOSTS[101], f"{root}/{P1LR_DECISION_IDENTITY}/seed101/"
                       "P1N_LR1E4/heartbeat.json")
    payload = json.loads(reader.files[key])
    payload["mode"] = "screen"
    reader.files[key] = json.dumps(payload)
    packet = _collect_p1lr(tmp_path, cpath, reader, p1lr_mode="decision",
                           p1lr_identity=P1LR_DECISION_IDENTITY)
    worker = (packet["fronts"]["f1_optimization"]["active_p1lr_factorial"]
              ["workers"]["101"])
    assert worker["heartbeat_mode_expected"] == "decision"
    assert worker["heartbeat_mode_declared"] == "screen"
    assert "does not belong" in worker["heartbeat_mode_mismatch"]


def test_completed_l1_factorial_stays_history_only_while_p1lr_leads(
        tmp_path, monkeypatch):
    """§7.7: the completed old L1 factorial (2de49ea9) is history only —
    with no fresh RUNNING launcher heartbeat it never renders active and
    never enters the executable queue, while the RUNNING P1LR screen
    leads it."""
    l1_cpath, l1_reader = _fixture(
        tmp_path, now=NOW - timedelta(days=1))  # heartbeats a day old
    p1lr_cpath, p1lr_reader = _p1lr_fixture(tmp_path)
    _paused_supervisor(monkeypatch)
    packet = _collect(
        tmp_path, l1_cpath, l1_reader,
        p1lr_contract_path=p1lr_cpath, p1lr_reader=p1lr_reader,
        p1lr_identity=P1LR_IDENTITY, p1lr_now_fn=lambda: NOW,
        p1lr_local_hostname="omega")
    f1 = packet["fronts"]["f1_optimization"]
    assert f1["active_p1lr_factorial"]["state"] == "active"
    assert f1["active_l1_factorial"]["state"] == "inactive_or_unknown"
    ids = [str(item["id"]) for item in packet["queue"]]
    assert ids[0] == f"p1lr-factorial-{P1LR_IDENTITY}"
    assert not any(item_id.startswith("l1-matched-factorial")
                   for item_id in ids)


def test_history_renders_even_without_l1_source(tmp_path, monkeypatch):
    _paused_supervisor(monkeypatch)
    packet = mfs.collect(
        snapshot_path=tmp_path / "m.json",
        watchdog_path=tmp_path / "m2.json",
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://mock", timeout=0.1,
        l0_heartbeat_path=tmp_path / "no-hb.json",
        l0_db_path=tmp_path / "no-l0.sqlite")
    f1 = packet["fronts"]["f1_optimization"]
    assert f1["active_l1_factorial"]["state"] == "unavailable"
    assert f1["doin_campaign_history"]["phase"] == "paused"
    fields = {entry["field"] for entry in packet["unavailable"]}
    assert "f1_optimization.active_l1_factorial" in fields
    # history alone never renders Front 1 as active work: the active
    # block stays explicitly unavailable, and no factorial queue entry
    # is invented
    assert not any(item["id"].startswith("l1-matched-factorial")
                   for item in packet["queue"])
