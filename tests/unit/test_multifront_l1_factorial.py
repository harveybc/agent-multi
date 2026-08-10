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
                 latest=None, unreachable=()):
        self.files = dict(files or {})
        self.mtimes = dict(mtimes or {})
        self.restarts = dict(restarts or {})
        self.latest = dict(latest or {})
        self.unreachable = set(unreachable)
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


def _fixture(tmp_path, *, now=NOW, epochs=None, streaks=None):
    contract = _contract(tmp_path)
    cpath = tmp_path / "contract.json"
    cpath.write_text(json.dumps(contract))
    root = contract["output_root"]
    files, mtimes, latest = {}, {}, {}
    for seed, host in HOSTS.items():
        heartbeat = {
            "schema": "agent_multi.l1_launcher_heartbeat.v2",
            "seed": seed, "cell": "L1_N_M10",
            "terminal_state": "RUNNING",
            "pid": 1000 + seed, "pid_start_identity": str(seed * 7),
            "assigned_gpu_uuid": f"GPU-{seed}",
            "cuda_visible_devices": f"GPU-{seed}",
            "observed_gpu_uuids": [f"GPU-{seed}"],
            "progress": "0/4 cells",
            "updated_utc": (now - timedelta(seconds=45)).isoformat(),
        }
        hb_path = f"{root}/{IDENTITY}/seed{seed}/launcher_heartbeat.json"
        files[(host, hb_path)] = json.dumps(heartbeat)
        latest.setdefault(host, hb_path)
        log_path = f"{root}/logs/seed{seed}.log"
        files[(host, log_path)] = _log(
            epoch=(epochs or {}).get(seed, 34),
            streak=(streaks or {}).get(seed, 0))
        mtimes[(host, log_path)] = (now - timedelta(seconds=30)).timestamp()
    reader = FakeReader(files=files, mtimes=mtimes, latest=latest)
    return cpath, reader


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
        "active_l1_factorial"]["workers"]["101"]["eta"]
    assert first["value"] == "unavailable"
    assert "epoch log lines carry no timestamps" in first["missing"]
    final = packets[-1]["fronts"]["f1_optimization"][
        "active_l1_factorial"]["workers"]["101"]["eta"]
    assert final["sample_size"] == {"value": 2, "unit": "observation_pairs"}
    assert final["seconds_per_epoch"]["median"] == 600.0
    assert final["remaining_epochs"] == 1996 - 32
    assert final["eta_seconds"]["value"] == 600.0 * (1996 - 32)
    assert "median(delta_seconds/delta_epochs" in final["formula"]
    assert "uncertainty" in final


def test_cell_eta_requires_observed_durations(tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path)
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader)
    cells_eta = packet["fronts"]["f1_optimization"][
        "active_l1_factorial"]["cells_eta"]
    assert cells_eta["value"] == "unavailable"
    assert "completed cell records" in cells_eta["missing"]


def test_landed_records_counted_and_cell_eta_derived(tmp_path, monkeypatch):
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
    cells_eta = active["cells_eta"]
    assert cells_eta["sample_size"] == {"value": 2, "unit": "completed_cells"}
    assert cells_eta["cells_remaining"] == 14
    assert cells_eta["eta_seconds"]["value"] == 5400.0 * 14
    assert "mean(finished_utc - started_utc" in cells_eta["formula"]


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


# ── identity discovery and history separation ──

def test_identity_discovered_from_latest_heartbeat(tmp_path, monkeypatch):
    cpath, reader = _fixture(tmp_path)
    _paused_supervisor(monkeypatch)
    packet = _collect(tmp_path, cpath, reader, l1_identity=None)
    active = packet["fronts"]["f1_optimization"]["active_l1_factorial"]
    assert active["identity"] == IDENTITY
    assert active["identity_basis"].startswith(
        "discovered_latest_heartbeat_mtime")


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
