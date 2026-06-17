#!/usr/bin/env python3
"""Keep Project 3 weekly pool services alive and recover stale claims."""
from __future__ import annotations

import argparse
import json
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from project3_weekly_pool import connect, init_db, status as pool_status  # noqa: E402


PYTHON_BIN = "/home/harveybc/anaconda3/envs/tensorflow/bin/python"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _unit_active(unit: str) -> bool:
    return _run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0


def _start_dashboard(args: argparse.Namespace) -> None:
    if _unit_active(args.dashboard_unit):
        return
    _run(["systemctl", "--user", "reset-failed", args.dashboard_unit])
    cmd = [
        "systemd-run",
        "--user",
        "--unit",
        args.dashboard_unit.removesuffix(".service"),
        "--same-dir",
        "--collect",
        args.python_bin,
        "-u",
        str(TOOLS_DIR / "project3_weekly_dashboard.py"),
        "--db",
        args.db,
        "--host",
        args.dashboard_host,
        "--port",
        str(args.dashboard_port),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        print(json.dumps({"event": "dashboard_start_failed", "stderr": proc.stderr[-1000:]}), flush=True)


def _start_worker(args: argparse.Namespace) -> None:
    if _unit_active(args.worker_unit):
        return
    _run(["systemctl", "--user", "reset-failed", args.worker_unit])
    existing = _run(["systemctl", "--user", "start", args.worker_unit])
    if existing.returncode == 0:
        return
    cmd = [
        "systemd-run",
        "--user",
        "--unit",
        args.worker_unit.removesuffix(".service"),
        "--same-dir",
        "--collect",
        args.python_bin,
        "-u",
        str(TOOLS_DIR / "project3_weekly_worker.py"),
        "--db",
        args.db,
        "--machine-id",
        args.machine_id,
        "--output-root",
        args.output_root,
        "--python-bin",
        args.python_bin,
        "--poll-sec",
        str(args.worker_poll_sec),
        "--max-subjobs",
        "0",
        "--idle-sleep-sec",
        str(args.worker_idle_sleep_sec),
        "--idle-cycles-before-exit",
        "0",
    ]
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", str(args.cuda_visible_devices)])
    proc = _run(cmd)
    if proc.returncode != 0:
        print(json.dumps({"event": "worker_start_failed", "stderr": proc.stderr[-1000:]}), flush=True)


def _start_phase_orchestrator(args: argparse.Namespace) -> None:
    if not args.phase_orchestrator_enabled:
        return
    if _unit_active(args.phase_orchestrator_unit):
        return
    _run(["systemctl", "--user", "reset-failed", args.phase_orchestrator_unit])
    existing = _run(["systemctl", "--user", "start", args.phase_orchestrator_unit])
    if existing.returncode == 0:
        return
    cmd = [
        "systemd-run",
        "--user",
        "--unit",
        args.phase_orchestrator_unit.removesuffix(".service"),
        "--same-dir",
        "--collect",
        args.python_bin,
        "-u",
        str(TOOLS_DIR / "project3_weekly_phase_orchestrator.py"),
        "--db",
        args.db,
        "--output-dir",
        args.phase_plan_dir,
        "--label-dir",
        args.phase_label_dir,
        "--python-bin",
        args.python_bin,
        "--min-backlog",
        str(args.phase_min_backlog),
        "--sleep-sec",
        str(args.phase_sleep_sec),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        print(json.dumps({"event": "phase_orchestrator_start_failed", "stderr": proc.stderr[-1000:]}), flush=True)


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _recover_stale_running(conn: sqlite3.Connection, *, stale_minutes: int, worker_active: bool) -> int:
    if worker_active or stale_minutes <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)
    stale: list[str] = []
    for row in conn.execute("SELECT external_id, heartbeat_at FROM subjobs WHERE status='running'"):
        heartbeat = _parse_utc(row["heartbeat_at"])
        if heartbeat is None or heartbeat < cutoff:
            stale.append(row["external_id"])
    if not stale:
        return 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with conn:
        for subjob_id in stale:
            conn.execute(
                """
                UPDATE subjobs
                SET status='pending',
                    claimed_by=NULL,
                    claimed_at=NULL,
                    heartbeat_at=NULL,
                    error=?,
                    updated_at=?
                WHERE external_id=?
                """,
                (f"requeued by supervisor after stale running heartbeat > {stale_minutes} minutes", now, subjob_id),
            )
            conn.execute(
                "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    "supervisor_requeue_stale",
                    subjob_id,
                    json.dumps({"stale_minutes": stale_minutes}, sort_keys=True),
                    now,
                ),
            )
    return len(stale)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--machine-id", default=socket.gethostname())
    ap.add_argument("--output-root", default=str(REPO_ROOT / "experiments" / "weekly_walkforward_pool"))
    ap.add_argument("--python-bin", default=PYTHON_BIN)
    ap.add_argument("--cuda-visible-devices")
    ap.add_argument("--worker-unit", default="project3-weekly-worker.service")
    ap.add_argument("--dashboard-unit", default="project3-weekly-dashboard.service")
    ap.add_argument("--dashboard-host", default="127.0.0.1")
    ap.add_argument("--dashboard-port", type=int, default=8787)
    ap.add_argument("--worker-poll-sec", type=int, default=20)
    ap.add_argument("--worker-idle-sleep-sec", type=int, default=60)
    ap.add_argument("--supervisor-sleep-sec", type=int, default=60)
    ap.add_argument("--stale-running-minutes", type=int, default=30)
    ap.add_argument("--phase-orchestrator-enabled", action="store_true")
    ap.add_argument("--phase-orchestrator-unit", default="project3-weekly-phase-orchestrator.service")
    ap.add_argument(
        "--phase-plan-dir",
        default=str(Path("/home/harveybc/Documents/GitHub/financial-data/experiments/weekly_walkforward_pool/auto_phase_plans")),
    )
    ap.add_argument(
        "--phase-label-dir",
        default=str(Path("/home/harveybc/Documents/GitHub/financial-data/experiments/oracle_behavior_pretraining/labels_auto")),
    )
    ap.add_argument("--phase-min-backlog", type=int, default=120)
    ap.add_argument("--phase-sleep-sec", type=int, default=300)
    args = ap.parse_args()

    conn = connect(args.db)
    init_db(conn)
    while True:
        dashboard_active = _unit_active(args.dashboard_unit)
        worker_active = _unit_active(args.worker_unit)
        recovered = _recover_stale_running(
            conn,
            stale_minutes=args.stale_running_minutes,
            worker_active=worker_active,
        )
        _start_dashboard(args)
        _start_worker(args)
        _start_phase_orchestrator(args)
        current = pool_status(conn)
        print(
            json.dumps(
                {
                    "event": "supervisor_tick",
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "dashboard_active": dashboard_active,
                    "worker_active": worker_active,
                    "phase_orchestrator_enabled": args.phase_orchestrator_enabled,
                    "recovered_stale_running": recovered,
                    "subjob_counts": current.get("subjob_counts", {}),
                    "dashboard_url": f"http://{args.dashboard_host}:{args.dashboard_port}",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(max(5, args.supervisor_sleep_sec))


if __name__ == "__main__":
    main()
