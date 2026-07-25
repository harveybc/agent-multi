#!/usr/bin/env python3
"""Install user-systemd services for the Project 3 evidence pool."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/harveybc/anaconda3/envs/trading-stack/bin/python")
USER_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def install_coordinator(args: argparse.Namespace) -> str:
    name = "project3-evidence-pool-api.service"
    unit = f"""
[Unit]
Description=Project 3 evidence-sweep transactional pool API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={REPO}
ExecStart={PYTHON} {REPO / 'tools/project3_evidence_pool_api.py'} --db {args.db} --host 0.0.0.0 --port {args.port} --token-file {args.token_file}
Restart=always
RestartSec=5
MemoryHigh=2G
MemoryMax=3G
NoNewPrivileges=true

[Install]
WantedBy=default.target
"""
    _write(USER_UNIT_DIR / name, unit)
    scheduler_name = "project3-evidence-scheduler.service"
    materialized_plan = Path(args.db).with_name("materialized_campaign.json")
    scheduler_unit = f"""
[Unit]
Description=Project 3 evidence stage scheduler
After={name}
Requires={name}

[Service]
Type=simple
WorkingDirectory={REPO}
ExecStart={PYTHON} {REPO / 'tools/project3_evidence_scheduler.py'} --db {args.db} --materialized-plan {materialized_plan} --poll-seconds 30
Restart=always
RestartSec=5
MemoryHigh=1G
MemoryMax=2G
NoNewPrivileges=true

[Install]
WantedBy=default.target
"""
    _write(USER_UNIT_DIR / scheduler_name, scheduler_unit)
    return name


def install_worker(args: argparse.Namespace) -> str:
    safe_id = args.machine_id.replace("/", "-")
    name = f"project3-evidence-worker-{safe_id}.service"
    unit = f"""
[Unit]
Description=Project 3 evidence worker {args.machine_id}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={REPO}
Environment=PYTHONUNBUFFERED=1
ExecStart={PYTHON} {REPO / 'tools/project3_evidence_worker.py'} --api-url {args.api_url} --token-file {args.token_file} --machine-id {args.machine_id} --poll-seconds 10 --heartbeat-seconds 30
Restart=always
RestartSec=5
MemoryHigh={args.memory_high}
MemoryMax={args.memory_max}
Nice=5
NoNewPrivileges=true

[Install]
WantedBy=default.target
"""
    _write(USER_UNIT_DIR / name, unit)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="role", required=True)
    coordinator = sub.add_parser("coordinator")
    coordinator.add_argument("--db", required=True)
    coordinator.add_argument("--token-file", required=True)
    coordinator.add_argument("--port", type=int, default=8796)
    worker = sub.add_parser("worker")
    worker.add_argument("--api-url", required=True)
    worker.add_argument("--token-file", required=True)
    worker.add_argument("--machine-id", required=True)
    worker.add_argument("--memory-high", default="6G")
    worker.add_argument("--memory-max", default="8G")
    args = parser.parse_args()
    name = install_coordinator(args) if args.role == "coordinator" else install_worker(args)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", name], check=True)
    if args.role == "coordinator":
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "project3-evidence-scheduler.service"],
            check=True,
        )
    subprocess.run(["systemctl", "--user", "--no-pager", "--full", "status", name], check=False)


if __name__ == "__main__":
    main()
