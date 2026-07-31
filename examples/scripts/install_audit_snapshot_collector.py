#!/usr/bin/env python3
"""Install the deterministic audit snapshot collector as a user timer."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SERVICE_NAME = "agent-multi-audit-snapshot.service"
TIMER_NAME = "agent-multi-audit-snapshot.timer"


def service_unit(repo_root: Path) -> str:
    script = repo_root / "tools/audit_snapshot_collector.py"
    return f"""[Unit]
Description=Collect a redacted agent-multi audit evidence snapshot
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={repo_root}
ExecStart=/usr/bin/python3 {script}
Nice=10
IOSchedulingClass=idle
CPUQuota=20%
MemoryMax=256M
TimeoutStartSec=150
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=%h/.local/state/agent-multi/audit-snapshots
"""


def timer_unit() -> str:
    return """[Unit]
Description=Collect redacted audit evidence every six hours

[Timer]
OnCalendar=*-*-* 00,06,12,18:15:00
Persistent=true
RandomizedDelaySec=10m
Unit=agent-multi-audit-snapshot.service

[Install]
WantedBy=timers.target
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    service = service_unit(repo_root)
    timer = timer_unit()
    if not args.apply:
        print(f"# {SERVICE_NAME}\n{service}")
        print(f"# {TIMER_NAME}\n{timer}")
        return 0

    unit_dir = Path.home() / ".config/systemd/user"
    state_dir = Path.home() / ".local/state/agent-multi/audit-snapshots"
    unit_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / SERVICE_NAME).write_text(service, encoding="utf-8")
    (unit_dir / TIMER_NAME).write_text(timer, encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", TIMER_NAME],
        check=True,
    )
    subprocess.run(
        ["systemctl", "--user", "start", SERVICE_NAME],
        check=True,
    )
    print(f"Installed {TIMER_NAME} and collected the initial snapshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
