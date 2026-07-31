#!/usr/bin/env python3
"""Install the bounded daily audit-test evidence runner on Omega."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    unit_dir = Path.home() / ".config/systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    service = f"""[Unit]
Description=Materialize bounded regression-test evidence for audits
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={repo}
ExecStart=/usr/bin/python3 {repo / "tools/audit_test_evidence.py"}
Nice=15
IOSchedulingClass=idle
CPUQuota=50%
MemoryMax=1G
TimeoutStartSec=3000
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=%h/.local/state/agent-multi/audit-test-evidence
"""
    timer = """[Unit]
Description=Collect bounded audit-test evidence daily

[Timer]
OnCalendar=*-*-* 03:30:00 America/Bogota
RandomizedDelaySec=10m
Persistent=true
Unit=agent-multi-audit-test-evidence.service

[Install]
WantedBy=timers.target
"""
    (unit_dir / "agent-multi-audit-test-evidence.service").write_text(
        service, encoding="utf-8"
    )
    (unit_dir / "agent-multi-audit-test-evidence.timer").write_text(
        timer, encoding="utf-8"
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        [
            "systemctl",
            "--user",
            "enable",
            "--now",
            "agent-multi-audit-test-evidence.timer",
        ],
        check=True,
    )
    print("Bounded audit-test evidence timer installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
