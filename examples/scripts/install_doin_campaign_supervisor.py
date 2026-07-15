#!/usr/bin/env python3
"""Install the per-host DOIN campaign supervisor as a user systemd service."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SERVICE_NAME = "doin-campaign-supervisor.service"


def service_text(*, repo: Path, python: Path, profile: Path) -> str:
    return f"""[Unit]
Description=Replicated DOIN campaign lifecycle supervisor
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={repo}
Environment=PYTHONPATH={repo}
ExecStart={python} -m app.campaign_supervisor --profile {profile}
Restart=always
RestartSec=3
TimeoutStopSec=20
# DOIN workers use separate process groups and survive a supervisor crash so
# the restarted supervisor can adopt them instead of duplicating a campaign.
KillMode=process

[Install]
WantedBy=default.target
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    profile = args.profile.expanduser().resolve()
    python = args.python.expanduser().resolve()
    if not profile.is_file():
        raise FileNotFoundError(profile)
    if not python.is_file():
        raise FileNotFoundError(python)
    unit_dir = Path.home() / ".config/systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / SERVICE_NAME
    unit_path.write_text(
        service_text(repo=repo, python=python, profile=profile),
        encoding="utf-8",
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    if args.enable:
        subprocess.run(["systemctl", "--user", "enable", SERVICE_NAME], check=True)
    if args.start:
        subprocess.run(["systemctl", "--user", "restart", SERVICE_NAME], check=True)
    print(unit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
