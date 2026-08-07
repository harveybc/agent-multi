#!/usr/bin/env python3
"""Install the bounded Hermes social-enrichment worker as user systemd units."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SERVICE = "agent-multi-social-enrichment.service"
TIMER = "agent-multi-social-enrichment.timer"


def service_text(repo: Path, python: Path) -> str:
    tool = repo / "tools/social_intelligence_enrichment.py"
    config = repo / "examples/config/social_intelligence/moltbook_observe_v1.json"
    prompt = repo / "examples/prompts/moltbook_enrichment_v1.txt"
    return f"""[Unit]
Description=Persist bounded Hermes Moltbook enrichment
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={repo}
ExecStart={python} {tool} --config {config} run --prompt {prompt}
Nice=15
IOSchedulingClass=idle
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=%h/.local/state/agent-multi %h/.hermes
MemoryMax=1G
CPUQuota=75%
TimeoutStartSec=300
StandardOutput=append:%h/.local/state/agent-multi/social-enrichment.log
StandardError=append:%h/.local/state/agent-multi/social-enrichment.log
"""


def timer_text(interval_minutes: int) -> str:
    return f"""[Unit]
Description=Run bounded social enrichment every {interval_minutes} minutes

[Timer]
OnBootSec=5m
OnUnitActiveSec={interval_minutes}m
RandomizedDelaySec=60
Persistent=true
Unit={SERVICE}

[Install]
WantedBy=timers.target
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path.home() / "anaconda3/envs/trading-stack/bin/python",
    )
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not 15 <= args.interval_minutes <= 1440:
        raise ValueError("interval_minutes must be within [15,1440]")
    repo = args.repo_root.resolve()
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    contents = {
        SERVICE: service_text(repo, python),
        TIMER: timer_text(args.interval_minutes),
    }
    if not args.apply:
        for name, text in contents.items():
            print(f"### {name}\n{text}")
        return 0
    unit_dir = Path.home() / ".config/systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    for name, text in contents.items():
        (unit_dir / name).write_text(text, encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", TIMER], check=True
    )
    print(f"Installed and enabled {TIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
