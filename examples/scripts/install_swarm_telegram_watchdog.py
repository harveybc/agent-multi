#!/usr/bin/env python3
"""Install the DOIN swarm Telegram watchdog in the user crontab."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BEGIN = "# BEGIN agent-multi swarm-telegram-watchdog"
END = "# END agent-multi swarm-telegram-watchdog"


def current_crontab() -> str:
    result = subprocess.run(
        ["crontab", "-l"], capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else ""


def replace_block(existing: str, block: str) -> str:
    output: list[str] = []
    skipping = False
    for line in existing.splitlines():
        if line.strip() == BEGIN:
            skipping = True
            continue
        if line.strip() == END:
            skipping = False
            continue
        if not skipping:
            output.append(line)
    while output and not output[-1].strip():
        output.pop()
    if output:
        output.append("")
    output.extend(block.splitlines())
    return "\n".join(output) + "\n"


def cron_block(
    *,
    repo_root: Path,
    profile: Path,
    interval_minutes: int,
    stale_minutes: float,
    stall_minutes: float,
    repeat_minutes: float,
) -> str:
    script = repo_root / "tools/swarm_telegram_watchdog.py"
    log_dir = "$HOME/.local/state/agent-multi/swarm-telegram-watchdog"
    command = (
        f"mkdir -p {log_dir} && "
        f"/usr/bin/python3 {script} "
        f"--profile {profile} "
        f"--stale-minutes {stale_minutes:g} "
        f"--stall-minutes {stall_minutes:g} "
        f"--repeat-minutes {repeat_minutes:g} "
        f">> {log_dir}/cron.log 2>&1"
    )
    return "\n".join([
        BEGIN,
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        f"*/{interval_minutes} * * * * {command}",
        END,
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--stale-minutes", type=float, default=10.0)
    parser.add_argument("--stall-minutes", type=float, default=120.0)
    parser.add_argument("--repeat-minutes", type=float, default=60.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.interval_minutes < 1 or 60 % args.interval_minutes:
        raise ValueError("interval must be a positive divisor of 60")
    profile = args.profile.expanduser().resolve()
    if not profile.is_file():
        raise FileNotFoundError(profile)
    block = cron_block(
        repo_root=args.repo_root.resolve(),
        profile=profile,
        interval_minutes=args.interval_minutes,
        stale_minutes=args.stale_minutes,
        stall_minutes=args.stall_minutes,
        repeat_minutes=args.repeat_minutes,
    )
    updated = replace_block(current_crontab(), block)
    if args.apply:
        subprocess.run(["crontab", "-"], input=updated, text=True, check=True)
        print("DOIN swarm Telegram watchdog cron installed")
    else:
        print(updated, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
