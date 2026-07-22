#!/usr/bin/env python3
"""Install the DOIN memory-pressure watchdog in the user crontab."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BEGIN = "# BEGIN agent-multi memory-pressure-watchdog"
END = "# END agent-multi memory-pressure-watchdog"


def current_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
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


def cron_block(repo_root: Path, interval_minutes: int, repeat_minutes: int) -> str:
    state_dir = "$HOME/.local/state/agent-multi/memory-pressure-watchdog"
    script = repo_root / "tools/memory_pressure_watchdog.py"
    command = (
        f"mkdir -p {state_dir} && /usr/bin/python3 {script} "
        f"--repeat-minutes {repeat_minutes} >> {state_dir}/cron.log 2>&1"
    )
    return "\n".join([
        BEGIN,
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        f"*/{interval_minutes} * * * * {command}",
        END,
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--interval-minutes", type=int, default=2)
    parser.add_argument("--repeat-minutes", type=int, default=60)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.interval_minutes < 1 or 60 % args.interval_minutes:
        raise ValueError("interval must be a positive divisor of 60")
    block = cron_block(args.repo_root.resolve(), args.interval_minutes, args.repeat_minutes)
    updated = replace_block(current_crontab(), block)
    if args.apply:
        subprocess.run(["crontab", "-"], input=updated, text=True, check=True)
        print("Memory-pressure watchdog cron installed")
    else:
        print(updated, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
