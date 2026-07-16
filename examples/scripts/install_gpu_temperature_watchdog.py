#!/usr/bin/env python3
"""Install the GPU temperature watchdog in the current user's crontab."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BEGIN = "# BEGIN agent-multi gpu-temperature-watchdog"
END = "# END agent-multi gpu-temperature-watchdog"


def current_crontab() -> str:
    result = subprocess.run(
        ["crontab", "-l"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def replace_block(existing: str, block: str) -> str:
    lines = existing.splitlines()
    output: list[str] = []
    skipping = False
    for line in lines:
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
    expected_gpus: int,
    interval_minutes: int,
    threshold: float,
    recovery_threshold: float,
    repeat_minutes: float,
) -> str:
    script = repo_root / "tools/gpu_temperature_watchdog.py"
    state_dir = "$HOME/.local/state/agent-multi/gpu-temperature-watchdog"
    command = (
        f"mkdir -p {state_dir} && "
        f"/usr/bin/python3 {script} "
        f"--expected-gpus {expected_gpus} "
        f"--threshold {threshold:g} "
        f"--recovery-threshold {recovery_threshold:g} "
        f"--repeat-minutes {repeat_minutes:g} "
        f">> {state_dir}/cron.log 2>&1"
    )
    return "\n".join([
        BEGIN,
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        f"*/{interval_minutes} * * * * {command}",
        END,
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=78.0)
    parser.add_argument("--recovery-threshold", type=float, default=72.0)
    parser.add_argument("--repeat-minutes", type=float, default=60.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.interval_minutes < 1 or 60 % args.interval_minutes:
        raise ValueError("interval must be a positive divisor of 60")
    block = cron_block(
        repo_root=args.repo_root.resolve(),
        expected_gpus=args.expected_gpus,
        interval_minutes=args.interval_minutes,
        threshold=args.threshold,
        recovery_threshold=args.recovery_threshold,
        repeat_minutes=args.repeat_minutes,
    )
    updated = replace_block(current_crontab(), block)
    if args.apply:
        subprocess.run(
            ["crontab", "-"],
            input=updated,
            text=True,
            check=True,
        )
        print("GPU temperature watchdog cron installed")
    else:
        print(updated, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
