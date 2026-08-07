#!/usr/bin/env python3
"""config-doctor — read-only facade over app/config_validation.

P2 of MUSASHI_DISPOSITION_SATOSHI_III_DETERMINISTIC_TOOLING_2026_08_06.
This CLI REPORTS; it never mutates a config, never authorizes a launch,
and holds no rule of its own — every rule lives in
app/config_validation.py, the same functions the campaign-launch
preflight executes. Run from the tooling environment it will honestly
report metric_resolvable as UNAVAILABLE (the pipeline is not importable
there); the authoritative preflight runs in the real runtime
environment, where the owning pipeline module supplies the implemented
set.

Exit codes: 0 PASS/WARNING · 2 BLOCK · 3 required UNAVAILABLE · 4 harness error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app import config_validation  # noqa: E402

DOCTOR_VERSION = "config_doctor.v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="config JSON files to examine")
    parser.add_argument(
        "--output", default=None, help="write the full report JSON here"
    )
    args = parser.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    implemented = config_validation.runtime_implemented_metrics()
    reports = {}
    worst = config_validation.PASS
    rank = {
        config_validation.PASS: 0,
        config_validation.WARNING: 1,
        config_validation.UNAVAILABLE: 2,
        config_validation.BLOCK: 3,
    }
    for path_value in args.configs:
        path = Path(path_value)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                json.dumps(
                    {
                        "harness_error": f"cannot read {path}: {exc}",
                        "doctor": DOCTOR_VERSION,
                    }
                )
            )
            return 4
        report = config_validation.evaluate(
            document, implemented_metrics=implemented
        )
        report["input"] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        reports[str(path)] = report
        if rank[report["overall"]] > rank[worst]:
            worst = report["overall"]

    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    envelope = {
        "schema": DOCTOR_VERSION,
        "overall": worst,
        "reports": reports,
        "provenance": {
            "doctor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "validators_sha256": hashlib.sha256(
                (REPO / "app/config_validation.py").read_bytes()
            ).hexdigest(),
            "repo_head": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "worktree_dirty": bool(dirty),
            "environment_python": sys.executable,
            "implemented_metrics_available": implemented is not None,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    text = json.dumps(envelope, indent=1, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return {
        config_validation.PASS: 0,
        config_validation.WARNING: 0,
        config_validation.UNAVAILABLE: 3,
        config_validation.BLOCK: 2,
    }[worst]


if __name__ == "__main__":
    raise SystemExit(main())
