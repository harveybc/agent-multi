#!/usr/bin/env python3
"""Emit a sanitized social evidence packet for a Hermes cron job."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    config = repo / "examples/config/social_intelligence/moltbook_observe_v1.json"
    tool = repo / "tools/social_intelligence.py"
    result = subprocess.run(
        [
            str(Path.home() / "anaconda3/envs/trading-stack/bin/python"),
            str(tool),
            "--config",
            str(config),
            "digest-context",
            "--hours",
            "8",
            "--limit",
            "30",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=45,
    )
    if result.returncode:
        print('{"wakeAgent":false,"reason":"social_context_unavailable"}')
        return result.returncode
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
