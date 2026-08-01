#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.live_parity import audit_experiment_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of research, live-inference and live-execution parity."
    )
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--live-contract", required=True)
    parser.add_argument(
        "--require",
        choices=("research", "live_inference", "live_execution"),
        default="research",
    )
    args = parser.parse_args()

    report = audit_experiment_files(args.experiment_config, args.live_contract)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report[f"{args.require}_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
