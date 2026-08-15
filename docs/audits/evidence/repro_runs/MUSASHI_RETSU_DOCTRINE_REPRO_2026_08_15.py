#!/usr/bin/env python3
"""Reproduce the concrete runtime and coin findings from the Retsu review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DOIN_CORE_CANDIDATES = (
    REPO_ROOT.parent / "doin-core",
    REPO_ROOT.parent.parent / "doin-core",
)
DOIN_CORE_ROOT = next(
    (candidate for candidate in DOIN_CORE_CANDIDATES if candidate.is_dir()),
    DOIN_CORE_CANDIDATES[0],
)
sys.path.insert(0, str(DOIN_CORE_ROOT / "src"))

from doin_core.models.coin import distribute_block_reward  # noqa: E402


def fee_conservation() -> dict[str, object]:
    fees = 10.0
    coinbase = distribute_block_reward(
        block_index=0,
        generator_id="generator-fixture",
        contributors=[],
        tx_fees=fees,
    )
    distributed = coinbase.total_distributed
    available = coinbase.block_reward + fees
    return {
        "available_total": available,
        "block_reward": coinbase.block_reward,
        "distributed_total": distributed,
        "excess": distributed - available,
        "outputs": [
            {
                "amount": output.amount,
                "reason": output.reason,
                "recipient": output.recipient,
            }
            for output in coinbase.outputs
        ],
        "reproduced": abs(distributed - available) > 1e-12,
        "transaction_fees": fees,
    }


def source_tree_drift() -> dict[str, object]:
    experiment_root = (
        Path.home()
        / ".local/share/agent-multi"
        / "p1_difficulty_lr_factorial_20260815_v2"
        / "14e7ce8208ac9776"
    )
    heartbeats = sorted(
        experiment_root.glob("seed101.failed_source_moved_*/P1*/heartbeat.json")
    )
    failures = []
    for heartbeat in heartbeats:
        payload = json.loads(heartbeat.read_text(encoding="utf-8"))
        error = payload.get("error")
        if (
            payload.get("terminal_state") == "CELL_FAILED"
            and isinstance(error, str)
            and "executing source tree moved" in error
        ):
            failures.append(
                {
                    "cell": payload.get("cell"),
                    "error": error,
                    "heartbeat": str(heartbeat),
                    "seed": payload.get("seed"),
                    "updated_utc": payload.get("updated_utc"),
                }
            )
    if not heartbeats:
        return {
            "available": False,
            "experiment_root": str(experiment_root),
            "reproduced": None,
        }
    return {
        "available": True,
        "cells_failed": sorted(failure["cell"] for failure in failures),
        "experiment_identity": "14e7ce8208ac9776",
        "failures": failures,
        "reproduced": len(failures) == 4,
        "seed": 101,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "fee_conservation": fee_conservation(),
        "schema": "agent_multi.musashi_retsu_doctrine_repro.v1",
        "source_tree_drift": source_tree_drift(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["fee_conservation"]["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
