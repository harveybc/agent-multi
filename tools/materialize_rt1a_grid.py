#!/usr/bin/env python3
"""Materialize the RT1-A grid (owner-amended, Musashi-ruled).

RT1-A (approved): cadences {2, 3, 6, 42} bars = {8, 12, 24, 168} h ×
lookbacks {1y, expanding} × four fixed non-overlapping 2024 blocks ×
two seeds, EACH with a paired frozen (no-update) control.

RT1-B (conditional, NOT materialized here): 18 bars (72 h) and 2y/4y
lookbacks, only under the audit's triggers.

This tool only WRITES the plan; it executes nothing. The performance
sweep is forbidden until Musashi independently verifies the corrected
runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CADENCES = (2, 3, 6, 42)                     # 8h, 12h, 24h, 168h
LOOKBACKS = ("1y", "expanding")
SEEDS = (101, 202)
# Four fixed, non-overlapping 28-day 2024 blocks, all inside the
# validation year and clear of the disclosed 2025 period.
BLOCKS = ("2024-02-01", "2024-05-01", "2024-08-01", "2024-10-01")
CONTROL_MODES = ("adaptive", "frozen")       # frozen = paired control
BLOCK_DAYS = 28


def build_plan(initial_steps: int, update_steps: int) -> dict:
    cells = []
    for block in BLOCKS:
        for cadence in CADENCES:
            for lookback in LOOKBACKS:
                for seed in SEEDS:
                    for mode in CONTROL_MODES:
                        cells.append({
                            "phase": "RT1",
                            "block_start": block,
                            "block_days": BLOCK_DAYS,
                            "cadence_bars": cadence,
                            "cadence_hours": cadence * 4,
                            "lookback": lookback,
                            "seed": seed,
                            "control_mode": mode,
                            "initial_steps": initial_steps,
                            "update_steps": (
                                update_steps if mode == "adaptive"
                                else 0),
                        })
    plan = {
        "schema": "agent_multi.rt1a_plan.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "MATERIALIZED_NOT_EXECUTED",
        "ruling": (
            "RT1-A approved by the owner on Musashi's advice"
            " 2026-08-06: cadences {8h,12h,24h,168h} x {1y,expanding}"
            " x 4 blocks x 2 seeds with paired no-update controls."
            " 72h and 2y/4y lookbacks move to conditional RT1-B."),
        "execution_gate": (
            "no performance sweep until Musashi independently verifies"
            " the corrected v2 runner"),
        "cells": cells,
        "cell_count": len(cells),
        "adaptive_cells": sum(1 for c in cells
                              if c["control_mode"] == "adaptive"),
        "paired_control_cells": sum(1 for c in cells
                                    if c["control_mode"] == "frozen"),
        "rt1b_conditional": {
            "cadences_bars": [18],
            "lookbacks": ["2y", "4y"],
            "status": "NOT_MATERIALIZED_PENDING_TRIGGERS",
        },
    }
    plan["plan_sha256"] = hashlib.sha256(json.dumps(
        plan, sort_keys=True, default=str).encode()).hexdigest()
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "examples/campaigns/rt1a_grid_plan.json")
    parser.add_argument("--initial-steps", type=int, default=20000)
    parser.add_argument("--update-steps", type=int, default=2000)
    args = parser.parse_args()
    plan = build_plan(args.initial_steps, args.update_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "cells": plan["cell_count"],
        "adaptive": plan["adaptive_cells"],
        "paired_controls": plan["paired_control_cells"],
        "plan_sha256": plan["plan_sha256"],
        "status": plan["status"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
