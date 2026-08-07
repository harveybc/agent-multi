#!/usr/bin/env python3
"""Aggregate the M0 mechanism screen and select EXACTLY ONE successor.

WP2/WP3 of the SAC inner-curriculum order §8.3/§9. Survival for an arm
requires the full fact set in >=3/4 seeds; the interpretation table is
frozen here and the selected successor branch is emitted as a queued
job spec. All-fail selects mechanism_fail (R0/R3 localization), never a
repeated broad run. No positive-profit gate exists at M0.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "agent_multi.m0_aggregation.v1"
SEEDS = (101, 202, 303, 404)
ARMS = ("N2_LR1", "E1_N1_LR1", "E1_N1_LR03", "E1_N1_LR01")
EASY_ARMS = ("E1_N1_LR1", "E1_N1_LR03", "E1_N1_LR01")
SURVIVAL_THRESHOLD = 3


def arm_survived_in_seed(record: dict) -> tuple[bool, list[str]]:
    """The §8.3 fact set; every requirement is checked from direct
    facts and a missing fact FAILS the requirement (never assumed)."""
    facts = record.get("decision_facts") or {}
    history = record.get("epoch_history") or []
    last = history[-1] if history else {}
    problems = []
    if facts.get("activity_survived_normal") is not True:
        problems.append("terminal validation trades not > 0")
    raw_std = last.get("val_action_raw_std")
    non_hold = last.get("val_action_non_hold_rate")
    if not (isinstance(raw_std, (int, float)) and raw_std > 0):
        problems.append("zero/unknown raw action dispersion")
    if not (isinstance(non_hold, (int, float)) and non_hold > 0):
        problems.append("no non-hold raw actions")
    submitted = last.get("val_entry_orders_submitted")
    if not (isinstance(submitted, int) and submitted > 0):
        problems.append("no protected entry submitted")
    if facts.get("weights_changed_from_anchor") is not True:
        problems.append("terminal weights equal anchor (or unknown)")
    updates = facts.get("normal_updates_applied")
    if not (isinstance(updates, int) and updates > 0):
        problems.append("no proven normal gradient updates")
    if record.get("terminal_sha256") is None:
        problems.append("terminal artifact missing/unloadable")
    if facts.get("terminal_usable") is not True:
        problems.append("terminal not usable")
    return (not problems), problems


def interpret(survival: dict[str, bool]) -> tuple[str, str]:
    """The frozen §8.3 interpretation table -> (branch, reason)."""
    n2 = survival["N2_LR1"]
    easy_any = any(survival[a] for a in EASY_ARMS)
    reduced_lr = any(survival[a] for a in ("E1_N1_LR03", "E1_N1_LR01"))
    if not n2 and reduced_lr:
        return ("mechanism_pass",
                "N2 fails while reduced-LR E/N survives: supports inner"
                " easy plus gentle normal fine-tuning")
    if n2 and not easy_any:
        return ("mechanism_fail",
                "N2 survives and every E/N fails: the easy handoff is"
                " harmful; localize in R0/R3")
    if not n2 and not easy_any:
        return ("mechanism_fail",
                "all arms fail: proceed to R0/R3 collapse localization,"
                " not larger curriculum confirmation")
    return ("mechanism_pass" if reduced_lr else "mechanism_fail",
            "mixed survival: branch selected by whether a reduced-LR"
            " E/N arm met the 3/4 rule; easy adds no demonstrated value"
            " if only parity was reached (normal-only control retained)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    args = parser.parse_args()

    per_arm: dict[str, dict] = {arm: {} for arm in ARMS}
    missing = []
    for seed in SEEDS:
        for arm in ARMS:
            path = args.root / f"seed{seed}" / arm / "m0_arm_record.json"
            if not path.is_file():
                missing.append(f"seed{seed}/{arm}")
                continue
            record = json.loads(path.read_text())
            ok, problems = arm_survived_in_seed(record)
            per_arm[arm][str(seed)] = {
                "survived": ok, "problems": problems}
    if missing:
        print(json.dumps({"outcome": "WAITING", "missing": missing}))
        return 0

    survival = {
        arm: sum(1 for v in per_arm[arm].values() if v["survived"])
        >= SURVIVAL_THRESHOLD
        for arm in ARMS
    }
    branch, reason = interpret(survival)
    envelope = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_arm": per_arm,
        "arm_survival_3_of_4": survival,
        "selected_successor_branch": branch,
        "interpretation": reason,
        "margin_attribution": (
            "forbidden: D1 margin telemetry was source_unavailable and"
            " M0 makes no solvency claim without direct margin events"),
    }
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    out = args.root / "m0_aggregation.json"
    out.write_text(json.dumps(envelope, indent=1, sort_keys=True) + "\n")
    queued = args.queue_dir / f"m0_successor_{branch}.json"
    queued.write_text(json.dumps({
        "schema": "agent_multi.m0_successor_job.v1",
        "branch": branch,
        "reason": reason,
        "aggregation": str(out),
        "launch_eligible": True,
        "queued_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=1) + "\n")
    print(json.dumps({"outcome": "AGGREGATED", "branch": branch,
                      "survival": survival, "queued": str(queued)},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
