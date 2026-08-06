#!/usr/bin/env python3
"""Aggregate per-seed decision packets into the paired comparison table.

Reports each paired seed and median paired differences (EN4_10 − N14 on
2024 validation), direction consistency and effect sizes. Never
replaces the raw table with a composite; the order key is transport
evidence only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

KEYS = ("mean_weekly_return", "annualized_return", "total_return",
        "max_drawdown_fraction", "trades_total")
EXPECTED_ARMS = ("N14", "EN4_10", "E4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expect-seeds", type=int, nargs="+",
                        default=[101, 202, 303, 404])
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="diagnostic only; the packet is then marked partial and "
             "cannot support a promotion claim")
    args = parser.parse_args()

    seeds = {}
    for packet_path in sorted(
            args.output_root.glob("seed*/seed_packet.json")):
        packet = json.loads(packet_path.read_text())
        seeds[packet["seed"]] = packet

    # AUD-F1-20260806-126: STRICT completeness. A campaign-level claim
    # requires exactly the declared seeds and arms, each present exactly
    # once. Anything else fails closed rather than publishing a partial
    # table that reads like a decision.
    if not args.allow_partial:
        missing = []
        if sorted(seeds) != sorted(args.expect_seeds):
            missing.append(
                f"seeds {sorted(seeds)} != expected"
                f" {sorted(args.expect_seeds)}")
        for seed, packet in sorted(seeds.items()):
            arms = sorted((packet.get("arms") or {}))
            if arms != sorted(EXPECTED_ARMS):
                missing.append(
                    f"seed {seed} arms {arms} != {sorted(EXPECTED_ARMS)}")
            for arm, record in (packet.get("arms") or {}).items():
                validation = (record.get("splits_raw") or {}).get(
                    "validation")
                if not validation:
                    missing.append(
                        f"seed {seed} arm {arm} has no validation"
                        " evidence")
        if missing:
            print(json.dumps({
                "aggregated": False,
                "reason": "incomplete decision packet",
                "problems": missing,
            }, indent=1))
            return 1

    rows = []
    paired = {key: [] for key in KEYS}
    for seed, packet in sorted(seeds.items()):
        row = {"seed": seed}
        for arm, record in (packet.get("arms") or {}).items():
            validation = (record.get("splits_raw") or {}).get(
                "validation") or {}
            row[arm] = {key: validation.get(key) for key in KEYS}
            terminal = ((record.get("best_checkpoint_vs_terminal") or {})
                        .get("terminal_evaluation") or {})
            terminal_val = (terminal.get("splits_raw") or {}).get(
                "validation") or {}
            row[f"{arm}__terminal"] = {
                key: terminal_val.get(key) for key in KEYS}
            row[f"{arm}__margin_telemetry"] = (
                record.get("margin_telemetry") or {}).get("validation")
        rows.append(row)
        en = row.get("EN4_10") or {}
        n = row.get("N14") or {}
        for key in KEYS:
            if en.get(key) is not None and n.get(key) is not None:
                paired[key].append(en[key] - n[key])

    def _median(values):
        return statistics.median(values) if values else None

    summary = {
        "schema": "agent_multi.eth_curriculum_decision_summary.v1",
        "complete": not args.allow_partial,
        "promotion_eligible": (
            not args.allow_partial
            and sorted(seeds) == sorted(args.expect_seeds)),
        "seeds": sorted(seeds),
        "per_seed_validation_raw": rows,
        "paired_differences_EN_minus_N": {
            key: {
                "values": paired[key],
                "median": _median(paired[key]),
                "direction_consistency": (
                    None if not paired[key] else
                    max(sum(v > 0 for v in paired[key]),
                        sum(v < 0 for v in paired[key]))
                    / len(paired[key])),
            }
            for key in KEYS
        },
        "note": ("raw same-scale 2024 validation values; four seeds"
                 " show direction consistency and effect size, not a"
                 " p-value; no composite replaces this table"),
    }
    out = args.output_root / "decision_summary.json"
    out.write_text(json.dumps(summary, indent=1, default=str) + "\n",
                   encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
