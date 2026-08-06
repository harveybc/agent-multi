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




REQUIRED_FINITE = ("mean_weekly_return", "total_return",
                   "max_drawdown_fraction")
PACKET_SCHEMA = "agent_multi.eth_curriculum_decision.v1"
RECORD_SCHEMA = "agent_multi.arm_record.v3"


def _finite(value) -> bool:
    try:
        import math
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_packets(seeds: dict, expect_seeds) -> list:
    problems = []
    if sorted(seeds) != sorted(expect_seeds):
        problems.append(
            f"seeds {sorted(seeds)} != expected {sorted(expect_seeds)}")
    identities = {}
    execution_ids = set()
    for seed, packet in sorted(seeds.items()):
        if packet.get("schema") != PACKET_SCHEMA:
            problems.append(
                f"seed {seed}: packet schema"
                f" {packet.get('schema')!r} != {PACKET_SCHEMA!r}")
        identities[seed] = (packet.get("data_sha256"),
                            packet.get("base_contract_sha256"),
                            json.dumps(packet.get("lineage"),
                                       sort_keys=True))
        arms = sorted((packet.get("arms") or {}))
        if arms != sorted(EXPECTED_ARMS):
            problems.append(
                f"seed {seed} arms {arms} != {sorted(EXPECTED_ARMS)}")
        for arm, record in (packet.get("arms") or {}).items():
            where = f"seed {seed} arm {arm}"
            if record.get("schema") != RECORD_SCHEMA:
                problems.append(
                    f"{where}: record schema"
                    f" {record.get('schema')!r} != {RECORD_SCHEMA!r}")
            execution_id = record.get("execution_id")
            if not execution_id:
                problems.append(f"{where}: no execution_id")
            elif execution_id in execution_ids:
                problems.append(
                    f"{where}: duplicate execution_id {execution_id[:16]}")
            else:
                execution_ids.add(execution_id)
            validation = (record.get("splits_raw") or {}).get(
                "validation") or {}
            for key in REQUIRED_FINITE:
                if not _finite(validation.get(key)):
                    problems.append(
                        f"{where}: validation {key} missing/non-finite")
            if arm != "E4":
                terminal = ((record.get("best_checkpoint_vs_terminal")
                             or {}).get("terminal_evaluation") or {})
                terminal_val = (terminal.get("splits_raw") or {}).get(
                    "validation") or {}
                if not terminal.get("artifact_sha256"):
                    problems.append(
                        f"{where}: no terminal artifact hash")
                for key in REQUIRED_FINITE:
                    if not _finite(terminal_val.get(key)):
                        problems.append(
                            f"{where}: terminal {key}"
                            " missing/non-finite")
                artifacts = record.get("artifacts") or {}
                if not artifacts.get("final", artifacts.get("best")):
                    if not artifacts:
                        problems.append(f"{where}: no artifacts map")
            if not record.get("margin_telemetry"):
                problems.append(f"{where}: no margin telemetry")
            if not record.get("return_trace_sha256"):
                problems.append(f"{where}: no return-trace hashes")
            if not record.get("resolved_config_sha256"):
                problems.append(f"{where}: no resolved-config hash")
    if len(set(identities.values())) > 1:
        problems.append(
            "packets carry DIFFERENT data/base/lineage identities:"
            f" {sorted(set(identities.values()))[:2]}...")
    return problems


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

    # AUD-F1-20260806-130/131: versioned packet schema + common
    # experiment identity + finite decision metrics + complete evidence.
    # A truthy mapping is NOT completeness.
    if not args.allow_partial:
        missing = _validate_packets(seeds, args.expect_seeds)
        if missing:
            print(json.dumps({
                "aggregated": False,
                "reason": "incomplete or invalid decision packet",
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
