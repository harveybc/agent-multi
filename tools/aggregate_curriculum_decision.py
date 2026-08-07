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
RECORD_SCHEMA = "agent_multi.arm_record.v6"


def _shared_validator(record: dict, arm: str) -> list:
    """The runner's validator, imported lazily so the aggregator stays
    usable without the training stack when only structure matters."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools.eth_curriculum_decision_experiment import (
            validate_arm_record)
    except Exception as exc:                       # noqa: BLE001
        return [f"shared validator unavailable: {exc}"]
    return validate_arm_record(record, arm)


def _is_hex64(value) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        c in "0123456789abcdefABCDEF" for c in text)


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
        data_sha = packet.get("data_sha256")
        base_sha = packet.get("base_contract_sha256")
        lineage = packet.get("lineage")
        # AUD-F1-20260806-137: empty/malformed identity is NOT a
        # matching identity. Hashes must be 64-hex and lineage non-empty.
        if not _is_hex64(data_sha):
            problems.append(
                f"seed {seed}: data_sha256 {data_sha!r} is not a"
                " 64-hex digest")
        if not _is_hex64(base_sha):
            problems.append(
                f"seed {seed}: base_contract_sha256 {base_sha!r} is not"
                " a 64-hex digest")
        if not isinstance(lineage, dict) or not lineage or not all(
                str(v).strip() for v in lineage.values()):
            problems.append(
                f"seed {seed}: lineage {lineage!r} is empty or invalid")
        identities[seed] = (data_sha, base_sha,
                            json.dumps(lineage, sort_keys=True))
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
            # AUD-F1-20260806-144: the aggregator runs the SAME
            # complete-record validator as the runner — including the
            # filesystem and artifact-load checks. A packet can no
            # longer promote records whose models do not exist or do
            # not load.
            for problem in _shared_validator(record, arm):
                problems.append(f"{where}: {problem}")
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
            before = record.get("code_revisions_before")
            after = record.get("code_revisions_after")
            if not before or not after:
                problems.append(
                    f"{where}: missing per-arm code revisions")
            elif before != after:
                problems.append(
                    f"{where}: code revisions CHANGED during the arm")
            elif isinstance(lineage, dict):
                shared = {k: v for k, v in (after or {}).items()
                          if k in lineage}
                mismatched = {k: (v, lineage[k]) for k, v in
                              shared.items() if v != lineage[k]}
                if mismatched:
                    problems.append(
                        f"{where}: arm lineage {mismatched} does not"
                        " match the packet lineage")
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
    packet_paths = {}
    duplicates = []
    for packet_path in sorted(
            args.output_root.rglob("seed_packet.json")):
        packet = json.loads(packet_path.read_text())
        seed = packet.get("seed")
        # AUD-F1-20260806-137: a second PHYSICAL packet for the same
        # seed must be rejected, never silently overwrite the first.
        if seed in seeds:
            duplicates.append(
                f"seed {seed} has multiple physical packets:"
                f" {packet_paths[seed]} and {packet_path}")
            continue
        seeds[seed] = packet
        packet_paths[seed] = str(packet_path)
    if duplicates and not args.allow_partial:
        print(json.dumps({
            "aggregated": False,
            "reason": "duplicate physical seed packets",
            "problems": duplicates,
        }, indent=1))
        return 1

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
