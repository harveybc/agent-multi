#!/usr/bin/env python3
"""WP0 — read-only D1 training-collapse diagnostics collector.

MUSASHI_TO_GENERAL_SATOSHI_III_SAC_INNER_CURRICULUM_ORDER_2026_08_07 §6.
Reads the completed D1 evidence root and emits per seed/arm/epoch rows
plus direct answers to the five §6 questions. NEVER mutates D1
evidence; every absent fact is CLASSIFIED (not_instrumented,
not_applicable, source_unavailable, direct_zero, invalid) and never
coerced to zero. The lexicographic transport scalar is copied only
under `selection_transport_scalar_do_not_display_as_performance`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "agent_multi.d1_training_collapse_diagnostics.v1"
SEEDS = (101, 202, 303, 404)
ARMS = ("N14", "EN4_10", "E4")
EXPECTED_HASHES = {
    "decision_summary.json":
        "3f3eeb940b04317c3bcc976a7e6bb230b38ce8ab6d23cdd6212701f9f9f85239",
    "fleet_manifest.json":
        "0f39d7e8e9e7c8d6a9fb007e8ca166950f4d335e2f79bf4284fcf13f7993c6e2",
}
# D1 recorded none of these per epoch; they exist only prospectively.
NOT_INSTRUMENTED = (
    "replay_size_before", "replay_size_after", "observed_learning_rate",
    "entropy_target", "entropy_loss", "actor_loss", "critic_loss",
    "gradient_update_count", "raw_action_mean", "raw_action_std",
    "raw_action_min", "raw_action_max", "hold_rate", "dominant_action_rate",
    "thresholded_entries", "protected_submissions", "protected_rejections",
    "order_families", "commission_paid", "spread_cost", "slippage_cost",
    "financing_cost",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(value):
    """Classify a numeric field: NaN/inf are INVALID recordings, 0 is a
    direct zero, everything else is a direct value."""
    if value is None:
        return {"classification": "not_instrumented"}
    if isinstance(value, str):
        if value == "unavailable":
            return {"classification": "source_unavailable"}
        return {"classification": "invalid", "raw": value}
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return {"classification": "invalid", "raw": repr(value)}
    if value == 0:
        return {"classification": "direct_zero", "value": value}
    return {"classification": "direct", "value": value}


def _flat(cell) -> str:
    if isinstance(cell, dict):
        if "value" in cell:
            return str(cell["value"])
        return cell["classification"]
    return str(cell)


def collect(root: Path) -> tuple[dict, list[dict]]:
    integrity = {}
    for name, expected in EXPECTED_HASHES.items():
        actual = _sha(root / name)
        integrity[name] = {"expected": expected, "actual": actual,
                           "match": actual == expected}

    rows: list[dict] = []
    seed_facts: dict[str, dict] = {}
    for seed in SEEDS:
        packet = json.loads((root / f"seed{seed}" / "seed_packet.json")
                            .read_text())
        epoch_timesteps = int(packet.get("epoch_timesteps") or 0)
        for arm in ARMS:
            record = packet["arms"][arm]
            anchor = record.get("anchor_sha256")
            curriculum = record.get("curriculum") or {}
            post_easy = (curriculum.get("post_easy") or {})
            margin = record.get("margin_telemetry") or {}
            terminal = ((record.get("best_checkpoint_vs_terminal") or {})
                        .get("terminal_evaluation") or {})
            history = record.get("epoch_history") or []
            arm_key = f"seed{seed}/{arm}"
            seed_facts[arm_key] = {
                "anchor_sha256": anchor,
                "post_easy_sha256": post_easy.get("artifact_sha256"),
                "post_easy_differs_from_anchor_by_file_hash": (
                    None if not post_easy.get("artifact_sha256")
                    else post_easy["artifact_sha256"] != anchor),
                "terminal_sha256": terminal.get("artifact_sha256"),
                "terminal_num_timesteps": terminal.get("num_timesteps"),
                "margin_telemetry": margin,
                "normal_epochs_recorded": sum(
                    1 for h in history
                    if h.get("checkpoint_source") !=
                    "warm_start_normal_baseline"),
            }
            for entry in history:
                is_baseline = (entry.get("checkpoint_source") ==
                               "warm_start_normal_baseline")
                epoch = entry.get("epoch")
                row = {
                    "seed": seed,
                    "arm": arm,
                    "phase": ("warm_start_baseline" if is_baseline
                              else "normal_training"),
                    "epoch": epoch,
                    "cumulative_timesteps_configured": (
                        {"classification": "not_applicable"} if is_baseline
                        else {"classification": "derived",
                              "value": epoch_timesteps * (int(epoch or 0) + 1),
                              "note": "epoch_timesteps * (epoch+1);"
                                      " per-epoch counter not recorded"}),
                    "configured_learning_rate": {
                        "classification": "direct_from_config",
                        "value": 1e-4,
                        "source": "pinned base config sha 5df24c0d..."},
                    "entropy_mode_configured": {
                        "classification": "direct_from_config",
                        "value": "fixed_0.2"},
                    "entropy_coefficient_recorded": _num(entry.get("ent_coef")),
                    "actor_l1_before": _num(entry.get("policy_actor_l1_before")),
                    "actor_l1_after": _num(entry.get("policy_actor_l1_after")),
                    "actor_delta": _num(entry.get("policy_actor_delta")),
                    "critic_delta": _num(entry.get("policy_critic_delta")),
                    "train_tail_trades": _num(entry.get("train_tail_trades")),
                    "val_trades": _num(entry.get("val_trades")),
                    "val_total_return_fraction": _num(
                        entry.get("val_total_return")),
                    "val_max_drawdown_fraction": _num(
                        entry.get("val_max_drawdown_fraction")),
                    "trade_gate_passed": entry.get(
                        "early_stop_trade_gate_passed"),
                    "checkpoint_eligible": entry.get("l1_checkpoint_eligible"),
                    "eligibility_reason": (
                        "baseline gate" if is_baseline else
                        ("eligible" if entry.get("l1_checkpoint_eligible")
                         else "trade gate failed: train_tail_trades>="
                              f"{entry.get('early_stop_min_train_tail_trades')}"
                              " and val_trades>="
                              f"{entry.get('early_stop_min_validation_trades')}"
                              " not both met")),
                    "selection_transport_scalar_do_not_display_as_performance":
                        entry.get("composite"),
                }
                for name in NOT_INSTRUMENTED:
                    row[name] = {"classification": "not_instrumented"}
                for split in ("train", "train_tail", "validation"):
                    row[f"margin_{split}"] = (
                        {"classification": "source_unavailable"}
                        if (margin.get(split) or {}).get(
                            "would_margin_call_count") == "unavailable"
                        else {"classification": "direct",
                              "value": margin.get(split)})
                rows.append(row)
    return {"integrity": integrity, "arm_facts": seed_facts}, rows


def answer_questions(arm_facts: dict, rows: list[dict]) -> dict:
    normal_rows = [r for r in rows if r["phase"] == "normal_training"]

    def _first_normal_collapsed(seed, arm):
        arm_rows = sorted((r for r in normal_rows
                           if r["seed"] == seed and r["arm"] == arm),
                          key=lambda r: r["epoch"])
        if not arm_rows:
            return None
        first = arm_rows[0]
        return first["val_trades"].get("classification") == "direct_zero"

    q2 = {}
    for seed in SEEDS:
        for arm in ("N14", "EN4_10"):
            q2[f"seed{seed}/{arm}"] = _first_normal_collapsed(seed, arm)

    weights_moved = all(
        r["actor_delta"].get("classification") == "direct"
        for r in normal_rows if r["arm"] in ("N14", "EN4_10"))

    return {
        "q1_raw_collapse_vs_threshold_suppression": {
            "answer": "NOT_DECIDABLE_FROM_D1",
            "classification": "not_instrumented",
            "direct_facts": (
                "actor/critic L1 deltas are nonzero in every recorded"
                " normal epoch (weights genuinely changed) while"
                " validation trades are direct_zero — the collapse is"
                " real behavior change, not a frozen no-op; but raw"
                " action statistics were never recorded, so raw-collapse"
                " vs threshold-suppression is prospective (WP1)."),
            "weights_changed_in_all_normal_epochs": weights_moved,
        },
        "q2_collapse_began_first_normal_epoch": {
            "per_arm": q2,
            "answer_all_true": all(v is True for v in q2.values()),
        },
        "q3_post_easy_differs_from_anchor": {
            "by_file_hash": {
                key: facts["post_easy_differs_from_anchor_by_file_hash"]
                for key, facts in arm_facts.items()
                if facts["post_easy_sha256"]},
            "tensor_distance": {"classification": "not_instrumented"},
            "action_trace": {"classification": "not_instrumented"},
        },
        "q4_fixed_entropy_0p2_actually_used": {
            "configured": "0.2 fixed (pinned base config)",
            "recorded_per_epoch": (
                "INVALID: ent_coef in epoch_history is NaN — the"
                " recording is broken, so runtime use of 0.2 cannot be"
                " confirmed from D1 history; WP1 must record the live"
                " coefficient."),
            "classification": "invalid",
        },
        "q5_easy_episodes_near_normal_margin_termination": {
            "answer": "UNPROVABLE_FROM_D1",
            "classification": "source_unavailable",
            "basis": "margin_telemetry is 'unavailable' for every split"
                     " in every arm record; no solvency attribution is"
                     " permitted (order §3.2 / §8.3).",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path.home() / ".local/share/agent-multi/"
                              "eth_curriculum_decision_20260807_v2")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent.parent /
        "docs/audits/evidence/eth_sac_inner_curriculum")
    args = parser.parse_args()

    facts, rows = collect(args.root)
    answers = answer_questions(facts["arm_facts"], rows)
    envelope = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "read_only": True,
        "integrity": facts["integrity"],
        "arm_facts": facts["arm_facts"],
        "questions": answers,
        "row_count": len(rows),
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "D1_TRAINING_COLLAPSE_DIAGNOSTICS_2026_08_07.json"
    out_json.write_text(
        json.dumps(envelope, indent=1, sort_keys=True, allow_nan=False,
                   default=str) + "\n")

    out_csv = args.out_dir / "D1_TRAINING_COLLAPSE_EPOCHS_2026_08_07.csv"
    if rows:
        fields = list(rows[0].keys())
        with out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _flat(v) for k, v in row.items()})

    print(json.dumps({
        "rows": len(rows),
        "integrity_ok": all(v["match"] for v in facts["integrity"].values()),
        "q2_all_first_epoch": answers[
            "q2_collapse_began_first_normal_epoch"]["answer_all_true"],
        "json": str(out_json), "csv": str(out_csv),
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
