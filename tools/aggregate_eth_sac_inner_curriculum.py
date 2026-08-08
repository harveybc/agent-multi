#!/usr/bin/env python3
"""Aggregate the M0 mechanism screen and select EXACTLY ONE successor.

Hardened per Musashi's in-flight order (2026-08-07): every one of the
16 records is VERIFIED before it may vote — contract binding, artifact
hashes on disk, uniform code revisions within and ACROSS records, real
compute, absence of 2025, loadable artifacts, protected entries and
absence of errors. A record that fails verification cannot count as
survival OR as failure evidence; it blocks aggregation with reasons.

Also emits the final per-seed/arm metrics table (CSV + markdown) and
the four-seed fleet manifest. Replication is a separate explicit phase
(--replicate) using the contract's replica topology. No positive-profit
gate exists at M0; margin events are typed, never coerced to zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCHEMA = "agent_multi.m0_aggregation.v2"
SEEDS = (101, 202, 303, 404)
ARMS = ("N2_LR1", "E1_N1_LR1", "E1_N1_LR03", "E1_N1_LR01")
EASY_ARMS = ("E1_N1_LR1", "E1_N1_LR03", "E1_N1_LR01")
SURVIVAL_THRESHOLD = 3
CONTRACT_PATH = REPO / "examples/config/phase_3_eth_sac_dynamics/m0_contract.json"
LEARNING_STARTS = 1000                    # D1-identical SAC config
EPOCH_TIMESTEPS = 20000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------- verification


def verify_m0_record(record: dict, arm: str, seed: int, *,
                     contract: dict, contract_sha: str,
                     load_proof: bool) -> list[str]:
    """The eight verification classes. Every problem is a reason string;
    an empty list means the record may vote."""
    problems: list[str] = []

    # 1. contract binding
    if record.get("m0_contract_sha256") != contract_sha:
        problems.append(
            f"contract hash {str(record.get('m0_contract_sha256'))[:16]}"
            f" != frozen {contract_sha[:16]}")
    spec = contract["arms"].get(arm) or {}
    if record.get("arm") != arm or record.get("seed") != seed:
        problems.append("record arm/seed does not match its location")
    if record.get("arm_spec") != spec:
        problems.append("arm_spec differs from the frozen contract")
    if record.get("easy_learning_rate") != contract["easy_learning_rate"]:
        problems.append("easy_learning_rate differs from contract")

    # 2. hashes: anchor binding + artifacts on disk
    anchor_expected = contract["anchors"][str(seed)]["sha256"]
    if record.get("anchor_sha256") != anchor_expected:
        problems.append("anchor sha does not match the contract anchor")
    terminal_sha = record.get("terminal_sha256")
    terminal_path = ((record.get("terminal_evaluation") or {})
                     .get("artifact_path"))
    if not terminal_sha or not terminal_path:
        problems.append("terminal artifact/hash missing from record")
    else:
        disk = Path(terminal_path)
        if not disk.is_file():
            problems.append(f"terminal artifact not on disk: {disk}")
        elif _sha(disk) != terminal_sha:
            problems.append("terminal artifact hash mismatch on disk")
        # 6. loadable artifacts
        elif load_proof:
            try:
                from stable_baselines3 import SAC
                SAC.load(str(disk), device="cpu")
            except Exception as exc:              # noqa: BLE001
                problems.append(
                    f"terminal artifact NOT loadable: "
                    f"{type(exc).__name__}: {str(exc)[:120]}")
        else:
            try:
                if zipfile.ZipFile(disk).testzip() is not None:
                    problems.append("terminal zip fails integrity test")
            except zipfile.BadZipFile:
                problems.append("terminal artifact is not a zip")

    # 3. uniform revision inside the record
    before = record.get("code_revisions_before")
    after = record.get("code_revisions_after")
    if not before or before != after:
        problems.append(
            f"code revisions moved during the arm: {before} -> {after}")

    # 4. real compute
    updates = (record.get("decision_facts") or {}).get(
        "normal_updates_applied")
    expected_updates = (spec.get("normal_epochs", 0) * EPOCH_TIMESTEPS
                        - LEARNING_STARTS)
    if updates != expected_updates:
        problems.append(
            f"gradient updates {updates!r} != expected"
            f" {expected_updates} (normal_epochs x {EPOCH_TIMESTEPS}"
            f" - learning_starts {LEARNING_STARTS})")
    history = record.get("epoch_history") or []
    normal_rows = [row for row in history
                   if row.get("checkpoint_source")
                   != "warm_start_normal_baseline"]
    if len(normal_rows) != spec.get("normal_epochs"):
        problems.append(
            f"{len(normal_rows)} normal epochs recorded, contract says"
            f" {spec.get('normal_epochs')}")

    # 5. absence of 2025 anywhere in evaluation evidence
    splits_raw = ((record.get("terminal_evaluation") or {})
                  .get("splits_raw") or {})
    if "test" in splits_raw:
        problems.append("terminal evaluation carries the 2025 test split")
    blob = json.dumps(splits_raw) + json.dumps(history[-1] if history else {})
    if "2025-" in blob:
        problems.append("a 2025 date appears in evaluation evidence")

    # 7. protected entries: diagnostics must be PRESENT; a survival
    # claim without a submitted protected entry is inconsistent
    last = normal_rows[-1] if normal_rows else {}
    submitted = last.get("val_entry_orders_submitted")
    if submitted is None:
        problems.append("entry-order diagnostics missing from last epoch")
    facts = record.get("decision_facts") or {}
    survived = facts.get("activity_survived_normal")
    if survived and not (isinstance(submitted, int) and submitted > 0):
        problems.append(
            "claims survival but no protected entry was submitted")

    # 8. absence of errors / completeness
    for key in ("execution_id", "started_utc", "finished_utc",
                "boundary_transfer_evidence"):
        if not record.get(key):
            problems.append(f"missing {key}")
    boundary = record.get("boundary_transfer_evidence") or {}
    if boundary and boundary.get(
            "policy_hash_matches_source_after_transfer") is not True:
        problems.append("boundary transfer hash proof absent or false")
    return problems


def verify_cross_record_uniformity(records: dict[str, dict]) -> list[str]:
    """One revision set and one contract hash across ALL 16 records."""
    problems = []
    revisions = {json.dumps(r.get("code_revisions_before"),
                            sort_keys=True) for r in records.values()}
    if len(revisions) > 1:
        problems.append(
            f"records span {len(revisions)} different code revisions")
    contracts = {r.get("m0_contract_sha256") for r in records.values()}
    if len(contracts) > 1:
        problems.append("records bind different contract hashes")
    return problems


# ------------------------------------------------------------ survival


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


# ------------------------------------------------------- table/manifest


def _metric(value):
    return value if value is not None else "unavailable"


def final_table_rows(records: dict[str, dict]) -> list[dict]:
    """Musashi task 5: simple per-seed/arm metrics — trades, return,
    Sharpe, drawdown, activity, action dispersion, updates, margin."""
    rows = []
    for key in sorted(records):
        record = records[key]
        seed, arm = key.split("/")
        history = record.get("epoch_history") or []
        last = history[-1] if history else {}
        validation = ((record.get("terminal_evaluation") or {})
                      .get("splits_raw") or {}).get("validation") or {}
        facts = record.get("decision_facts") or {}
        margin = validation.get("would_margin_call_count")
        rows.append({
            "seed": seed,
            "arm": arm,
            "terminal_val_trades": _metric(validation.get("trades_total")),
            "terminal_val_total_return": _metric(
                validation.get("total_return")),
            "terminal_val_mean_weekly_return": _metric(
                validation.get("mean_weekly_return")),
            "terminal_val_sharpe": _metric(validation.get("sharpe_ratio")),
            "terminal_val_max_drawdown_fraction": _metric(
                validation.get("max_drawdown_fraction")),
            "activity_survived_normal": facts.get(
                "activity_survived_normal"),
            "raw_action_std": _metric(last.get("val_action_raw_std")),
            "non_hold_rate": _metric(last.get("val_action_non_hold_rate")),
            "normal_updates_applied": _metric(
                facts.get("normal_updates_applied")),
            "margin_events": (margin if margin is not None
                              else "unavailable"),
            "terminal_usable": facts.get("terminal_usable"),
            "anchor_selected_as_best": facts.get("anchor_selected_as_best"),
        })
    return rows


def write_table(rows: list[dict], root: Path) -> dict:
    import csv

    csv_path = root / "m0_final_table.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = root / "m0_final_table.md"
    header = "| " + " | ".join(rows[0].keys()) + " |"
    sep = "|" + "|".join(["---"] * len(rows[0])) + "|"
    lines = [header, sep] + [
        "| " + " | ".join(str(v) for v in row.values()) + " |"
        for row in rows
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return {"csv": str(csv_path), "markdown": str(md_path)}


def build_fleet_manifest(root: Path, records: dict[str, dict],
                         contract_sha: str) -> Path:
    """Musashi task 4: the four-seed final manifest, content-hashed."""
    manifest = {
        "schema": "agent_multi.m0_fleet_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "m0_contract_sha256": contract_sha,
        "seeds": {},
    }
    for seed in SEEDS:
        packet = root / f"seed{seed}" / "m0_seed_packet.json"
        manifest["seeds"][str(seed)] = {
            "seed_packet_sha256": _sha(packet) if packet.is_file()
            else "missing",
            "arms": {
                arm: {
                    "record_sha256": _sha(
                        root / f"seed{seed}" / arm / "m0_arm_record.json"),
                    "terminal_sha256": records[
                        f"seed{seed}/{arm}"].get("terminal_sha256"),
                    "anchor_sha256": records[
                        f"seed{seed}/{arm}"].get("anchor_sha256"),
                }
                for arm in ARMS
            },
        }
    path = root / "m0_fleet_manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return path


def replicate_evidence(root: Path, contract: dict) -> dict:
    """Musashi task 4: replicate records + manifest per the contract
    topology with an INDEPENDENT remote observation (151/159/160
    discipline; namespaced by experiment/seed/arm — no collisions)."""
    from tools.eth_curriculum_decision_experiment import _replicate_to_remote

    observations = {}
    for seed in SEEDS:
        worker = contract["workers"][str(seed)]
        authority = contract["replica_topology"][worker]
        source = root / f"seed{seed}" / "m0_seed_packet.json"
        label = f"eth_sac_inner_curriculum_m0/seed{seed}/m0_seed_packet.json"
        observations[label] = _replicate_to_remote(source, label, authority)
        for arm in ARMS:
            source = root / f"seed{seed}" / arm / "m0_arm_record.json"
            label = (f"eth_sac_inner_curriculum_m0/seed{seed}/{arm}/"
                     "m0_arm_record.json")
            observations[label] = _replicate_to_remote(
                source, label, authority)
    manifest_path = root / "m0_fleet_manifest.json"
    if manifest_path.is_file():
        observations["eth_sac_inner_curriculum_m0/m0_fleet_manifest.json"] = (
            _replicate_to_remote(
                manifest_path,
                "eth_sac_inner_curriculum_m0/m0_fleet_manifest.json",
                contract["replica_topology"]["omega"]))
    return observations


# --------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--load-proof", action="store_true",
                        help="prove every terminal artifact loads as a"
                             " real SAC (slower; required for the final"
                             " packet)")
    parser.add_argument("--replicate", action="store_true",
                        help="replicate records per the contract"
                             " topology and record remote observations")
    args = parser.parse_args()

    contract = json.loads(CONTRACT_PATH.read_text())
    contract_sha = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()

    records: dict[str, dict] = {}
    verification: dict[str, list[str]] = {}
    missing = []
    for seed in SEEDS:
        for arm in ARMS:
            path = args.root / f"seed{seed}" / arm / "m0_arm_record.json"
            if not path.is_file():
                missing.append(f"seed{seed}/{arm}")
                continue
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                verification[f"seed{seed}/{arm}"] = [f"corrupt json: {exc}"]
                continue
            key = f"seed{seed}/{arm}"
            records[key] = record
            verification[key] = verify_m0_record(
                record, arm, seed, contract=contract,
                contract_sha=contract_sha, load_proof=args.load_proof)
    if missing:
        print(json.dumps({"outcome": "WAITING", "missing": missing,
                          "records_landed": len(records)}))
        return 0

    cross = verify_cross_record_uniformity(records)
    invalid = {k: v for k, v in verification.items() if v}
    if invalid or cross:
        print(json.dumps({
            "outcome": "VERIFICATION_FAILED",
            "cross_record_problems": cross,
            "invalid_records": invalid,
            "note": "an unverifiable record can neither survive nor"
                    " count as failure evidence; aggregation refused",
        }, indent=1))
        return 2

    per_arm: dict[str, dict] = {arm: {} for arm in ARMS}
    for key, record in records.items():
        seed, arm = key.split("/")
        ok, problems = arm_survived_in_seed(record)
        per_arm[arm][seed] = {"survived": ok, "problems": problems}
    survival = {
        arm: sum(1 for v in per_arm[arm].values() if v["survived"])
        >= SURVIVAL_THRESHOLD
        for arm in ARMS
    }
    branch, reason = interpret(survival)

    rows = final_table_rows(records)
    table = write_table(rows, args.root)
    manifest = build_fleet_manifest(args.root, records, contract_sha)
    replicas = None
    if args.replicate:
        replicas = replicate_evidence(args.root, contract)

    envelope = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification": {"all_16_verified": True,
                         "load_proof": bool(args.load_proof)},
        "per_arm": per_arm,
        "arm_survival_3_of_4": survival,
        "selected_successor_branch": branch,
        "interpretation": reason,
        "final_table": table,
        "fleet_manifest": str(manifest),
        "replica_observations": replicas,
        "margin_attribution": (
            "forbidden: margin telemetry is typed unavailable unless"
            " directly exported; no solvency claim is made"),
    }
    out = args.root / "m0_aggregation.json"
    out.write_text(json.dumps(envelope, indent=1, sort_keys=True,
                              default=str) + "\n")
    args.queue_dir.mkdir(parents=True, exist_ok=True)
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
                      "survival": survival, "queued": str(queued),
                      "table": table["markdown"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
