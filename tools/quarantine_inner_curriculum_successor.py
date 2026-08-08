#!/usr/bin/env python3
"""WP0 — atomic quarantine of the invalid M0 mechanism_pass successor.

AUD-F1-20260808-159: the successor was emitted from an experiment whose
easy treatment never executed (all 12 easy arms handed the epoch-0
anchor to normal training). The successor must become permanently
launch-ineligible while every historical byte is preserved append-only.

Behavior (emergency repair spec §3):
1. exclusive file lock; atomic + fsynced replacement;
2. original bytes moved to queue/retired/<sha256>/...;
3. superseding record written at the original path;
4. idempotent — a second invocation changes no bytes;
5. campaign/supervisor ledgers inspected for any consumer claim;
6. m0_correction_envelope_v1.json emitted beside the aggregation,
   binding hashes of aggregation, final table, manifest, supersession.

It never edits m0_aggregation.json, the 16 records or any model ZIP.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SUPERSESSION_SCHEMA = "agent_multi.inner_curriculum_successor_supersession.v1"
ENVELOPE_SCHEMA = "agent_multi.m0_correction_envelope.v1"
DEFAULT_ROOT = Path.home() / (
    ".local/share/agent-multi/eth_sac_inner_curriculum_m0_20260807_v1")
LEDGER_CANDIDATES = (
    Path.home() / ".local/state/agent-multi/doin-campaigns",
    Path.home() / ".local/share/agent-multi/eth_curriculum_decision_20260807_v2",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def inspect_consumers(successor_sha: str) -> dict:
    """Search launch/claim ledgers for any reference to the invalid
    successor. Absence of ledgers is reported as such, never as a
    silent 'no consumer'."""
    findings = {"claimed": False, "references": [], "ledgers_scanned": [],
                "ledgers_missing": []}
    for root in LEDGER_CANDIDATES:
        if not root.exists():
            findings["ledgers_missing"].append(str(root))
            continue
        findings["ledgers_scanned"].append(str(root))
        for path in root.rglob("*.json"):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if ("m0_successor_mechanism_pass" in text
                    or successor_sha in text):
                findings["claimed"] = True
                findings["references"].append(str(path))
    return findings


def quarantine(root: Path) -> dict:
    queue = root / "queue"
    target = queue / "m0_successor_mechanism_pass.json"
    lock_path = queue / ".quarantine.lock"
    queue.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        if not target.exists():
            return {"outcome": "REFUSED",
                    "reason": f"successor not found at {target}"}
        current = json.loads(target.read_text())
        if current.get("schema") == SUPERSESSION_SCHEMA:
            # idempotency: already quarantined; prove zero byte change
            retired = queue / "retired" / current["supersedes_sha256"] / target.name
            return {
                "outcome": "ALREADY_QUARANTINED",
                "supersession_sha256": _sha(target),
                "retired_original_present": retired.is_file(),
                "retired_original_sha256": (
                    _sha(retired) if retired.is_file() else None),
                "bytes_changed": 0,
            }

        original_sha = _sha(target)
        retired_dir = queue / "retired" / original_sha
        retired_dir.mkdir(parents=True, exist_ok=True)
        retired = retired_dir / target.name
        if not retired.exists():
            retired.write_bytes(target.read_bytes())
            with retired.open("rb") as handle:
                os.fsync(handle.fileno())
        if _sha(retired) != original_sha:
            return {"outcome": "REFUSED",
                    "reason": "retired copy hash mismatch; aborting"}

        supersession = {
            "schema": SUPERSESSION_SCHEMA,
            "launch_eligible": False,
            "supersedes_sha256": original_sha,
            "reason_finding": "AUD-F1-20260808-159",
            "preserved_observation": (
                "reduced normal LR/duration retained activity; easy"
                " contribution unmeasured"),
            "superseded_at_utc": datetime.now(timezone.utc).isoformat(),
            "retired_path": str(retired),
        }
        _atomic_write(target, supersession)

        consumers = inspect_consumers(original_sha)

        aggregation = root / "m0_aggregation.json"
        envelope = {
            "schema": ENVELOPE_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reason_findings": ["AUD-F1-20260808-159",
                                "AUD-F1-20260808-160"],
            "historical_evidence_immutable": True,
            "bindings": {
                "m0_aggregation_sha256": (
                    _sha(aggregation) if aggregation.is_file() else None),
                "m0_final_table_csv_sha256": (
                    _sha(root / "m0_final_table.csv")
                    if (root / "m0_final_table.csv").is_file() else None),
                "m0_fleet_manifest_sha256": (
                    _sha(root / "m0_fleet_manifest.json")
                    if (root / "m0_fleet_manifest.json").is_file() else None),
                "successor_supersession_sha256": _sha(target),
                "retired_successor_sha256": original_sha,
            },
            "consumer_inspection": consumers,
            "withdrawn_claim": (
                "mechanism_pass as an easy-vs-normal conclusion is"
                " WITHDRAWN; the preserved narrower observation is"
                " normal fine-tuning rate/duration evidence only"),
        }
        envelope_path = root / "m0_correction_envelope_v1.json"
        _atomic_write(envelope_path, envelope)

        return {
            "outcome": "QUARANTINED",
            "original_sha256": original_sha,
            "retired_path": str(retired),
            "supersession_sha256": _sha(target),
            "envelope_path": str(envelope_path),
            "envelope_sha256": _sha(envelope_path),
            "consumer_inspection": consumers,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = quarantine(args.root)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["outcome"] in ("QUARANTINED",
                                      "ALREADY_QUARANTINED") else 2


if __name__ == "__main__":
    sys.exit(main())
