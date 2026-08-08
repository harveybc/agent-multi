#!/usr/bin/env python3
"""Independent adversarial reproduction for Satoshi III WP0 quarantine."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools import quarantine_inner_curriculum_successor as subject


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_original(root: Path) -> Path:
    queue = root / "queue"
    queue.mkdir(parents=True)
    target = queue / "m0_successor_mechanism_pass.json"
    target.write_text(
        json.dumps(
            {
                "schema": "agent_multi.m0_successor_job.v1",
                "branch": "mechanism_pass",
                "launch_eligible": True,
            }
        )
    )
    return target


def main() -> None:
    report: dict[str, object] = {
        "schema": "agent_multi.audit.satoshi_iii_wp0_quarantine_repro.v1",
        "runtime_mutation": False,
        "network_used": False,
        "scenarios": {},
    }

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "malformed_already"
        queue = root / "queue"
        queue.mkdir(parents=True)
        target = queue / "m0_successor_mechanism_pass.json"
        target.write_text(
            json.dumps(
                {
                    "schema": subject.SUPERSESSION_SCHEMA,
                    "launch_eligible": True,
                    "supersedes_sha256": "a" * 64,
                }
            )
        )
        result = subject.quarantine(root)
        report["scenarios"]["malformed_supersession_accepted"] = {
            "result": result,
            "launch_eligible_after": json.loads(target.read_text())["launch_eligible"],
            "reproduced": (
                result["outcome"] == "ALREADY_QUARANTINED"
                and result["retired_original_present"] is False
                and json.loads(target.read_text())["launch_eligible"] is True
            ),
        }

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "missing_bindings"
        write_original(root)
        old_ledgers = subject.LEDGER_CANDIDATES
        subject.LEDGER_CANDIDATES = ()
        try:
            result = subject.quarantine(root)
        finally:
            subject.LEDGER_CANDIDATES = old_ledgers
        envelope = json.loads((root / "m0_correction_envelope_v1.json").read_text())
        missing = [
            name for name, value in envelope["bindings"].items()
            if value is None
        ]
        report["scenarios"]["missing_canonical_evidence_accepted"] = {
            "result_outcome": result["outcome"],
            "missing_bindings": missing,
            "historical_evidence_immutable": envelope["historical_evidence_immutable"],
            "reproduced": (
                result["outcome"] == "QUARANTINED"
                and len(missing) == 3
                and envelope["historical_evidence_immutable"] is True
            ),
        }

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "missing_envelope_on_retry"
        write_original(root)
        for name, data in (
            ("m0_aggregation.json", "{}"),
            ("m0_final_table.csv", "seed,arm\n"),
            ("m0_fleet_manifest.json", "{}"),
        ):
            (root / name).write_text(data)
        old_ledgers = subject.LEDGER_CANDIDATES
        subject.LEDGER_CANDIDATES = ()
        try:
            first = subject.quarantine(root)
            envelope = root / "m0_correction_envelope_v1.json"
            envelope.unlink()
            second = subject.quarantine(root)
        finally:
            subject.LEDGER_CANDIDATES = old_ledgers
        report["scenarios"]["missing_envelope_retry_accepted"] = {
            "first_outcome": first["outcome"],
            "second": second,
            "envelope_present_after_retry": envelope.exists(),
            "reproduced": (
                second["outcome"] == "ALREADY_QUARANTINED"
                and not envelope.exists()
            ),
        }

    with tempfile.TemporaryDirectory() as raw:
        ledger_root = Path(raw) / "real_ledger_format"
        ledger_root.mkdir()
        successor_sha = "b" * 64
        database = ledger_root / "campaign_history.sqlite"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE events (payload TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO events(payload) VALUES(?)",
            (json.dumps({"successor_sha256": successor_sha}),),
        )
        connection.commit()
        connection.close()
        old_ledgers = subject.LEDGER_CANDIDATES
        subject.LEDGER_CANDIDATES = (ledger_root,)
        try:
            inspection = subject.inspect_consumers(successor_sha)
        finally:
            subject.LEDGER_CANDIDATES = old_ledgers
        report["scenarios"]["sqlite_claim_ignored"] = {
            "inspection": inspection,
            "sqlite_contains_successor": successor_sha in database.read_bytes().decode(
                "latin-1"
            ),
            "reproduced": (
                inspection["claimed"] is False
                and str(ledger_root) in inspection["ledgers_scanned"]
                and successor_sha in database.read_bytes().decode("latin-1")
            ),
        }

    runtime_root = Path.home() / (
        ".local/share/agent-multi/eth_sac_inner_curriculum_m0_20260807_v1"
    )
    target = runtime_root / "queue/m0_successor_mechanism_pass.json"
    envelope_path = runtime_root / "m0_correction_envelope_v1.json"
    real: dict[str, object] = {
        "runtime_root": str(runtime_root),
        "target_present": target.is_file(),
        "envelope_present": envelope_path.is_file(),
    }
    if target.is_file() and envelope_path.is_file():
        supersession = json.loads(target.read_text())
        envelope = json.loads(envelope_path.read_text())
        retired = Path(supersession["retired_path"])
        computed = {
            "m0_aggregation_sha256": sha256(runtime_root / "m0_aggregation.json"),
            "m0_final_table_csv_sha256": sha256(runtime_root / "m0_final_table.csv"),
            "m0_fleet_manifest_sha256": sha256(runtime_root / "m0_fleet_manifest.json"),
            "successor_supersession_sha256": sha256(target),
            "retired_successor_sha256": sha256(retired),
        }
        real.update(
            {
                "schema": supersession.get("schema"),
                "launch_eligible": supersession.get("launch_eligible"),
                "retired_present": retired.is_file(),
                "computed_bindings": computed,
                "recorded_bindings": envelope.get("bindings"),
                "bindings_match": computed == envelope.get("bindings"),
                "containment_verified": (
                    supersession.get("schema") == subject.SUPERSESSION_SCHEMA
                    and supersession.get("launch_eligible") is False
                    and retired.is_file()
                    and computed == envelope.get("bindings")
                ),
            }
        )
    report["runtime_observation"] = real
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
