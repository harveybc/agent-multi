#!/usr/bin/env python3
"""Verify Project 3 terminal-record integrity (Musashi order 2026-08-04).

Project 3 is terminal: 16,019 archived jobs. This verifier proves, from
direct evidence, that the retained OLAP, final backup and scheduling
silence still match the terminal record. Integrity loss emits a P2
incident; scheduling resurrection emits P3; a healthy pass emits
recoveries. Completion, health and progress never page.

Checks:

1. the OLAP exists and its job count equals the recorded 16,019;
2. the recorded final backup manifest exists, its recorded sha256 matches
   the terminal record, and the snapshot bytes rehash to that sha256
   (``--skip-rehash`` skips the expensive rehash for frequent runs);
3. no project3 systemd unit is enabled or active on this host;
4. no project3 crontab entry exists on this host.

All paths and expectations come from the versioned terminal record; this
tool has no write access to any Project 3 artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORD = REPO_ROOT / "records/project3_terminal_record.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incident_emit  # noqa: E402


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


def check_olap(record: dict) -> list[str]:
    olap = record["olap"]
    path = expand(olap["path"])
    if not path.is_file():
        return [f"OLAP missing: {path}"]
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
        count = conn.execute(
            f"SELECT COUNT(*) FROM {olap['jobs_table']}").fetchone()[0]
        conn.close()
    except sqlite3.Error as exc:
        return [f"OLAP unreadable: {exc}"]
    if count != int(olap["expected_job_count"]):
        return [f"OLAP job count {count} != recorded"
                f" {olap['expected_job_count']}"]
    return []


def check_backup(record: dict, *, skip_rehash: bool) -> list[str]:
    backup = record["final_backup"]
    backup_dir = expand(backup["backups_dir"]) / backup["backup_id"]
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.is_file():
        return [f"final backup manifest missing: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"final backup manifest unreadable: {exc}"]
    recorded = backup["snapshot_sha256"]
    if manifest.get("snapshot", {}).get("sha256") != recorded:
        return ["final backup manifest sha256 does not match the terminal"
                " record"]
    snapshot = backup_dir / backup["snapshot_filename"]
    if not snapshot.is_file():
        return [f"final backup snapshot missing: {snapshot}"]
    if snapshot.stat().st_size != int(backup["snapshot_size_bytes"]):
        return ["final backup snapshot size differs from the terminal"
                " record"]
    if not skip_rehash:
        digest = hashlib.sha256()
        with snapshot.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != recorded:
            return ["final backup snapshot bytes do not rehash to the"
                    " recorded sha256"]
    return []


def check_scheduling() -> list[str]:
    violations = []
    result = subprocess.run(
        ["systemctl", "--user", "list-units", "project3*", "--all",
         "--no-legend", "--plain"],
        capture_output=True, text=True, timeout=20)
    active = [line for line in result.stdout.splitlines() if line.strip()]
    if active:
        violations.append(
            f"project3 systemd units loaded/active: {len(active)}")
    result = subprocess.run(
        ["systemctl", "--user", "list-unit-files", "project3*",
         "--no-legend", "--plain"],
        capture_output=True, text=True, timeout=20)
    enabled = []
    for line in result.stdout.splitlines():
        fields = line.split()
        # Columns: UNIT-FILE STATE [PRESET]; only STATE decides.
        if len(fields) >= 2 and fields[1] == "enabled":
            enabled.append(fields[0])
    if enabled:
        violations.append(f"project3 unit files enabled: {enabled}")
    result = subprocess.run(["crontab", "-l"], capture_output=True,
                            text=True, timeout=20)
    if result.returncode == 0 and "project3" in result.stdout:
        violations.append("crontab contains project3 entries")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--skip-rehash", action="store_true")
    parser.add_argument("--no-emit", action="store_true")
    args = parser.parse_args()

    record = json.loads(args.record.read_text(encoding="utf-8"))
    if record.get("schema") != "agent_multi.project_terminal_record.v1":
        print("terminal record schema mismatch", file=sys.stderr)
        return 2

    integrity = check_olap(record) + check_backup(
        record, skip_rehash=args.skip_rehash)
    scheduling = check_scheduling()

    if not args.no_emit:
        if integrity:
            incident_emit.observe_incident(
                source="project3_terminal_verify",
                event_code="project3_evidence_integrity_loss",
                severity="P2", front="front4",
                summary="Project 3 retained evidence failed verification",
                payload={"violations": integrity,
                         "operator_action":
                         "restore from the recorded final backup; do not"
                         " regenerate evidence"})
        else:
            incident_emit.recover_incident(
                source="project3_terminal_verify",
                event_code="project3_evidence_integrity_loss",
                evidence={"olap_jobs": record["olap"]["expected_job_count"],
                          "backup_id":
                          record["final_backup"]["backup_id"]})
        if scheduling:
            incident_emit.observe_incident(
                source="project3_terminal_verify",
                event_code="project3_scheduling_resurrected",
                severity="P3", front="front4",
                summary="Project 3 scheduling reappeared on this host",
                payload={"violations": scheduling})
        else:
            incident_emit.recover_incident(
                source="project3_terminal_verify",
                event_code="project3_scheduling_resurrected",
                evidence={"scheduling": "silent"})

    verdict = {
        "project": record["project"],
        "state": record["state"],
        "integrity_violations": integrity,
        "scheduling_violations": scheduling,
    }
    print(json.dumps(verdict, sort_keys=True))
    return 0 if not integrity and not scheduling else 1


if __name__ == "__main__":
    raise SystemExit(main())
