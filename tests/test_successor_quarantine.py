"""WP0 quarantine — corrected per findings 166-168.

The 12 fixture classes of the correction order §4: complete-state
validation on retries, containment-first behavior under incomplete
evidence, typed per-source consumer inspection (JSON/JSONL/SQLite),
unavailable-not-false semantics, byte stability and safe convergence.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from tools.quarantine_inner_curriculum_successor import (
    PRESERVED_OBSERVATION,
    REASON_FINDING,
    SUCCESSOR_NAME,
    SUPERSESSION_SCHEMA,
    inspect_consumers,
    quarantine,
    validate_quarantined_state,
)


def _ledgers(tmp_path: Path) -> tuple:
    ledgers = tmp_path / "isolated_ledgers"
    ledgers.mkdir(exist_ok=True)
    (ledgers / "state.json").write_text(json.dumps({"ok": 1}))
    return (ledgers,)


def _root(tmp_path: Path, with_evidence: bool = True) -> Path:
    root = tmp_path / "m0root"
    (root / "queue").mkdir(parents=True)
    (root / "queue" / SUCCESSOR_NAME).write_text(
        json.dumps({"schema": "agent_multi.m0_successor_job.v1",
                    "branch": "mechanism_pass",
                    "launch_eligible": True}))
    if with_evidence:
        (root / "m0_aggregation.json").write_text(json.dumps({"a": 1}))
        (root / "m0_final_table.csv").write_text("seed,arm\n")
        (root / "m0_fleet_manifest.json").write_text(json.dumps({"m": 1}))
    return root


class TestFirstRunAndRetry:
    def test_first_run_quarantines_with_complete_evidence(self, tmp_path):
        root = _root(tmp_path)
        result = quarantine(root, ledger_roots=_ledgers(root.parent))
        assert result["outcome"] == "QUARANTINED"
        assert result["validation_problems"] == []
        problems, _ = validate_quarantined_state(root)
        assert problems == []

    def test_valid_second_invocation_changes_zero_bytes(self, tmp_path):
        root = _root(tmp_path)
        quarantine(root, ledger_roots=_ledgers(root.parent))
        target = root / "queue" / SUCCESSOR_NAME
        envelope = root / "m0_correction_envelope_v1.json"
        target_before = target.read_bytes()
        envelope_before = envelope.read_bytes()
        second = quarantine(root, ledger_roots=_ledgers(root.parent))
        assert second["outcome"] == "ALREADY_QUARANTINED"
        assert second["bytes_changed"] == 0
        assert target.read_bytes() == target_before
        assert envelope.read_bytes() == envelope_before

    def test_missing_successor_refuses(self, tmp_path):
        root = tmp_path / "empty"
        (root / "queue").mkdir(parents=True)
        assert quarantine(root, ledger_roots=_ledgers(root.parent))["outcome"] == "REFUSED"

    def test_historical_files_untouched(self, tmp_path):
        root = _root(tmp_path)
        agg_before = (root / "m0_aggregation.json").read_bytes()
        quarantine(root, ledger_roots=_ledgers(root.parent))
        quarantine(root, ledger_roots=_ledgers(root.parent))
        assert (root / "m0_aggregation.json").read_bytes() == agg_before


class TestMalformedStateRefusals:
    """Finding 167: schema-string matching is never containment proof."""

    def _quarantined(self, tmp_path):
        root = _root(tmp_path)
        quarantine(root, ledger_roots=_ledgers(root.parent))
        return root

    def test_launch_eligible_true_supersession_is_not_accepted(self, tmp_path):
        root = _root(tmp_path)
        target = root / "queue" / SUCCESSOR_NAME
        target.write_text(json.dumps({
            "schema": SUPERSESSION_SCHEMA,
            "launch_eligible": True,
            "supersedes_sha256": "ab" * 32,
            "reason_finding": REASON_FINDING,
            "preserved_observation": PRESERVED_OBSERVATION,
        }))
        result = quarantine(root, ledger_roots=_ledgers(root.parent))
        assert result["outcome"] != "ALREADY_QUARANTINED"
        after = json.loads(target.read_text())
        assert after["launch_eligible"] is False
        corrupt = list((root / "queue" / "retired").glob("corrupt-*"))
        assert corrupt, "corrupted state must be preserved append-only"

    def test_missing_retired_original_refuses_success(self, tmp_path):
        root = self._quarantined(tmp_path)
        target = root / "queue" / SUCCESSOR_NAME
        sha = json.loads(target.read_text())["supersedes_sha256"]
        (root / "queue" / "retired" / sha / SUCCESSOR_NAME).unlink()
        problems, _ = validate_quarantined_state(root)
        assert any("retired original missing" in p for p in problems)
        result = quarantine(root, ledger_roots=_ledgers(root.parent))
        assert result["outcome"] != "ALREADY_QUARANTINED"

    def test_wrong_retired_hash_refuses(self, tmp_path):
        root = self._quarantined(tmp_path)
        target = root / "queue" / SUCCESSOR_NAME
        sha = json.loads(target.read_text())["supersedes_sha256"]
        (root / "queue" / "retired" / sha / SUCCESSOR_NAME).write_text("tampered")
        problems, _ = validate_quarantined_state(root)
        assert any("hash" in p for p in problems)

    def test_traversal_escape_in_retired_path_refuses(self, tmp_path):
        root = self._quarantined(tmp_path)
        target = root / "queue" / SUCCESSOR_NAME
        record = json.loads(target.read_text())
        record["retired_path"] = str(tmp_path / "outside" / SUCCESSOR_NAME)
        target.write_text(json.dumps(record))
        problems, _ = validate_quarantined_state(root)
        assert any("escape" in p or "match" in p for p in problems)

    def test_bad_sha_syntax_refuses(self, tmp_path):
        root = self._quarantined(tmp_path)
        target = root / "queue" / SUCCESSOR_NAME
        record = json.loads(target.read_text())
        record["supersedes_sha256"] = "NOT-A-SHA"
        target.write_text(json.dumps(record))
        problems, _ = validate_quarantined_state(root)
        assert any("lowercase sha256" in p for p in problems)


class TestEnvelopeAndEvidence:
    """Finding 168: containment first, but no success without evidence."""

    def test_deleted_envelope_refuses_success_and_retry_repairs(self, tmp_path):
        root = _root(tmp_path)
        quarantine(root, ledger_roots=_ledgers(root.parent))
        (root / "m0_correction_envelope_v1.json").unlink()
        problems, _ = validate_quarantined_state(root)
        assert any("envelope missing" in p for p in problems)
        retry = quarantine(root, ledger_roots=_ledgers(root.parent))
        assert retry["outcome"] == "QUARANTINED"          # repaired
        assert (root / "m0_correction_envelope_v1.json").is_file()

    def test_missing_canonical_evidence_yields_typed_incomplete(self, tmp_path):
        root = _root(tmp_path, with_evidence=False)
        result = quarantine(root, ledger_roots=_ledgers(root.parent))
        assert result["outcome"] == "QUARANTINED_EVIDENCE_INCOMPLETE"
        assert result["historical_evidence_immutable"] == "unavailable"
        # containment still holds
        after = json.loads((root / "queue" / SUCCESSOR_NAME).read_text())
        assert after["launch_eligible"] is False
        envelope = json.loads(
            (root / "m0_correction_envelope_v1.json").read_text())
        assert envelope["historical_evidence_immutable"] == "unavailable"
        assert envelope["complete"] is False

    def test_each_single_missing_file_is_incomplete(self, tmp_path):
        for missing in ("m0_aggregation.json", "m0_final_table.csv",
                        "m0_fleet_manifest.json"):
            root = _root(tmp_path / missing.replace(".", "_"))
            (root / missing).unlink()
            result = quarantine(root, ledger_roots=_ledgers(root.parent))
            assert result["outcome"] == "QUARANTINED_EVIDENCE_INCOMPLETE", missing

    def test_retry_completes_previously_incomplete_envelope(self, tmp_path):
        root = _root(tmp_path, with_evidence=False)
        assert quarantine(root, ledger_roots=_ledgers(root.parent))["outcome"] == "QUARANTINED_EVIDENCE_INCOMPLETE"
        (root / "m0_aggregation.json").write_text(json.dumps({"a": 1}))
        (root / "m0_final_table.csv").write_text("seed,arm\n")
        (root / "m0_fleet_manifest.json").write_text(json.dumps({"m": 1}))
        retry = quarantine(root, ledger_roots=_ledgers(root.parent))
        assert retry["outcome"] == "QUARANTINED"
        envelope = json.loads(
            (root / "m0_correction_envelope_v1.json").read_text())
        assert envelope["complete"] is True
        assert envelope["historical_evidence_immutable"] is True

    def test_interrupted_write_converges_on_retry(self, tmp_path):
        root = _root(tmp_path)
        quarantine(root, ledger_roots=_ledgers(root.parent))
        # simulate a crash that left a stale tmp beside the envelope
        (root / "m0_correction_envelope_v1.json.tmp").write_text("garbage")
        retry = quarantine(root, ledger_roots=_ledgers(root.parent))
        assert retry["outcome"] in ("ALREADY_QUARANTINED", "QUARANTINED")
        problems, _ = validate_quarantined_state(root)
        assert problems == []


class TestConsumerInventory:
    """Finding 166: typed per-source evidence, never directory claims."""

    def test_sqlite_claim_is_detected(self, tmp_path):
        ledger_root = tmp_path / "ledgers"
        ledger_root.mkdir()
        db = ledger_root / "campaign_history.sqlite"
        connection = sqlite3.connect(db)
        connection.execute(
            "CREATE TABLE campaigns (job_id TEXT PRIMARY KEY,"
            " domain_id TEXT NOT NULL, plan_hash TEXT NOT NULL,"
            " status TEXT NOT NULL, artifact_sha256 TEXT)")
        connection.execute(
            "INSERT INTO campaigns VALUES ('j1', 'd', 'p', 'launched',"
            " 'deadbeef' )")
        connection.execute(
            "CREATE TABLE worker_events (id INTEGER PRIMARY KEY,"
            " job_id TEXT, node_id TEXT, worker_id TEXT, event TEXT,"
            " detail_json TEXT)")
        secret_sha = hashlib.sha256(b"claimed").hexdigest()
        connection.execute(
            "INSERT INTO worker_events VALUES (1, 'j1', 'omega', 'w',"
            f" 'claimed', '{json.dumps({'ref': secret_sha})}')")
        connection.commit()
        connection.close()
        result = inspect_consumers(secret_sha, roots=(ledger_root,))
        assert result["claimed"] is True
        formats = {s["format"] for s in result["sources"]}
        assert "sqlite" in formats

    def test_jsonl_claim_is_detected(self, tmp_path):
        ledger_root = tmp_path / "ledgers"
        ledger_root.mkdir()
        sha = hashlib.sha256(b"x").hexdigest()
        (ledger_root / "events.jsonl").write_text(
            json.dumps({"event": "noise"}) + "\n"
            + json.dumps({"event": "launch", "successor": sha}) + "\n")
        result = inspect_consumers(sha, roots=(ledger_root,))
        assert result["claimed"] is True

    def test_unknown_format_yields_unavailable_not_false(self, tmp_path):
        ledger_root = tmp_path / "ledgers"
        ledger_root.mkdir()
        (ledger_root / "opaque.bin").write_bytes(b"\x00\x01binary")
        result = inspect_consumers("ab" * 32, roots=(ledger_root,))
        assert result["claimed"] == "unavailable"
        assert str(ledger_root) not in result["roots_fully_scanned"]

    def test_clean_typed_roots_yield_false(self, tmp_path):
        ledger_root = tmp_path / "ledgers"
        ledger_root.mkdir()
        (ledger_root / "state.json").write_text(json.dumps({"ok": 1}))
        (ledger_root / "events.jsonl").write_text(
            json.dumps({"event": "noise"}) + "\n")
        result = inspect_consumers("ab" * 32, roots=(ledger_root,))
        assert result["claimed"] is False
        assert str(ledger_root) in result["roots_fully_scanned"]
        assert result["code_level"]["revision"] != "unavailable"

    def test_unreadable_sqlite_yields_unavailable(self, tmp_path):
        ledger_root = tmp_path / "ledgers"
        ledger_root.mkdir()
        (ledger_root / "broken.sqlite").write_bytes(b"not a database")
        result = inspect_consumers("ab" * 32, roots=(ledger_root,))
        assert result["claimed"] == "unavailable"
