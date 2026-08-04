"""Project 3 terminalization proofs (Musashi order §6 item 17): stale
memory loses to the terminal record; integrity checks fail closed on
tamper; the runtime-facts block is idempotent and precedence-bearing."""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for name in ("fleet_status_context", "project3_terminal_verify"):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
context = sys.modules["fleet_status_context"]
verify = sys.modules["project3_terminal_verify"]

NOW = datetime(2026, 8, 4, 19, 0, 0, tzinfo=timezone.utc)
RECORD_PATH = REPO_ROOT / "records/project3_terminal_record.json"


def real_record() -> dict:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def test_terminal_record_is_valid_and_matches_owner_facts():
    record = real_record()
    assert record["schema"] == "agent_multi.project_terminal_record.v1"
    assert record["state"] == "terminal"
    assert record["completed_jobs"] == 16019
    assert record["final_backup"]["snapshot_sha256"].startswith("73c46d56")


def test_stale_memory_loses_to_terminal_record(tmp_path):
    """Order §6 item 17: memory says running; the record says complete.
    The deterministic context layer must present COMPLETE with explicit
    precedence, and must not describe ongoing work."""
    stale_memory = (
        "# Notes\n"
        "Project 3 is running generation 12 with 4 workers; expect\n"
        "completion next week.\n"
    )
    memory_file = tmp_path / "MEMORY.md"
    memory_file.write_text(stale_memory, encoding="utf-8")

    block = context.render_block([real_record()], ["inventory: no project3"
                                                   " process"], NOW)
    merged = context.merge_managed_block(
        memory_file.read_text(encoding="utf-8"), block)
    memory_file.write_text(merged, encoding="utf-8")

    text = memory_file.read_text(encoding="utf-8")
    managed = text.split(context.BEGIN_MARK)[1].split(context.END_MARK)[0]
    assert "TERMINAL" in managed
    assert "16,019 archived jobs" in managed
    assert "PRECEDENCE" in text.split(context.END_MARK)[0]
    assert "must not be repeated" in managed
    # The stale prose survives outside the managed block but is
    # subordinated by the precedence rule that precedes it.
    assert text.index(context.BEGIN_MARK) < text.index("generation 12")


def test_managed_block_merge_is_idempotent(tmp_path):
    block_one = context.render_block([real_record()], ["fact one"], NOW)
    block_two = context.render_block([real_record()], ["fact two"], NOW)
    merged = context.merge_managed_block("existing notes\n", block_one)
    merged = context.merge_managed_block(merged, block_two)
    assert merged.count(context.BEGIN_MARK) == 1
    assert "fact two" in merged and "fact one" not in merged
    assert "existing notes" in merged


def _fixture_record(tmp_path, *, job_count=3, tamper_snapshot=False,
                    wrong_manifest=False):
    olap = tmp_path / "evidence.sqlite"
    import sqlite3
    conn = sqlite3.connect(olap)
    conn.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY)")
    conn.executemany("INSERT INTO jobs VALUES (?)",
                     [(i,) for i in range(job_count)])
    conn.commit()
    conn.close()

    backup_id = "project3-evidence-test"
    backup_dir = tmp_path / "backups" / backup_id
    backup_dir.mkdir(parents=True)
    snapshot = backup_dir / "snap.sqlite"
    snapshot.write_bytes(b"snapshot-bytes")
    digest = hashlib.sha256(b"snapshot-bytes").hexdigest()
    if tamper_snapshot:
        snapshot.write_bytes(b"tampered-bytes")
    manifest_sha = "0" * 64 if wrong_manifest else digest
    (backup_dir / "manifest.json").write_text(json.dumps(
        {"snapshot": {"sha256": manifest_sha}}), encoding="utf-8")
    return {
        "schema": "agent_multi.project_terminal_record.v1",
        "project": "project3",
        "state": "terminal",
        "olap": {"path": str(olap), "jobs_table": "jobs",
                 "expected_job_count": 3},
        "final_backup": {
            "backup_id": backup_id,
            "backups_dir": str(tmp_path / "backups"),
            "snapshot_filename": "snap.sqlite",
            "snapshot_sha256": digest,
            "snapshot_size_bytes": len(b"snapshot-bytes"),
        },
    }


def test_intact_fixture_passes_integrity(tmp_path):
    record = _fixture_record(tmp_path)
    assert verify.check_olap(record) == []
    assert verify.check_backup(record, skip_rehash=False) == []


def test_snapshot_tamper_is_integrity_loss(tmp_path):
    record = _fixture_record(tmp_path, tamper_snapshot=True)
    violations = verify.check_backup(record, skip_rehash=False)
    assert any("size" in v or "rehash" in v for v in violations)


def test_manifest_tamper_is_integrity_loss(tmp_path):
    record = _fixture_record(tmp_path, wrong_manifest=True)
    violations = verify.check_backup(record, skip_rehash=True)
    assert any("does not match" in v for v in violations)


def test_missing_olap_and_wrong_count_fail(tmp_path):
    record = _fixture_record(tmp_path)
    record["olap"]["expected_job_count"] = 999
    assert verify.check_olap(record)
    record["olap"]["path"] = str(tmp_path / "absent.sqlite")
    assert verify.check_olap(record)
