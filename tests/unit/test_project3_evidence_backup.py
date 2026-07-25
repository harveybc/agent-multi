from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from project3_evidence_backup import (  # noqa: E402
    BACKUP_PREFIX,
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    SNAPSHOT_NAME,
    create_backup,
)


def _source_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    connection.execute(
        "CREATE TABLE evidence (id INTEGER PRIMARY KEY, metric_name TEXT, value REAL)"
    )
    connection.executemany(
        "INSERT INTO evidence(metric_name, value) VALUES (?, ?)",
        [("mean_weekly_return", 0.01), ("annualized_return", 0.52)],
    )
    connection.commit()
    return connection


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshot_is_consistent_and_source_remains_usable(tmp_path: Path) -> None:
    source = tmp_path / "evidence.sqlite"
    source_connection = _source_database(source)
    source_connection.execute(
        "INSERT INTO evidence(metric_name, value) VALUES (?, ?)",
        ("uncommitted_during_backup", 99.0),
    )

    result = create_backup(source, tmp_path / "backups", retention_count=3)
    snapshot = Path(result["snapshot_path"])
    with sqlite3.connect(snapshot) as snapshot_connection:
        rows = snapshot_connection.execute(
            "SELECT metric_name, value FROM evidence ORDER BY id"
        ).fetchall()
        assert rows == [
            ("mean_weekly_return", 0.01),
            ("annualized_return", 0.52),
        ]
        assert snapshot_connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"

    source_connection.commit()
    source_connection.execute(
        "INSERT INTO evidence(metric_name, value) VALUES (?, ?)",
        ("annual_rap", 0.31),
    )
    source_connection.commit()
    assert source_connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert source_connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 4
    source_connection.close()


def test_manifest_matches_snapshot_hash_and_atomic_layout(tmp_path: Path) -> None:
    source = tmp_path / "evidence.sqlite"
    _source_database(source).close()
    destination = tmp_path / "backups"

    result = create_backup(source, destination)
    backup_dir = Path(result["backup_dir"])
    manifest = json.loads(
        (backup_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    snapshot = backup_dir / SNAPSHOT_NAME

    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["backup_id"] == backup_dir.name
    assert manifest["quick_check"] == "ok"
    assert manifest["snapshot"]["size_bytes"] == snapshot.stat().st_size
    assert manifest["snapshot"]["sha256"] == _digest(snapshot)
    assert sorted(path.name for path in backup_dir.iterdir()) == [
        MANIFEST_NAME,
        SNAPSHOT_NAME,
    ]
    assert not list(destination.glob(".*.tmp"))


def test_retention_keeps_newest_valid_backups_only(tmp_path: Path) -> None:
    source = tmp_path / "evidence.sqlite"
    connection = _source_database(source)
    destination = tmp_path / "backups"
    base_time = datetime(2026, 7, 25, tzinfo=timezone.utc)

    backup_ids = []
    for offset in range(4):
        result = create_backup(
            source,
            destination,
            retention_count=2,
            created_at=base_time + timedelta(seconds=offset),
        )
        backup_ids.append(result["backup_id"])
        connection.execute(
            "INSERT INTO evidence(metric_name, value) VALUES (?, ?)",
            (f"metric_{offset}", float(offset)),
        )
        connection.commit()

    retained = sorted(
        path.name
        for path in destination.glob(f"{BACKUP_PREFIX}*")
        if path.is_dir()
    )
    assert retained == sorted(backup_ids[-2:])
    assert connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 6
    connection.close()
