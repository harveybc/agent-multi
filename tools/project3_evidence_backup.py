#!/usr/bin/env python3
"""Create bounded, consistent backups of the Project 3 evidence OLAP."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "project3.evidence.backup.v1"
BACKUP_PREFIX = "project3-evidence-"
SNAPSHOT_NAME = "project3_evidence.sqlite"
MANIFEST_NAME = "manifest.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quick_check(path: Path) -> None:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30) as connection:
        rows = [str(row[0]) for row in connection.execute("PRAGMA quick_check")]
    if rows != ["ok"]:
        raise RuntimeError(f"snapshot quick_check failed: {rows}")


def _backup_id(created_at: datetime) -> str:
    timestamp = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{BACKUP_PREFIX}{timestamp}-{uuid.uuid4().hex[:8]}"


def _published_backups(destination: Path) -> list[tuple[str, Path]]:
    backups: list[tuple[str, Path]] = []
    for candidate in destination.glob(f"{BACKUP_PREFIX}*"):
        if not candidate.is_dir():
            continue
        try:
            manifest = json.loads(
                (candidate / MANIFEST_NAME).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        if (
            manifest.get("schema_version") != MANIFEST_SCHEMA
            or manifest.get("backup_id") != candidate.name
            or not (candidate / SNAPSHOT_NAME).is_file()
        ):
            continue
        backups.append((str(manifest.get("created_at", "")), candidate))
    return sorted(backups, key=lambda item: (item[0], item[1].name))


def _apply_retention(destination: Path, retention_count: int) -> None:
    backups = _published_backups(destination)
    for _, expired in backups[:-retention_count]:
        shutil.rmtree(expired)
    _fsync_directory(destination)


def create_backup(
    source_db: str | Path,
    destination_dir: str | Path,
    *,
    retention_count: int = 24,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Create, verify, and atomically publish one SQLite snapshot."""
    if retention_count < 1:
        raise ValueError("retention_count must be at least 1")

    source = Path(source_db).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    destination = Path(destination_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    created = created_at or _utc_now()
    if created.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")

    backup_id = _backup_id(created)
    staging = destination / f".{backup_id}.tmp"
    published = destination / backup_id
    staging.mkdir(mode=0o700)
    snapshot = staging / SNAPSHOT_NAME

    try:
        source_uri = f"{source.as_uri()}?mode=ro"
        with sqlite3.connect(source_uri, uri=True, timeout=30) as source_connection:
            source_connection.execute("PRAGMA query_only=ON")
            with sqlite3.connect(snapshot, timeout=30) as snapshot_connection:
                source_connection.backup(snapshot_connection, pages=256, sleep=0.05)
                journal_mode = snapshot_connection.execute(
                    "PRAGMA journal_mode=DELETE"
                ).fetchone()[0]
                if str(journal_mode).lower() != "delete":
                    raise RuntimeError(
                        f"could not make snapshot standalone: {journal_mode}"
                    )

        snapshot.chmod(0o600)
        _quick_check(snapshot)
        snapshot_sha256 = _sha256(snapshot)
        _fsync_file(snapshot)

        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "backup_id": backup_id,
            "created_at": created.astimezone(timezone.utc).isoformat(
                timespec="microseconds"
            ),
            "source_db": str(source),
            "retention_count": retention_count,
            "quick_check": "ok",
            "snapshot": {
                "filename": SNAPSHOT_NAME,
                "size_bytes": snapshot.stat().st_size,
                "sha256": snapshot_sha256,
            },
        }
        manifest_path = staging / MANIFEST_NAME
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest_path.chmod(0o600)
        _fsync_directory(staging)

        os.replace(staging, published)
        _fsync_directory(destination)
        _apply_retention(destination, retention_count)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        **manifest,
        "backup_dir": str(published),
        "manifest_path": str(published / MANIFEST_NAME),
        "snapshot_path": str(published / SNAPSHOT_NAME),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--destination-dir", required=True, type=Path)
    parser.add_argument("--retention-count", type=int, default=24)
    args = parser.parse_args()
    result = create_backup(
        args.source_db,
        args.destination_dir,
        retention_count=args.retention_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
