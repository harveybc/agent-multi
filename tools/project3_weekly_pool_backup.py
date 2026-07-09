#!/usr/bin/env python3
"""Create a portable Project 3 weekly-pool failover backup.

The SQLite database remains the canonical source of job state and OLAP metrics.
This tool creates a consistent SQLite snapshot, gzips it for replication, and
exports small CSV/SQL companions for quick inspection from another workstation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_DB = Path(
    "/home/harveybc/Documents/GitHub/financial-data/experiments/"
    "weekly_walkforward_pool/project3_weekly_pool.sqlite"
)
DEFAULT_BACKUP_ROOT = DEFAULT_DB.parent / "backups"

EXPORT_OBJECTS = (
    "jobs",
    "subjobs",
    "machine_heartbeats",
    "weekly_result_olap",
    "weekly_result_test_year_olap",
    "weekly_result_validation_year_olap",
    "weekly_result_full_year_protocol_olap",
    "weekly_result_artifact_olap",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_count(conn: sqlite3.Connection, name: str) -> int | None:
    try:
        return int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
    except sqlite3.Error:
        return None


def backup_sqlite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(src))
    try:
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def write_schema(conn: sqlite3.Connection, path: Path) -> None:
    rows = conn.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE sql IS NOT NULL
        ORDER BY type, name
        """
    ).fetchall()
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f"-- {row[0]} {row[1]} table={row[2]}\n")
            fh.write(str(row[3]).rstrip() + ";\n\n")


def export_csv(conn: sqlite3.Connection, name: str, path: Path) -> int | None:
    try:
        cur = conn.execute(f'SELECT * FROM "{name}"')
    except sqlite3.Error:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([item[0] for item in cur.description])
        for row in cur:
            writer.writerow(list(row))
            count += 1
    return count


def gzip_file(path: Path, *, remove_source: bool) -> Path:
    gz_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    if remove_source:
        path.unlink()
    return gz_path


def copy_to_targets(paths: Iterable[Path], targets: Iterable[Path]) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for path in paths:
            dst = target / path.name
            shutil.copy2(path, dst)
            copied.append({"source": str(path), "target": str(dst)})
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    ap.add_argument("--label", default="")
    ap.add_argument("--keep-sqlite", action="store_true")
    ap.add_argument("--copy-to", action="append", type=Path, default=[])
    args = ap.parse_args()

    if not args.db.exists():
        raise FileNotFoundError(args.db)

    label = f"_{args.label}" if args.label else ""
    out_dir = args.backup_root / f"{utc_stamp()}{label}"
    out_dir.mkdir(parents=True, exist_ok=False)

    snapshot = out_dir / "project3_weekly_pool.sqlite"
    backup_sqlite(args.db, snapshot)
    integrity = sqlite3.connect(str(snapshot)).execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"backup integrity_check failed: {integrity}")

    conn = sqlite3.connect(str(snapshot))
    conn.row_factory = sqlite3.Row
    try:
        schema_path = out_dir / "schema.sql"
        write_schema(conn, schema_path)
        csv_counts: dict[str, int | None] = {}
        csv_paths: list[Path] = []
        for name in EXPORT_OBJECTS:
            csv_path = out_dir / f"{name}.csv.gz"
            csv_counts[name] = export_csv(conn, name, csv_path)
            if csv_counts[name] is not None:
                csv_paths.append(csv_path)
        status_counts = {
            row["status"]: int(row["n"])
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM subjobs GROUP BY status")
        }
        object_counts = {name: row_count(conn, name) for name in EXPORT_OBJECTS}
    finally:
        conn.close()

    gz_snapshot = gzip_file(snapshot, remove_source=not args.keep_sqlite)
    kept_snapshot = snapshot if args.keep_sqlite else None

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_db": str(args.db),
        "backup_dir": str(out_dir),
        "sqlite_snapshot": str(kept_snapshot) if kept_snapshot else None,
        "sqlite_snapshot_gz": str(gz_snapshot),
        "schema_sql": str(schema_path),
        "status_counts": status_counts,
        "object_counts": object_counts,
        "csv_counts": csv_counts,
        "sha256": {
            "sqlite_gz": sha256(gz_snapshot),
            "schema_sql": sha256(schema_path),
            **{path.name: sha256(path) for path in csv_paths},
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    copied = copy_to_targets([gz_snapshot, schema_path, manifest_path, *csv_paths], args.copy_to)
    if copied:
        manifest["copied"] = copied
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
