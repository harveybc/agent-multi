#!/usr/bin/env python3
"""Backfill weekly-pool run artifacts into the SQLite OLAP cube.

This indexes existing run outputs without deleting or moving files. The
`subjobs.result_json` field remains the canonical metric summary; this tool
only fills missing summary keys from a local results file and records artifacts
in `result_artifacts` for auditability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from project3_weekly_pool import _json, connect, init_db
from project3_weekly_worker import summarize_result


JSON_PAYLOAD_TYPES = {
    "results_json",
    "summary_json",
    "training_progress_json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_json_load(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tail(path: Path, max_chars: int) -> str | None:
    if max_chars <= 0:
        return None
    try:
        data = path.read_bytes()
    except Exception:
        return None
    text = data[-max_chars:].decode("utf-8", errors="replace")
    return text


def _artifact_type(path: Path, run_dir: Path) -> str:
    rel = path.relative_to(run_dir) if path.is_relative_to(run_dir) else path.name
    name = path.name
    text = str(rel)
    if name == "results.json":
        return "results_json"
    if name == "summary.json":
        return "summary_json"
    if name.startswith("subprocess_stdout") and name.endswith(".log"):
        return "stdout_log"
    if name == "training_progress.json":
        return "training_progress_json"
    if name == "policy.zip":
        return "policy_zip"
    if name == "config_out.json":
        return "config_out_json"
    if text.startswith("return_traces/"):
        return "return_trace"
    if "context_embedding" in text and name.endswith(".json"):
        return "context_embedding_json"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".log"):
        return "log"
    return "artifact"


def _iter_run_artifacts(run_dir: Path) -> Iterable[Path]:
    direct_names = [
        "results.json",
        "summary.json",
        "subprocess_stdout.log",
        "training_progress.json",
        "policy.zip",
        "config_out.json",
    ]
    for name in direct_names:
        path = run_dir / name
        if path.exists() and path.is_file():
            yield path
    for path in sorted(run_dir.glob("subprocess_stdout*.log")):
        if path.is_file() and path.name != "subprocess_stdout.log":
            yield path
    for folder in ("return_traces", "context_embedding"):
        root = run_dir / folder
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    yield path


def _metadata_for(
    path: Path,
    artifact_type: str,
    *,
    max_json_bytes: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if path.suffix == ".json":
        payload = _safe_json_load(path)
        if isinstance(payload, dict):
            metadata["json_keys"] = sorted(str(k) for k in payload.keys())
            if artifact_type in JSON_PAYLOAD_TYPES and path.stat().st_size <= max_json_bytes:
                metadata["payload"] = payload
        elif payload is not None and path.stat().st_size <= max_json_bytes:
            metadata["payload"] = payload
    return metadata


def _upsert_artifact(
    conn: sqlite3.Connection,
    *,
    subjob_id: str,
    artifact_type: str,
    path: Path,
    max_log_tail_chars: int,
    max_json_bytes: int,
    now: str,
) -> bool:
    stat = path.stat()
    metadata = _metadata_for(path, artifact_type, max_json_bytes=max_json_bytes)
    content_tail = _tail(path, max_log_tail_chars) if artifact_type.endswith("log") else None
    sha = _sha256(path)
    cur = conn.execute(
        """
        INSERT INTO result_artifacts (
            subjob_id, artifact_type, path, size_bytes, mtime, sha256,
            content_tail, metadata_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(subjob_id, artifact_type, path) DO UPDATE SET
            size_bytes=excluded.size_bytes,
            mtime=excluded.mtime,
            sha256=excluded.sha256,
            content_tail=excluded.content_tail,
            metadata_json=excluded.metadata_json,
            updated_at=excluded.updated_at
        """,
        (
            subjob_id,
            artifact_type,
            str(path),
            stat.st_size,
            datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
            sha,
            content_tail,
            _json(metadata),
            now,
            now,
        ),
    )
    return cur.rowcount > 0


def _load_config(row: sqlite3.Row, run_dir: Path) -> dict[str, Any]:
    candidates = []
    if row["config_path"]:
        candidates.append(Path(str(row["config_path"])))
    candidates.extend([run_dir / "config_out.json", run_dir / "config.json"])
    for path in candidates:
        if path.exists():
            payload = _safe_json_load(path)
            if isinstance(payload, dict):
                return payload
    return {}


def _merge_result_summary(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    run_dir: Path,
    *,
    now: str,
    refresh_summaries: bool,
    mark_done_from_results: bool,
) -> bool:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        results_path = run_dir / "summary.json"
    if not results_path.exists():
        return False

    config = _load_config(row, run_dir)
    try:
        summary = summarize_result(results_path, config)
    except Exception:
        return False
    summary.update(
        {
            "subjob_id": row["external_id"],
            "config_path": row["config_path"],
            "run_dir": str(run_dir),
            "results_file": str(results_path),
        }
    )
    stdout = run_dir / "subprocess_stdout.log"
    if stdout.exists():
        summary["stdout_log"] = str(stdout)

    existing: dict[str, Any] = {}
    if row["result_json"]:
        try:
            existing = json.loads(row["result_json"])
        except Exception:
            existing = {}

    merged = dict(existing)
    changed = False
    for key, value in summary.items():
        if refresh_summaries or key not in merged or merged.get(key) is None:
            merged[key] = value
            if existing.get(key) != value:
                changed = True
    artifact_count = conn.execute(
        "SELECT COUNT(*) AS n FROM result_artifacts WHERE subjob_id=?",
        (row["external_id"],),
    ).fetchone()["n"]
    if merged.get("artifact_count") != artifact_count:
        merged["artifact_count"] = artifact_count
        changed = True
    if merged.get("artifact_backfilled_at") is None:
        merged["artifact_backfilled_at"] = now
        changed = True

    if not changed:
        return False

    status = row["status"]
    completed_at = row["completed_at"]
    if mark_done_from_results and status != "running":
        status = "done"
        completed_at = completed_at or now
    conn.execute(
        """
        UPDATE subjobs
        SET result_json=?, status=?, completed_at=?, updated_at=?
        WHERE external_id=?
        """,
        (_json(merged), status, completed_at, now, row["external_id"]),
    )
    conn.execute(
        "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
        (
            "artifact_backfill_result_summary",
            row["external_id"],
            _json({"results_file": str(results_path), "refresh_summaries": refresh_summaries}),
            now,
        ),
    )
    return True


def backfill(
    conn: sqlite3.Connection,
    *,
    runs_root: Path,
    apply: bool,
    max_log_tail_chars: int,
    max_json_bytes: int,
    refresh_summaries: bool,
    mark_done_from_results: bool,
) -> dict[str, int]:
    init_db(conn)
    now = utc_now()
    report = {
        "subjobs_seen": 0,
        "run_dirs_seen": 0,
        "artifact_files_seen": 0,
        "artifact_rows_upserted": 0,
        "result_summaries_updated": 0,
        "orphan_run_dirs": 0,
    }
    rows = conn.execute("SELECT * FROM subjobs ORDER BY id").fetchall()
    known_subjobs = {row["external_id"] for row in rows}

    if apply:
        conn.execute("BEGIN IMMEDIATE")
    try:
        for row in rows:
            report["subjobs_seen"] += 1
            run_dir = Path(str(row["run_dir"])) if row["run_dir"] else runs_root / str(row["external_id"])
            if not run_dir.exists() or not run_dir.is_dir():
                continue
            report["run_dirs_seen"] += 1
            for path in _iter_run_artifacts(run_dir):
                report["artifact_files_seen"] += 1
                if apply:
                    if _upsert_artifact(
                        conn,
                        subjob_id=str(row["external_id"]),
                        artifact_type=_artifact_type(path, run_dir),
                        path=path,
                        max_log_tail_chars=max_log_tail_chars,
                        max_json_bytes=max_json_bytes,
                        now=now,
                    ):
                        report["artifact_rows_upserted"] += 1
            if apply and _merge_result_summary(
                conn,
                row,
                run_dir,
                now=now,
                refresh_summaries=refresh_summaries,
                mark_done_from_results=mark_done_from_results,
            ):
                report["result_summaries_updated"] += 1

        if runs_root.exists():
            for run_dir in runs_root.iterdir():
                if run_dir.is_dir() and run_dir.name not in known_subjobs:
                    report["orphan_run_dirs"] += 1
                    if apply:
                        for path in _iter_run_artifacts(run_dir):
                            report["artifact_files_seen"] += 1
                            if _upsert_artifact(
                                conn,
                                subjob_id=run_dir.name,
                                artifact_type=_artifact_type(path, run_dir),
                                path=path,
                                max_log_tail_chars=max_log_tail_chars,
                                max_json_bytes=max_json_bytes,
                                now=now,
                            ):
                                report["artifact_rows_upserted"] += 1

        if apply:
            conn.execute(
                "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
                ("artifact_backfill", str(runs_root), _json(report), now),
            )
            conn.commit()
    except Exception:
        if apply:
            conn.rollback()
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--apply", action="store_true", help="write DB changes; default is dry-run")
    parser.add_argument("--max-log-tail-chars", type=int, default=20000)
    parser.add_argument("--max-json-bytes", type=int, default=2_000_000)
    parser.add_argument("--refresh-summaries", action="store_true", help="replace existing summary keys from results.json")
    parser.add_argument(
        "--mark-done-from-results",
        action="store_true",
        help="mark non-running subjobs done when a results file can be summarized",
    )
    args = parser.parse_args()

    conn = connect(args.db)
    report = backfill(
        conn,
        runs_root=Path(args.runs_root),
        apply=args.apply,
        max_log_tail_chars=args.max_log_tail_chars,
        max_json_bytes=args.max_json_bytes,
        refresh_summaries=args.refresh_summaries,
        mark_done_from_results=args.mark_done_from_results,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
