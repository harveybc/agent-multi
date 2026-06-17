#!/usr/bin/env python3
"""SQLite job pool for Project 3 weekly walk-forward experiments."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "project3_weekly_walkforward_pool_v1"
HELDOUT_START = datetime.fromisoformat("2025-01-01T00:00:00+00:00")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_dt(value: Any) -> datetime:
    if value is None:
        raise ValueError("missing datetime value")
    text = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT NOT NULL UNIQUE,
            candidate_id TEXT NOT NULL,
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            model_family TEXT NOT NULL,
            train_years INTEGER NOT NULL,
            training_policy TEXT NOT NULL,
            input_data_file TEXT NOT NULL,
            feature_count INTEGER NOT NULL,
            config_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS subjobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT NOT NULL UNIQUE,
            job_id INTEGER NOT NULL REFERENCES jobs(id),
            weekly_anchor_id TEXT NOT NULL,
            train_start TEXT NOT NULL,
            train_end TEXT NOT NULL,
            validation_start TEXT NOT NULL,
            validation_end TEXT NOT NULL,
            test_start TEXT NOT NULL,
            test_end TEXT NOT NULL,
            train_rows INTEGER,
            validation_rows INTEGER,
            test_rows INTEGER,
            depends_on_subjob_id TEXT,
            warm_start_parent_subjob_id TEXT,
            priority INTEGER NOT NULL DEFAULT 100,
            status TEXT NOT NULL DEFAULT 'pending',
            claimed_by TEXT,
            claimed_at TEXT,
            heartbeat_at TEXT,
            completed_at TEXT,
            config_path TEXT,
            run_dir TEXT,
            result_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS machine_heartbeats (
            machine_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            current_subjob_id TEXT,
            gpu_summary TEXT,
            message TEXT,
            heartbeat_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pool_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            subject_id TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_subjobs_status_priority
            ON subjobs(status, priority, id);
        CREATE INDEX IF NOT EXISTS idx_subjobs_job_id ON subjobs(job_id);
        """
    )
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(subjobs)")
    }
    for column in ("depends_on_subjob_id", "warm_start_parent_subjob_id"):
        if column not in existing:
            conn.execute(f"ALTER TABLE subjobs ADD COLUMN {column} TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_subjobs_depends_on ON subjobs(depends_on_subjob_id)"
    )
    conn.commit()


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("stage_c_access") != "DENIED":
        raise ValueError("plan must carry stage_c_access='DENIED'")
    if plan.get("training_launched") not in (False, None):
        raise ValueError("plan must carry training_launched=false before enqueue")
    subjob_ids = {
        subjob.get("subjob_id")
        for job in plan.get("jobs", [])
        for subjob in job.get("subjobs", [])
    }
    for job in plan.get("jobs", []):
        for subjob in job.get("subjobs", []):
            parent = subjob.get("depends_on_subjob_id") or subjob.get("warm_start_parent_subjob_id")
            if parent and parent not in subjob_ids:
                raise ValueError(
                    f"subjob {subjob.get('subjob_id')} depends on unknown parent {parent}"
                )
            dates = [
                _parse_dt(subjob["train_start"]),
                _parse_dt(subjob["train_end"]),
                _parse_dt(subjob["validation_start"]),
                _parse_dt(subjob["validation_end"]),
                _parse_dt(subjob["test_start"]),
                _parse_dt(subjob["test_end"]),
            ]
            if not dates[0] < dates[1] <= dates[2] < dates[3] <= dates[4] < dates[5]:
                raise ValueError(f"invalid split ordering in {subjob.get('subjob_id')}")
            if dates[5] > HELDOUT_START:
                raise ValueError(f"subjob {subjob.get('subjob_id')} reaches Stage C heldout")


def enqueue_plan(conn: sqlite3.Connection, plan_path: str | Path) -> dict[str, int]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    _validate_plan(plan)
    now = utc_now()
    inserted_jobs = 0
    inserted_subjobs = 0
    with conn:
        for job in plan.get("jobs", []):
            feature_columns = job.get("feature_columns") or job.get("selected_features") or []
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    external_id, candidate_id, asset, timeframe, model_family,
                    train_years, training_policy, input_data_file, feature_count,
                    config_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job["job_id"],
                    job.get("candidate_id", job["job_id"]),
                    job["asset"],
                    job["timeframe"],
                    job.get("model_family", "sac"),
                    int(job["train_years"]),
                    job.get("training_policy", "scratch_n_years"),
                    job["input_data_file"],
                    len(feature_columns),
                    _json(job),
                    now,
                    now,
                ),
            )
            inserted_jobs += cur.rowcount
            row = conn.execute("SELECT id FROM jobs WHERE external_id = ?", (job["job_id"],)).fetchone()
            if row is None:
                raise RuntimeError(f"job disappeared after enqueue: {job['job_id']}")
            for subjob in job.get("subjobs", []):
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO subjobs (
                        external_id, job_id, weekly_anchor_id,
                        train_start, train_end, validation_start, validation_end,
                        test_start, test_end, train_rows, validation_rows, test_rows,
                        depends_on_subjob_id, warm_start_parent_subjob_id,
                        priority, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        subjob["subjob_id"],
                        row["id"],
                        subjob["weekly_anchor_id"],
                        subjob["train_start"],
                        subjob["train_end"],
                        subjob["validation_start"],
                        subjob["validation_end"],
                        subjob["test_start"],
                        subjob["test_end"],
                        subjob.get("train_rows"),
                        subjob.get("validation_rows"),
                        subjob.get("test_rows"),
                        subjob.get("depends_on_subjob_id"),
                        subjob.get("warm_start_parent_subjob_id"),
                        int(subjob.get("priority", 100)),
                        now,
                        now,
                    ),
                )
                inserted_subjobs += cur.rowcount
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("enqueue_plan", str(plan_path), _json({"jobs": inserted_jobs, "subjobs": inserted_subjobs}), now),
        )
    return {"inserted_jobs": inserted_jobs, "inserted_subjobs": inserted_subjobs}


def claim_subjob(conn: sqlite3.Connection, machine_id: str) -> dict[str, Any] | None:
    now = utc_now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            """
            SELECT s.*, j.external_id AS job_external_id, j.config_json
            FROM subjobs s
            JOIN jobs j ON j.id = s.job_id
            WHERE s.status = 'pending'
              AND (
                s.depends_on_subjob_id IS NULL
                OR EXISTS (
                    SELECT 1
                    FROM subjobs parent
                    WHERE parent.external_id = s.depends_on_subjob_id
                      AND parent.status = 'done'
                )
              )
            ORDER BY s.priority ASC, s.id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO machine_heartbeats(machine_id, status, current_subjob_id, heartbeat_at)
                VALUES (?, 'idle', NULL, ?)
                ON CONFLICT(machine_id) DO UPDATE SET
                    status='idle', current_subjob_id=NULL, heartbeat_at=excluded.heartbeat_at
                """,
                (machine_id, now),
            )
            conn.commit()
            return None
        conn.execute(
            """
            UPDATE subjobs
            SET status='running', claimed_by=?, claimed_at=?, heartbeat_at=?, updated_at=?
            WHERE id=?
            """,
            (machine_id, now, now, now, row["id"]),
        )
        conn.execute(
            """
            INSERT INTO machine_heartbeats(machine_id, status, current_subjob_id, heartbeat_at)
            VALUES (?, 'running', ?, ?)
            ON CONFLICT(machine_id) DO UPDATE SET
                status='running',
                current_subjob_id=excluded.current_subjob_id,
                heartbeat_at=excluded.heartbeat_at
            """,
            (machine_id, row["external_id"], now),
        )
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("claim", row["external_id"], _json({"machine_id": machine_id}), now),
        )
        conn.commit()
        out = dict(row)
        out["job_config"] = json.loads(row["config_json"])
        out.pop("config_json", None)
        return out
    except Exception:
        conn.rollback()
        raise


def heartbeat(
    conn: sqlite3.Connection,
    machine_id: str,
    subjob_id: str | None,
    status: str,
    message: str | None = None,
    gpu_summary: str | None = None,
) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            """
            INSERT INTO machine_heartbeats(machine_id, status, current_subjob_id, message, gpu_summary, heartbeat_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(machine_id) DO UPDATE SET
                status=excluded.status,
                current_subjob_id=excluded.current_subjob_id,
                message=excluded.message,
                gpu_summary=excluded.gpu_summary,
                heartbeat_at=excluded.heartbeat_at
            """,
            (machine_id, status, subjob_id, message, gpu_summary, now),
        )
        if subjob_id:
            conn.execute(
                "UPDATE subjobs SET heartbeat_at=?, updated_at=? WHERE external_id=?",
                (now, now, subjob_id),
            )


def complete_subjob(conn: sqlite3.Connection, subjob_id: str, result: dict[str, Any]) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            """
            UPDATE subjobs
            SET status='done', completed_at=?, heartbeat_at=?, updated_at=?, result_json=?
            WHERE external_id=?
            """,
            (now, now, now, _json(result), subjob_id),
        )
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("complete", subjob_id, _json(result), now),
        )


def fail_subjob(conn: sqlite3.Connection, subjob_id: str, error: str) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            """
            UPDATE subjobs
            SET status='failed', completed_at=?, heartbeat_at=?, updated_at=?, error=?
            WHERE external_id=?
            """,
            (now, now, now, error, subjob_id),
        )
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("fail", subjob_id, _json({"error": error}), now),
        )


def status(conn: sqlite3.Connection) -> dict[str, Any]:
    counts = {
        row["status"]: row["n"]
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM subjobs GROUP BY status")
    }
    jobs = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    machines = [dict(row) for row in conn.execute("SELECT * FROM machine_heartbeats ORDER BY machine_id")]
    running = [
        dict(row)
        for row in conn.execute(
            """
            SELECT s.external_id, s.weekly_anchor_id, s.claimed_by, s.heartbeat_at,
                   s.train_start, s.train_end, s.validation_start, s.validation_end, s.test_start, s.test_end,
                   s.depends_on_subjob_id, s.warm_start_parent_subjob_id,
                   j.asset, j.timeframe, j.model_family, j.train_years, j.training_policy
            FROM subjobs s
            JOIN jobs j ON j.id = s.job_id
            WHERE s.status='running'
            ORDER BY s.claimed_at
            """
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "job_count": jobs,
        "subjob_counts": counts,
        "machines": machines,
        "running": running,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("--plan", required=True)
    claim = sub.add_parser("claim")
    claim.add_argument("--machine-id", required=True)
    hb = sub.add_parser("heartbeat")
    hb.add_argument("--machine-id", required=True)
    hb.add_argument("--subjob-id")
    hb.add_argument("--status", default="running")
    hb.add_argument("--message")
    hb.add_argument("--gpu-summary")
    done = sub.add_parser("complete")
    done.add_argument("--subjob-id", required=True)
    done.add_argument("--result-json", default="{}")
    fail = sub.add_parser("fail")
    fail.add_argument("--subjob-id", required=True)
    fail.add_argument("--error", required=True)
    sub.add_parser("status")
    args = ap.parse_args()

    conn = connect(args.db)
    init_db(conn)
    if args.cmd == "init":
        print(json.dumps({"ok": True, "db": args.db}, indent=2))
    elif args.cmd == "enqueue":
        print(json.dumps(enqueue_plan(conn, args.plan), indent=2))
    elif args.cmd == "claim":
        print(json.dumps(claim_subjob(conn, args.machine_id), indent=2))
    elif args.cmd == "heartbeat":
        heartbeat(conn, args.machine_id, args.subjob_id, args.status, args.message, args.gpu_summary)
        print(json.dumps({"ok": True}, indent=2))
    elif args.cmd == "complete":
        complete_subjob(conn, args.subjob_id, json.loads(args.result_json))
        print(json.dumps({"ok": True}, indent=2))
    elif args.cmd == "fail":
        fail_subjob(conn, args.subjob_id, args.error)
        print(json.dumps({"ok": True}, indent=2))
    elif args.cmd == "status":
        print(json.dumps(status(conn), indent=2))


if __name__ == "__main__":
    main()
