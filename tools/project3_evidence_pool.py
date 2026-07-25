#!/usr/bin/env python3
"""Transactional job pool and OLAP store for Project 3 evidence sweeps."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from project3_evidence_metrics import METRIC_SCHEMA, metric_rows_from_result


SCHEMA_VERSION = "project3.evidence.pool.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS campaigns (
            campaign_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            plan_sha256 TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT NOT NULL UNIQUE,
            campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
            stage TEXT NOT NULL,
            task_type TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 100,
            config_sha256 TEXT NOT NULL,
            config_json TEXT NOT NULL,
            eligible_machines_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'pending',
            claimed_by TEXT,
            claimed_at TEXT,
            heartbeat_at TEXT,
            lease_until TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            started_at TEXT,
            completed_at TEXT,
            result_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS job_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL REFERENCES jobs(id),
            attempt_number INTEGER NOT NULL,
            machine_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            error TEXT,
            UNIQUE(job_id, attempt_number)
        );

        CREATE TABLE IF NOT EXISTS machine_heartbeats (
            machine_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            current_job_id TEXT,
            message TEXT,
            cpu_summary_json TEXT,
            gpu_summary_json TEXT,
            heartbeat_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metric_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL REFERENCES jobs(id),
            metric_schema TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL,
            unit TEXT NOT NULL,
            horizon TEXT NOT NULL,
            aggregation TEXT NOT NULL,
            split TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(job_id, metric_schema, metric_name, split, horizon, aggregation)
        );

        CREATE TABLE IF NOT EXISTS parameter_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL REFERENCES jobs(id),
            parameter_path TEXT NOT NULL,
            value_json TEXT NOT NULL,
            value_numeric REAL,
            value_text TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(job_id, parameter_path)
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL REFERENCES jobs(id),
            artifact_type TEXT NOT NULL,
            path TEXT NOT NULL,
            sha256 TEXT,
            size_bytes INTEGER,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(job_id, artifact_type, path)
        );

        CREATE TABLE IF NOT EXISTS pool_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            subject_id TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_evidence_jobs_claim
            ON jobs(status, priority, id);
        CREATE INDEX IF NOT EXISTS idx_evidence_jobs_campaign_stage
            ON jobs(campaign_id, stage, status);
        CREATE INDEX IF NOT EXISTS idx_evidence_metric_lookup
            ON metric_facts(metric_name, split, horizon);
        CREATE INDEX IF NOT EXISTS idx_evidence_parameter_lookup
            ON parameter_facts(parameter_path);

        DROP VIEW IF EXISTS evidence_result_olap;
        CREATE VIEW evidence_result_olap AS
        SELECT
            j.external_id AS job_id,
            j.campaign_id,
            j.stage,
            j.task_type,
            j.status,
            j.claimed_by AS machine_id,
            j.attempt_count,
            j.started_at,
            j.completed_at,
            j.config_sha256,
            j.config_json,
            j.result_json,
            MAX(CASE WHEN m.metric_name='mean_weekly_return' AND m.split='validation'
                     THEN m.value END) AS validation_mean_weekly_return,
            MAX(CASE WHEN m.metric_name='annualized_return' AND m.split='validation'
                     THEN m.value END) AS validation_annualized_return,
            MAX(CASE WHEN m.metric_name='mean_weekly_rap' AND m.split='validation'
                     THEN m.value END) AS validation_mean_weekly_rap,
            MAX(CASE WHEN m.metric_name='annual_rap' AND m.split='validation'
                     THEN m.value END) AS validation_annual_rap,
            MAX(CASE WHEN m.metric_name='max_drawdown' AND m.split='validation'
                     THEN m.value END) AS validation_max_drawdown,
            MAX(CASE WHEN m.metric_name='mean_weekly_return' AND m.split='test'
                     THEN m.value END) AS test_mean_weekly_return,
            MAX(CASE WHEN m.metric_name='annualized_return' AND m.split='test'
                     THEN m.value END) AS test_annualized_return,
            MAX(CASE WHEN m.metric_name='mean_weekly_rap' AND m.split='test'
                     THEN m.value END) AS test_mean_weekly_rap,
            MAX(CASE WHEN m.metric_name='annual_rap' AND m.split='test'
                     THEN m.value END) AS test_annual_rap,
            MAX(CASE WHEN m.metric_name='max_drawdown' AND m.split='test'
                     THEN m.value END) AS test_max_drawdown,
            MAX(CASE WHEN m.metric_name='optimization_score' THEN m.value END)
                AS optimization_score_dimensionless
        FROM jobs j
        LEFT JOIN metric_facts m ON m.job_id=j.id
        GROUP BY j.id;

        DROP VIEW IF EXISTS evidence_parameter_effect_olap;
        CREATE VIEW evidence_parameter_effect_olap AS
        SELECT
            j.campaign_id,
            j.stage,
            p.parameter_path,
            p.value_json,
            m.metric_name,
            m.split,
            m.horizon,
            m.unit,
            COUNT(*) AS job_count,
            AVG(m.value) AS mean_value,
            MIN(m.value) AS min_value,
            MAX(m.value) AS max_value
        FROM parameter_facts p
        JOIN jobs j ON j.id=p.job_id
        JOIN metric_facts m ON m.job_id=j.id
        WHERE j.status='completed'
        GROUP BY
            j.campaign_id, j.stage, p.parameter_path, p.value_json,
            m.metric_name, m.split, m.horizon, m.unit;

        DROP VIEW IF EXISTS evidence_machine_olap;
        CREATE VIEW evidence_machine_olap AS
        SELECT
            h.machine_id,
            h.status,
            h.current_job_id,
            h.message,
            h.cpu_summary_json,
            h.gpu_summary_json,
            h.heartbeat_at,
            SUM(CASE WHEN j.status='completed' THEN 1 ELSE 0 END) AS completed_jobs,
            SUM(CASE WHEN j.status='failed' THEN 1 ELSE 0 END) AS failed_jobs
        FROM machine_heartbeats h
        LEFT JOIN jobs j ON j.claimed_by=h.machine_id
        GROUP BY h.machine_id;
        """
    )
    conn.commit()


def _flatten(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(value[key], child)
    elif isinstance(value, list):
        yield prefix, value
    else:
        yield prefix, value


def _write_parameter_facts(conn: sqlite3.Connection, job_id: int, config: dict[str, Any]) -> None:
    now = utc_now()
    for path, value in _flatten(config):
        numeric = None
        text = None
        if isinstance(value, bool):
            text = str(value).lower()
        elif isinstance(value, (int, float)):
            numeric = float(value)
        elif value is not None:
            text = str(value)
        conn.execute(
            """
            INSERT OR REPLACE INTO parameter_facts(
                job_id, parameter_path, value_json, value_numeric, value_text, created_at
            ) VALUES(?,?,?,?,?,?)
            """,
            (job_id, path, _json(value), numeric, text, now),
        )


def enqueue_plan(conn: sqlite3.Connection, plan: dict[str, Any]) -> dict[str, int]:
    campaign_id = str(plan["campaign_id"])
    now = utc_now()
    plan_sha = _sha256_json(plan)
    conn.execute(
        """
        INSERT INTO campaigns(
            campaign_id, schema_version, plan_sha256, plan_json, status, created_at, updated_at
        ) VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(campaign_id) DO UPDATE SET
            plan_sha256=excluded.plan_sha256,
            plan_json=excluded.plan_json,
            updated_at=excluded.updated_at
        """,
        (campaign_id, SCHEMA_VERSION, plan_sha, _json(plan), "running", now, now),
    )
    inserted = 0
    existing = 0
    for job in plan.get("jobs") or []:
        config = dict(job.get("config") or {})
        config_sha = _sha256_json(config)
        external_id = str(job["job_id"])
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO jobs(
                external_id, campaign_id, stage, task_type, priority,
                config_sha256, config_json, eligible_machines_json,
                max_attempts, created_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                external_id,
                campaign_id,
                str(job["stage"]),
                str(job["task_type"]),
                int(job.get("priority", 100)),
                config_sha,
                _json(config),
                _json(job.get("eligible_machines") or []),
                int(job.get("max_attempts", 3)),
                now,
                now,
            ),
        )
        if cursor.rowcount:
            inserted += 1
            _write_parameter_facts(conn, int(cursor.lastrowid), config)
        else:
            existing += 1
            row = conn.execute(
                "SELECT config_sha256 FROM jobs WHERE external_id=?",
                (external_id,),
            ).fetchone()
            if row and row["config_sha256"] != config_sha:
                raise ValueError(f"job {external_id} already exists with a different config")
    conn.execute(
        "INSERT INTO pool_events(event_type,subject_id,payload_json,created_at) VALUES(?,?,?,?)",
        ("plan_enqueued", campaign_id, _json({"inserted": inserted, "existing": existing}), now),
    )
    conn.commit()
    return {"inserted": inserted, "existing": existing}


def requeue_expired(conn: sqlite3.Connection, *, commit: bool = True) -> int:
    now = utc_now()
    rows = conn.execute(
        """
        SELECT id, external_id, attempt_count, max_attempts
        FROM jobs
        WHERE status='running' AND lease_until IS NOT NULL AND lease_until < ?
        """,
        (now,),
    ).fetchall()
    for row in rows:
        new_status = "pending" if row["attempt_count"] < row["max_attempts"] else "failed"
        conn.execute(
            """
            UPDATE jobs SET status=?, claimed_by=NULL, lease_until=NULL,
                error=?, updated_at=? WHERE id=?
            """,
            (new_status, "worker lease expired", now, row["id"]),
        )
        conn.execute(
            """
            UPDATE job_attempts SET status='lease_expired', completed_at=?, error=?
            WHERE job_id=? AND attempt_number=? AND completed_at IS NULL
            """,
            (now, "worker lease expired", row["id"], row["attempt_count"]),
        )
    if rows and commit:
        conn.commit()
    return len(rows)


def claim_job(
    conn: sqlite3.Connection,
    machine_id: str,
    *,
    lease_seconds: int = 300,
) -> dict[str, Any] | None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        requeue_expired(conn, commit=False)
        candidates = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status='pending' AND attempt_count < max_attempts
            ORDER BY priority ASC, id ASC
            """
        ).fetchall()
        row = next(
            (
                item
                for item in candidates
                if not json.loads(item["eligible_machines_json"])
                or machine_id in json.loads(item["eligible_machines_json"])
            ),
            None,
        )
        if row is None:
            conn.commit()
            return None
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat(timespec="seconds")
        lease_until = (now_dt + timedelta(seconds=lease_seconds)).isoformat(timespec="seconds")
        attempt = int(row["attempt_count"]) + 1
        cursor = conn.execute(
            """
            UPDATE jobs SET status='running', claimed_by=?, claimed_at=?,
                heartbeat_at=?, lease_until=?, attempt_count=?, started_at=COALESCE(started_at,?),
                error=NULL, updated_at=?
            WHERE id=? AND status='pending'
            """,
            (machine_id, now, now, lease_until, attempt, now, now, row["id"]),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            return None
        conn.execute(
            """
            INSERT INTO job_attempts(
                job_id, attempt_number, machine_id, started_at, heartbeat_at, status
            ) VALUES(?,?,?,?,?,'running')
            """,
            (row["id"], attempt, machine_id, now, now),
        )
        conn.commit()
        claimed = conn.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
        return {
            "job_id": claimed["external_id"],
            "stage": claimed["stage"],
            "task_type": claimed["task_type"],
            "attempt": claimed["attempt_count"],
            "config": json.loads(claimed["config_json"]),
        }
    except Exception:
        conn.rollback()
        raise


def heartbeat(
    conn: sqlite3.Connection,
    machine_id: str,
    job_external_id: str | None,
    *,
    status: str,
    message: str = "",
    cpu_summary: dict[str, Any] | None = None,
    gpu_summary: dict[str, Any] | None = None,
    lease_seconds: int = 300,
) -> None:
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat(timespec="seconds")
    if job_external_id:
        lease_until = (now_dt + timedelta(seconds=lease_seconds)).isoformat(timespec="seconds")
        row = conn.execute(
            "SELECT id, attempt_count, claimed_by, status FROM jobs WHERE external_id=?",
            (job_external_id,),
        ).fetchone()
        if row is None or row["claimed_by"] != machine_id or row["status"] != "running":
            raise PermissionError(f"{machine_id} does not own running job {job_external_id}")
        conn.execute(
            "UPDATE jobs SET heartbeat_at=?, lease_until=?, updated_at=? WHERE id=?",
            (now, lease_until, now, row["id"]),
        )
        conn.execute(
            """
            UPDATE job_attempts SET heartbeat_at=?
            WHERE job_id=? AND attempt_number=? AND completed_at IS NULL
            """,
            (now, row["id"], row["attempt_count"]),
        )
    conn.execute(
        """
        INSERT INTO machine_heartbeats(
            machine_id,status,current_job_id,message,cpu_summary_json,gpu_summary_json,heartbeat_at
        ) VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(machine_id) DO UPDATE SET
            status=excluded.status,
            current_job_id=excluded.current_job_id,
            message=excluded.message,
            cpu_summary_json=excluded.cpu_summary_json,
            gpu_summary_json=excluded.gpu_summary_json,
            heartbeat_at=excluded.heartbeat_at
        """,
        (
            machine_id,
            status,
            job_external_id,
            message,
            _json(cpu_summary or {}),
            _json(gpu_summary or {}),
            now,
        ),
    )
    conn.commit()


def complete_job(
    conn: sqlite3.Connection,
    machine_id: str,
    job_external_id: str,
    result: dict[str, Any],
) -> None:
    now = utc_now()
    row = conn.execute(
        "SELECT id, attempt_count, claimed_by, status FROM jobs WHERE external_id=?",
        (job_external_id,),
    ).fetchone()
    if row is None or row["claimed_by"] != machine_id or row["status"] != "running":
        raise PermissionError(f"{machine_id} does not own running job {job_external_id}")
    conn.execute(
        """
        UPDATE jobs SET status='completed', completed_at=?, result_json=?,
            lease_until=NULL, updated_at=? WHERE id=?
        """,
        (now, _json(result), now, row["id"]),
    )
    conn.execute(
        """
        UPDATE job_attempts SET status='completed', completed_at=?
        WHERE job_id=? AND attempt_number=?
        """,
        (now, row["id"], row["attempt_count"]),
    )
    for metric in metric_rows_from_result(result):
        conn.execute(
            """
            INSERT OR REPLACE INTO metric_facts(
                job_id,metric_schema,metric_name,value,unit,horizon,aggregation,split,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                row["id"],
                metric["metric_schema"],
                metric["metric_name"],
                metric["value"],
                metric["unit"],
                metric["horizon"],
                metric["aggregation"],
                metric["split"],
                now,
            ),
        )
    for artifact in result.get("artifacts") or []:
        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts(
                job_id,artifact_type,path,sha256,size_bytes,metadata_json,created_at
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                row["id"],
                str(artifact["artifact_type"]),
                str(artifact["path"]),
                artifact.get("sha256"),
                artifact.get("size_bytes"),
                _json(artifact.get("metadata") or {}),
                now,
            ),
        )
    conn.execute(
        """
        INSERT INTO pool_events(event_type,subject_id,payload_json,created_at)
        VALUES('job_completed',?,?,?)
        """,
        (job_external_id, _json({"machine_id": machine_id}), now),
    )
    conn.commit()


def fail_job(
    conn: sqlite3.Connection,
    machine_id: str,
    job_external_id: str,
    error: str,
    *,
    retry: bool = True,
) -> None:
    now = utc_now()
    row = conn.execute(
        "SELECT id, attempt_count, max_attempts, claimed_by, status FROM jobs WHERE external_id=?",
        (job_external_id,),
    ).fetchone()
    if row is None or row["claimed_by"] != machine_id or row["status"] != "running":
        raise PermissionError(f"{machine_id} does not own running job {job_external_id}")
    status = "pending" if retry and row["attempt_count"] < row["max_attempts"] else "failed"
    conn.execute(
        """
        UPDATE jobs SET status=?, claimed_by=NULL, lease_until=NULL, error=?, updated_at=?
        WHERE id=?
        """,
        (status, error[-20000:], now, row["id"]),
    )
    conn.execute(
        """
        UPDATE job_attempts SET status=?, completed_at=?, error=?
        WHERE job_id=? AND attempt_number=?
        """,
        (status, now, error[-20000:], row["id"], row["attempt_count"]),
    )
    conn.commit()


def requeue_machine(conn: sqlite3.Connection, machine_id: str, reason: str) -> int:
    """Return a stopped machine's owned jobs to the pool immediately."""
    now = utc_now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            """
            SELECT id, external_id, attempt_count, max_attempts
            FROM jobs WHERE status='running' AND claimed_by=?
            """,
            (machine_id,),
        ).fetchall()
        for row in rows:
            status_value = "pending" if row["attempt_count"] < row["max_attempts"] else "failed"
            conn.execute(
                """
                UPDATE jobs SET status=?, claimed_by=NULL, lease_until=NULL,
                    error=?, updated_at=? WHERE id=?
                """,
                (status_value, reason, now, row["id"]),
            )
            conn.execute(
                """
                UPDATE job_attempts SET status='operator_requeued', completed_at=?, error=?
                WHERE job_id=? AND attempt_number=? AND completed_at IS NULL
                """,
                (now, reason, row["id"], row["attempt_count"]),
            )
            conn.execute(
                """
                INSERT INTO pool_events(event_type,subject_id,payload_json,created_at)
                VALUES('job_operator_requeued',?,?,?)
                """,
                (
                    row["external_id"],
                    _json({"machine_id": machine_id, "reason": reason}),
                    now,
                ),
            )
        conn.execute(
            """
            UPDATE machine_heartbeats SET status='stopped', current_job_id=NULL,
                message=?, heartbeat_at=? WHERE machine_id=?
            """,
            (reason, now, machine_id),
        )
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise


def status(conn: sqlite3.Connection) -> dict[str, Any]:
    requeue_expired(conn)
    counts = {
        row["status"]: int(row["n"])
        for row in conn.execute("SELECT status,COUNT(*) AS n FROM jobs GROUP BY status")
    }
    stages = [
        dict(row)
        for row in conn.execute(
            """
            SELECT stage,status,COUNT(*) AS jobs
            FROM jobs GROUP BY stage,status ORDER BY stage,status
            """
        )
    ]
    machines = [dict(row) for row in conn.execute("SELECT * FROM evidence_machine_olap ORDER BY machine_id")]
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_schema": METRIC_SCHEMA,
        "counts": counts,
        "stages": stages,
        "machines": machines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("--plan", required=True)
    claim = sub.add_parser("claim")
    claim.add_argument("--machine-id", required=True)
    complete = sub.add_parser("complete")
    complete.add_argument("--machine-id", required=True)
    complete.add_argument("--job-id", required=True)
    complete.add_argument("--result", required=True)
    requeue = sub.add_parser("requeue-machine")
    requeue.add_argument("--machine-id", required=True)
    requeue.add_argument("--reason", default="operator requeue")
    sub.add_parser("status")
    args = parser.parse_args()
    conn = connect(args.db)
    init_db(conn)
    if args.command == "init":
        output: Any = {"ok": True, "db": args.db}
    elif args.command == "enqueue":
        output = enqueue_plan(conn, json.loads(Path(args.plan).read_text(encoding="utf-8")))
    elif args.command == "claim":
        output = claim_job(conn, args.machine_id)
    elif args.command == "complete":
        result = json.loads(Path(args.result).read_text(encoding="utf-8"))
        complete_job(conn, args.machine_id, args.job_id, result)
        output = {"ok": True}
    elif args.command == "requeue-machine":
        output = {
            "ok": True,
            "requeued": requeue_machine(conn, args.machine_id, args.reason),
        }
    else:
        output = status(conn)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
