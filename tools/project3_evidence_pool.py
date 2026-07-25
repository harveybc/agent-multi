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
PROXY_RESULT_STAGES = {
    "E1_BASE_SOURCE_SCREEN",
    "E1_EXTERNAL_SOURCE_SCREEN",
    "E2_PREPROCESSING_CONTEXT",
    "E2_INTERACTION_CONFIRMATION",
    "E3_PROXY_MODEL_SCREEN",
}
CANONICAL_TRADING_METRICS = {
    "mean_weekly_return": ("fraction", "week"),
    "annualized_return": ("fraction", "year"),
    "mean_weekly_rap": ("fraction", "week"),
    "annual_rap": ("fraction", "year"),
    "max_drawdown": ("fraction", "evaluation_period"),
    "evaluation_weeks": ("count", "evaluation_period"),
}

PARAMETER_PATH_ALIASES = {
    "asset": "data.asset",
    "timeframe": "data.timeframe",
    "base_feature_bundle": "data.base_feature_bundle",
    "external_context_bundle": "data.external_context_bundle",
    "external_context_lag_hours": "data.external_context_lag_hours",
    "missing_value_policy": "data.missing_value_policy",
    "max_staleness_hours": "data.max_staleness_hours",
    "cross_asset_reference_set": "data.cross_asset_reference_set",
    "cross_asset_volatility_window_hours": "features.cross_asset_volatility_window_hours",
    "target_horizon_hours": "data.target_horizon_hours",
    "target_definition": "data.target_definition",
    "target_barrier_volatility_window_hours": "data.target_barrier_volatility_window_hours",
    "feature_selection_method": "selection.method",
    "feature_budget": "selection.feature_budget",
    "redundancy_threshold": "selection.redundancy_threshold",
    "stability_folds": "selection.stability_folds",
    "selection_regime_volatility_window_hours": "selection.regime_volatility_window_hours",
    "preprocessing_mode": "preprocessing.mode",
    "scaling_history_hours": "preprocessing.scaling_history_hours",
    "clip_value": "preprocessing.clip_value",
    "log_transform_positive_features": "preprocessing.log_transform_positive_features",
    "transform_volatility_window_hours": "features.transform_volatility_window_hours",
    "transform_detrend_window_hours": "features.transform_detrend_window_hours",
    "transform_sample_interval_hours": "features.transform_sample_interval_hours",
    "transform_input_signal": "features.transform_input_signal",
    "wavelet_family": "features.wavelet_family",
    "wavelet_base_scale_hours": "features.wavelet_base_scale_hours",
    "wavelet_levels": "features.wavelet_levels",
    "hilbert_input_signal": "features.hilbert_input_signal",
    "hilbert_window_hours": "features.hilbert_window_hours",
    "multitaper_input_signal": "features.multitaper_input_signal",
    "multitaper_window_hours": "features.multitaper_window_hours",
    "multitaper_time_bandwidth": "features.multitaper_time_bandwidth",
    "multitaper_taper_count": "features.multitaper_taper_count",
    "emd_input_signal": "features.emd_input_signal",
    "emd_backend": "features.emd_backend",
    "emd_window_hours": "features.emd_window_hours",
    "fracdiff_input_signal": "features.fracdiff_input_signal",
    "fracdiff_d": "features.fracdiff_d",
    "fracdiff_weight_threshold": "features.fracdiff_weight_threshold",
    "fracdiff_max_history_hours": "features.fracdiff_max_history_hours",
    "context_hours": "observation.context_hours",
    "context_representation": "observation.context_representation",
    "ridge_alpha": "proxy.ridge_alpha",
    "proxy_model_family": "proxy.model_family",
    "proxy_latent_dimension": "proxy.latent_dimension",
    "proxy_random_seed": "proxy.random_seed",
    "proxy_max_train_rows": "proxy.max_train_rows",
    "action_threshold_quantile": "proxy.action_threshold_quantile",
    "include_price_window": "observation.include_price_window",
    "include_agent_state": "observation.include_agent_state",
    "transaction_cost_fraction": "execution.commission_fraction",
    "risk_penalty_lambda": "reward.risk_penalty_lambda",
    "minimum_split_rows": "evaluation.minimum_split_rows",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, probability))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _duration_label(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


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
            json_extract(j.result_json, '$.evaluation_protocol_id')
                AS evaluation_protocol_id,
            json_extract(j.result_json, '$.evaluation_protocol_hash')
                AS evaluation_protocol_hash,
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
            MAX(CASE WHEN m.metric_name='evaluation_weeks' AND m.split='validation'
                     THEN m.value END) AS validation_evaluation_weeks,
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
            MAX(CASE WHEN m.metric_name='evaluation_weeks' AND m.split='test'
                     THEN m.value END) AS test_evaluation_weeks,
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


def _write_parameter_fact(
    conn: sqlite3.Connection,
    job_id: int,
    path: str,
    value: Any,
    now: str,
) -> None:
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


def _write_parameter_facts(conn: sqlite3.Connection, job_id: int, config: dict[str, Any]) -> None:
    now = utc_now()
    for path, value in _flatten(config):
        _write_parameter_fact(conn, job_id, path, value, now)
        alias = PARAMETER_PATH_ALIASES.get(path)
        if alias and alias != path:
            _write_parameter_fact(conn, job_id, alias, value, now)


def backfill_parameter_facts(conn: sqlite3.Connection) -> int:
    rows = conn.execute("SELECT id,config_json,result_json FROM jobs").fetchall()
    for row in rows:
        config = json.loads(row["config_json"])
        if row["result_json"]:
            result = json.loads(row["result_json"])
            config.update(dict(result.get("resolved_parameters") or {}))
        _write_parameter_facts(conn, int(row["id"]), config)
    conn.commit()
    return len(rows)


def _validate_terminal_result(stage: str, result: dict[str, Any]) -> None:
    if stage in PROXY_RESULT_STAGES:
        summary = result.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"{stage} result requires a summary object")
        if str(summary.get("screen_status") or "").startswith("blocked_"):
            return
        if not result.get("evaluation_protocol_id"):
            raise ValueError(f"{stage} result requires evaluation_protocol_id")
        if not result.get("evaluation_protocol_hash"):
            raise ValueError(f"{stage} result requires evaluation_protocol_hash")
        rows = metric_rows_from_result(result)
        indexed = {
            (row["metric_name"], row["split"]): row
            for row in rows
        }
        missing: list[str] = []
        invalid: list[str] = []
        for split in ("validation", "test"):
            for name, (unit, horizon) in CANONICAL_TRADING_METRICS.items():
                row = indexed.get((name, split))
                label = f"{split}.{name}"
                if row is None or row["value"] is None:
                    missing.append(label)
                    continue
                if row["unit"] != unit or row["horizon"] != horizon:
                    invalid.append(
                        f"{label}={row['unit']}/{row['horizon']}; "
                        f"expected={unit}/{horizon}"
                    )
        if missing:
            raise ValueError(
                f"{stage} result missing canonical metrics: {', '.join(missing)}"
            )
        if invalid:
            raise ValueError(
                f"{stage} result has mislabeled canonical metrics: "
                + "; ".join(invalid)
            )
        if stage == "E3_PROXY_MODEL_SCREEN":
            if summary.get("selection_source") != "frozen_upstream_contract":
                raise ValueError(
                    "E3_PROXY_MODEL_SCREEN requires frozen upstream features"
                )

    if stage == "E4_ASSET_POLICY_TRAINING":
        artifacts = list(result.get("artifacts") or [])
        champion = next(
            (
                item
                for item in artifacts
                if item.get("artifact_type") == "champion_model"
            ),
            None,
        )
        if champion is None:
            raise ValueError(
                "E4_ASSET_POLICY_TRAINING requires champion_model artifact"
            )
        for field in ("path", "sha256", "size_bytes"):
            if not champion.get(field):
                raise ValueError(
                    "E4_ASSET_POLICY_TRAINING champion_model requires "
                    + field
                )
        for field in (
            "resolved_config",
            "selected_features",
            "selected_features_sha256",
            "data_contract_sha256",
        ):
            if not result.get(field):
                raise ValueError(
                    f"E4_ASSET_POLICY_TRAINING requires {field}"
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
    *,
    attempt_number: int | None = None,
) -> None:
    now = utc_now()
    serialized_result = _json(result)
    row = conn.execute(
        """
        SELECT id, stage, attempt_count, claimed_by, status, result_json
        FROM jobs WHERE external_id=?
        """,
        (job_external_id,),
    ).fetchone()
    if row is None:
        raise PermissionError(f"unknown job {job_external_id}")
    expected_attempt = int(attempt_number or row["attempt_count"])
    previous = conn.execute(
        """
        SELECT status FROM job_attempts
        WHERE job_id=? AND attempt_number=? AND machine_id=?
        """,
        (row["id"], expected_attempt, machine_id),
    ).fetchone()
    if (
        row["status"] == "completed"
        and row["result_json"] == serialized_result
        and previous is not None
        and previous["status"] == "completed"
    ):
        return
    if (
        row["claimed_by"] != machine_id
        or row["status"] != "running"
        or int(row["attempt_count"]) != expected_attempt
    ):
        raise PermissionError(f"{machine_id} does not own running job {job_external_id}")
    _validate_terminal_result(str(row["stage"]), result)
    conn.execute(
        """
        UPDATE jobs SET status='completed', completed_at=?, result_json=?,
            lease_until=NULL, updated_at=? WHERE id=?
        """,
        (now, serialized_result, now, row["id"]),
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
    resolved_parameters = result.get("resolved_parameters")
    if isinstance(resolved_parameters, dict):
        _write_parameter_facts(conn, int(row["id"]), resolved_parameters)
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
    attempt_number: int | None = None,
) -> None:
    now = utc_now()
    row = conn.execute(
        "SELECT id, attempt_count, max_attempts, claimed_by, status FROM jobs WHERE external_id=?",
        (job_external_id,),
    ).fetchone()
    if row is None:
        raise PermissionError(f"unknown job {job_external_id}")
    expected_attempt = int(attempt_number or row["attempt_count"])
    previous = conn.execute(
        """
        SELECT status,error FROM job_attempts
        WHERE job_id=? AND attempt_number=? AND machine_id=?
        """,
        (row["id"], expected_attempt, machine_id),
    ).fetchone()
    truncated_error = error[-20000:]
    if (
        previous is not None
        and previous["status"] in {"pending", "failed"}
        and previous["error"] == truncated_error
    ):
        return
    if (
        row["claimed_by"] != machine_id
        or row["status"] != "running"
        or int(row["attempt_count"]) != expected_attempt
    ):
        raise PermissionError(f"{machine_id} does not own running job {job_external_id}")
    status = "pending" if retry and row["attempt_count"] < row["max_attempts"] else "failed"
    conn.execute(
        """
        UPDATE jobs SET status=?, claimed_by=NULL, lease_until=NULL, error=?, updated_at=?
        WHERE id=?
        """,
        (status, truncated_error, now, row["id"]),
    )
    conn.execute(
        """
        UPDATE job_attempts SET status=?, completed_at=?, error=?
        WHERE job_id=? AND attempt_number=?
        """,
        (status, now, truncated_error, row["id"], row["attempt_count"]),
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


def invalidate_stages(
    conn: sqlite3.Connection,
    stages: list[str],
    reason: str,
) -> int:
    """Invalidate protocol-incompatible results and restart those jobs."""
    normalized = sorted({str(stage).strip() for stage in stages if str(stage).strip()})
    if not normalized:
        raise ValueError("at least one stage is required")
    now = utc_now()
    placeholders = ",".join("?" for _ in normalized)
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            f"SELECT id,external_id FROM jobs WHERE stage IN ({placeholders})",
            normalized,
        ).fetchall()
        ids = [int(row["id"]) for row in rows]
        if ids:
            id_placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"DELETE FROM metric_facts WHERE job_id IN ({id_placeholders})",
                ids,
            )
            conn.execute(
                f"DELETE FROM artifacts WHERE job_id IN ({id_placeholders})",
                ids,
            )
            conn.execute(
                f"""
                UPDATE job_attempts
                SET status='invalidated_protocol',
                    completed_at=COALESCE(completed_at,?),
                    error=?
                WHERE job_id IN ({id_placeholders})
                """,
                (now, reason, *ids),
            )
            conn.execute(
                f"""
                UPDATE jobs
                SET status='pending',
                    claimed_by=NULL,
                    claimed_at=NULL,
                    heartbeat_at=NULL,
                    lease_until=NULL,
                    max_attempts=max_attempts+attempt_count,
                    started_at=NULL,
                    completed_at=NULL,
                    result_json=NULL,
                    error=NULL,
                    updated_at=?
                WHERE id IN ({id_placeholders})
                """,
                (now, *ids),
            )
        conn.execute(
            """
            INSERT INTO pool_events(event_type,subject_id,payload_json,created_at)
            VALUES('stages_invalidated',?,?,?)
            """,
            (
                ",".join(normalized),
                _json({"stages": normalized, "reason": reason, "jobs": len(rows)}),
                now,
            ),
        )
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise


def status(conn: sqlite3.Connection) -> dict[str, Any]:
    requeue_expired(conn)
    now = datetime.now(timezone.utc)
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
    completed_durations: dict[tuple[str, str], list[float]] = {}
    stage_durations: dict[str, list[float]] = {}
    for row in conn.execute(
        """
        SELECT a.machine_id,j.stage,a.started_at,a.completed_at
        FROM job_attempts a
        JOIN jobs j ON j.id=a.job_id
        WHERE a.status='completed' AND a.completed_at IS NOT NULL
        """
    ):
        started = _parse_utc(row["started_at"])
        completed = _parse_utc(row["completed_at"])
        if started is None or completed is None:
            continue
        duration = max(0.0, (completed - started).total_seconds())
        key = (str(row["machine_id"]), str(row["stage"]))
        completed_durations.setdefault(key, []).append(duration)
        stage_durations.setdefault(str(row["stage"]), []).append(duration)

    active_workers = sum(bool(machine.get("current_job_id")) for machine in machines)
    for machine in machines:
        current_job_id = machine.get("current_job_id")
        machine["candidate_eta"] = {
            "status": "idle",
            "remaining_range_seconds": None,
            "remaining_range_human": None,
            "sample_count": 0,
        }
        if not current_job_id:
            continue
        job = conn.execute(
            "SELECT stage,claimed_at FROM jobs WHERE external_id=?",
            (current_job_id,),
        ).fetchone()
        if job is None:
            machine["candidate_eta"]["status"] = "job_not_found"
            continue
        stage = str(job["stage"])
        machine["current_stage"] = stage
        samples = completed_durations.get((str(machine["machine_id"]), stage))
        source = "machine_stage"
        if not samples:
            samples = stage_durations.get(stage, [])
            source = "stage_all_machines"
        claimed_at = _parse_utc(job["claimed_at"])
        elapsed = max(0.0, (now - claimed_at).total_seconds()) if claimed_at else 0.0
        median = _quantile(samples, 0.5)
        p90 = _quantile(samples, 0.9)
        if median is None or p90 is None:
            machine["candidate_eta"] = {
                "status": "uncalibrated",
                "elapsed_seconds": round(elapsed, 1),
                "remaining_range_seconds": None,
                "remaining_range_human": None,
                "sample_count": 0,
                "sample_source": source,
            }
            continue
        low = max(0.0, median - elapsed)
        high = max(0.0, p90 - elapsed)
        estimate_status = "calibrated"
        if elapsed > p90:
            estimate_status = "exceeded_observed_p90"
        machine["candidate_eta"] = {
            "status": estimate_status,
            "elapsed_seconds": round(elapsed, 1),
            "expected_duration_p50_seconds": round(median, 1),
            "expected_duration_p90_seconds": round(p90, 1),
            "remaining_range_seconds": [round(low, 1), round(high, 1)],
            "remaining_range_human": [
                _duration_label(low),
                _duration_label(high),
            ],
            "sample_count": len(samples),
            "sample_source": source,
        }

    stage_estimates = []
    uncalibrated_stages = []
    total_low = 0.0
    total_high = 0.0
    worker_divisor = max(1, active_workers)
    for row in conn.execute(
        """
        SELECT stage,
               SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending_jobs,
               SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running_jobs
        FROM jobs
        GROUP BY stage
        HAVING pending_jobs > 0 OR running_jobs > 0
        ORDER BY MIN(priority),stage
        """
    ):
        stage = str(row["stage"])
        remaining_jobs = int(row["pending_jobs"] or 0) + int(row["running_jobs"] or 0)
        samples = stage_durations.get(stage, [])
        median = _quantile(samples, 0.5)
        p90 = _quantile(samples, 0.9)
        estimate = {
            "stage": stage,
            "pending_jobs": int(row["pending_jobs"] or 0),
            "running_jobs": int(row["running_jobs"] or 0),
            "remaining_jobs": remaining_jobs,
            "sample_count": len(samples),
        }
        if median is None or p90 is None:
            estimate.update(
                {
                    "status": "uncalibrated",
                    "remaining_range_seconds": None,
                    "remaining_range_human": None,
                }
            )
            uncalibrated_stages.append(stage)
        else:
            low = remaining_jobs * median / worker_divisor
            high = remaining_jobs * p90 / worker_divisor
            total_low += low
            total_high += high
            estimate.update(
                {
                    "status": "calibrated",
                    "job_duration_p50_seconds": round(median, 1),
                    "job_duration_p90_seconds": round(p90, 1),
                    "remaining_range_seconds": [round(low, 1), round(high, 1)],
                    "remaining_range_human": [
                        _duration_label(low),
                        _duration_label(high),
                    ],
                }
            )
        stage_estimates.append(estimate)

    total_jobs = sum(counts.values())
    remaining_jobs = int(counts.get("pending", 0)) + int(counts.get("running", 0))
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_schema": METRIC_SCHEMA,
        "display_metric_contract": {
            "return_and_risk_scale": "percent",
            "mean_weekly_return_percent": "week",
            "annualized_return_percent": "year",
            "mean_weekly_rap_percent": "week",
            "annual_rap_percent": "year",
            "max_drawdown_percent": "evaluation_period",
            "evaluation_weeks": "count",
            "optimization_score_dimensionless": "dimensionless",
        },
        "counts": counts,
        "stages": stages,
        "machines": machines,
        "eta": {
            "generated_at": now.isoformat(timespec="seconds"),
            "status": (
                "fully_calibrated"
                if not uncalibrated_stages
                else "partially_calibrated"
            ),
            "active_workers": active_workers,
            "total_jobs_in_pool": total_jobs,
            "remaining_jobs_in_pool": remaining_jobs,
            "calibrated_remaining_range_seconds": [
                round(total_low, 1),
                round(total_high, 1),
            ],
            "calibrated_remaining_range_human": [
                _duration_label(total_low),
                _duration_label(total_high),
            ],
            "uncalibrated_stages": uncalibrated_stages,
            "stage_estimates": stage_estimates,
        },
    }


def canonical_leaderboard(
    conn: sqlite3.Connection,
    *,
    split: str = "validation",
    limit: int = 100,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    prefix = f"{split}_"
    rows = conn.execute(
        f"""
        SELECT
            job_id,
            stage,
            machine_id,
            completed_at,
            evaluation_protocol_id,
            evaluation_protocol_hash,
            json_extract(config_json, '$.asset') AS asset,
            json_extract(config_json, '$.timeframe') AS timeframe,
            json_extract(config_json, '$.base_feature_bundle') AS base_feature_bundle,
            json_extract(config_json, '$.external_context_bundle') AS external_context_bundle,
            {prefix}mean_weekly_return AS mean_weekly_return,
            {prefix}annualized_return AS annualized_return,
            {prefix}mean_weekly_rap AS mean_weekly_rap,
            {prefix}annual_rap AS annual_rap,
            {prefix}max_drawdown AS max_drawdown,
            {prefix}evaluation_weeks AS evaluation_weeks,
            optimization_score_dimensionless
        FROM evidence_result_olap
        WHERE status='completed'
          AND {prefix}annual_rap IS NOT NULL
        ORDER BY {prefix}annual_rap DESC, job_id ASC
        LIMIT ?
        """,
        (max(1, min(int(limit), 5000)),),
    ).fetchall()

    def percent(value: Any) -> float | None:
        return None if value is None else float(value) * 100.0

    results = []
    for row in rows:
        results.append(
            {
                "job_id": row["job_id"],
                "stage": row["stage"],
                "machine_id": row["machine_id"],
                "completed_at": row["completed_at"],
                "asset": row["asset"],
                "timeframe": row["timeframe"],
                "base_feature_bundle": row["base_feature_bundle"],
                "external_context_bundle": row["external_context_bundle"],
                "evaluation_protocol_id": row["evaluation_protocol_id"],
                "evaluation_protocol_hash": row["evaluation_protocol_hash"],
                "mean_weekly_return_percent": percent(
                    row["mean_weekly_return"]
                ),
                "annualized_return_percent": percent(row["annualized_return"]),
                "mean_weekly_rap_percent": percent(row["mean_weekly_rap"]),
                "annual_rap_percent": percent(row["annual_rap"]),
                "max_drawdown_percent": percent(row["max_drawdown"]),
                "evaluation_weeks": row["evaluation_weeks"],
                "optimization_score_dimensionless": row[
                    "optimization_score_dimensionless"
                ],
            }
        )
    return {
        "metric_schema": METRIC_SCHEMA,
        "split": split,
        "return_and_risk_scale": "percent",
        "results": results,
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
    invalidate = sub.add_parser("invalidate-stages")
    invalidate.add_argument("--stage", action="append", required=True)
    invalidate.add_argument("--reason", required=True)
    sub.add_parser("backfill-parameter-facts")
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
    elif args.command == "invalidate-stages":
        output = {
            "ok": True,
            "invalidated": invalidate_stages(conn, args.stage, args.reason),
        }
    elif args.command == "backfill-parameter-facts":
        output = {
            "ok": True,
            "jobs": backfill_parameter_facts(conn),
        }
    else:
        output = status(conn)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
