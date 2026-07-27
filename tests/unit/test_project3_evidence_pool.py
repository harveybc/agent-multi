from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import base64
import hashlib


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from project3_evidence_pool import (  # noqa: E402
    PARAMETER_PATH_ALIASES,
    _validate_terminal_result,
    claim_job,
    complete_job,
    backfill_parameter_facts,
    canonical_leaderboard,
    connect,
    enqueue_plan,
    fail_job,
    heartbeat,
    init_db,
    invalidate_stages,
    requeue_machine,
    requeue_orphaned_jobs,
    status,
)
from project3_evidence_screen import RESOLVED_PARAMETER_KEYS  # noqa: E402


def test_every_executed_parameter_has_a_canonical_olap_registry_path() -> None:
    registry_path = (
        Path(__file__).resolve().parents[2]
        / "examples/config/evidence_sweep/project3_parameter_registry_v1.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registered = {item["path"] for item in registry["parameters"]}
    missing_aliases = sorted(set(RESOLVED_PARAMETER_KEYS) - set(PARAMETER_PATH_ALIASES))
    assert not missing_aliases
    missing_registry = sorted(
        PARAMETER_PATH_ALIASES[key]
        for key in RESOLVED_PARAMETER_KEYS
        if PARAMETER_PATH_ALIASES[key] not in registered
    )
    assert not missing_registry


def test_proxy_completion_rejects_missing_or_mislabeled_canonical_metrics() -> None:
    result = {
        "evaluation_protocol_id": "protocol",
        "evaluation_protocol_hash": "hash",
        "summary": {"selection_source": "train_only_selector"},
        "metric_rows": [
            {
                "metric_name": "annual_rap",
                "value": 0.2,
                "unit": "fraction",
                "horizon": "week",
                "aggregation": "bad_fixture",
                "split": "validation",
            }
        ],
    }
    try:
        _validate_terminal_result("E2_PREPROCESSING_CONTEXT", result)
    except ValueError as exc:
        assert "canonical metrics" in str(exc)
    else:
        raise AssertionError("incomplete canonical metrics should be rejected")


def test_asset_policy_completion_requires_loadable_champion_contract() -> None:
    try:
        _validate_terminal_result(
            "E4_ASSET_POLICY_TRAINING",
            {
                "summary": {"screen_status": "blocked_test_fixture"},
                "artifacts": [],
            },
        )
    except ValueError as exc:
        assert "champion_model" in str(exc)
    else:
        raise AssertionError("missing champion artifact should be rejected")


def test_asset_policy_completion_accepts_verified_artifact_and_metrics() -> None:
    content = b"stable-baselines3-fixture"
    metrics = []
    values = {
        "mean_weekly_return": (0.01, "fraction", "week"),
        "annualized_return": (0.52, "fraction", "year"),
        "mean_weekly_rap": (0.005, "fraction", "week"),
        "annual_rap": (0.26, "fraction", "year"),
        "max_drawdown": (0.08, "fraction", "evaluation_period"),
        "evaluation_weeks": (52.0, "count", "evaluation_period"),
    }
    for split in ("validation", "test"):
        for name, (value, unit, horizon) in values.items():
            metrics.append(
                {
                    "metric_name": name,
                    "value": value,
                    "unit": unit,
                    "horizon": horizon,
                    "aggregation": "fixture",
                    "split": split,
                }
            )
    _validate_terminal_result(
        "E4_ASSET_POLICY_TRAINING",
        {
            "evaluation_protocol_id": "asset-policy-protocol",
            "evaluation_protocol_hash": "a" * 64,
            "summary": {},
            "metric_rows": metrics,
            "artifacts": [
                {
                    "artifact_type": "champion_model",
                    "path": "/tmp/champion.zip",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                    "content_base64": base64.b64encode(content).decode("ascii"),
                    "metadata": {
                        "format": "stable_baselines3_zip",
                        "load_tested": True,
                    },
                }
            ],
            "resolved_config": {"experiment": {"name": "fixture"}},
            "selected_features": ["signal"],
            "selected_features_sha256": "b" * 64,
            "observation_columns": ["signal__last"],
            "observation_columns_sha256": "c" * 64,
            "data_contract_sha256": "d" * 64,
            "source_manifest": [{"relative_path": "input.csv"}],
        },
    )


def _plan() -> dict:
    return {
        "campaign_id": "test-campaign",
        "jobs": [
            {
                "job_id": "job-a",
                "stage": "E0",
                "task_type": "data_contract_audit",
                "config": {"asset": "BTCUSDT", "context_hours": 168},
            },
            {
                "job_id": "job-b",
                "stage": "E1",
                "task_type": "feature_proxy_screen",
                "config": {"asset": "ETHUSDT", "context_hours": 72},
            },
        ],
    }


def test_pool_claim_is_atomic_and_materializes_olap_facts(tmp_path: Path) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    assert enqueue_plan(conn, _plan()) == {"inserted": 2, "existing": 0}

    first = claim_job(conn, "omega")
    second = claim_job(conn, "dragon")
    assert first and second
    assert first["job_id"] != second["job_id"]
    assert claim_job(conn, "gamma") is None

    complete_job(
        conn,
        "omega",
        first["job_id"],
        {
            "metric_rows": [
                {
                    "metric_name": "mean_weekly_return",
                    "value": 0.01,
                    "unit": "fraction",
                    "horizon": "week",
                    "aggregation": "arithmetic_mean",
                    "split": "validation",
                },
                {
                    "metric_name": "evaluation_weeks",
                    "value": 52.0,
                    "unit": "count",
                    "horizon": "evaluation_period",
                    "aggregation": "count",
                    "split": "validation",
                },
            ]
        },
    )
    row = conn.execute(
        """
        SELECT validation_mean_weekly_return,validation_evaluation_weeks
        FROM evidence_result_olap WHERE job_id=?
        """,
        (first["job_id"],),
    ).fetchone()
    assert row["validation_mean_weekly_return"] == 0.01
    assert row["validation_evaluation_weeks"] == 52.0
    canonical_parameter = conn.execute(
        """
        SELECT value_numeric FROM parameter_facts
        WHERE job_id=(SELECT id FROM jobs WHERE external_id=?)
          AND parameter_path='observation.context_hours'
        """,
        (first["job_id"],),
    ).fetchone()
    assert canonical_parameter["value_numeric"] == 168.0
    assert status(conn)["counts"] == {"completed": 1, "running": 1}


def test_terminal_reports_are_idempotent_by_attempt(tmp_path: Path) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(conn, _plan())
    completed = claim_job(conn, "omega")
    result = {"metric_rows": []}
    complete_job(
        conn,
        "omega",
        completed["job_id"],
        result,
        attempt_number=completed["attempt"],
    )
    complete_job(
        conn,
        "omega",
        completed["job_id"],
        result,
        attempt_number=completed["attempt"],
    )

    failed = claim_job(conn, "dragon")
    fail_job(
        conn,
        "dragon",
        failed["job_id"],
        "transient failure",
        attempt_number=failed["attempt"],
    )
    fail_job(
        conn,
        "dragon",
        failed["job_id"],
        "transient failure",
        attempt_number=failed["attempt"],
    )
    assert status(conn)["counts"] == {"completed": 1, "pending": 1}


def test_enqueue_rejects_changed_config_for_existing_job(tmp_path: Path) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(conn, _plan())
    changed = _plan()
    changed["jobs"][0]["config"]["context_hours"] = 999
    try:
        enqueue_plan(conn, changed)
    except ValueError as exc:
        assert "different config" in str(exc)
    else:
        raise AssertionError("changed config should have been rejected")


def test_operator_can_requeue_stopped_machine_without_waiting_for_lease(tmp_path: Path) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(conn, _plan())
    claimed = claim_job(conn, "gamma-5090")
    assert claimed
    assert requeue_machine(conn, "gamma-5090", "rolling service update") == 1
    replacement = claim_job(conn, "dragon")
    assert replacement
    assert replacement["job_id"] == claimed["job_id"]


def test_newer_worker_heartbeat_recovers_old_orphan_claim(tmp_path: Path) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(conn, _plan())
    old_claim = claim_job(conn, "gamma")
    assert old_claim
    claimed_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    heartbeat_at = datetime.now(timezone.utc)
    conn.execute(
        "UPDATE jobs SET claimed_at=? WHERE external_id=?",
        (claimed_at.isoformat(timespec="seconds"), old_claim["job_id"]),
    )
    conn.execute(
        """
        INSERT INTO machine_heartbeats(
            machine_id,status,current_job_id,message,cpu_summary_json,
            gpu_summary_json,heartbeat_at
        ) VALUES(?,?,?,?,?,?,?)
        """,
        (
            "gamma",
            "running",
            "a-newer-job",
            "restarted worker",
            "{}",
            "{}",
            heartbeat_at.isoformat(timespec="seconds"),
        ),
    )
    conn.commit()

    assert requeue_orphaned_jobs(conn) == 1
    recovered = conn.execute(
        "SELECT status,claimed_by,max_attempts FROM jobs WHERE external_id=?",
        (old_claim["job_id"],),
    ).fetchone()
    assert dict(recovered) == {
        "status": "pending",
        "claimed_by": None,
        "max_attempts": 3,
    }
    attempt = conn.execute(
        """
        SELECT status FROM job_attempts
        WHERE job_id=(SELECT id FROM jobs WHERE external_id=?)
        """,
        (old_claim["job_id"],),
    ).fetchone()
    assert attempt["status"] == "orphaned_worker_requeue"


def test_protocol_invalidation_removes_results_and_requeues_stage(tmp_path: Path) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(conn, _plan())
    claimed = claim_job(conn, "omega")
    assert claimed and claimed["stage"] == "E0"
    complete_job(
        conn,
        "omega",
        claimed["job_id"],
        {
            "metric_rows": [
                {
                    "metric_name": "annual_rap",
                    "value": 1.0,
                    "unit": "fraction",
                    "horizon": "year",
                    "aggregation": "weekly_mean_x_52",
                    "split": "validation",
                }
            ]
        },
    )
    assert invalidate_stages(conn, ["E0"], "realized-return protocol correction") == 1
    row = conn.execute(
        "SELECT status,result_json,attempt_count,max_attempts FROM jobs WHERE external_id=?",
        (claimed["job_id"],),
    ).fetchone()
    assert dict(row) == {
        "status": "pending",
        "result_json": None,
        "attempt_count": 1,
        "max_attempts": 4,
    }
    assert conn.execute("SELECT COUNT(*) FROM metric_facts").fetchone()[0] == 0
    replacement = claim_job(conn, "dragon")
    assert replacement and replacement["job_id"] == claimed["job_id"]
    assert replacement["attempt"] == 2


def test_resolved_defaults_are_written_as_canonical_parameter_facts(tmp_path: Path) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(conn, _plan())
    claimed = claim_job(conn, "omega")
    complete_job(
        conn,
        "omega",
        claimed["job_id"],
        {
            "resolved_parameters": {
                "target_definition": "forward_return",
                "cross_asset_reference_set": "none",
            },
            "metric_rows": [],
        },
    )
    backfill_parameter_facts(conn)
    rows = {
        row["parameter_path"]: row["value_text"]
        for row in conn.execute(
            """
            SELECT parameter_path,value_text FROM parameter_facts
            WHERE job_id=(SELECT id FROM jobs WHERE external_id=?)
            """,
            (claimed["job_id"],),
        )
    }
    assert rows["data.target_definition"] == "forward_return"
    assert rows["data.cross_asset_reference_set"] == "none"


def test_leaderboard_always_exposes_labeled_percent_scales(tmp_path: Path) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(
        conn,
        {
            "campaign_id": "leaderboard-test",
            "jobs": [
                {
                    "job_id": "leader",
                    "stage": "TEST_STAGE",
                    "task_type": "test",
                    "config": {
                        "asset": "BTCUSDT",
                        "timeframe": "1h",
                        "base_feature_bundle": "baseline_12",
                    },
                }
            ],
        },
    )
    claimed = claim_job(conn, "omega")
    rows = []
    metrics = {
        "mean_weekly_return": (0.01, "fraction", "week"),
        "annualized_return": (0.6777, "fraction", "year"),
        "mean_weekly_rap": (0.005, "fraction", "week"),
        "annual_rap": (0.26, "fraction", "year"),
        "max_drawdown": (0.12, "fraction", "evaluation_period"),
        "evaluation_weeks": (52.0, "count", "evaluation_period"),
    }
    for name, (value, unit, horizon) in metrics.items():
        rows.append(
            {
                "metric_name": name,
                "value": value,
                "unit": unit,
                "horizon": horizon,
                "aggregation": "fixture",
                "split": "validation",
            }
        )
    complete_job(
        conn,
        "omega",
        claimed["job_id"],
        {"metric_rows": rows},
    )
    payload = canonical_leaderboard(conn, split="validation")
    assert payload["return_and_risk_scale"] == "percent"
    result = payload["results"][0]
    assert result["mean_weekly_return_percent"] == 1.0
    assert result["annualized_return_percent"] == 67.77
    assert result["mean_weekly_rap_percent"] == 0.5
    assert result["annual_rap_percent"] == 26.0
    assert result["max_drawdown_percent"] == 12.0
    assert result["evaluation_weeks"] == 52.0


def test_status_exposes_candidate_and_pool_eta_from_observed_durations(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(
        conn,
        {
            "campaign_id": "eta-test",
            "jobs": [
                {
                    "job_id": f"job-{index}",
                    "stage": "E1",
                    "task_type": "screen",
                    "config": {"asset": "BTCUSDT"},
                }
                for index in range(3)
            ],
        },
    )
    now = datetime.now(timezone.utc)
    for index, duration in enumerate((60, 120)):
        claimed = claim_job(conn, "omega")
        assert claimed
        complete_job(
            conn,
            "omega",
            claimed["job_id"],
            {"metric_rows": []},
        )
        conn.execute(
            """
            UPDATE job_attempts
            SET started_at=?,completed_at=?
            WHERE job_id=(SELECT id FROM jobs WHERE external_id=?)
            """,
            (
                (now - timedelta(seconds=duration)).isoformat(timespec="seconds"),
                now.isoformat(timespec="seconds"),
                f"job-{index}",
            ),
        )
        conn.commit()
    current = claim_job(conn, "omega")
    assert current
    claimed_at = (now - timedelta(seconds=30)).isoformat(timespec="seconds")
    conn.execute(
        "UPDATE jobs SET claimed_at=? WHERE external_id=?",
        (claimed_at, current["job_id"]),
    )
    conn.commit()
    heartbeat(conn, "omega", current["job_id"], status="running")

    payload = status(conn)
    assert payload["eta"]["status"] == "fully_calibrated"
    assert payload["eta"]["total_jobs_in_pool"] == 3
    assert payload["eta"]["remaining_jobs_in_pool"] == 1
    assert payload["eta"]["stage_estimates"][0]["sample_count"] == 2
    assert payload["eta"]["stage_estimates"][0]["remaining_range_seconds"] == [
        90.0,
        114.0,
    ]
    candidate = payload["machines"][0]["candidate_eta"]
    assert candidate["status"] == "calibrated"
    assert candidate["sample_count"] == 2
    assert 59.0 <= candidate["remaining_range_seconds"][0] <= 60.0
    assert 83.0 <= candidate["remaining_range_seconds"][1] <= 84.0
