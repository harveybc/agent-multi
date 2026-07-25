from __future__ import annotations

import json
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from project3_evidence_pool import (  # noqa: E402
    PARAMETER_PATH_ALIASES,
    _validate_terminal_result,
    claim_job,
    complete_job,
    backfill_parameter_facts,
    connect,
    enqueue_plan,
    fail_job,
    init_db,
    invalidate_stages,
    requeue_machine,
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
            {"artifacts": []},
        )
    except ValueError as exc:
        assert "champion_model" in str(exc)
    else:
        raise AssertionError("missing champion artifact should be rejected")


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
