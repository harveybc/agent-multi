from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from project3_evidence_pool import (  # noqa: E402
    claim_job,
    complete_job,
    connect,
    enqueue_plan,
    init_db,
    requeue_machine,
    status,
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
                }
            ]
        },
    )
    row = conn.execute(
        "SELECT validation_mean_weekly_return FROM evidence_result_olap WHERE job_id=?",
        (first["job_id"],),
    ).fetchone()
    assert row["validation_mean_weekly_return"] == 0.01
    assert status(conn)["counts"] == {"completed": 1, "running": 1}


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
