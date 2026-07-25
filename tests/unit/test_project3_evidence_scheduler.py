from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from project3_evidence_pool import claim_job, complete_job, connect, enqueue_plan, init_db  # noqa: E402
from project3_evidence_scheduler import (  # noqa: E402
    promote_e2,
    promote_e2_interactions,
)


def test_scheduler_materializes_bounded_e2_after_e1_finishes(tmp_path: Path) -> None:
    base_config = {
        "asset": "BTCUSDT",
        "timeframe": "1h",
        "base_feature_bundle": "baseline_12",
        "external_context_bundle": "none",
        "feature_selection_method": "rank_ic_topk",
        "feature_budget": 32,
        "preprocessing_mode": "rolling_zscore",
        "scaling_history_hours": 168,
        "clip_value": 10,
        "context_hours": 168,
        "context_representation": "summary",
    }
    plan = {
        "campaign_id": "scheduler-test",
        "jobs": [
            {
                "job_id": "e1-job",
                "stage": "E1_BASE_SOURCE_SCREEN",
                "task_type": "feature_proxy_screen",
                "config": base_config,
            }
        ],
    }
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(conn, plan)
    assert promote_e2(conn)["status"] == "waiting_for_e1"
    claimed = claim_job(conn, "omega")
    complete_job(
        conn,
        "omega",
        claimed["job_id"],
        {
            "metric_rows": [
                {
                    "metric_name": "annual_rap",
                    "value": -0.2,
                    "unit": "fraction",
                    "horizon": "year",
                    "aggregation": "weekly_mean_x_52",
                    "split": "validation",
                }
            ]
        },
    )
    materialized = tmp_path / "materialized.json"
    result = promote_e2(conn, materialized_plan=materialized)
    assert result["status"] == "enqueued"
    assert 165 <= result["jobs"] <= 200
    assert materialized.exists()
    stages = {
        row["stage"]: row["n"]
        for row in conn.execute("SELECT stage,COUNT(*) AS n FROM jobs GROUP BY stage")
    }
    assert stages["E2_PREPROCESSING_CONTEXT"] == result["jobs"]


def test_scheduler_materializes_interactions_from_ranked_e2(tmp_path: Path) -> None:
    base_config = {
        "asset": "BTCUSDT",
        "timeframe": "1h",
        "base_feature_bundle": "baseline_12",
        "external_context_bundle": "none",
        "feature_selection_method": "rank_ic_topk",
        "feature_budget": 32,
        "preprocessing_mode": "rolling_zscore",
        "scaling_history_hours": 168,
        "clip_value": 10,
        "context_hours": 168,
        "context_representation": "summary",
    }
    jobs = []
    for index, patch in enumerate(
        (
            {},
            {"clip_value": 5},
            {"context_hours": 336},
            {"feature_budget": 48},
            {"preprocessing_mode": "rolling_robust"},
        )
    ):
        jobs.append(
            {
                "job_id": f"e2-{index}",
                "stage": "E2_PREPROCESSING_CONTEXT",
                "task_type": "feature_proxy_screen",
                "config": {**base_config, **patch},
            }
        )
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(conn, {"campaign_id": "interaction-test", "jobs": jobs})
    for index in range(len(jobs)):
        claimed = claim_job(conn, "omega")
        complete_job(
            conn,
            "omega",
            claimed["job_id"],
            {
                "metric_rows": [
                    {
                        "metric_name": "annual_rap",
                        "value": float(index),
                        "unit": "fraction",
                        "horizon": "year",
                        "aggregation": "weekly_mean_x_52",
                        "split": "validation",
                    }
                ]
            },
        )
    result = promote_e2_interactions(conn)
    assert result["status"] == "enqueued"
    assert result["jobs"] >= 5
    assert result["inserted"] == result["jobs"]
