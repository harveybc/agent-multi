from __future__ import annotations

import json
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from project3_evidence_pool import (  # noqa: E402
    claim_job,
    complete_job,
    connect,
    enqueue_plan,
    heartbeat,
    init_db,
)
from project3_evidence_scheduler import (  # noqa: E402
    _eligible_machines,
    E3_PROXY_STAGE,
    _executor_fleet_ready,
    promote_e2,
    promote_e2_interactions,
    promote_e3,
)
from project3_evidence_screen import (  # noqa: E402
    EVIDENCE_EXECUTOR_VERSION,
    FEATURE_PROXY_PROTOCOL,
    FEATURE_PROXY_PROTOCOL_HASH,
)


def _proxy_result(
    *,
    annual_rap: float,
    selected_features: list[str] | None = None,
) -> dict:
    metric_rows = []
    values = {
        "mean_weekly_return": (annual_rap / 52.0, "fraction", "week"),
        "annualized_return": (annual_rap, "fraction", "year"),
        "mean_weekly_rap": (annual_rap / 52.0, "fraction", "week"),
        "annual_rap": (annual_rap, "fraction", "year"),
        "max_drawdown": (0.1, "fraction", "evaluation_period"),
        "evaluation_weeks": (52.0, "count", "evaluation_period"),
    }
    for split in ("validation", "test"):
        for name, (value, unit, horizon) in values.items():
            metric_rows.append(
                {
                    "metric_name": name,
                    "value": value,
                    "unit": unit,
                    "horizon": horizon,
                    "aggregation": "test_fixture",
                    "split": split,
                }
            )
    return {
        "evaluation_protocol_id": FEATURE_PROXY_PROTOCOL,
        "evaluation_protocol_hash": FEATURE_PROXY_PROTOCOL_HASH,
        "summary": {
            "selected_features": selected_features or ["signal"],
            "selection_source": "train_only_selector",
        },
        "metric_rows": metric_rows,
    }


def test_executor_fleet_gate_requires_fresh_matching_worker_versions(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(
        conn,
        {
            "campaign_id": "fleet-gate",
            "required_workers": ["omega", "dragon"],
            "required_executor_version": EVIDENCE_EXECUTOR_VERSION,
            "jobs": [],
        },
    )
    ready, problems = _executor_fleet_ready(conn)
    assert not ready
    assert problems == {
        "dragon": "missing_heartbeat",
        "omega": "missing_heartbeat",
    }
    for machine_id in ("omega", "dragon"):
        heartbeat(
            conn,
            machine_id,
            None,
            status="idle",
            cpu_summary={"evidence_executor_version": EVIDENCE_EXECUTOR_VERSION},
        )
    assert _executor_fleet_ready(conn) == (True, {})


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
        _proxy_result(annual_rap=-0.2),
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
            _proxy_result(annual_rap=float(index)),
        )
    result = promote_e2_interactions(conn)
    assert result["status"] == "enqueued"
    assert result["jobs"] >= 5
    assert result["inserted"] == result["jobs"]


def test_scheduler_freezes_e2_features_for_bounded_e3_proxy_screen(
    tmp_path: Path,
) -> None:
    base_config = {
        "asset": "BTCUSDT",
        "timeframe": "1h",
        "base_feature_bundle": "baseline_12",
        "external_context_bundle": "none",
        "feature_selection_method": "rank_ic_topk",
        "feature_budget": 2,
        "preprocessing_mode": "rolling_zscore",
        "scaling_history_hours": 168,
        "clip_value": 10,
        "context_hours": 168,
        "context_representation": "summary",
    }
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(
        conn,
        {
            "campaign_id": "e3-test",
            "jobs": [
                {
                    "job_id": "e2i-winner",
                    "stage": "E2_INTERACTION_CONFIRMATION",
                    "task_type": "feature_proxy_screen",
                    "config": base_config,
                }
            ],
        },
    )
    claimed = claim_job(conn, "omega")
    complete_job(
        conn,
        "omega",
        claimed["job_id"],
        _proxy_result(
            annual_rap=0.25,
            selected_features=["signal", "volume"],
        ),
    )
    result = promote_e3(conn)
    assert result == {
        "status": "enqueued",
        "jobs": 36,
        "inserted": 36,
        "existing": 1,
    }
    rows = conn.execute(
        "SELECT stage,task_type,config_json FROM jobs WHERE stage=?",
        (E3_PROXY_STAGE,),
    ).fetchall()
    assert len(rows) == 36
    for row in rows:
        config = json.loads(row["config_json"])
        assert row["task_type"] == "feature_proxy_screen"
        assert config["upstream_selected_features"] == ["signal", "volume"]
        assert config["upstream_evaluation_protocol_id"] == FEATURE_PROXY_PROTOCOL
        assert (
            config["upstream_evaluation_protocol_hash"]
            == FEATURE_PROXY_PROTOCOL_HASH
        )


def test_widest_15m_jobs_exclude_second_gamma_worker() -> None:
    assert _eligible_machines(
        {
            "timeframe": "15m",
            "external_context_bundle": "all_non_cryptoquant",
        }
    ) == ["omega", "dragon", "gamma-5090"]
    assert _eligible_machines(
        {
            "timeframe": "1h",
            "external_context_bundle": "all_non_cryptoquant",
        }
    ) == []
