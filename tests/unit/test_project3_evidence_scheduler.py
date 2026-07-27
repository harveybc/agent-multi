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
    apply_resource_eligibility,
    E3_PROXY_STAGE,
    E4_POLICY_STAGE,
    _executor_fleet_ready,
    promote_e2,
    promote_e2_interactions,
    promote_e3,
    promote_e4,
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
    metric_rows.append(
        {
            "metric_name": "turnover_events",
            "value": 25.0,
            "unit": "count",
            "horizon": "evaluation_period",
            "aggregation": "test_fixture",
            "split": "validation",
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


def test_scheduler_selects_robust_e3_cells_and_materializes_e4_training(
    tmp_path: Path,
) -> None:
    assets = (
        ("NZDUSD", "1h"),
        ("BNBUSDT", "1h"),
        ("EURUSD", "1h"),
        ("ETHUSDT", "1h"),
        ("ADAUSDT", "4h"),
        ("BTCUSDT", "4h"),
        ("XRPUSDT", "4h"),
        ("USDCAD", "4h"),
    )
    jobs = []
    for asset_index, (asset, timeframe) in enumerate(assets):
        for seed in (1701, 1702, 1703):
            jobs.append(
                {
                    "job_id": f"e3-{asset}-{timeframe}-{seed}",
                    "stage": E3_PROXY_STAGE,
                    "task_type": "feature_proxy_screen",
                    "config": {
                        "asset": asset,
                        "timeframe": timeframe,
                        "base_feature_bundle": "baseline_12",
                        "proxy_model_family": "ridge",
                        "proxy_random_seed": seed,
                        "upstream_selected_features": ["signal", "volume"],
                    },
                }
            )
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(conn, {"campaign_id": "e4-test", "jobs": jobs})
    for index in range(len(jobs)):
        claimed = claim_job(conn, "omega")
        result = _proxy_result(
            annual_rap=0.5 - index / 1000.0,
            selected_features=["signal", "volume"],
        )
        result["summary"]["selection_source"] = "frozen_upstream_contract"
        complete_job(conn, "omega", claimed["job_id"], result)

    promoted = promote_e4(conn)

    assert promoted["status"] == "enqueued"
    assert promoted["selected_cells"] == 8
    assert promoted["jobs"] == 24
    rows = conn.execute(
        "SELECT task_type,config_json FROM jobs WHERE stage=?",
        (E4_POLICY_STAGE,),
    ).fetchall()
    assert len(rows) == 24
    assert {row["task_type"] for row in rows} == {"asset_policy_training"}
    configs = [json.loads(row["config_json"]) for row in rows]
    assert {config["training_seed"] for config in configs} == {2701, 2702, 2703}
    assert {config["portfolio_role"] for config in configs} == {
        "short_horizon",
        "long_horizon",
    }
    assert all(config["upstream_selected_features"] for config in configs)


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


def test_resource_eligibility_never_leaves_a_write_transaction_open(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "pool.sqlite")
    init_db(conn)
    enqueue_plan(
        conn,
        {
            "campaign_id": "memory-policy-test",
            "jobs": [
                {
                    "job_id": "heavy",
                    "stage": "E2_PREPROCESSING_CONTEXT",
                    "task_type": "feature_proxy_screen",
                    "config": {
                        "timeframe": "15m",
                        "external_context_bundle": "all_non_cryptoquant",
                    },
                }
            ],
        },
    )
    assert apply_resource_eligibility(conn) == 1
    assert conn.in_transaction is False
    assert apply_resource_eligibility(conn) == 0
    assert conn.in_transaction is False
