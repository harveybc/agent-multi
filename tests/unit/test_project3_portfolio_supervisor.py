from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from project3_portfolio_supervisor import fetch_strategy_weeks, simulate_portfolio  # noqa: E402
from project3_weekly_pool import connect, enqueue_plan, init_db  # noqa: E402


def _result(composite: float, test: float, gate: bool = True) -> str:
    return json.dumps(
        {
            "train_validation_composite_score": composite,
            "train_tail_total_return": composite * 0.8,
            "validation_total_return": composite * 1.2,
            "test_total_return": test,
            "trade_gate_passed": gate,
        }
    )


def _seed_pool(db: Path) -> sqlite3.Connection:
    plan = db.with_name("plan.json")
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": "btc_best",
                        "candidate_id": "btc_best",
                        "asset": "btcusdt",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 1,
                        "training_policy": "scratch_n_years",
                        "input_data_file": "/tmp/btc.csv",
                        "feature_columns": ["f1"],
                        "subjobs": [
                            {
                                "subjob_id": "btc_best_w1",
                                "weekly_anchor_id": "2023-12-04",
                                "train_start": "2022-12-04 00:00:00",
                                "train_end": "2023-12-04 00:00:00",
                                "validation_start": "2023-12-04 00:00:00",
                                "validation_end": "2023-12-11 00:00:00",
                                "test_start": "2023-12-11 00:00:00",
                                "test_end": "2023-12-18 00:00:00",
                            },
                            {
                                "subjob_id": "btc_best_w2",
                                "weekly_anchor_id": "2023-12-11",
                                "train_start": "2022-12-11 00:00:00",
                                "train_end": "2023-12-11 00:00:00",
                                "validation_start": "2023-12-11 00:00:00",
                                "validation_end": "2023-12-18 00:00:00",
                                "test_start": "2023-12-18 00:00:00",
                                "test_end": "2023-12-25 00:00:00",
                            },
                        ],
                    },
                    {
                        "job_id": "btc_weak",
                        "candidate_id": "btc_weak",
                        "asset": "btcusdt",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 1,
                        "training_policy": "scratch_n_years",
                        "input_data_file": "/tmp/btc.csv",
                        "feature_columns": ["f1"],
                        "subjobs": [
                            {
                                "subjob_id": "btc_weak_w1",
                                "weekly_anchor_id": "2023-12-04",
                                "train_start": "2022-12-04 00:00:00",
                                "train_end": "2023-12-04 00:00:00",
                                "validation_start": "2023-12-04 00:00:00",
                                "validation_end": "2023-12-11 00:00:00",
                                "test_start": "2023-12-11 00:00:00",
                                "test_end": "2023-12-18 00:00:00",
                            }
                        ],
                    },
                    {
                        "job_id": "eth_best",
                        "candidate_id": "eth_best",
                        "asset": "ethusdt",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 1,
                        "training_policy": "scratch_n_years",
                        "input_data_file": "/tmp/eth.csv",
                        "feature_columns": ["f1"],
                        "subjobs": [
                            {
                                "subjob_id": "eth_best_w1",
                                "weekly_anchor_id": "2023-12-04",
                                "train_start": "2022-12-04 00:00:00",
                                "train_end": "2023-12-04 00:00:00",
                                "validation_start": "2023-12-04 00:00:00",
                                "validation_end": "2023-12-11 00:00:00",
                                "test_start": "2023-12-11 00:00:00",
                                "test_end": "2023-12-18 00:00:00",
                            },
                            {
                                "subjob_id": "eth_best_w2",
                                "weekly_anchor_id": "2023-12-11",
                                "train_start": "2022-12-11 00:00:00",
                                "train_end": "2023-12-11 00:00:00",
                                "validation_start": "2023-12-11 00:00:00",
                                "validation_end": "2023-12-18 00:00:00",
                                "test_start": "2023-12-18 00:00:00",
                                "test_end": "2023-12-25 00:00:00",
                            },
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)
    updates = {
        "btc_best_w1": _result(0.04, 0.01),
        "btc_best_w2": _result(0.03, 0.02),
        "btc_weak_w1": _result(0.01, 0.99),
        "eth_best_w1": _result(0.02, -0.01),
        "eth_best_w2": _result(0.05, 0.03),
    }
    with conn:
        for subjob_id, result in updates.items():
            conn.execute(
                """
                UPDATE subjobs
                SET status='done', result_json=?, completed_at='2023-12-01T00:00:00+00:00'
                WHERE external_id=?
                """,
                (result, subjob_id),
            )
    return conn


def test_portfolio_supervisor_selects_best_stream_per_asset(tmp_path):
    conn = _seed_pool(tmp_path / "pool.sqlite")

    weeks = fetch_strategy_weeks(conn)
    result = simulate_portfolio(
        weeks,
        method="equal_weight",
        max_assets=5,
        min_signal=0.0,
        lookback_weeks=12,
        risk_floor=1e-4,
        max_weight=0.5,
    )

    first_week_allocations = [
        row for row in result["allocations"] if row["test_start"] == "2023-12-11 00:00:00"
    ]
    assert {row["job_id"] for row in first_week_allocations} == {"btc_best", "eth_best"}
    assert sum(row["weight"] for row in first_week_allocations) == pytest.approx(1.0)
    assert result["weekly"][0]["portfolio_return"] == pytest.approx(0.0)


def test_portfolio_supervisor_score_weight_uses_known_composite_signal(tmp_path):
    conn = _seed_pool(tmp_path / "pool.sqlite")

    weeks = fetch_strategy_weeks(conn)
    result = simulate_portfolio(
        weeks,
        method="score_weight",
        max_assets=5,
        min_signal=0.0,
        lookback_weeks=12,
        risk_floor=1e-4,
        max_weight=0.8,
    )

    first_week_allocations = [
        row for row in result["allocations"] if row["test_start"] == "2023-12-11 00:00:00"
    ]
    btc_weight = next(row["weight"] for row in first_week_allocations if row["job_id"] == "btc_best")
    eth_weight = next(row["weight"] for row in first_week_allocations if row["job_id"] == "eth_best")
    assert btc_weight > eth_weight
    assert result["summary"]["weeks"] == 2
    assert result["summary"]["allocations"] == 4
