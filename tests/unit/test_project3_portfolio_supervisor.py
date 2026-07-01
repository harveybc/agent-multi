from __future__ import annotations

import json
import sqlite3
import sys
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from project3_portfolio_supervisor import (  # noqa: E402
    StrategyWeek,
    allocate_week,
    fetch_strategy_weeks,
    simulate_portfolio,
)
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


# ---------------------------------------------------------------------------
# Allocator / activation hardening (Part B)
# ---------------------------------------------------------------------------

WEEKS = [
    "2023-01-02 00:00:00",
    "2023-01-09 00:00:00",
    "2023-01-16 00:00:00",
    "2023-01-23 00:00:00",
    "2023-01-30 00:00:00",
    "2023-02-06 00:00:00",
]


def _sw(job_id: str, asset_key: str, test_start: str, composite: float, test_ret: float) -> StrategyWeek:
    return StrategyWeek(
        job_id=job_id,
        asset=asset_key.split("_")[0],
        timeframe="4h",
        asset_key=asset_key,
        weekly_anchor_id=test_start,
        test_start=test_start,
        test_end=test_start,
        composite_signal=composite,
        train_tail_return=composite,
        validation_return=composite,
        realized_test_return=test_ret,
        trade_gate_passed=True,
    )


def _by_job_history(specs: dict[str, tuple[str, list[float]]]) -> dict[str, list[StrategyWeek]]:
    by_job: dict[str, list[StrategyWeek]] = {}
    for job_id, (asset_key, returns) in specs.items():
        by_job[job_id] = [
            _sw(job_id, asset_key, WEEKS[k], 0.03, ret) for k, ret in enumerate(returns)
        ]
    return by_job


def test_allocation_never_uses_current_week_test_return():
    by_job = _by_job_history(
        {
            "A": ("aaa_4h", [0.01, -0.02, 0.03, -0.01, 0.02]),
            "B": ("bbb_4h", [-0.01, 0.02, -0.03, 0.01, -0.02]),
        }
    )
    current = WEEKS[5]
    for method in ("min_variance", "min_semivariance", "score_inverse_vol", "score_inverse_cvar"):
        low = [
            _sw("A", "aaa_4h", current, 0.03, -5.0),
            _sw("B", "bbb_4h", current, 0.03, -5.0),
        ]
        high = [
            _sw("A", "aaa_4h", current, 0.03, 5.0),
            _sw("B", "bbb_4h", current, 0.03, 5.0),
        ]
        w_low = allocate_week(
            low, by_job=by_job, method=method, lookback_weeks=12, risk_floor=1e-6, max_weight=1.0
        )
        w_high = allocate_week(
            high, by_job=by_job, method=method, lookback_weeks=12, risk_floor=1e-6, max_weight=1.0
        )
        assert w_low == w_high, f"{method} allocation leaked current test return"


def test_weights_sum_to_one_for_every_method():
    by_job = _by_job_history(
        {
            "A": ("aaa_4h", [0.01, -0.02, 0.03, -0.01, 0.02]),
            "B": ("bbb_4h", [-0.01, 0.02, -0.03, 0.01, -0.02]),
            "C": ("ccc_4h", [0.02, 0.01, -0.01, 0.02, -0.02]),
        }
    )
    current = WEEKS[5]
    selected = [
        _sw("A", "aaa_4h", current, 0.04, 0.0),
        _sw("B", "bbb_4h", current, 0.03, 0.0),
        _sw("C", "ccc_4h", current, 0.02, 0.0),
    ]
    for method in (
        "equal_weight",
        "score_weight",
        "score_inverse_vol",
        "score_inverse_cvar",
        "min_variance",
        "min_semivariance",
    ):
        weights = allocate_week(
            selected, by_job=by_job, method=method, lookback_weeks=12, risk_floor=1e-6, max_weight=1.0
        )
        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(value >= -1e-12 for value in weights.values())


def test_min_variance_respects_max_weight_cap():
    by_job = _by_job_history(
        {
            "LOW": ("low_4h", [0.001, 0.001, 0.0011, 0.001, 0.0009]),
            "MED": ("med_4h", [0.05, -0.04, 0.03, -0.05, 0.04]),
            "HIGH": ("high_4h", [0.06, -0.05, 0.04, -0.06, 0.05]),
        }
    )
    current = WEEKS[5]
    selected = [
        _sw("LOW", "low_4h", current, 0.03, 0.0),
        _sw("MED", "med_4h", current, 0.03, 0.0),
        _sw("HIGH", "high_4h", current, 0.03, 0.0),
    ]
    weights = allocate_week(
        selected, by_job=by_job, method="min_variance", lookback_weeks=12, risk_floor=1e-9, max_weight=0.5
    )
    assert max(weights.values()) <= 0.5 + 1e-9
    assert sum(weights.values()) == pytest.approx(1.0)
    # The near-constant (lowest-variance) stream is pinned at the cap.
    assert weights["LOW"] == pytest.approx(0.5, abs=1e-9)


def test_covariance_and_downside_allocators_are_deterministic():
    by_job = _by_job_history(
        {
            "A": ("aaa_4h", [0.01, -0.02, 0.03, -0.01, 0.02]),
            "B": ("bbb_4h", [-0.01, 0.02, -0.03, 0.01, -0.02]),
            "C": ("ccc_4h", [0.02, 0.01, -0.01, 0.02, -0.02]),
        }
    )
    current = WEEKS[5]
    selected = [
        _sw("A", "aaa_4h", current, 0.03, 0.0),
        _sw("B", "bbb_4h", current, 0.03, 0.0),
        _sw("C", "ccc_4h", current, 0.03, 0.0),
    ]
    for method in ("min_variance", "min_semivariance"):
        first = allocate_week(
            selected, by_job=by_job, method=method, lookback_weeks=12, risk_floor=1e-6, max_weight=1.0
        )
        second = allocate_week(
            selected, by_job=by_job, method=method, lookback_weeks=12, risk_floor=1e-6, max_weight=1.0
        )
        assert first == second
        assert sum(first.values()) == pytest.approx(1.0)
    # With 5 aligned observations and numpy present the true covariance path runs
    # (not the diagonal fallback), so weights are not the equal-weight vector.
    covariance_weights = allocate_week(
        selected, by_job=by_job, method="min_variance", lookback_weeks=12, risk_floor=1e-6, max_weight=1.0
    )
    assert any(abs(value - 1.0 / 3.0) > 1e-6 for value in covariance_weights.values())


def test_no_trade_activation_min_observations():
    weeks = [
        _sw("A", "aaa_4h", "2023-01-08 00:00:00", 0.05, 0.02),
        _sw("A", "aaa_4h", "2023-01-15 00:00:00", 0.05, 0.03),
    ]
    result = simulate_portfolio(
        weeks,
        method="equal_weight",
        max_assets=5,
        min_signal=0.0,
        lookback_weeks=12,
        risk_floor=1e-4,
        max_weight=0.5,
        min_observations=1,
    )
    by_week = {row["test_start"]: row for row in result["weekly"]}
    assert by_week["2023-01-08 00:00:00"]["selected_assets"] == 0
    assert by_week["2023-01-08 00:00:00"]["inactive_assets"] == 1
    assert by_week["2023-01-08 00:00:00"]["portfolio_return"] == pytest.approx(0.0)
    assert by_week["2023-01-15 00:00:00"]["selected_assets"] == 1
    first_week = next(
        row for row in result["activations"] if row["test_start"] == "2023-01-08 00:00:00"
    )
    assert first_week["active"] is False
    assert "insufficient_lookback_observations" in first_week["reasons"]


def test_no_trade_activation_recent_drawdown():
    weeks = [
        _sw("B", "bbb_4h", "2023-01-01 00:00:00", 0.05, 0.20),
        _sw("B", "bbb_4h", "2023-01-08 00:00:00", 0.05, -0.50),
        _sw("B", "bbb_4h", "2023-01-15 00:00:00", 0.05, 0.0),
    ]
    result = simulate_portfolio(
        weeks,
        method="equal_weight",
        max_assets=5,
        min_signal=0.0,
        lookback_weeks=12,
        risk_floor=1e-4,
        max_weight=0.5,
        max_drawdown_threshold=0.30,
    )
    by_week = {row["test_start"]: row for row in result["weekly"]}
    assert by_week["2023-01-15 00:00:00"]["selected_assets"] == 0
    decision = next(
        row for row in result["activations"] if row["test_start"] == "2023-01-15 00:00:00"
    )
    assert decision["active"] is False
    assert "recent_drawdown_exceeds_threshold" in decision["reasons"]


def test_portfolio_supervisor_db_output_is_reproducible_by_run_id(tmp_path):
    conn = _seed_pool(tmp_path / "pool.sqlite")
    from project3_portfolio_supervisor import _write_db

    weeks = fetch_strategy_weeks(conn)
    result = simulate_portfolio(
        weeks,
        method="min_variance",
        max_assets=5,
        min_signal=0.0,
        lookback_weeks=12,
        risk_floor=1e-4,
        max_weight=0.5,
    )
    config = {"method": "min_variance"}
    _write_db(conn, "run_xyz", config, result)
    _write_db(conn, "run_xyz", config, result)  # replace, not append

    runs = conn.execute("SELECT COUNT(*) AS n FROM portfolio_runs WHERE run_id='run_xyz'").fetchone()["n"]
    weekly = conn.execute(
        "SELECT COUNT(*) AS n FROM portfolio_weekly_returns WHERE run_id='run_xyz'"
    ).fetchone()["n"]
    assert runs == 1
    assert weekly == len(result["weekly"])


def test_v2_method_aliases_and_turnover_cost_reduce_net_return():
    weeks = [
        _sw("A", "aaa_4h", "2023-01-08 00:00:00", 0.05, 0.02),
        _sw("B", "bbb_4h", "2023-01-08 00:00:00", 0.04, 0.02),
        _sw("A", "aaa_4h", "2023-01-15 00:00:00", 0.05, 0.02),
        _sw("B", "bbb_4h", "2023-01-15 00:00:00", 0.04, 0.02),
    ]
    result = simulate_portfolio(
        weeks,
        method="score_inverse_vol_turnover_penalty",
        max_assets=5,
        min_signal=0.0,
        lookback_weeks=12,
        risk_floor=1e-4,
        max_weight=0.8,
        turnover_penalty_bps=100,
    )

    first_week = result["weekly"][0]
    assert first_week["portfolio_gross_return"] == pytest.approx(0.02)
    assert first_week["portfolio_turnover"] == pytest.approx(1.0)
    assert first_week["portfolio_rebalance_cost"] == pytest.approx(0.01)
    assert first_week["portfolio_return"] == pytest.approx(0.01)
    assert result["summary"]["mean_weekly_gross_return"] > result["summary"]["mean_weekly_return"]


def test_v2_asset_cap_limits_same_asset_multi_stream_exposure():
    weeks = [
        _sw("ETH_4H", "ethusdt_4h", "2023-01-08 00:00:00", 0.90, 0.0),
        _sw("ETH_1H", "ethusdt_1h", "2023-01-08 00:00:00", 0.80, 0.0),
        _sw("BTC_4H", "btcusdt_4h", "2023-01-08 00:00:00", 0.10, 0.0),
    ]
    result = simulate_portfolio(
        weeks,
        method="score_weight_top_k_capped",
        max_assets=5,
        min_signal=0.0,
        lookback_weeks=12,
        risk_floor=1e-4,
        max_weight=1.0,
        max_asset_weight=0.60,
    )

    first_week = [row for row in result["allocations"] if row["test_start"] == "2023-01-08 00:00:00"]
    eth_weight = sum(row["weight"] for row in first_week if row["asset"] == "ethusdt")
    assert eth_weight <= 0.60 + 1e-9
    assert sum(row["weight"] for row in first_week) == pytest.approx(1.0)


def test_v2_cutoff_manifest_and_activation_gates_are_recorded():
    weeks = [
        _sw("A", "aaa_4h", "2023-01-08 00:00:00", 0.05, 0.02),
        _sw("A", "aaa_4h", "2023-01-15 00:00:00", 0.05, 0.03),
    ]
    weak = weeks[0]
    weeks[0] = StrategyWeek(
        **{
            **weak.__dict__,
            "validation_trades": 0,
            "train_tail_trades": 5,
            "action_entropy": 0.5,
        }
    )
    strong = weeks[1]
    weeks[1] = StrategyWeek(
        **{
            **strong.__dict__,
            "validation_trades": 3,
            "train_tail_trades": 5,
            "action_entropy": 0.5,
        }
    )
    result = simulate_portfolio(
        weeks,
        method="equal_weight_top_k_capped",
        max_assets=5,
        min_signal=0.0,
        lookback_weeks=12,
        risk_floor=1e-4,
        max_weight=0.5,
        min_validation_trades=1,
    )

    first_decision = next(row for row in result["activations"] if row["test_start"] == "2023-01-08 00:00:00")
    assert first_decision["active"] is False
    assert "insufficient_validation_trades" in first_decision["reasons"]
    manifest = result["cutoff_manifests"][0]
    assert manifest["same_anchor_test_used"] is False
    assert manifest["future_anchor_used"] is False
    assert manifest["stage_c_access"] == "DENIED"


def test_v2_db_writes_cutoff_manifest_and_cost_columns(tmp_path):
    conn = _seed_pool(tmp_path / "pool.sqlite")
    from project3_portfolio_supervisor import _write_db

    weeks = fetch_strategy_weeks(conn)
    result = simulate_portfolio(
        weeks,
        method="score_inverse_vol_turnover_penalty",
        max_assets=5,
        min_signal=0.0,
        lookback_weeks=12,
        risk_floor=1e-4,
        max_weight=0.8,
        turnover_penalty_bps=50,
    )
    _write_db(conn, "run_v2", {"method": "score_inverse_vol_turnover_penalty"}, result)

    manifest_count = conn.execute(
        "SELECT COUNT(*) AS n FROM portfolio_cutoff_manifests WHERE run_id='run_v2'"
    ).fetchone()["n"]
    weekly = conn.execute(
        """
        SELECT portfolio_gross_return, portfolio_rebalance_cost, portfolio_return
        FROM portfolio_weekly_returns
        WHERE run_id='run_v2'
        ORDER BY test_start
        LIMIT 1
        """
    ).fetchone()
    assert manifest_count == len(result["weekly"])
    assert weekly["portfolio_rebalance_cost"] >= 0.0
    assert weekly["portfolio_return"] <= weekly["portfolio_gross_return"]


def test_portfolio_supervisor_runner_refreshes_stable_auto_latest_runs(tmp_path):
    conn = _seed_pool(tmp_path / "pool.sqlite")
    conn.close()
    from project3_portfolio_supervisor_runner import run_sweep

    args = Namespace(
        db=tmp_path / "pool.sqlite",
        output_dir=tmp_path / "portfolio",
        run_prefix="auto_latest_test",
        methods=["equal_weight_top_k_capped", "score_inverse_cvar_top_k_capped"],
        sqlite_timeout_sec=5.0,
        write_db=True,
        max_assets=5,
        min_signal=0.0,
        lookback_weeks=12,
        risk_floor=1e-4,
        max_weight=0.5,
        max_asset_weight=0.6,
        max_cluster_weight=0.8,
        turnover_penalty_bps=0.0,
        portfolio_vol_target=None,
        min_test_start=None,
        allow_failed_trade_gate=False,
        min_observations=0,
        max_drawdown_threshold=None,
        drawdown_lookback_weeks=0,
        activation_min_signal=0.0,
        min_train_tail_trades=0,
        min_validation_trades=0,
        min_action_entropy=None,
        max_cost_to_gross_edge=None,
        require_no_broker_violations=False,
        require_no_friday_violations=False,
    )

    first = run_sweep(args)
    second = run_sweep(args)
    assert [row["run_id"] for row in first] == [
        "auto_latest_test_equal_weight_top_k_capped",
        "auto_latest_test_score_inverse_cvar_top_k_capped",
    ]
    assert [row["run_id"] for row in second] == [row["run_id"] for row in first]

    conn = sqlite3.connect(tmp_path / "pool.sqlite")
    count = conn.execute(
        "SELECT COUNT(*) FROM portfolio_runs WHERE run_id LIKE 'auto_latest_test_%'"
    ).fetchone()[0]
    weekly_count = conn.execute(
        "SELECT COUNT(*) FROM portfolio_weekly_returns WHERE run_id='auto_latest_test_equal_weight_top_k_capped'"
    ).fetchone()[0]
    conn.close()
    assert count == 2
    assert weekly_count > 0
