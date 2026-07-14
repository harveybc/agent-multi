from __future__ import annotations

import csv

import pytest

from app.weekly_promotion import aggregate_weekly_results, build_week_config, build_week_windows


def _base_config() -> dict:
    return {
        "schema_version": "trading_experiment.v1",
        "experiment": {"name": "sol_policy"},
        "data": {
            "train_start": "2021-01-01T00:00:00",
            "test_start": "2023-01-01T00:00:00",
            "test_end": "2023-12-31T23:59:59",
        },
        "training": {"learning_rate": 0.0001},
        "asset_policy": {"continuous_action_threshold": 0.1},
        "optimization": {"enabled": True},
    }


def _candidate() -> dict:
    return {
        "candidate_id": "sha256:candidate",
        "parameters": {
            "learning_rate": 0.0002,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "train_freq": 2,
            "gradient_steps": 4,
            "continuous_action_threshold": 0.2,
        },
    }


def test_week_windows_cover_2023_with_causal_boundaries() -> None:
    windows = build_week_windows(
        train_start="2021-01-01T00:00:00",
        protected_test_start="2023-01-01T00:00:00",
        protected_test_end="2023-12-31T23:59:59",
        validation_days=182,
    )

    assert len(windows) == 52
    assert windows[0].label == "2023-01-02"
    assert windows[-1].label == "2023-12-25"
    for window in windows:
        assert window.train_start < window.train_end <= window.validation_start
        assert window.validation_start < window.validation_end == window.test_start < window.test_end


def test_week_config_freezes_candidate_and_never_selects_on_test(tmp_path) -> None:
    window = build_week_windows(
        train_start="2021-01-01T00:00:00",
        protected_test_start="2023-01-01T00:00:00",
        protected_test_end="2023-12-31T23:59:59",
        validation_days=182,
    )[0]
    config = build_week_config(
        base_config=_base_config(),
        candidate=_candidate(),
        window=window,
        output_root=tmp_path,
        min_test_rows=36,
    )

    assert config["optimization"]["enabled"] is False
    assert config["training"]["evaluate_test_split"] is True
    assert config["walk_forward"]["selection_uses_test"] is False
    assert config["training"]["learning_rate"] == 0.0002
    assert config["asset_policy"]["continuous_action_threshold"] == 0.2
    assert config["data"]["validation_end"] == config["data"]["test_start"]


def test_aggregate_weekly_results_compounds_return_and_penalizes_drawdown() -> None:
    result = aggregate_weekly_results(
        [
            {
                "week_start": "2023-01-02T00:00:00",
                "total_return": 0.10,
                "risk_adjusted_total_return": 0.08,
                "max_drawdown_fraction": 0.02,
                "trades_total": 5,
            },
            {
                "week_start": "2023-01-09T00:00:00",
                "total_return": -0.05,
                "risk_adjusted_total_return": -0.08,
                "max_drawdown_fraction": 0.03,
                "trades_total": 4,
            },
        ],
        expected_weeks=2,
    )

    assert result["annual_return"] == pytest.approx(0.045)
    assert result["annual_max_drawdown_fraction"] == pytest.approx(0.05)
    assert result["annual_rap"] == pytest.approx(-0.005)
    assert result["trades_total"] == 9


def test_aggregate_weekly_results_uses_concatenated_equity_traces(tmp_path) -> None:
    first_trace = tmp_path / "first.csv"
    second_trace = tmp_path / "second.csv"
    for path, equities in ((first_trace, [10000.0, 11000.0]), (second_trace, [10000.0, 9500.0])):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["equity"])
            writer.writeheader()
            writer.writerows({"equity": value} for value in equities)

    result = aggregate_weekly_results(
        [
            {
                "week_start": "2023-01-02T00:00:00",
                "total_return": 0.10,
                "risk_adjusted_total_return": 0.08,
                "max_drawdown_fraction": 0.02,
                "trades_total": 5,
                "final_equity": 11000.0,
                "return_trace_file": str(first_trace),
            },
            {
                "week_start": "2023-01-09T00:00:00",
                "total_return": -0.05,
                "risk_adjusted_total_return": -0.08,
                "max_drawdown_fraction": 0.03,
                "trades_total": 4,
                "final_equity": 9500.0,
                "return_trace_file": str(second_trace),
            },
        ],
        expected_weeks=2,
    )

    assert result["annual_max_drawdown_fraction"] == pytest.approx(0.05)
    assert result["annual_drawdown_method"] == "observed_concatenated_intraperiod_equity_trace"
    assert result["annual_drawdown_trace_observations"] == 4
