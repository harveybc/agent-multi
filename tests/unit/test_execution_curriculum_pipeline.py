from __future__ import annotations

import math

import pytest

from pipeline_plugins.rl_pipeline_with_execution_curriculum import (
    PipelinePlugin,
    _compact_scenario_summary,
    canonical_weekly_metrics_from_trace,
)
from pipeline_plugins.rl_pipeline_with_validation import _selection_value


def test_trace_metrics_preserve_weekly_and_annual_units() -> None:
    rows = []
    equity = 100.0
    for day in range(15):
        equity *= 1.001
        rows.append(
            {
                "timestamp": f"2024-01-{day + 1:02d}T00:00:00Z",
                "equity": equity,
            }
        )

    metrics = canonical_weekly_metrics_from_trace(
        rows,
        initial_cash=100.0,
        risk_penalty_lambda=1.0,
    )

    assert metrics["evaluation_weeks"] == 3
    assert math.isfinite(metrics["annualized_return"])
    assert metrics["annual_return"] == metrics["annualized_return"]
    assert metrics["annual_rap"] == pytest.approx(
        52.0 * metrics["mean_weekly_rap"]
    )
    assert metrics["metric_schema"] == "trading.execution_robust.v1"


def test_robust_weekly_fitness_is_a_valid_l1_selection_metric() -> None:
    assert _selection_value(
        {"robust_weekly_rap_fitness": 0.0025},
        selection_metric="robust_weekly_rap_fitness",
        risk_lambda=1.0,
    ) == pytest.approx(0.0025)


def test_robust_selection_fails_closed_when_missing() -> None:
    with pytest.raises(ValueError, match="missing finite"):
        _selection_value(
            {},
            selection_metric="robust_weekly_rap_fitness",
            risk_lambda=1.0,
        )


def test_compact_scenario_summary_excludes_weekly_trace_rows() -> None:
    compact = _compact_scenario_summary(
        "nominal_reference",
        {
            "metric_schema": "trading.execution_robust.v1",
            "mean_weekly_return": 0.01,
            "mean_weekly_rap": 0.008,
            "weekly_rows": [{"week": "2024-01-01/2024-01-07"}],
            "large_native_diagnostics": {"orders": [1, 2, 3]},
        },
    )

    assert compact == {
        "metric_schema": "trading.execution_robust.v1",
        "cost_scenario_id": "nominal_reference",
        "mean_weekly_return": 0.01,
        "mean_weekly_rap": 0.008,
    }


def test_final_result_surfaces_robust_validation_fitness() -> None:
    class StubPipeline(PipelinePlugin):
        def _eval_on_split(self, *args, **kwargs):
            return {
                "total_return": 0.10,
                "mean_weekly_return": 0.002,
                "annualized_return": 0.11,
                "annual_return": 0.11,
                "mean_weekly_rap": 0.0015,
                "annual_rap": 0.078,
                "robust_weekly_rap_fitness": 0.0012,
                "worst_scenario_weekly_rap": 0.0008,
                "lower_tail_cvar_weekly_rap": 0.0009,
                "scenario_weekly_rap_dispersion": 0.0002,
                "max_drawdown_fraction": 0.02,
                "trades_total": 5,
                "final_equity": 11_000.0,
            }

    result = StubPipeline()._final_eval(
        agent_plugin=None,
        model=None,
        train_env=None,
        env_plugin_name="unused",
        paths={"train": "a", "train_tail": "b", "val": "c", "test": "d"},
        config={
            "selection_metric": "robust_weekly_rap_fitness",
            "risk_penalty_lambda": 1.0,
            "evaluate_test_split": False,
            "write_results_sidecar": False,
            "eval_seed": 1,
        },
        agent_plugin_for_wrap=None,
    )

    assert result["robust_weekly_rap_fitness"] == pytest.approx(0.0012)
    assert result["mean_weekly_return"] == pytest.approx(0.002)
    assert result["annual_return"] == pytest.approx(0.11)
    assert result["mean_weekly_rap"] == pytest.approx(0.0015)
    assert result["annual_rap"] == pytest.approx(0.078)
