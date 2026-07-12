from __future__ import annotations

import pytest

from app.metrics import compute_optimization_fitness


class _Agent:
    def fitness(self, summary, config):
        return 123.0


def test_default_metric_preserves_agent_plugin_fitness() -> None:
    assert compute_optimization_fitness({"total_return": 0.2}, {}, _Agent()) == 123.0


def test_risk_adjusted_metric_uses_fractional_drawdown() -> None:
    value = compute_optimization_fitness(
        {"total_return": 0.20, "max_drawdown_pct": 10.0},
        {"optimization_metric": "risk_adjusted_return", "risk_lambda": 1.5},
        _Agent(),
    )
    assert value == pytest.approx(0.05)


def test_weekly_rap_metric_fails_closed_when_missing() -> None:
    with pytest.raises(ValueError, match="mean_weekly_rap"):
        compute_optimization_fitness(
            {"total_return": 0.2},
            {"optimization_metric": "mean_weekly_rap"},
            _Agent(),
        )


def test_gap_penalized_l1_metric_is_selectable() -> None:
    value = compute_optimization_fitness(
        {"train_validation_l1_score": 0.123},
        {
            "optimization_metric": "train_validation_l1_score",
            "metric_missing_policy": "fail",
        },
        _Agent(),
    )
    assert value == pytest.approx(0.123)
