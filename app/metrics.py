"""Configurable optimization-objective selector shared by local and DOIN runs.

The default path preserves each agent plugin's historical ``fitness`` method.
An experiment opts into a named trading metric explicitly, so changing the
objective cannot silently alter an existing local experiment. Raw simulator
metrics belong to the selected ``gym-fx`` ``metrics.plugins`` implementation;
this module only maps its summary to the scalar the local optimizer reports.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def compute_optimization_fitness(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    agent_plugin: Any,
) -> float:
    metric = config.get("optimization_metric") or config.get("metric_type")
    if not metric or str(metric).lower() in {"agent_fitness", "plugin_fitness"}:
        return float(agent_plugin.fitness(dict(summary), dict(config)))

    name = str(metric).strip().lower()
    if name in {"total_return", "return"}:
        return _number(summary, "total_return")
    if name in {"mean_weekly_rap", "weekly_rap", "rap"}:
        return _number(summary, "mean_weekly_rap", missing=config)
    if name in {"annual_rap", "annualized_rap"}:
        return _number(summary, "annual_rap", missing=config)
    if name in {"train_validation_l1_score", "l1_score"}:
        return _number(summary, "train_validation_l1_score", missing=config)
    if name in {
        "train_validation_risk_adjusted_composite_score",
        "risk_adjusted_composite_score",
    }:
        return _number(
            summary,
            "train_validation_risk_adjusted_composite_score",
            missing=config,
        )
    if name in {"risk_adjusted_return", "risk_adjusted_total_return"}:
        total_return = _number(summary, "total_return")
        drawdown = _drawdown_fraction(summary)
        return total_return - float(config.get("risk_lambda", 1.0)) * drawdown

    custom = config.get("optimization_metric_callable")
    if callable(custom):
        return float(custom(dict(summary), dict(config)))
    raise ValueError(f"unknown optimization_metric={metric!r}")


def _number(
    summary: Mapping[str, Any],
    key: str,
    *,
    missing: Mapping[str, Any] | None = None,
) -> float:
    value = summary.get(key)
    if value is None or not math.isfinite(float(value)):
        if missing and str(missing.get("metric_missing_policy", "fail")) == "fail":
            raise ValueError(f"summary does not contain finite metric {key!r}")
        return float(missing.get("metric_missing_value", -1e9)) if missing else 0.0
    return float(value)


def _drawdown_fraction(summary: Mapping[str, Any]) -> float:
    if summary.get("max_drawdown_fraction") is not None:
        return max(0.0, float(summary["max_drawdown_fraction"]))
    if summary.get("max_drawdown_pct") is not None:
        return max(0.0, float(summary["max_drawdown_pct"]) / 100.0)
    if summary.get("max_drawdown") is not None:
        return abs(float(summary["max_drawdown"]))
    return 0.0
