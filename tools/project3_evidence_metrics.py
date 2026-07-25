#!/usr/bin/env python3
"""Canonical Project 3 metric calculations and OLAP row materialization.

All return and risk values are stored as decimal fractions. Presentation code
may multiply values by 100, but it must retain the explicit unit and horizon
fields emitted here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


METRIC_SCHEMA = "project3.evidence.metrics.v2"
WEEKS_PER_YEAR = 52.0
DAYS_PER_YEAR = 365.2425


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: float | None
    unit: str
    horizon: str
    aggregation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_schema": METRIC_SCHEMA,
            "metric_name": self.name,
            "value": self.value,
            "unit": self.unit,
            "horizon": self.horizon,
            "aggregation": self.aggregation,
        }


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _max_drawdown(equity: pd.Series) -> float:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    peaks = clean.cummax()
    drawdown = (peaks - clean) / peaks.replace(0.0, np.nan)
    return float(drawdown.replace([np.inf, -np.inf], np.nan).fillna(0.0).max())


def _weekly_slices(
    timestamps: Iterable[Any],
    equity: Iterable[float],
) -> list[dict[str, float | str]]:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(list(timestamps), utc=True, errors="coerce"),
            "equity": pd.to_numeric(pd.Series(list(equity)), errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return []
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    initial_equity = float(frame["equity"].iloc[0])
    prior_equity = initial_equity
    rows: list[dict[str, float | str]] = []
    week_period = frame["timestamp"].dt.tz_localize(None).dt.to_period("W-SUN")
    for period, group in frame.groupby(week_period, sort=True):
        values = group["equity"].astype(float)
        start_equity = prior_equity
        end_equity = float(values.iloc[-1])
        weekly_path = pd.concat(
            [pd.Series([start_equity], dtype=float), values.reset_index(drop=True)],
            ignore_index=True,
        )
        weekly_return = end_equity / start_equity - 1.0 if start_equity else 0.0
        rows.append(
            {
                "week": str(period),
                "return_fraction": float(weekly_return),
                "drawdown_fraction": _max_drawdown(weekly_path),
            }
        )
        prior_equity = end_equity
    return rows


def canonical_trading_metrics(
    *,
    timestamps: Iterable[Any],
    equity: Iterable[float],
    risk_penalty_lambda: float = 1.0,
) -> dict[str, Any]:
    """Return comparable weekly and annual metrics from an equity curve."""
    ts = pd.to_datetime(list(timestamps), utc=True, errors="coerce")
    eq = pd.to_numeric(pd.Series(list(equity)), errors="coerce")
    frame = pd.DataFrame({"timestamp": ts, "equity": eq}).dropna()
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if len(frame) < 2:
        raise ValueError("at least two finite equity observations are required")

    initial = float(frame["equity"].iloc[0])
    final = float(frame["equity"].iloc[-1])
    if initial <= 0.0:
        raise ValueError("initial equity must be positive")
    elapsed_days = max(
        (frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]).total_seconds() / 86400.0,
        1.0 / 24.0,
    )
    total_return = final / initial - 1.0
    annualized_return: float | None = None
    if final > 0.0 and total_return > -1.0:
        annualized_return = (1.0 + total_return) ** (DAYS_PER_YEAR / elapsed_days) - 1.0

    weekly = _weekly_slices(frame["timestamp"], frame["equity"])
    weekly_returns = np.asarray([row["return_fraction"] for row in weekly], dtype=float)
    weekly_drawdowns = np.asarray([row["drawdown_fraction"] for row in weekly], dtype=float)
    weekly_rap = weekly_returns - float(risk_penalty_lambda) * weekly_drawdowns
    mean_weekly_return = float(np.mean(weekly_returns)) if weekly_returns.size else None
    mean_weekly_drawdown = float(np.mean(weekly_drawdowns)) if weekly_drawdowns.size else None
    mean_weekly_rap = float(np.mean(weekly_rap)) if weekly_rap.size else None
    annual_rap = mean_weekly_rap * WEEKS_PER_YEAR if mean_weekly_rap is not None else None

    metrics = [
        MetricValue("total_return", total_return, "fraction", "evaluation_period", "compound"),
        MetricValue("mean_weekly_return", mean_weekly_return, "fraction", "week", "arithmetic_mean"),
        MetricValue("annualized_return", annualized_return, "fraction", "year", "compound"),
        MetricValue("max_drawdown", _max_drawdown(frame["equity"]), "fraction", "evaluation_period", "maximum"),
        MetricValue("mean_weekly_drawdown", mean_weekly_drawdown, "fraction", "week", "arithmetic_mean"),
        MetricValue("mean_weekly_rap", mean_weekly_rap, "fraction", "week", "arithmetic_mean"),
        MetricValue("annual_rap", annual_rap, "fraction", "year", "weekly_mean_x_52"),
        MetricValue("evaluation_weeks", float(len(weekly)), "count", "evaluation_period", "count"),
        MetricValue("evaluation_days", elapsed_days, "count", "evaluation_period", "elapsed"),
    ]
    return {
        "metric_schema": METRIC_SCHEMA,
        "risk_penalty_lambda": float(risk_penalty_lambda),
        "metrics": [metric.as_dict() for metric in metrics],
        "weekly_rows": weekly,
        "canonical": {metric.name: _finite(metric.value) for metric in metrics},
    }


def metric_rows_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("metric_rows") or []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        value = _finite(row.get("value"))
        normalized.append(
            {
                "metric_schema": str(row.get("metric_schema") or METRIC_SCHEMA),
                "metric_name": str(row["metric_name"]),
                "value": value,
                "unit": str(row.get("unit") or "dimensionless"),
                "horizon": str(row.get("horizon") or "unspecified"),
                "aggregation": str(row.get("aggregation") or "unspecified"),
                "split": str(row.get("split") or "unspecified"),
            }
        )
    return normalized


def display_percent(value: Any) -> str:
    number = _finite(value)
    return "N/A" if number is None else f"{number * 100.0:+.4f}%"
