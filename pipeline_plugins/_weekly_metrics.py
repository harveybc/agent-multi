"""Canonical weekly/annual return and RAP metrics from an equity trace."""
from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Mapping

import pandas as pd


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0.0:
            worst = max(worst, (peak - value) / peak)
    return worst


def canonical_weekly_metrics_from_trace(
    rows: list[Mapping[str, Any]],
    *,
    initial_cash: float,
    risk_penalty_lambda: float,
    metric_schema: str = "trading.execution_robust.v1",
) -> dict[str, Any]:
    """Calculate comparable weekly and annual metrics from one equity trace."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [row.get("timestamp") for row in rows],
                utc=True,
                errors="coerce",
            ),
            "equity": pd.to_numeric(
                pd.Series([row.get("equity") for row in rows]),
                errors="coerce",
            ),
        }
    ).dropna()
    frame = frame.sort_values("timestamp").drop_duplicates(
        "timestamp", keep="last"
    )
    initial = float(initial_cash)
    if initial <= 0.0 or frame.empty:
        raise ValueError("weekly metrics require positive cash and a finite trace")

    prior = initial
    weekly_rows: list[dict[str, Any]] = []
    periods = frame["timestamp"].dt.tz_localize(None).dt.to_period("W-SUN")
    for period, group in frame.groupby(periods, sort=True):
        path = [prior, *group["equity"].astype(float).tolist()]
        end = path[-1]
        weekly_return = end / prior - 1.0 if prior else 0.0
        drawdown = _max_drawdown(path)
        weekly_rows.append(
            {
                "week": str(period),
                "return_fraction": weekly_return,
                "drawdown_fraction": drawdown,
                "rap_fraction": (
                    weekly_return - float(risk_penalty_lambda) * drawdown
                ),
            }
        )
        prior = end
    if not weekly_rows:
        raise ValueError("weekly metrics require at least one observed week")

    elapsed_days = max(
        (
            frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]
        ).total_seconds()
        / 86_400.0,
        1.0 / 24.0,
    )
    final = float(frame["equity"].iloc[-1])
    total_return = final / initial - 1.0
    annualized_return = None
    if final > 0.0 and total_return > -1.0:
        annualized_return = (1.0 + total_return) ** (
            365.25 / elapsed_days
        ) - 1.0
    if annualized_return is None or not math.isfinite(annualized_return):
        raise ValueError("trace does not produce a finite annualized return")

    mean_weekly_return = fmean(
        row["return_fraction"] for row in weekly_rows
    )
    mean_weekly_drawdown = fmean(
        row["drawdown_fraction"] for row in weekly_rows
    )
    mean_weekly_rap = fmean(row["rap_fraction"] for row in weekly_rows)
    return {
        "metric_schema": metric_schema,
        "total_return": total_return,
        "mean_weekly_return": mean_weekly_return,
        "annualized_return": annualized_return,
        "annual_return": 52.0 * mean_weekly_return,
        "annual_return_method": "weekly_arithmetic_mean_x_52",
        "mean_weekly_drawdown": mean_weekly_drawdown,
        "max_drawdown_fraction": _max_drawdown(
            [initial, *frame["equity"].astype(float).tolist()]
        ),
        "mean_weekly_rap": mean_weekly_rap,
        "annual_rap": 52.0 * mean_weekly_rap,
        "annual_rap_method": "weekly_arithmetic_mean_x_52",
        "evaluation_weeks": len(weekly_rows),
        "evaluation_days": elapsed_days,
        "weekly_rows": weekly_rows,
    }
