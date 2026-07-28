"""Validation pipeline with visible cost curriculum and robust weekly fitness."""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Mapping

import pandas as pd

from env_plugins.execution_cost_curriculum import load_curriculum
from pipeline_plugins._execution_curriculum import (
    RobustFitnessConfig,
    aggregate_robust_weekly_fitness,
)
from pipeline_plugins.rl_pipeline_with_validation import (
    PipelinePlugin as ValidationPipeline,
)


ROBUST_METRIC_SCHEMA = "trading.execution_robust.v1"
SCENARIO_METRIC_KEYS = (
    "metric_schema",
    "cost_scenario_id",
    "total_return",
    "mean_weekly_return",
    "annualized_return",
    "annual_return",
    "mean_weekly_drawdown",
    "max_drawdown_fraction",
    "mean_weekly_rap",
    "annual_rap",
    "evaluation_weeks",
    "evaluation_days",
    "trades_total",
    "turnover",
    "cost_drag",
    "fill_ratio",
    "expiration_count",
)


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
) -> dict[str, Any]:
    """Calculate comparable weekly/annual metrics from one equity trace."""
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
        "metric_schema": ROBUST_METRIC_SCHEMA,
        "total_return": total_return,
        "mean_weekly_return": mean_weekly_return,
        "annualized_return": annualized_return,
        "annual_return": annualized_return,
        "mean_weekly_drawdown": mean_weekly_drawdown,
        "max_drawdown_fraction": _max_drawdown(
            [initial, *frame["equity"].astype(float).tolist()]
        ),
        "mean_weekly_rap": mean_weekly_rap,
        "annual_rap": 52.0 * mean_weekly_rap,
        "evaluation_weeks": len(weekly_rows),
        "evaluation_days": elapsed_days,
        "weekly_rows": weekly_rows,
    }


def _compact_scenario_summary(
    scenario_id: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep blockchain metrics small while traces retain weekly evidence."""
    compact = {
        key: summary[key]
        for key in SCENARIO_METRIC_KEYS
        if key in summary
    }
    compact["cost_scenario_id"] = scenario_id
    return compact


class PipelinePlugin(ValidationPipeline):
    plugin_params = {
        **ValidationPipeline.plugin_params,
        "execution_cost_curriculum": None,
        "execution_cost_curriculum_fingerprint": None,
        "robust_validation_scenarios": None,
        "robust_fitness_config": {
            "lower_tail_fraction": 0.25,
            "downside_penalty_weight": 1.0,
            "dispersion_penalty_weight": 0.5,
            "annualization_weeks": 52.0,
        },
    }

    plugin_debug_vars = [
        *ValidationPipeline.plugin_debug_vars,
        "execution_cost_curriculum",
        "execution_cost_curriculum_fingerprint",
        "robust_validation_scenarios",
        "robust_fitness_config",
    ]

    @staticmethod
    def _load_curriculum(config: Dict[str, Any]):
        value = config.get("execution_cost_curriculum")
        if not value:
            raise ValueError(
                "execution curriculum pipeline requires execution_cost_curriculum"
            )
        curriculum = load_curriculum(
            value,
            base_dir=Path(__file__).resolve().parents[1],
        )
        expected = config.get("execution_cost_curriculum_fingerprint")
        if expected and str(expected) != curriculum.fingerprint:
            raise ValueError(
                "execution cost curriculum fingerprint mismatch: "
                f"expected {expected}, got {curriculum.fingerprint}"
            )
        return curriculum

    def _eval_on_split(
        self,
        env_plugin_name: str,
        config: Dict[str, Any],
        csv_path: str,
        agent_plugin,
        model,
        seed: int,
        split_name: str,
    ) -> Dict[str, Any]:
        if config.get("execution_cost_fixed_scenario_id"):
            return super()._eval_on_split(
                env_plugin_name,
                config,
                csv_path,
                agent_plugin,
                model,
                seed,
                split_name,
            )

        curriculum = self._load_curriculum(config)
        requested = config.get("robust_validation_scenarios")
        scenario_ids = (
            [str(value) for value in requested]
            if isinstance(requested, list) and requested
            else [scenario.scenario_id for scenario in curriculum.scenarios]
        )
        known = {scenario.scenario_id for scenario in curriculum.scenarios}
        unknown = sorted(set(scenario_ids) - known)
        if unknown:
            raise ValueError(f"unknown robust validation scenarios: {unknown}")
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("robust validation scenarios must be unique")

        scenario_summaries: dict[str, dict[str, Any]] = {}
        compact_summaries: dict[str, dict[str, Any]] = {}
        for scenario_id in scenario_ids:
            scenario_config = deepcopy(config)
            scenario_config["execution_cost_fixed_scenario_id"] = scenario_id
            scenario_config["_retain_return_trace_rows"] = True
            trace_dir = scenario_config.get("return_trace_dir")
            if trace_dir:
                scenario_config["return_trace_dir"] = str(
                    Path(str(trace_dir)) / scenario_id
                )
            summary = super()._eval_on_split(
                env_plugin_name,
                scenario_config,
                csv_path,
                agent_plugin,
                model,
                seed,
                split_name,
            )
            trace_rows = summary.pop("_return_trace_rows", None)
            if not isinstance(trace_rows, list) or not trace_rows:
                raise ValueError(
                    f"{scenario_id} did not emit an equity return trace"
                )
            summary.update(
                canonical_weekly_metrics_from_trace(
                    trace_rows,
                    initial_cash=float(config.get("initial_cash", 10_000.0)),
                    risk_penalty_lambda=float(
                        config.get("risk_penalty_lambda", 1.0)
                    ),
                )
            )
            scenario_summaries[scenario_id] = summary
            compact_summaries[scenario_id] = _compact_scenario_summary(
                scenario_id,
                summary,
            )

        raw_fitness = config.get("robust_fitness_config") or {}
        if not isinstance(raw_fitness, Mapping):
            raise ValueError("robust_fitness_config must be an object")
        robust = aggregate_robust_weekly_fitness(
            scenario_summaries,
            config=RobustFitnessConfig(**dict(raw_fitness)),
        )
        representative = deepcopy(scenario_summaries[scenario_ids[0]])
        representative.update(robust)
        representative.update(
            {
                "metric_schema": ROBUST_METRIC_SCHEMA,
                "cost_scenario_order": scenario_ids,
                "cost_scenarios": compact_summaries,
                "total_return": fmean(
                    float(item["total_return"])
                    for item in scenario_summaries.values()
                ),
                "max_drawdown_fraction": max(
                    float(item["max_drawdown_fraction"])
                    for item in scenario_summaries.values()
                ),
                "trades_total": fmean(
                    float(item.get("trades_total") or 0.0)
                    for item in scenario_summaries.values()
                ),
                "execution_cost_curriculum_fingerprint": (
                    curriculum.fingerprint
                ),
            }
        )
        return representative
