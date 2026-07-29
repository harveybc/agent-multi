"""Validation pipeline with visible cost curriculum and robust weekly fitness."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

from env_plugins.execution_cost_curriculum import load_curriculum
from pipeline_plugins._execution_curriculum import (
    RobustFitnessConfig,
    aggregate_robust_weekly_fitness,
)
from pipeline_plugins._weekly_metrics import (
    canonical_weekly_metrics_from_trace,
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
