from __future__ import annotations

from typing import Any

import pytest

from optimizer_plugins.default_optimizer import Plugin


class _Agent:
    def hparam_schema(self):
        return [
            ("score", 0.0, 1.0, "float"),
            ("steps", 1.0, 8.0, "int"),
        ]

    def set_params(self, **kwargs: Any) -> None:
        self.params = kwargs

    def fitness(self, summary, config):
        return float(summary["total_return"])


class _Env:
    def close(self) -> None:
        return None


class _Pipeline:
    def run_pipeline(self, *, config, env_plugin, agent_plugin, mode):
        return {
            "total_return": float(config["score"]) + float(config["steps"]) / 100.0,
            "mode": mode,
        }


def _config() -> dict[str, Any]:
    return {
        "higher_is_better": True,
        "ga_population": 4,
        "ga_cxpb": 0.5,
        "ga_mutpb": 0.2,
        "ga_eval_timesteps": 1,
        "ga_fitness_split": "train",
        "optimization_patience": 2,
        "initial_candidate_params": {"score": 0.7, "steps": 3},
        "optimization_stages": [
            {"name": "score", "params": ["score"], "generations": 1, "patience": 2},
            {"name": "steps", "params": ["steps"], "generations": 2, "patience": 2},
        ],
    }


def _optimizer() -> Plugin:
    optimizer = Plugin()
    optimizer.setup_shared_mode(
        env_plugin=_Env(),
        agent_plugin=_Agent(),
        pipeline_plugin=_Pipeline(),
        config=_config(),
    )
    return optimizer


def test_shared_population_is_deterministic_and_json_serializable() -> None:
    first = _optimizer().create_shared_population(4, seed=71)
    second = _optimizer().create_shared_population(4, seed=71)

    assert first == second
    assert len(first["population"]) == 4
    assert first["stage_schedule"][0]["active_params"] == ["score"]
    assert all("fitness" not in genome for genome in first["population"])


def test_shared_candidate_uses_existing_local_evaluation_path() -> None:
    optimizer = _optimizer()
    state = optimizer.create_shared_population(4, seed=71)

    result = optimizer.evaluate_candidate(state["population"][0], generation=0)

    assert result["hyper_dict"] == {"score": pytest.approx(0.7), "steps": 3}
    assert result["fitness"] == pytest.approx(0.73)
    assert result["optimization_metric"] == "agent_fitness"


def test_shared_reproduction_advances_stages_deterministically() -> None:
    optimizer = _optimizer()
    state = optimizer.create_shared_population(4, seed=71)
    evaluated = []
    for genome in state["population"]:
        result = optimizer.evaluate_candidate(genome, generation=0)
        evaluated.append({
            "parameters": result["hyper_dict"],
            "fitness": result["fitness"],
        })

    first = optimizer.reproduce_shared(
        evaluated,
        generation=0,
        seed=101,
        innovation_tracker_data=state["innovation_tracker"],
        stage_schedule=state["stage_schedule"],
        param_defaults=state["param_defaults"],
        current_stage_idx=0,
        no_improve_count=0,
    )
    second = optimizer.reproduce_shared(
        evaluated,
        generation=0,
        seed=101,
        innovation_tracker_data=state["innovation_tracker"],
        stage_schedule=state["stage_schedule"],
        param_defaults=state["param_defaults"],
        current_stage_idx=0,
        no_improve_count=0,
    )

    assert first == second
    assert first["generation"] == 1
    assert first["stage_idx"] == 1
    assert first["stage_advanced"] is True
    assert all("fitness" not in genome for genome in first["population"])
