from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces

from env_plugins.execution_cost_curriculum import (
    ExecutionCostCurriculumWrapper,
)
from pipeline_plugins._execution_curriculum import ExecutionCostCurriculum


def _curriculum() -> ExecutionCostCurriculum:
    scenarios = [
        {
            "scenario_id": name,
            "commission_fraction_per_side": commission,
            "full_spread_rate": spread,
            "slippage_bps_per_side": slippage,
        }
        for name, commission, spread, slippage in (
            ("easy", 0.00001, 0.00002, 0.1),
            ("nominal_a", 0.0001, 0.0002, 1.0),
            ("nominal_b", 0.0002, 0.0004, 2.0),
            ("stress", 0.0005, 0.001, 5.0),
        )
    ]
    return ExecutionCostCurriculum.from_mapping(
        {
            "contract_version": "execution_cost_curriculum.v1",
            "curriculum_id": "wrapper_test",
            "normalization_bounds": {
                "commission_fraction_per_side": 0.001,
                "full_spread_rate": 0.002,
                "slippage_bps_per_side": 10.0,
            },
            "scenarios": scenarios,
            "phases": [
                {
                    "name": "easy_nonzero",
                    "start_progress": 0.0,
                    "end_progress": 0.3,
                    "scenario_weights": [
                        {"scenario_id": "easy", "weight": 1.0}
                    ],
                },
                {
                    "name": "nominal_randomized",
                    "start_progress": 0.3,
                    "end_progress": 0.8,
                    "scenario_weights": [
                        {"scenario_id": "nominal_a", "weight": 1.0},
                        {"scenario_id": "nominal_b", "weight": 1.0},
                    ],
                },
                {
                    "name": "stress",
                    "start_progress": 0.8,
                    "end_progress": 1.0,
                    "scenario_weights": [
                        {"scenario_id": "stress", "weight": 1.0}
                    ],
                },
            ],
        }
    )


class _Env(gym.Env):
    observation_space = spaces.Box(0.0, 1.0, shape=(1,))
    action_space = spaces.Discrete(1)

    def __init__(self):
        self.contexts = []

    def set_execution_cost_context(self, **kwargs):
        self.contexts.append(kwargs)

    def reset(self, *, seed=None, options=None):
        return [0.0], {}

    def step(self, action):
        return [0.0], 0.0, True, False, {}


def test_wrapper_applies_visible_scenario_before_each_reset() -> None:
    base = _Env()
    wrapper = ExecutionCostCurriculumWrapper(
        base,
        curriculum=_curriculum(),
        seed=7,
    )
    wrapper.set_training_progress(0.9)
    wrapper.reset()

    assert wrapper.last_selection.scenario_id == "stress"
    assert base.contexts[-1]["metadata"]["episode_index"] == 0
    assert len(base.contexts[-1]["observable_vector"]) == 5


def test_wrapper_can_pin_immutable_validation_scenario() -> None:
    base = _Env()
    wrapper = ExecutionCostCurriculumWrapper(
        base,
        curriculum=_curriculum(),
        seed=7,
        fixed_scenario_id="nominal_b",
    )
    wrapper.reset()
    wrapper.set_training_progress(1.0)
    wrapper.reset()

    assert [context["metadata"]["scenario_id"] for context in base.contexts] == [
        "nominal_b",
        "nominal_b",
    ]
