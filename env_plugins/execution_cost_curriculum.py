"""Gym wrapper applying a visible, deterministic execution-cost curriculum."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym

from pipeline_plugins._execution_curriculum import ExecutionCostCurriculum


class ExecutionCostCurriculumWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        curriculum: ExecutionCostCurriculum,
        seed: int,
        fixed_scenario_id: str | None = None,
    ):
        super().__init__(env)
        self.curriculum = curriculum
        self.curriculum_seed = int(seed)
        self.fixed_scenario_id = fixed_scenario_id
        self.training_progress = 0.0
        self.episode_index = 0
        self.last_selection = None

    def set_training_progress(self, progress: float) -> None:
        value = float(progress)
        if not 0.0 <= value <= 1.0:
            raise ValueError("training progress must be in [0, 1]")
        self.training_progress = value

    def reset(self, **kwargs):
        selection = self.curriculum.select(
            seed=self.curriculum_seed,
            training_progress=self.training_progress,
            episode_index=self.episode_index,
            scenario_id=self.fixed_scenario_id,
        )
        setter = getattr(self.env.unwrapped, "set_execution_cost_context", None)
        if not callable(setter):
            raise RuntimeError(
                "gym-fx environment does not implement set_execution_cost_context"
            )
        setter(
            observable_names=selection.observable_names,
            observable_vector=selection.observable_vector,
            cost_patch=selection.cost_patch,
            metadata={
                "execution_cost_curriculum_contract": selection.contract_version,
                "execution_cost_curriculum_id": selection.curriculum_id,
                "execution_cost_curriculum_fingerprint": (
                    selection.contract_fingerprint
                ),
                "execution_cost_phase": selection.phase_name,
                "scenario_id": selection.scenario_id,
                "training_progress": selection.training_progress,
                "episode_index": selection.episode_index,
            },
        )
        self.last_selection = selection
        self.episode_index += 1
        return self.env.reset(**kwargs)


def load_curriculum(
    value: str | Path | Mapping[str, Any],
    *,
    base_dir: Path,
) -> ExecutionCostCurriculum:
    if isinstance(value, Mapping):
        return ExecutionCostCurriculum.from_mapping(value)
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return ExecutionCostCurriculum.from_json_file(str(path))
