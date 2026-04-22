"""
random_agent.py — uniform random action baseline (sanity check).
"""
from __future__ import annotations

import random
from typing import Any, Dict


class Plugin:
    plugin_params: Dict[str, Any] = {
        "total_timesteps": 0,
        "train_seed": 0,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        self._rng = random.Random(0)
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update({k: v for k, v in kwargs.items() if k in self.params})
        self._rng = random.Random(int(self.params.get("train_seed", 0)))

    def build(self, env, config: Dict[str, Any]):
        return {"env": env, "action_space": env.action_space}

    def train(self, model, config: Dict[str, Any]):
        return model

    def predict(self, model, obs, deterministic: bool = True):
        return int(model["action_space"].sample())

    def save(self, model, path: str) -> None:
        # Nothing to persist; write a stub file for consistency.
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("random_agent\n")

    def load(self, path: str, env):
        return {"env": env, "action_space": env.action_space}

    def fitness(self, summary: Dict[str, Any], config: Dict[str, Any]) -> float:
        return float(summary.get("total_return") or 0.0)
