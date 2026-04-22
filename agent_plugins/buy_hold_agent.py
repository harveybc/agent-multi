"""
buy_hold_agent.py — long on first step, then hold.

Simple baseline to verify that env rewards + metrics line up with gym-fx's
buy-hold smoke test.
"""
from __future__ import annotations

from typing import Any, Dict


class Plugin:
    plugin_params: Dict[str, Any] = {"total_timesteps": 0}

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update({k: v for k, v in kwargs.items() if k in self.params})

    def build(self, env, config: Dict[str, Any]):
        return {"env": env, "step": 0}

    def train(self, model, config: Dict[str, Any]):
        return model

    def predict(self, model, obs, deterministic: bool = True):
        step = model["step"]
        model["step"] = step + 1
        return 1 if step == 0 else 0

    def save(self, model, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("buy_hold_agent\n")

    def load(self, path: str, env):
        return {"env": env, "step": 0}

    def fitness(self, summary: Dict[str, Any], config: Dict[str, Any]) -> float:
        return float(summary.get("total_return") or 0.0)
