"""Always-flat deterministic baseline for Stage B evidence runs."""
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
        return {"env": env}

    def train(self, model, config: Dict[str, Any]):
        return model

    def predict(self, model, obs, deterministic: bool = True):
        return 0

    def save(self, model, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("no_trade_agent\n")

    def load(self, path: str, env):
        return {"env": env}

    def fitness(self, summary: Dict[str, Any], config: Dict[str, Any]) -> float:
        return float(summary.get("total_return") or 0.0)
