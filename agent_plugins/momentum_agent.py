"""Simple deterministic momentum/reversal baselines for Stage B diagnostics."""
from __future__ import annotations

from typing import Any, Dict


def _last_delta(obs: Any, feature_index: int) -> float:
    try:
        import numpy as np

        arr = np.asarray(obs, dtype=float)
        if arr.ndim >= 2 and arr.shape[0] >= 2:
            idx = feature_index
            return float(arr[-1, idx] - arr[-2, idx])
        flat = arr.reshape(-1)
        if flat.size >= 2:
            return float(flat[-1] - flat[-2])
    except Exception:
        pass
    return 0.0


class Plugin:
    plugin_params: Dict[str, Any] = {
        "total_timesteps": 0,
        "momentum_threshold": 0.0,
        "momentum_feature_index": -1,
        "momentum_reversal": False,
    }

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
        delta = _last_delta(obs, int(self.params.get("momentum_feature_index", -1)))
        threshold = float(self.params.get("momentum_threshold") or 0.0)
        if abs(delta) <= threshold:
            action = 0
        elif delta > 0:
            action = 1
        else:
            action = 2
        if self.params.get("momentum_reversal"):
            if action == 1:
                return 2
            if action == 2:
                return 1
        return action

    def save(self, model, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("momentum_agent\n")

    def load(self, path: str, env):
        return {"env": env}

    def fitness(self, summary: Dict[str, Any], config: Dict[str, Any]) -> float:
        return float(summary.get("total_return") or 0.0)
