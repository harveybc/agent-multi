from __future__ import annotations

import base64
from pathlib import Path

import pytest

from optimizer_plugins.default_optimizer import Plugin


class _Agent:
    def hparam_schema(self):
        return [("score", 0.0, 1.0, "float")]

    def set_params(self, **kwargs):
        self.params = kwargs

    def fitness(self, summary, config):
        return float(summary["total_return"])


class _Pipeline:
    def run_pipeline(self, *, config, env_plugin, agent_plugin, mode):
        model_path = config.get("save_model")
        if model_path:
            Path(model_path).write_bytes(b"candidate-model")
        return {
            "total_return": float(config["score"]),
            "max_drawdown_pct": 5.0,
            "metric_schema": "trading.metrics.v1",
        }


def test_optimizer_preserves_metric_vector_and_champion_artifact() -> None:
    candidates = []
    champions = []
    result = Plugin().optimize(
        env_plugin=object(),
        agent_plugin=_Agent(),
        pipeline_plugin=_Pipeline(),
        config={
            "ga_population": 2,
            "ga_generations": 0,
            "ga_seed": 1,
            "initial_candidate_params": {"score": 0.99},
            "optimization_capture_model_artifact": True,
            "optimization_require_model_artifact": True,
            "optimization_callbacks": {
                "on_candidate_evaluated": candidates.append,
                "on_new_champion": lambda *args: champions.append(args),
            },
            "quiet_mode": True,
        },
    )

    assert result["_best_fitness"] == pytest.approx(0.99)
    assert result["_best_metrics"]["total_return"] == pytest.approx(0.99)
    assert result["_best_metrics"]["max_drawdown_pct"] == pytest.approx(5.0)
    assert base64.b64decode(result["_best_model_b64"]) == b"candidate-model"
    assert candidates[0]["metrics"]["metric_schema"] == "trading.metrics.v1"
    assert "_model_b64" not in candidates[0]["metrics"]
    assert champions[0][2]["_model_b64"]
