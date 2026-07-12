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
    progress = []
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
                "on_between_candidates": lambda *args: progress.append(args),
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
    assert progress[0][2]["fitness"] is None
    assert progress[-1][2]["fitness"] is not None
    assert candidates[0]["stage"] == 1
    assert candidates[0]["total_eval"] == 1


def test_staged_optimizer_freezes_parameters_and_writes_resume(tmp_path: Path) -> None:
    calls = []
    stages = []

    class StagedAgent(_Agent):
        def hparam_schema(self):
            return [("alpha", 0.0, 1.0, "float"), ("beta", 0.0, 1.0, "float")]

    class StagedPipeline(_Pipeline):
        def run_pipeline(self, *, config, env_plugin, agent_plugin, mode):
            calls.append((float(config["alpha"]), float(config["beta"])))
            Path(config["save_model"]).write_bytes(
                f"{config['alpha']:.8f}:{config['beta']:.8f}".encode()
            )
            return {"total_return": 10.0 * config["alpha"] + config["beta"]}

    config = {
        "alpha": 0.2,
        "beta": 0.3,
        "ga_population": 2,
        "ga_seed": 3,
        "ga_cxpb": 0.0,
        "ga_mutpb": 0.0,
        "optimization_stages": [
            {"name": "alpha_stage", "params": ["alpha"], "generations": 0},
            {"name": "beta_stage", "params": ["beta"], "generations": 0},
        ],
        "optimization_capture_model_artifact": True,
        "optimization_require_model_artifact": True,
        "optimization_resume_file": str(tmp_path / "resume.json"),
        "optimization_candidate_history": str(tmp_path / "history.csv"),
        "optimization_statistics": str(tmp_path / "stats.json"),
        "optimization_parameters_file": str(tmp_path / "params.json"),
        "optimization_champion_model_file": str(tmp_path / "champion.zip"),
        "optimization_callbacks": {
            "on_new_champion": lambda *args: stages.append(args[4]["stage_name"]),
        },
        "quiet_mode": True,
    }

    result = Plugin().optimize(
        env_plugin=object(), agent_plugin=StagedAgent(),
        pipeline_plugin=StagedPipeline(), config=config,
    )

    assert len(calls) == 3
    assert calls[0][1] == pytest.approx(0.3)
    assert calls[1][1] == pytest.approx(0.3)
    assert calls[2][0] == pytest.approx(result["alpha"])
    assert stages[0] == "alpha_stage"
    assert (tmp_path / "history.csv").read_text().count("\n") == 4
    assert (tmp_path / "resume.json").is_file()
    assert (tmp_path / "stats.json").is_file()
    assert (tmp_path / "params.json").is_file()
    assert (tmp_path / "champion.zip").is_file()

    resumed = Plugin().optimize(
        env_plugin=object(), agent_plugin=StagedAgent(),
        pipeline_plugin=StagedPipeline(),
        config={
            **config,
            "optimization_resume": True,
            "canonical_config_hash": "sha256:resume-toggle-changes-canonical-hash",
        },
    )
    assert base64.b64decode(resumed["_best_model_b64"])
    assert len(calls) == 3


def test_declared_bounds_are_executable_and_cannot_exceed_agent_contract() -> None:
    schema = [("alpha", 0.0, 1.0, "float"), ("count", 1, 8, "int")]
    effective = Plugin._effective_schema(
        schema,
        {"hyperparameter_bounds": {"alpha": [0.2, 0.4], "count": [2, 5]}},
    )
    assert effective == [
        ("alpha", 0.2, 0.4, "float"),
        ("count", 2.0, 5.0, "int"),
    ]

    with pytest.raises(ValueError, match="exceeds agent safety bounds"):
        Plugin._effective_schema(
            schema,
            {"hyperparameter_bounds": {"alpha": [-1.0, 0.4]}},
        )


def test_all_failed_generation_aborts_without_publishing_champion() -> None:
    class FailingPipeline(_Pipeline):
        def run_pipeline(self, **kwargs):
            raise RuntimeError("candidate exploded")

    champions = []
    with pytest.raises(RuntimeError, match="all candidates failed"):
        Plugin().optimize(
            env_plugin=object(),
            agent_plugin=_Agent(),
            pipeline_plugin=FailingPipeline(),
            config={
                "ga_population": 2,
                "ga_generations": 0,
                "optimization_callbacks": {
                    "on_new_champion": lambda *args: champions.append(args),
                },
                "quiet_mode": True,
            },
        )
    assert champions == []
