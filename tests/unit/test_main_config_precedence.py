from __future__ import annotations

from app import main as main_module


class _PluginBase:
    plugin_params: dict = {}

    def __init__(self, config: dict) -> None:
        self.params: dict = {}

    def set_params(self, **config) -> None:
        self.params.update(config)


class _Environment(_PluginBase):
    plugin_params = {"total_timesteps": 999, "environment_default": "present"}


class _Agent(_PluginBase):
    plugin_params = {"total_timesteps": 888, "agent_default": "present"}


class _Pipeline(_PluginBase):
    plugin_params = {"total_timesteps": 777, "pipeline_default": "present"}

    def run_pipeline(self, *, config, env_plugin, agent_plugin, mode):
        return {
            "total_timesteps": config["total_timesteps"],
            "environment_default": config["environment_default"],
            "agent_default": config["agent_default"],
            "pipeline_default": config["pipeline_default"],
            "mode": mode,
        }


def test_loaded_values_win_over_plugin_defaults(monkeypatch) -> None:
    plugins = {
        "env.plugins": _Environment,
        "agent.plugins": _Agent,
        "pipeline.plugins": _Pipeline,
    }

    def fake_load_plugin(group: str, name: str):
        return plugins[group], None

    monkeypatch.setattr(main_module, "load_plugin", fake_load_plugin)
    config = {
        "env_plugin": "env",
        "agent_plugin": "agent",
        "pipeline_plugin": "pipeline",
        "optimizer_plugin": "unused",
        "use_optimizer": False,
        "load_model": None,
        "mode": "train",
        "quiet_mode": True,
        "total_timesteps": 123,
    }
    summary = main_module._run(config)
    assert summary["total_timesteps"] == 123
    assert summary["environment_default"] == "present"
    assert summary["agent_default"] == "present"
    assert summary["pipeline_default"] == "present"
    assert summary["mode"] == "train"


def test_optimizer_can_return_persisted_champion_without_retraining(
    monkeypatch,
    tmp_path,
) -> None:
    class Optimizer(_PluginBase):
        def optimize(self, **kwargs):
            return {
                "encoded_gene": 1,
                "_best_fitness": 0.25,
                "_best_model_b64": "large-model-payload",
                "_best_metrics": {"validation_score": 0.2},
            }

        def resolve_best_config(self, optimal, config):
            return {**config, "decoded_gene": "winner"}

    class Pipeline(_Pipeline):
        def run_pipeline(self, **kwargs):
            raise AssertionError("the persisted champion must not be retrained")

    plugins = {
        "env.plugins": _Environment,
        "agent.plugins": _Agent,
        "pipeline.plugins": Pipeline,
        "optimizer.plugins": Optimizer,
    }
    monkeypatch.setattr(
        main_module,
        "load_plugin",
        lambda group, name: (plugins[group], None),
    )
    output = tmp_path / "optimizer.json"
    summary = main_module._run(
        {
            "env_plugin": "env",
            "agent_plugin": "agent",
            "pipeline_plugin": "pipeline",
            "optimizer_plugin": "optimizer",
            "use_optimizer": True,
            "load_model": None,
            "mode": "train",
            "quiet_mode": True,
            "optimization_run_final_pipeline": False,
            "optimization_champion_model_file": str(tmp_path / "champion.zip"),
            "optimizer_output_file": str(output),
        }
    )

    assert summary["mode"] == "optimization"
    assert summary["validation_score"] == 0.2
    assert summary["best_fitness"] == 0.25
    assert summary["best_parameters"] == {"encoded_gene": 1}
    assert "_best_model_b64" not in output.read_text(encoding="utf-8")
