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
