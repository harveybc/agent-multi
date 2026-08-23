from __future__ import annotations

import numpy as np
import pytest


def _architecture():
    return {
        "feature_columns": ["return_1", "rsi", "atr"],
        "branches": [
            {
                "name": "returns",
                "features": ["return_1"],
                "plugin": "tcn_branch",
                "params": {"channels": [8], "kernel_size": 2},
            },
            {
                "name": "oscillators",
                "features": ["rsi"],
                "plugin": "gru_branch",
                "params": {"hidden_size": 6},
            },
            {
                "name": "volatility",
                "features": ["atr"],
                "plugin": "transformer_branch",
                "params": {"model_dim": 8, "num_heads": 2, "num_layers": 1},
            },
        ],
        "state_keys": ["position", "equity_norm"],
        "state_branch": {
            "plugin": "mlp_branch",
            "params": {"hidden_dims": [4], "output_dim": 4},
        },
        "fusion": {
            "plugin": "gated_fusion",
            "params": {"common_dim": 8, "output_dim": 12},
        },
    }


@pytest.fixture
def local_plugins(monkeypatch):
    import importlib
    import agent_plugins.grouped_features_extractor as grouped

    modules = {
        ("feature_branch.plugins", name): f"feature_branch_plugins.{name}"
        for name in ("mlp_branch", "gru_branch", "tcn_branch", "transformer_branch")
    }
    modules.update({
        ("feature_fusion.plugins", name): f"feature_fusion_plugins.{name}"
        for name in ("concat_fusion", "gated_fusion")
    })

    def load(group, name):
        cls = importlib.import_module(modules[(group, name)]).Plugin
        return cls, list(cls.plugin_params)

    monkeypatch.setattr(grouped, "load_plugin", load)


def _space():
    from gymnasium import spaces

    return spaces.Dict({
        "features": spaces.Box(-10, 10, shape=(8, 3), dtype=np.float32),
        "position": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        "equity_norm": spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
    })


def test_grouped_extractor_preserves_temporal_shape(local_plugins):
    import torch
    from agent_plugins.grouped_features_extractor import build_grouped_extractor_class

    extractor = build_grouped_extractor_class()(_space(), _architecture())
    observations = {
        "features": torch.randn(5, 8, 3),
        "position": torch.zeros(5, 1),
        "equity_norm": torch.zeros(5, 1),
    }
    assert extractor(observations).shape == (5, 12)
    assert extractor.features_dim == 12


def test_every_feature_must_have_exactly_one_branch(local_plugins):
    from agent_plugins.grouped_features_extractor import build_grouped_extractor_class

    architecture = _architecture()
    architecture["branches"][2]["features"] = ["rsi"]
    with pytest.raises(ValueError, match="multiple branches"):
        build_grouped_extractor_class()(_space(), architecture)


def test_unknown_nested_parameter_refuses(local_plugins):
    from agent_plugins.grouped_features_extractor import build_grouped_extractor_class

    architecture = _architecture()
    architecture["branches"][0]["params"]["kernal_size"] = 3
    with pytest.raises(ValueError, match="unknown"):
        build_grouped_extractor_class()(_space(), architecture)


def test_sac_builds_multi_input_actor_and_twin_critics(local_plugins):
    import torch
    from gymnasium import Env, spaces
    from agent_plugins.sac_agent import Plugin

    class TinyEnv(Env):
        observation_space = _space()
        action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            return {
                key: value.sample()
                for key, value in self.observation_space.spaces.items()
            }, {}

        def step(self, action):
            observation, _ = self.reset()
            return observation, 0.0, False, False, {}

    plugin = Plugin({
        "feature_extractor_plugin": "grouped_features_extractor",
        "feature_extractor_config": _architecture(),
        "buffer_size": 100,
        "learning_starts": 1,
        "batch_size": 2,
        "device": "cpu",
    })
    env = plugin.wrap_env(TinyEnv(), {})
    model = plugin.build(env, {})
    observation, _ = env.reset()
    action, _ = model.predict(observation)
    assert model.policy.__class__.__name__ == "MultiInputPolicy"
    assert model.policy.actor.features_extractor.features_dim == 12
    assert len(model.policy.critic.q_networks) == 2
    assert action.shape == (1,)
