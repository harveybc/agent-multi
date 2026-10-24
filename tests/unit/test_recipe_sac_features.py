import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces

from agent_plugins.sac_agent import Plugin
from agent_plugins.recipe_features import RecipeTemporalExtractor
from tests.unit.test_frozen_recipe_inputs import recipe


class ProbeEnv(gym.Env):
    observation_space = spaces.Dict({"features": spaces.Box(-np.inf, np.inf, (3, 2), dtype=np.float32)})
    action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"features": np.arange(6, dtype=np.float32).reshape(3, 2)}, {}

    def step(self, action):
        raise AssertionError("NO_TRAINING_OR_SIMULATION_IN_PROBE")


def test_sac_actor_critics_receive_exact_features_and_reload(tmp_path):
    cfg = {"observation_encoder": "shared_recipe", "shared_recipe": recipe(),
           "net_arch": [16], "device": "cpu", "buffer_size": 1, "train_seed": 7}
    plugin = Plugin(cfg)
    env = plugin.wrap_env(ProbeEnv(), cfg)
    model = plugin.build(env, cfg)
    obs, _ = env.reset()
    seen = []
    hook = model.actor.features_extractor.register_forward_pre_hook(
        lambda module, args: seen.append(args[0]["features"].detach().numpy().copy()))
    action = model.predict(obs, deterministic=True)[0]
    hook.remove()
    np.testing.assert_array_equal(seen[0][0], obs["features"])
    assert isinstance(model.actor.features_extractor, RecipeTemporalExtractor)
    assert isinstance(model.critic.features_extractor, RecipeTemporalExtractor)
    assert len(model.critic.q_networks) == 2
    assert model.actor.features_extractor is not model.critic.features_extractor
    path = tmp_path / "sac.zip"
    model.save(path)
    restored = plugin.load(path, env)
    np.testing.assert_array_equal(action, restored.predict(obs, deterministic=True)[0])
    assert model.num_timesteps == 0


def test_feedback_and_expert_pretraining_refused():
    bad = spaces.Dict({"features": ProbeEnv.observation_space["features"], "position": spaces.Box(-1., 1., (1,))})
    with pytest.raises(ValueError, match="FEATURES_ONLY"):
        RecipeTemporalExtractor(bad, recipe())
    cfg = {"observation_encoder": "shared_recipe", "shared_recipe": recipe(), "oracle_behavior_pretrain_enabled": True}
    with pytest.raises(ValueError, match="NO_EXPERT"):
        Plugin(cfg).build(ProbeEnv(), cfg)
