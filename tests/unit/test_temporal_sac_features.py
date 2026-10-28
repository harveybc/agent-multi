import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces

from agent_plugins.sac_agent import Plugin
from agent_plugins.temporal_features import PerFeatureTemporalExtractor


class ToyEnv(gym.Env):
    observation_space = spaces.Dict({
        "features": spaces.Box(-10, 10, (13, 3), dtype=np.float32),
        "position": spaces.Box(-1, 1, (1,), dtype=np.float32),
    })
    action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {k: np.zeros(s.shape, dtype=np.float32)
                for k, s in self.observation_space.spaces.items()}, {}

    def step(self, action):
        obs, _ = self.reset()
        return obs, 0.0, False, False, {}


def test_temporal_shape_prefix_and_origin():
    torch.manual_seed(7)
    extractor = PerFeatureTemporalExtractor(ToyEnv.observation_space)
    # Positive deterministic weights make the origin-dependence control decisive.
    for p in extractor.parameters():
        torch.nn.init.constant_(p, 0.1)
    x = torch.ones(2, 13, 3)
    y = extractor.temporal_forward(x)
    changed = x.clone()
    changed[:, 8:, :] += 10
    altered = extractor.temporal_forward(changed)
    assert y.shape == (2, 8, 13)
    assert torch.equal(y[:, :, :8], altered[:, :, :8])
    changed = x.clone()
    changed[:, -1, :] += 10
    altered = extractor.temporal_forward(changed)
    assert torch.equal(y[:, :, :-1], altered[:, :, :-1])
    assert not torch.equal(y[:, :, -1], altered[:, :, -1])
    assert not any(isinstance(m, (torch.nn.Linear, torch.nn.BatchNorm1d))
                   for m in extractor.modules())
    assert extractor({"features": x, "position": torch.zeros(2, 1)}).shape == (2, 105)


@pytest.mark.parametrize("encoder", ["flat", "per_feature_conv1d"])
def test_real_sb3_build_gradient_save_reload(tmp_path, encoder):
    cfg = {"observation_encoder": encoder, "net_arch": [8], "device": "cpu",
           "buffer_size": 32, "batch_size": 2, "learning_starts": 2,
           "train_seed": 7}
    plugin = Plugin(cfg)
    env = plugin.wrap_env(ToyEnv(), cfg)
    model = plugin.build(env, cfg)
    obs, _ = env.reset(seed=7)
    before = model.predict(obs, deterministic=True)[0]
    path = tmp_path / "policy.zip"
    model.save(path)
    restored = plugin.load(path, env)
    np.testing.assert_array_equal(before, restored.predict(obs, deterministic=True)[0])
    # Real SAC gradient path on synthetic observations only; no financial fit.
    model.learn(total_timesteps=4)
    assert model._n_updates == 2
    assert np.isfinite(model.predict(obs, deterministic=True)[0]).all()
    env.close()


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_bad_dimensions_rejected(bad):
    with pytest.raises(ValueError, match="POSITIVE_INTEGER"):
        PerFeatureTemporalExtractor(ToyEnv.observation_space, branch_filters=bad)


def test_unknown_encoder_rejected():
    with pytest.raises(ValueError, match="UNKNOWN"):
        Plugin().wrap_env(ToyEnv(), {"observation_encoder": "typo"})
