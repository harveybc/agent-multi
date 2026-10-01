"""RL03: SAC actor and critics, and DQN online/target networks, consume
compatible representations; shared versus separate encoders and optimizer
ownership are explicit; no accidental double updates."""
from __future__ import annotations

import pytest

from ._fixtures import FEATURES, env_config, flat_env, modular_config, write_synthetic_csv
from ._mechanisms import assert_checkout_resolution, require

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("backtrader")


@pytest.fixture
def csv(tmp_path):
    return write_synthetic_csv(tmp_path / "fixture.csv", rows=200)


def _build(csv, algorithm, share):
    build = require("RL03", "rl_temporal.consumers", "build_model",
                    "SB3 model builder with the modular extractor plugged into the policy")
    mode = "continuous" if algorithm == "SAC" else "discrete"
    env = flat_env(env_config(csv, action_space_mode=mode))
    model = build(algorithm, env, modular_config=modular_config(), feature_order=FEATURES,
                  share_features_extractor=share, device="cpu", seed=0,
                  learning_starts=8, buffer_size=64, batch_size=8)
    return env, model


@pytest.mark.parametrize("share", [True, False])
def test_sac_actor_and_critics_ownership_is_explicit(csv, share):
    assert_checkout_resolution("RL03")
    ownership = require("RL03", "rl_temporal.donor_contract", "optimizer_ownership",
                        "optimizer ownership report")
    env, model = _build(csv, "SAC", share)
    try:
        report = ownership(model)
        assert report["algorithm"] == "SAC"
        assert report["shared_features_extractor"] is share
        if share:
            assert report["extractor_owner"] == {"actor": True, "critic": False}
            assert model.critic.features_extractor is model.actor.features_extractor
        else:
            assert report["extractor_owner"] == {"actor": True, "critic": True}
            assert model.critic.features_extractor is not model.actor.features_extractor
        assert report["target_extractor_in_any_optimizer"] is False
        assert report["params_in_more_than_one_optimizer"] == []
        obs, _ = env.reset(seed=0)
        x = torch.as_tensor(obs[None])
        a = model.actor.features_extractor(x)
        c = model.critic.features_extractor(x)
        assert a.shape == c.shape == (1, model.actor.features_dim)
    finally:
        env.close()


def test_dqn_online_and_target_consume_the_same_representation(csv):
    ownership = require("RL03", "rl_temporal.donor_contract", "optimizer_ownership",
                        "optimizer ownership report")
    env, model = _build(csv, "DQN", True)
    try:
        report = ownership(model)
        assert report["algorithm"] == "DQN"
        assert report["extractor_owner"] == {"q_net": True}
        assert report["target_extractor_in_any_optimizer"] is False
        assert report["params_in_more_than_one_optimizer"] == []
        obs, _ = env.reset(seed=0)
        x = torch.as_tensor(obs[None])
        online = model.q_net.features_extractor(x)
        target = model.q_net_target.features_extractor(x)
        assert online.shape == target.shape
        assert torch.allclose(online, target), "target starts as a copy of online"
    finally:
        env.close()


def test_a_gradient_step_updates_each_extractor_parameter_at_most_once(csv):
    count = require("RL03", "rl_temporal.donor_contract", "count_parameter_updates",
                    "per-parameter update counter across one train() call")
    env, model = _build(csv, "SAC", False)
    try:
        model.learn(total_timesteps=16)
        updates = count(model, gradient_steps=1, batch_size=8)
        assert updates, "no extractor parameter was observed"
        assert max(updates.values()) <= 1, {k: v for k, v in updates.items() if v > 1}
    finally:
        env.close()
