from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from agent_plugins.sac_agent import Plugin, _transfer_expanded_policy_state


def test_actor_observation_columns_expand_neutrally() -> None:
    source = {"actor.latent_pi.0.weight": torch.arange(12.0).reshape(3, 4)}
    target = {"actor.latent_pi.0.weight": torch.ones((3, 9))}

    result, expanded = _transfer_expanded_policy_state(
        source,
        target,
        source_observation_dim=4,
        target_observation_dim=9,
        action_dim=1,
    )

    torch.testing.assert_close(
        result["actor.latent_pi.0.weight"][:, :4],
        source["actor.latent_pi.0.weight"],
    )
    assert torch.count_nonzero(result["actor.latent_pi.0.weight"][:, 4:]) == 0
    assert expanded == ["actor.latent_pi.0.weight"]


def test_critic_preserves_action_column_after_observation_expansion() -> None:
    source_weight = torch.arange(10.0).reshape(2, 5)
    target_weight = torch.ones((2, 10))

    result, _expanded = _transfer_expanded_policy_state(
        {"critic.qf0.0.weight": source_weight},
        {"critic.qf0.0.weight": target_weight},
        source_observation_dim=4,
        target_observation_dim=9,
        action_dim=1,
    )

    resolved = result["critic.qf0.0.weight"]
    torch.testing.assert_close(resolved[:, :4], source_weight[:, :4])
    assert torch.count_nonzero(resolved[:, 4:9]) == 0
    torch.testing.assert_close(resolved[:, -1:], source_weight[:, -1:])


def test_expansion_rejects_unexplained_shape_change() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _transfer_expanded_policy_state(
            {"x": torch.zeros((2, 7))},
            {"x": torch.zeros((2, 9))},
            source_observation_dim=4,
            target_observation_dim=9,
            action_dim=1,
        )


def test_training_load_builds_candidate_then_transfers_policy(monkeypatch) -> None:
    from stable_baselines3 import SAC

    class FakeTensor:
        shape = (1,)

        class Data:
            @staticmethod
            def copy_(_value):
                return None

        data = Data()

    class FakePolicy:
        def __init__(self, state):
            self.state = state
            self.loaded = None

        def state_dict(self):
            return self.state

        def load_state_dict(self, state, strict):
            self.loaded = (state, strict)

    class FakeModel:
        def __init__(self, state, *, ent_coef, automatic):
            self.policy = FakePolicy(state)
            self.ent_coef = ent_coef
            self.ent_coef_optimizer = object() if automatic else None
            self.log_ent_coef = FakeTensor() if automatic else None

    source = FakeModel({"weight": "champion"}, ent_coef=0.2, automatic=False)
    target = FakeModel({"weight": "random"}, ent_coef="auto_0.2", automatic=True)
    captured = {"build_config": None}

    def fake_load(path, *, device):
        captured.update({"path": path, "device": device})
        return source

    plugin = Plugin()
    monkeypatch.setattr(plugin, "_require_continuous", lambda _env: None)
    monkeypatch.setattr(
        plugin,
        "build",
        lambda _env, config: captured.update({"build_config": config}) or target,
    )
    monkeypatch.setattr(SAC, "load", fake_load)
    env = object()
    result = plugin.load_for_training(
        "anchor.zip",
        env,
        {
            "learning_rate": 1e-4,
            "batch_size": 512,
            "gamma": 0.97,
            "tau": 0.003,
            "net_arch": [512, 256],
        },
    )

    assert result is target
    assert captured["path"] == "anchor.zip"
    assert captured["build_config"]["learning_rate"] == 1e-4
    assert captured["build_config"]["batch_size"] == 512
    assert captured["build_config"]["ent_coef"] == "auto_0.2"
    assert target.policy.loaded == ({"weight": "champion"}, True)
    assert result.warm_start_transfer_evidence["optimizer_state_transferred"] is False


def test_fixed_entropy_anchor_can_warm_start_automatic_entropy(tmp_path) -> None:
    import gymnasium as gym
    from stable_baselines3 import SAC

    env = gym.make("Pendulum-v1")
    source = SAC(
        "MlpPolicy",
        env,
        ent_coef=0.2,
        policy_kwargs={"net_arch": [8]},
        buffer_size=100,
        learning_starts=1,
        batch_size=8,
        verbose=0,
    )
    source_path = tmp_path / "fixed_entropy_sac"
    source.save(source_path)

    plugin = Plugin()
    target = plugin.load_for_training(
        str(source_path),
        env,
        {
            "ent_coef": "auto",
            "net_arch": [8],
            "buffer_size": 100,
            "learning_starts": 1,
            "batch_size": 8,
            "device": "cpu",
        },
    )

    assert target.ent_coef == "auto_0.2"
    assert target.ent_coef_optimizer is not None
    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    for key in source_state:
        torch.testing.assert_close(
            source_state[key].detach().cpu(),
            target_state[key].detach().cpu(),
        )
    assert target.warm_start_transfer_evidence == {
        "source_model": str(source_path.resolve()),
        "source_entropy_mode": "fixed",
        "target_entropy_mode": "automatic",
        "optimizer_state_transferred": False,
        "policy_state_transferred": True,
    }
    env.close()
