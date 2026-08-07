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


def test_training_load_builds_candidate_then_transfers_policy(monkeypatch, tmp_path) -> None:
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
            # mimic a real transfer so the boundary-evidence hash sees
            # the source tensors on the target afterwards
            self.state = dict(state)

    class FakeOptimizerOwner:
        class optimizer:  # noqa: N801 - mimic SB3 attribute shape
            param_groups = [{"lr": 1e-4}]

    class FakeModel:
        def __init__(self, state, *, ent_coef, automatic):
            self.policy = FakePolicy(state)
            self.ent_coef = ent_coef
            self.ent_coef_optimizer = object() if automatic else None
            self.log_ent_coef = FakeTensor() if automatic else None
            self.actor = FakeOptimizerOwner()
            self.critic = FakeOptimizerOwner()
            self.replay_buffer = None

    champion = {"actor.weight": torch.ones(2)}
    source = FakeModel(champion, ent_coef=0.2, automatic=False)
    target = FakeModel(
        {"actor.weight": torch.zeros(2)}, ent_coef="auto_0.2", automatic=True
    )
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
    anchor_path = tmp_path / "anchor.zip"
    anchor_path.write_bytes(b"fake anchor bytes")
    result = plugin.load_for_training(
        str(anchor_path),
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
    assert captured["path"].endswith("anchor.zip")
    assert captured["build_config"]["learning_rate"] == 1e-4
    assert captured["build_config"]["batch_size"] == 512
    assert captured["build_config"]["ent_coef"] == "auto_0.2"
    loaded_state, loaded_strict = target.policy.loaded
    assert loaded_strict is True
    torch.testing.assert_close(loaded_state["actor.weight"], champion["actor.weight"])
    evidence = result.warm_start_transfer_evidence
    assert evidence["optimizer_state_transferred"] is False
    assert evidence["policy_hash_matches_source_after_transfer"] is True
    assert evidence["replay_transitions_transferred"] == 0


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
    evidence = target.warm_start_transfer_evidence
    assert evidence["source_model"] == str(source_path.resolve())
    assert evidence["source_entropy_mode"] == "fixed"
    assert evidence["target_entropy_mode"] == "automatic"
    assert evidence["optimizer_state_transferred"] is False
    assert evidence["policy_state_transferred"] is True
    # M0 boundary proofs on a REAL SAC: tensor-hash equality, zero
    # component distances, fresh empty replay, and the exact optimizer
    # learning rates the fine-tune will use.
    assert evidence["policy_hash_matches_source_after_transfer"] is True
    for component, distance in evidence[
        "component_l1_distances_after_transfer"
    ].items():
        assert distance == 0.0, component
    assert evidence["replay_transitions_transferred"] == 0
    assert evidence["replay_size_at_boundary"] == 0
    assert evidence["target_actor_optimizer_lr"] > 0
    assert evidence["target_critic_optimizer_lr"] > 0
    assert evidence["source_entropy_value"] == 0.2
    assert len(evidence["source_artifact_sha256"]) == 64
    env.close()
