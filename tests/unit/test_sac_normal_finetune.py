"""M0 normal fine-tuning boundary proofs (SAC inner-curriculum order §7/§12).

On a REAL tiny SAC: the normal-phase learning rate must actually reach
both actor and critic Adam optimizers (the LR03/LR01 arms are
meaningless otherwise), optimizer moments must be fresh at the
boundary, and replay must be empty at the boundary and fill from
normal collection only.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from agent_plugins.sac_agent import Plugin


@pytest.fixture(scope="module")
def trained_source(tmp_path_factory):
    from stable_baselines3 import SAC

    env = gym.make("Pendulum-v1")
    source = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        ent_coef=0.2,
        policy_kwargs={"net_arch": [8]},
        buffer_size=128,
        learning_starts=8,
        batch_size=8,
        verbose=0,
    )
    # a short REAL learn so the source carries optimizer moments and a
    # non-empty replay buffer — the things that must NOT cross
    source.learn(total_timesteps=48)
    path = tmp_path_factory.mktemp("anchor") / "anchor_sac"
    source.save(path)
    yield {"source": source, "path": path, "env": env}
    env.close()


def _finetune_target(trained_source, learning_rate):
    plugin = Plugin()
    return plugin.load_for_training(
        str(trained_source["path"]),
        trained_source["env"],
        {
            "learning_rate": learning_rate,
            "ent_coef": 0.2,
            "net_arch": [8],
            "buffer_size": 128,
            "learning_starts": 8,
            "batch_size": 8,
            "device": "cpu",
        },
    )


class TestNormalFinetuneBoundary:
    def test_reduced_lr_reaches_actor_and_critic_optimizers(self, trained_source):
        for lr in (1e-4, 3e-5, 1e-5):
            target = _finetune_target(trained_source, lr)
            actor_lr = target.actor.optimizer.param_groups[0]["lr"]
            critic_lr = target.critic.optimizer.param_groups[0]["lr"]
            assert actor_lr == pytest.approx(lr), "actor optimizer missed the LR"
            assert critic_lr == pytest.approx(lr), "critic optimizer missed the LR"
            evidence = target.warm_start_transfer_evidence
            assert evidence["target_actor_optimizer_lr"] == pytest.approx(lr)
            assert evidence["target_critic_optimizer_lr"] == pytest.approx(lr)

    def test_policy_tensors_match_source_exactly_after_transfer(self, trained_source):
        target = _finetune_target(trained_source, 3e-5)
        evidence = target.warm_start_transfer_evidence
        assert evidence["policy_hash_matches_source_after_transfer"] is True
        for component, distance in evidence[
            "component_l1_distances_after_transfer"
        ].items():
            assert distance == 0.0, component

    def test_optimizer_moments_are_fresh_not_transferred(self, trained_source):
        source = trained_source["source"]
        # the source genuinely carries Adam moments after learning
        assert len(source.actor.optimizer.state_dict()["state"]) > 0
        target = _finetune_target(trained_source, 3e-5)
        assert target.warm_start_transfer_evidence[
            "optimizer_state_transferred"] is False
        assert len(target.actor.optimizer.state_dict()["state"]) == 0
        assert len(target.critic.optimizer.state_dict()["state"]) == 0

    def test_replay_is_empty_at_boundary_and_fills_from_normal_only(
        self, trained_source
    ):
        source = trained_source["source"]
        assert source.replay_buffer.size() > 0
        target = _finetune_target(trained_source, 3e-5)
        evidence = target.warm_start_transfer_evidence
        assert evidence["replay_transitions_transferred"] == 0
        assert evidence["replay_size_at_boundary"] == 0
        assert target.replay_buffer.size() == 0
        # normal collection is the ONLY replay source
        target.learn(total_timesteps=16)
        assert 0 < target.replay_buffer.size() <= 16

    def test_fixed_entropy_value_is_recorded(self, trained_source):
        target = _finetune_target(trained_source, 1e-5)
        evidence = target.warm_start_transfer_evidence
        assert evidence["source_entropy_mode"] == "fixed"
        assert evidence["target_entropy_mode"] == "fixed"
        assert evidence["source_entropy_value"] == 0.2
        assert evidence["target_entropy_value"] == 0.2
