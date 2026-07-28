from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from agent_plugins.sac_agent import _transfer_expanded_policy_state


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
