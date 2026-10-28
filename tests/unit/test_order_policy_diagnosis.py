import numpy as np
import pytest

from tools.order_policy_diagnosis import command_bins, sample_indices, counterfactual_actions, output_root, policy_probe


def test_bins_float32_edges():
    x = np.array([-1, -.6, -.2, .2, .6, 1], dtype=np.float32)
    assert command_bins(x).tolist() == [0, 1, 2, 2, 3, 4]
    with pytest.raises(ValueError):
        command_bins(np.array([np.nan]))


def test_sampling_is_unique_complete_when_small_and_spans_support():
    assert sample_indices(4, 256).tolist() == [0, 1, 2, 3]
    x = sample_indices(1000, 256)
    assert len(x) == len(set(x)) == 256 and x[0] == 0 and x[-1] == 999
    with pytest.raises(ValueError):
        sample_indices(0, 256)


def test_counterfactual_changes_only_command():
    a = np.array([[.8, .2, -.3], [-.4, .7, -.1]], np.float32)
    grid, names = counterfactual_actions(a)
    assert grid.shape == (2, 8, 3)
    np.testing.assert_array_equal(grid[:, -1], a)
    for j in range(8):
        np.testing.assert_array_equal(grid[:, j, 1:], a[:, 1:])
    # Positive control: a critic with Q=-action^2 must prefer the HOLD center.
    scores = -(grid[:, :, 0] ** 2)
    assert [names[i] for i in scores.argmax(axis=1)] == ['HOLD', 'HOLD']


def test_output_cannot_change_parent_or_existing_evidence(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    for bad in (parent, parent / 'child', tmp_path):
        with pytest.raises(ValueError):
            output_root(parent, bad)
    out = tmp_path / 'out'
    output_root(parent, out)
    (out / 'existing').write_text('retained')
    with pytest.raises(ValueError):
        output_root(parent, out)
    assert (out / 'existing').read_text() == 'retained'


def test_native_sac_probe_preserves_weights_rng_and_modes():
    import gymnasium as gym
    import torch
    from stable_baselines3 import SAC
    from tools.recipe_sac_queue import tensor_hash

    class Toy(gym.Env):
        observation_space = gym.spaces.Dict({'features': gym.spaces.Box(-1, 1, (3,), np.float32)})
        action_space = gym.spaces.Box(-1, 1, (3,), np.float32)

    model = SAC('MultiInputPolicy', Toy(), policy_kwargs={'net_arch': [8]}, device='cpu', seed=7)
    before, rng = tensor_hash(model), torch.get_rng_state().clone()
    modes = [m.training for m in model.policy.modules()]
    report = policy_probe(model, {'features': np.zeros((4, 3), np.float32)})
    assert tensor_hash(model) == before and torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in model.policy.modules()]
    assert sum(report['stochastic_commands'].values()) == 128
    assert sum(report['q_argmax_counts'].values()) == 4
    assert report['dropout_mode_probe']['command_change_fraction'] == 0
