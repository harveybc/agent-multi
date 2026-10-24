from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from env_plugins.recipe_native import RecipeNativeEnv, DurationReplayBuffer


class Engine:
    def __init__(self):
        self.first, self.last = 0, 6
        self.cfg = SimpleNamespace(capital=100., tp_bounds=(.25, 3.), sl_bounds=(.5, 3.))
        self.v = np.arange(7) * 3600 * 10**9
        self.c = np.ones(7)
        self.reset()

    def reset(self):
        self.balance, self.pos, self.calls = 100., None, []

    def step(self, b, action, tp, sl):
        self.calls.append((b, action))
        self.balance += b + 1
        return {}


class Inputs:
    windows = np.arange(7, dtype=np.float32).reshape(7, 1, 1)
    valid = np.array([False, True, False, False, True, True, False])
    row_ts_ns = (np.arange(7) + 1) * 3600 * 10**9

    def observation(self, row):
        assert self.valid[row]
        return {'features': self.windows[row].copy()}


def test_missing_intervals_preserve_bars_reward_and_terminal():
    e = Engine()
    env = RecipeNativeEnv(e, Inputs(), gamma=.5)
    obs, info = env.reset()
    assert obs['features'].item() == 1
    assert e.calls == [(0, 0)]
    obs, r, done, trunc, info = env.step(np.array([1., 0., 0.]))
    assert e.calls == [(0, 0), (1, 1), (2, 0), (3, 0)]
    assert r == 2 + .5 * 3 + .25 * 4
    assert info['discount'] == .5 ** 3 and info['elapsed_bars'] == 3
    assert obs['features'].item() == 4 and not done and not trunc
    env.step(np.zeros(3))
    obs, r, done, _, info = env.step(np.zeros(3))
    assert done and info['terminal_observation_placeholder']
    assert obs['features'].item() == 5
    assert e.calls[-1][0] == 5
    assert env.coverage['engine_bars'] == 6


def test_open_position_gap_uses_hold_not_target_flat():
    e = Engine()
    env = RecipeNativeEnv(e, Inputs(), gamma=.99)
    env.reset()
    e.pos = SimpleNamespace(direction=1, units=1, entry_price=1., entry_cost={})
    env.step(np.array([1., 0., 0.]))
    assert e.calls[-3:] == [(1, 0), (2, 0), (3, 0)]


def test_replay_preserves_per_transition_discount():
    space = gym.spaces.Dict({'features': gym.spaces.Box(-np.inf, np.inf, (1, 1), np.float32)})
    action = gym.spaces.Box(-1, 1, (3,), np.float32)
    replay = DurationReplayBuffer(4, space, action, device='cpu')
    obs = {'features': np.array([[[5.]]], np.float32)}
    replay.add(obs, obs, np.zeros((1, 3)), np.array([4.5]), np.array([False]), [{'discount': .125}])
    sample = replay._get_samples(np.array([0]))
    assert sample.discounts.item() == .125 and sample.rewards.item() == 4.5
    assert sample.dones.item() == 0
    with pytest.raises(ValueError, match='DISCOUNT'):
        replay.add(obs, obs, np.zeros((1, 3)), np.array([0.]), np.array([True]), [{}])


def test_refuse_missing_all_observations_and_invalid_action():
    x = Inputs()
    x.valid = np.zeros(7, bool)
    with pytest.raises(ValueError, match='NO_DECISION'):
        RecipeNativeEnv(Engine(), x)
    env = RecipeNativeEnv(Engine(), Inputs())
    env.reset()
    with pytest.raises(ValueError, match='ACTION'):
        env.step(np.array([np.nan, 0., 0.]))
