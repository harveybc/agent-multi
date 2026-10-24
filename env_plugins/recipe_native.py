"""Label-free frozen observations over the shared bar accounting engine."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer

from app.frozen_recipe_inputs import require


class DurationReplayBuffer(DictReplayBuffer):
    """The environment aggregates automatic HOLD bars; SAC bootstraps at gamma**n."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        require(self.n_envs == 1, 'SINGLE_ENV_REPLAY')
        self.discounts = np.zeros((self.buffer_size, 1), np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        value = infos[0].get('discount')
        require(value is not None and np.isfinite(value) and 0 <= value <= 1, 'DISCOUNT')
        self.discounts[self.pos, 0] = value
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        sample = super()._get_samples(batch_inds, env)
        return sample._replace(discounts=self.to_torch(self.discounts[batch_inds]))


def equity(engine, bar):
    p = engine.pos
    if p is None:
        return float(engine.balance)
    return float(engine.balance + p.direction * p.units * (engine.c[bar] - p.entry_price)
                 - sum(p.entry_cost.values()))


class RecipeNativeEnv(gym.Env):
    """No observation synthesis. Missing windows disable decisions, not accounting.

    Engine bars have start timestamps; input timestamps are completed-bar times.
    The engine owns position, costs and all entry/exit restrictions. None of its
    state is exposed to the policy. End-of-period is a true terminal transition.
    """

    metadata = {}

    def __init__(self, engine, inputs, gamma=.99):
        super().__init__()
        require(0 < gamma <= 1, 'GAMMA')
        self.engine, self.inputs, self.gamma = engine, inputs, gamma
        stamps = engine.v + 3600 * 10**9
        rows = np.searchsorted(inputs.row_ts_ns, stamps)
        bounded = np.minimum(rows, len(inputs.row_ts_ns) - 1)
        matched = (rows < len(inputs.row_ts_ns)) & (inputs.row_ts_ns[bounded] == stamps)
        self.rows = bounded
        self.valid = matched & inputs.valid[bounded]
        require(self.valid[engine.first:engine.last].any(), 'NO_DECISION')
        self.observation_space = gym.spaces.Dict({'features': gym.spaces.Box(
            -np.inf, np.inf, inputs.windows.shape[1:], np.float32)})
        self.action_space = gym.spaces.Box(-1., 1., (3,), np.float32)
        self.finished = True

    def _observe(self):
        self.last_observation = self.inputs.observation(int(self.rows[self.bar]))
        return {k: v.copy() for k, v in self.last_observation.items()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        self.bar, self.finished = self.engine.first, False
        self.coverage = dict(engine_bars=0, automatic_hold_bars=0, policy_decisions=0,
                             prefix_bars=0)
        while not self.valid[self.bar]:
            self.engine.step(self.bar, 0, 1.5, 2.)
            self.bar += 1
            self.coverage['engine_bars'] += 1
            self.coverage['prefix_bars'] += 1
        return self._observe(), {'bar': self.bar}

    def _decode(self, x):
        require(np.shape(x) == (3,) and np.isfinite(x).all()
                and (np.abs(x) <= 1.000001).all(), 'ACTION')
        target = 1 if x[0] > 1 / 3 else (-1 if x[0] < -1 / 3 else 0)
        position = self.engine.pos.direction if self.engine.pos is not None else 0
        action = 0 if target == position else (3 if target == 0 else (1 if target > 0 else 2))
        tp_lo, tp_hi = self.engine.cfg.tp_bounds
        sl_lo, sl_hi = self.engine.cfg.sl_bounds
        return action, tp_lo + (float(x[1]) + 1) / 2 * (tp_hi - tp_lo), \
            sl_lo + (float(x[2]) + 1) / 2 * (sl_hi - sl_lo)

    def step(self, action):
        require(not self.finished, 'STEP_AFTER_TERMINAL')
        decoded = self._decode(action)
        start = self.bar
        total, n = 0., 0
        self.coverage['policy_decisions'] += 1
        while True:
            previous = equity(self.engine, self.bar)
            self.engine.step(self.bar, *decoded)
            self.bar += 1
            reward = 100. * (equity(self.engine, self.bar) - previous) / self.engine.cfg.capital
            require(np.isfinite(reward), 'NONFINITE_REWARD')
            total += self.gamma ** n * reward
            n += 1
            self.coverage['engine_bars'] += 1
            if self.bar >= self.engine.last:
                self.finished = True
                break
            if self.valid[self.bar]:
                break
            decoded = (0, 1.5, 2.)  # Explicit HOLD, never the continuous target-flat action.
            self.coverage['automatic_hold_bars'] += 1
        placeholder = self.finished and not self.valid[self.bar]
        obs = ({k: v.copy() for k, v in self.last_observation.items()}
               if placeholder else self._observe())
        return obs, total, self.finished, False, {
            'discount': self.gamma ** n, 'elapsed_bars': n,
            'decision_bar': start, 'bar': self.bar,
            'terminal_observation_placeholder': bool(placeholder)}
