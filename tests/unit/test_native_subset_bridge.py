import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from tests.unit.test_rl_subset import split_recipe
from app.rl_subset import seal_recipe
from tools.rl_subset_bridge import build_model, progress_callback, native_metrics


class Synthetic(gym.Env):
    observation_space = gym.spaces.Dict({
        'features': gym.spaces.Box(-np.inf, np.inf, (13, 2), np.float32),
        'action_mask': gym.spaces.Box(0, 1, (5,), np.float32)})
    action_space = gym.spaces.Box(np.array([0, 0, 0, 0], np.float32),
                                 np.array([4, 1, 1, 1], np.float32))

    def obs(self):
        return dict(features=np.ones((13, 2), np.float32),
                    action_mask=np.array([1, 1, 1, 0, 0], np.float32))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.n = 0
        return self.obs(), {}

    def step(self, action):
        assert action[0] == int(action[0]) and int(action[0]) in (0, 1, 2)
        self.n += 1
        return self.obs(), float(action[0] == 1), self.n == 8, False, {'elapsed_bars': 1}


def test_real_ppo_updates_and_reload(tmp_path):
    r = split_recipe()
    r['tensor']['window']['observations'] = 13
    seal_recipe(r)
    m = build_model(Synthetic(), r, 7, 'cpu', {'total_timesteps': 32, 'n_steps': 16, 'batch_size': 16, 'n_epochs': 1})
    before = [p.detach().clone() for p in m.policy.parameters()]
    m.learn(32, callback=progress_callback(tmp_path, 'test', 32))
    assert any(not a.equal(b) for a, b in zip(before, m.policy.parameters()))
    m.save(tmp_path / 'model.zip')
    reloaded = PPO.load(tmp_path / 'model.zip', device='cpu')
    obs = Synthetic().obs()
    np.testing.assert_array_equal(m.predict(obs, deterministic=True)[0],
                                  reloaded.predict(obs, deterministic=True)[0])
    assert m._n_updates == 2


def test_native_costs_and_undefined_no_trade_rates():
    report = dict(net_profit_usd=10., sharpe_daily_sqrt365=2., max_drawdown_pct=5.,
                  orders_per_week=2., cost_spread_usd=1., cost_slippage_usd=2.,
                  cost_commission_usd=3., cost_financing_usd=-1.)
    trades = [dict(direction=1, net_usd=10., close_reason='AGENT_CLOSE')]
    m, costs = native_metrics(report, trades)
    assert m['costs'] == 5 and m['max_drawdown'] == .05
    assert m['win_rate'] == 1 and m['sell_win_rate'] is None
    assert m['early_closes'] == 1 and costs['financing'] == -1
    m, _ = native_metrics(report, [])
    assert m['sharpe'] is None and m['win_rate'] is None
