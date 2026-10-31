"""Known causal planted signal through learned surfaces and the native executor."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from agent_plugins.discrete_curve_sac import tensor_observation
from app.dynamic_account_panel import World
from app.train_native_pareto import bid_environment_class
from env_plugins.discrete_curve_native import CurveNativeEnv


def planted_control(model, recipe, risk, mirror=1):
    base = World(dict(seed=791, pattern='trend', volatility='decreasing', mirror=1), risk)
    old = base.env
    n = len(old.v)
    signal = mirror * np.where((np.arange(n) // 6) % 2 == 0, 1, -1)
    close = 1.1 + np.r_[0., np.cumsum(signal[:-1] * .00008)]
    op = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame(dict(OPEN=op, HIGH=np.maximum(op, close) + .000005,
        LOW=np.minimum(op, close) - .000005, CLOSE=close), index=old.bars.index)
    profile = replace(old.cfg.profile, spread_schedule=((0, 24, .00005),), weekly_open_spread=.00005)
    engine = bid_environment_class()(bars, old.cal, np.full(n, .001), np.full(n, .001),
        bars.index[0], bars.index[-1] + pd.Timedelta(hours=1), replace(old.cfg, profile=profile), tick=.00001)
    shape = (recipe['tensor']['window']['observations'], recipe['list']['n'])
    features = np.zeros((n, *shape), np.float32)
    features[:, -1, :] = signal[:, None]
    inputs = SimpleNamespace(row_ts_ns=engine.v + 3600 * 10**9, valid=np.ones(n, bool),
        windows=features, observation=lambda row: {'features': features[row].copy()})
    env = CurveNativeEnv(engine, inputs, risk, 6)
    obs = env.reset()
    while True:
        with torch.no_grad():
            action = int(model.distribution(tensor_observation(obs, model.device)).probs.argmax(-1)[0])
        obs, _, done, _ = env.step(action)
        if done:
            break
    engine.flatten_at_end()
    directions = {t['direction'] for t in engine.trades}
    closes = {t['direction'] for t in engine.trades if t['close_reason'] == 'AGENT_CLOSE'}
    profit = float(engine.balance - engine.cfg.capital)
    cash_error = abs(profit - sum(t['net_usd'] for t in engine.trades))
    return dict(scope='PLANTED_CAUSAL_SIGNAL_NATIVE_CONTROL_NOT_MARKET_PERFORMANCE',
        mirror=mirror, profit=profit, trades=len(engine.trades),
        buy=sum(t['direction'] == 1 for t in engine.trades),
        sell=sum(t['direction'] == -1 for t in engine.trades),
        early_buy=sum(t['direction'] == 1 and t['close_reason'] == 'AGENT_CLOSE' for t in engine.trades),
        early_sell=sum(t['direction'] == -1 and t['close_reason'] == 'AGENT_CLOSE' for t in engine.trades),
        cash_error=cash_error, passed=directions == {-1, 1} and closes == {-1, 1}
            and profit > 0 and cash_error < 1e-6)
