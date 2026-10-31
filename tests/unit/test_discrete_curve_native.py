import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.recipe_runtime import attach_engine
from app.dynamic_account_panel import World
from env_plugins.discrete_curve_native import CurveNativeEnv


@pytest.fixture
def curve_env():
    import os
    engine = os.environ.get('STRATEGY_ENGINE_ROOT')
    if not engine:
        pytest.skip('explicit native engine fixture required')
    attach_engine(engine)
    risk = json.loads((Path(__file__).resolve().parents[2] /
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
    world = World(dict(seed=711, pattern='overlap', volatility='decreasing', mirror=1), risk)
    e = world.env
    features = np.zeros((len(e.v), 13, 2), np.float32)
    inputs = SimpleNamespace(row_ts_ns=e.v + 3600 * 10**9, valid=np.ones(len(e.v), bool),
        windows=features, observation=lambda row: {'features': features[row].copy()})
    return CurveNativeEnv(e, inputs, risk, 6)


@pytest.mark.parametrize('action,direction', [(1, 1), (2, -1)])
def test_native_two_directions_fixed_risk_and_early_close(curve_env, action, direction):
    env = curve_env
    obs = env.reset()
    while not obs['mask'][action]:
        obs, _, _, _ = env.step(0)
    obs, _, _, _ = env.step(action)
    if env.engine.pos is not None:
        p = env.engine.pos
        assert p.direction == direction
        assert direction * (p.tp - p.entry_price) > 0
        assert direction * (p.entry_price - p.sl) > 0
        assert 0 < p.relative_size <= 1
        obs, _, _, _ = env.step(1)
        assert env.engine.pos is None
    else:
        assert env.engine.trades
    while not env.finished:
        obs, _, _, _ = env.step(0)
    assert abs(env.engine.balance - env.engine.cfg.capital -
               sum(t['net_usd'] for t in env.engine.trades)) < 1e-6
    assert all(t['direction'] == direction for t in env.engine.trades)


def test_no_invalid_action_or_phase_leak_to_features(curve_env):
    obs = curve_env.reset()
    assert not obs['mask'][1]
    with pytest.raises(ValueError, match='MASKED_ACTION'):
        curve_env.step(1)
    assert set(obs) == {'features', 'route', 'progress', 'mask'}
    assert np.array_equal(obs['features'], np.zeros((13, 2)))


def test_missing_windows_hold_and_preserve_discount_duration(curve_env):
    env = curve_env
    env.reset()
    start = env.bar
    env.valid[start + 1:start + 4] = False
    _, _, _, duration = env.step(0)
    assert duration == 4
    assert env.bar == start + 4
    assert env.auto_holds == 3


@pytest.mark.parametrize('mirror', [1, -1])
def test_planted_native_positive_fixture_has_realisable_profit(curve_env, mirror):
    import torch
    from app.curve_probe_control import planted_control
    from tests.unit.test_double_tcn_probe import probe_recipe

    class Ideal:
        device = 'cpu'

        def distribution(self, obs):
            signal = int(obs['features'][0, -1, 0])
            route = int(obs['route'][0, 0])
            action = {-1: 2, 0: 0, 1: 1}[signal] if route == 0 else int(signal == (-1 if route == 1 else 1))
            if not obs['mask'][0, action]:
                action = 0
            probs = torch.zeros(1, 3)
            probs[0, action] = 1
            return SimpleNamespace(probs=probs)
    report = planted_control(Ideal(), probe_recipe(), curve_env.risk_config, mirror)
    assert report['passed']
    assert report['early_buy'] > 0 and report['early_sell'] > 0
