import numpy as np
import pytest
import torch

from app.economic_curve_arbiter import route, choose, economic_vector, calibrate, rollout_reference
from agent_plugins.discrete_curve_sac import route_surface
from tests.unit.test_suppressed_close_counterfactuals import risk  # noqa: F401
from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('z', [-.2, 0., .13, .5, 1., 1.2])
def test_numpy_routing_matches_frozen_torch(direction, z):
    x = np.random.default_rng(7).normal(size=(3, 39)).astype(np.float32)
    wanted = route_surface(torch.tensor(x), torch.full((3, 1), 1 if direction == 1 else 2),
                           torch.full((3, 1), z)).numpy()[:, :2]
    assert np.allclose(route(x, direction, z), wanted, atol=1e-6)


def test_pareto_account_choice_and_conservative_ties():
    assert choose({'NOP': [0., 0.], 'BUY:3': [-1., .1]})['action'] == 'NOP'
    assert choose({'NOP': [0., 0.], 'BUY:3': [3., 1.], 'SELL:6': [3., 1.]})['action'] == 'NOP'
    assert choose({'HOLD': [2., 1.], 'CLOSE': [2., 1.]})['action'] == 'HOLD'
    assert choose({'HOLD': [-2., 2.], 'CLOSE': [-.2, .1]})['action'] == 'CLOSE'
    assert choose({'HOLD': [4., .5], 'CLOSE': [.1, .1]})['action'] == 'HOLD'
    assert choose({'NOP': [0., 0.], 'BUY:3': [2., .1], 'SELL:24': [4., .2]})['action'] == 'SELL:24'


def test_economic_units_and_split_purge():
    assert economic_vector([100., 95., 110.]) == pytest.approx([10., 5.])
    rows = [dict(key='x', x=[float(i)], y=[float(i), .1], origin='2022-01-01',
                 deadline='2022-01-02') for i in range(20)]
    rows.append(dict(key='x', x=[10000.], y=[10000., 3.], origin='2023-01-01', deadline='2023-01-02'))
    model, report = calibrate(rows, minimum=4)
    assert model['x'].named_steps['scale'].mean_[0] == 9.5
    assert report['x']['fit_rows'] == 20 and report['x']['check_rows'] == 1
    rows.append(dict(key='x', x=[1.], y=[1., 1.], origin='2022-12-31', deadline='2023-01-02'))
    _, report = calibrate(rows, minimum=4)
    assert report['x']['purged_rows'] == 1


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize('favorable', [False, True])
def test_native_known_hold_close_and_state_isolation(request, side, favorable):
    from app.sequential_account_pareto import fixture
    _, e, _ = fixture(request.getfixturevalue('risk'), 'smooth', side if favorable else -side, side, steps=4)

    class Passive:
        def reference(self, env, owner):
            return 'HOLD' if env.pos else 'NOP'

        def step(self, env, owner, action):
            info = env.step(env.next_decision, 3 if action == 'CLOSE' else 0)
            return owner if env.pos else None, info

    before = (e.next_decision, e.balance, e.pos, tuple(e.trades))
    bids = {a: rollout_reference(e, 12, a, Passive())['vector'] for a in ('HOLD', 'CLOSE')}
    assert choose(bids)['action'] == ('HOLD' if favorable else 'CLOSE')
    assert before == (e.next_decision, e.balance, e.pos, tuple(e.trades))
    from types import SimpleNamespace
    from app.economic_curve_arbiter import Reports, EconomicController, HORIZONS
    from env_plugins.discrete_curve_native import CurveNativeEnv, HOUR
    inputs = SimpleNamespace(row_ts_ns=e.v + HOUR, valid=np.ones(len(e.v), bool))
    values = {h: {b: np.arange(117, dtype=float).reshape(3, 39) + h
                  for b in range(len(e.v))} for h in HORIZONS}
    reports = Reports(values, {h: CurveNativeEnv(e, inputs, request.getfixturevalue('risk'), h)
                              for h in HORIZONS})
    seen = []

    class KnownValue:
        def __init__(self, h, action):
            self.h, self.action = h, action
        def predict(self, x):
            seen.append(self.h)
            np.testing.assert_allclose(x[0], reports.features(e, self.h))
            return np.array([bids[self.action]])

    models = {f'close_12_{side}_{h}_{action}': KnownValue(h, action)
              for h in HORIZONS for action in ('HOLD', 'CLOSE')}
    choice = EconomicController(reports, models).decide(e, 12)
    assert choice['action'] == ('HOLD' if favorable else 'CLOSE')
    assert sorted(seen) == sorted(list(HORIZONS) * 2)
    assert before == (e.next_decision, e.balance, e.pos, tuple(e.trades))


def test_same_position_all_reporters_and_no_future_access(request):
    from app.economic_curve_arbiter import Reports, EconomicController, HORIZONS
    from env_plugins.discrete_curve_native import CurveNativeEnv
    base = request.getfixturevalue('curve_env')
    base.reset()
    while not base.last_observation['mask'][1]:
        base.step(0)
    values = {h: {int(r): np.arange(117, dtype=float).reshape(3, 39) + h
                  for r in base.rows} for h in HORIZONS}
    reports = Reports(values, {h: CurveNativeEnv(base.engine, base.inputs, base.risk_config, h)
                              for h in HORIZONS})

    class Constant:
        def __init__(self, y):
            self.y = y
        def predict(self, x):
            return np.array([self.y])

    models = {f'open_{h}_{a}': Constant([2., .1] if h == 12 and a == 2 else [0., 0.])
              for h in HORIZONS for a in (0, 1, 2)}
    controller = EconomicController(reports, models)
    decision = controller.decide(base.engine, None)
    assert decision['action'] == 'SELL:12'
    assert decision['reporters'] == list(HORIZONS)
    current = int(base.rows[base.bar])
    for h in HORIZONS:
        for r in values[h]:
            if r > current:
                values[h][r][:] = 1e9
    assert controller.decide(base.engine, None) == decision
    reports.arrays[24].pop(current)
    assert controller.decide(base.engine, None)['action'] == 'NOP'


@pytest.mark.parametrize('direction', [1, -1])
def test_joint_native_account_and_posthoc_close_replay(request, tmp_path, direction):
    from types import SimpleNamespace
    from tools.run_economic_curve_arbiter import run_account, JointEnv
    from app.economic_curve_arbiter import Reports, HORIZONS
    from env_plugins.discrete_curve_native import CurveNativeEnv
    from unittest.mock import patch
    base = request.getfixturevalue('curve_env')
    values = {h: {int(r): np.zeros((3, 39)) for r in base.rows} for h in HORIZONS}
    reports = Reports(values, {h: CurveNativeEnv(base.engine, base.inputs, base.risk_config, h)
                              for h in HORIZONS})

    class Constant:
        def __init__(self, result):
            self.result = result
        def predict(self, x):
            return np.array([self.result])

    models = {f'open_{h}_{a}': Constant([2., .1] if h == 24 and a == (1 if direction == 1 else 2)
                                      else [0., 0.]) for h in HORIZONS for a in (0, 1, 2)}
    models.update({f'close_24_{direction}_{h}_{a}': Constant([1., 0.] if a == 'CLOSE' else [0., .1])
                   for h in HORIZONS for a in ('CLOSE', 'HOLD')})
    with patch('tools.run_economic_curve_arbiter.native_environment', return_value=base):
        result = run_account(SimpleNamespace(inputs=base.inputs), base.risk_config, reports,
                             models, 'TRAIN', tmp_path, 'joint', 1.)
    assert result['trades'] > 0
    assert result['horizon_counts']['24'] == result['trades']
    assert result['buy' if direction == 1 else 'sell']['trades'] == result['trades']
    assert result['close_diagnostic']['exact_replay']
    assert result['close_diagnostic']['proposals'] > 0
    assert result['cash_error'] < 1e-6


def test_invalid_numeric_config_rejected():
    import json
    from pathlib import Path
    from tools.run_economic_curve_arbiter import valid_design
    config = json.loads((Path(__file__).resolve().parents[2] /
        'examples/config/real_curve_economics_v1.json').read_text())
    valid_design(config)
    config['ridge_alpha'] = float('nan')
    with pytest.raises(ValueError, match='ECONOMIC_PILOT_DESIGN'):
        valid_design(config)


def test_closed_resume_authenticates_and_never_rewrites(tmp_path):
    from tools.run_economic_curve_arbiter import closed_and_verified, write
    from app.frozen_recipe_inputs import file_sha
    assert not closed_and_verified(tmp_path, 'frozen')
    write(tmp_path / 'RESULTS.json', {'net': -1})
    write(tmp_path / 'CLOSE.json', {'identity': 'frozen',
          'files': {'RESULTS.json': file_sha(tmp_path / 'RESULTS.json')}})
    before = (tmp_path / 'CLOSE.json').read_bytes()
    assert closed_and_verified(tmp_path, 'frozen')
    assert (tmp_path / 'CLOSE.json').read_bytes() == before
    with pytest.raises(ValueError, match='CLOSURE_IDENTITY'):
        closed_and_verified(tmp_path, 'other')
    write(tmp_path / 'RESULTS.json', {'net': 1})
    with pytest.raises(ValueError, match='CLOSURE_CHANGED'):
        closed_and_verified(tmp_path, 'frozen')
