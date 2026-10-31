import json
import pytest

from tools.diagnose_curve_closes import summarize, check_record, diagnose
from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401


def row(delta, regret, hours=1., direction=1, preempted=False):
    return dict(delta_vs_hold=delta, regret_risk=regret,
                delta_vs_hold_risk=delta / 10., holding_hours=hours,
                direction=direction, preempted=preempted, hold_reason='SL',
                immediate=2., risk=10., alternatives=[])


def test_signs_preemption_and_no_portfolio_claim():
    result = summarize([row(2., 0.), row(-3., .5, direction=-1),
                        row(0., .2, hours=2.), row(0., 0., preempted=True)])
    assert result['actual_closes'] == 3 and result['preempted'] == 1
    assert result['helpful'] == result['harmful'] == result['tied'] == 1
    assert result['at_first_hour'] == 2
    assert result['sum_position_delta_usd_not_portfolio_profit'] == -1.
    assert result['mean_regret_risk'] == pytest.approx(.7 / 3)
    assert summarize([])['harmful_fraction'] is None


def test_action_trace_mismatch_rejected():
    record = dict(timestamp=123, route=1, action=1)
    check_record(record, 123, 1, [True, True, False])
    with pytest.raises(ValueError, match='BASELINE_ACTION_TRACE'):
        check_record(record, 124, 1, [True, True, False])
    with pytest.raises(ValueError, match='BASELINE_ACTION_TRACE'):
        check_record(record, 123, 0, [True, True, False])
    with pytest.raises(ValueError, match='BASELINE_ACTION_TRACE'):
        check_record(record, 123, 1, [True, False, False])


@pytest.mark.parametrize('opening', [1, 2])
def test_native_complete_replay_and_bad_ledger_rejected(request, opening):
    env = request.getfixturevalue('curve_env')
    obs = env.reset()
    actions = []
    while True:
        route = int(obs['route'][0])
        action = 1 if route else opening if obs['mask'][opening] else 0
        row = dict(timestamp=int(env.engine.v[env.bar]) + 3600 * 10**9,
                   route=route, action=action, probabilities=[0., 1., 0.])
        obs, reward, done, duration = env.step(action)
        row.update(reward=reward, duration=duration)
        actions.append(row)
        if done:
            break
    archived = json.loads(json.dumps(env.engine.trades, default=str))
    report = diagnose(env, actions, archived)
    assert report['baseline_exact']
    assert report['summary']['actual_closes'] > 0
    assert all(r['regret_risk'] >= 0 for r in report['rows'])
    with pytest.raises(ValueError, match='BASELINE_LEDGER_NOT_EXACT'):
        diagnose(env, actions, [])
