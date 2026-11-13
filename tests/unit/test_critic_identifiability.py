import copy
import numpy as np
import pytest

from app.critic_identifiability import ambiguity, baseline_predictions, select_origins, forked_state_test
from tests.unit.test_discrete_curve_native import curve_env


def test_conflict_and_compatible_action_sets_and_shift_invariance():
    y=np.array([[0.,2.,0.],[0.,0.,2.]])
    a=ambiguity(y)
    assert a['common_optimal_actions']==[] and a['minimum_shared_mean_regret']==1.
    assert a['conflicting_pairs']==1
    shifted=ambiguity(y+np.array([[10.],[-7.]]))
    assert shifted['minimum_shared_mean_regret']==a['minimum_shared_mean_regret']
    assert ambiguity([[0,2,2],[0,0,2]])['common_optimal_actions']==[2]
    assert ambiguity([[0,0,0]])['minimum_shared_mean_regret']==0
    with pytest.raises(ValueError):ambiguity([[np.nan,0,0]])


def test_baselines_never_consume_held_labels_and_report_missing_states():
    records=[dict(role='fit' if i<5 else 'monitor',year=2020 if i<5 else 2023,
        kind='flat_both_allowed' if i<6 else 'open_long',
        outcomes=[dict(reward_return=float(i+j)) for j in range(3)]) for i in range(7)]
    x=np.arange(7.).reshape(7,1,1)
    p,d=baseline_predictions(x,records,list(range(5)),[5,6])
    assert d['nearest_fit_record']==[4,4] and d['missing_state_predictions']==1
    changed=copy.deepcopy(records)
    for r in changed[5:]:
        for o in r['outcomes']:o['reward_return']=float('nan')
    p2,d2=baseline_predictions(x,changed,list(range(5)),[5,6])
    for k in p:np.testing.assert_equal(p[k],p2[k])
    assert d==d2
    with pytest.raises(ValueError):baseline_predictions(x,records,[0,5],[5,6])


def test_origin_selection_ignores_targets():
    rows=[dict(timestamp=i,year=2020+i//6,kind='flat_both_allowed') for i in range(24)]
    a,c=select_origins(rows,list(range(24)))
    assert len(a)==16 and all(v['selected']==4 for v in c.values())
    assert (a,c)==select_origins([dict(r,future_profit=999) for r in rows],list(range(24)))


def test_native_forks_use_same_real_market_and_leave_parent_untouched(curve_env):
    from app.three_signal_learning import synthetic_env
    from app.decision_attribution import state_view
    from tests.unit.test_double_tcn_probe import probe_recipe
    env,regime=synthetic_env(probe_recipe(),curve_env.risk_config,21,1,.99)
    env.reset()
    while not env.engine.entry_gate(env.bar)[0]:env.step(0)
    before=state_view(env)
    result=forked_state_test(env,np.where(regime>0,1,np.where(regime<0,2,0)),repeat=True)
    assert result['status']=='COMPLETE' and result['same_market_tensor'] and result['repeat_checked']
    assert state_view(env)==before
    assert len(result['outcomes'])==3 and all(len(v)==3 for v in result['outcomes'])
    assert all(o['parent_unchanged'] for row in result['outcomes'] for o in row)
