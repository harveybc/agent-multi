import numpy as np
import pytest

from app.refinement_decision_audit import boundary_terms, classify_event, mechanism_summary
from tests.unit.test_anchored_action_attribution import fixture


def test_weight_bias_and_posterior_reconstruct_boundary():
    h=np.array([2.,3.]);pw=np.zeros((3,2));pb=np.array([.01,-1.,0.])
    w=pw.copy();w[2,0]=.02;b=pb.copy();b[2]+=.03
    logits=w@h+b;logits[2]+=.001
    r=boundary_terms(0,2,h,pw,pb,w,b,pb,logits)
    assert r['weight_term']==pytest.approx(.04)
    assert r['bias_term']==pytest.approx(.03)
    assert r['posterior_term']==pytest.approx(.001)
    assert r['log_odds_change']==pytest.approx(.071)
    with pytest.raises(ValueError,match='PARENT'):
        boundary_terms(0,2,h,pw,pb,w,b,pb+1,logits)


def test_soft_target_is_not_confused_with_cash_or_plain_reward():
    e=fixture();e['isolated_state']=dict(position=None)
    e['soft_continuations']=[dict(seed=s,old=dict(soft_return=1.,reward_return=.1),
        new=dict(soft_return=.7,reward_return=.2)) for s in (7,19,43)]
    r=classify_event(e,np.array([.01,-1.,0.]),np.array([0.,-1.,.01]),
        np.array([[0.,-.1,.2],[.1,-.2,.3]]),.005,1,{})
    assert r['q_soft_all_draws_conflict']
    assert not r['q_reward_sign_conflict']
    assert r['calibration_position_mismatch']
    assert r['outcome']=='harmful'
    assert mechanism_summary([r])['harmful_counts']['q_soft_all_draws_conflict']==1


def test_mixed_draws_are_not_all_draws_conflict():
    e=fixture();e['isolated_state']=dict(position=None)
    e['soft_continuations']=[dict(seed=s,old=dict(soft_return=1.,reward_return=0.),
        new=dict(soft_return=2. if s==19 else .5,reward_return=0.)) for s in (7,19,43)]
    r=classify_event(e,np.array([.01,-1.,0.]),np.array([0.,-1.,.01]),
        np.array([[0.,-.1,.2],[.1,-.2,.3]]),.005,0,{})
    assert not r['q_soft_all_draws_conflict']
    assert not r['all_soft_draws_favor_change']
    assert not r['all_soft_draws_oppose_change']


def test_soft_continuations_cache_and_cash_identity(curve_env,tmp_path):
    from tests.unit.test_double_tcn_probe import probe_recipe
    from app.three_signal_learning import synthetic_env,evaluate
    from tools.attribute_ranking_improvement import attribute_pair
    def make():return synthetic_env(probe_recipe(),curve_env.risk_config,17,1,.99)[0]
    env=make();n=len(env.inputs.valid)
    a=np.tile([3.,0.,0.],(n,1));b=a.copy();b[::11]=[0.,3.,0.]
    evaluate(make(),a,save=tmp_path/'a');evaluate(make(),b,save=tmp_path/'b')
    policies={s:(np.zeros(n,dtype=int),np.tile([.5,.25,.25],(n,1))) for s in (7,19,43)}
    cache={}
    run=lambda:attribute_pair(make,env.inputs.row_ts_ns,tmp_path/'a',tmp_path/'b',
        lambda **kw:None,soft_policies=policies,alpha=.02,isolated_cache=cache,cache_namespace='fixture')
    first=run();second=run()
    assert first==second and cache
    assert first['reconciliation']['error_usd']==pytest.approx(0.,abs=1e-7)
    assert all(len(e['soft_continuations'])==3 for e in first['events'])
    assert any(e['soft_continuations'][0]['old']['soft_return']!=e['soft_continuations'][0]['old']['reward_return']
               for e in first['events'])
    with pytest.raises(ValueError,match='NAMESPACE'):
        attribute_pair(make,env.inputs.row_ts_ns,tmp_path/'a',tmp_path/'b',lambda **kw:None,isolated_cache={})


from tests.unit.test_discrete_curve_native import curve_env
