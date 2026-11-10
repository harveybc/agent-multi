import numpy as np
import pytest
import torch

from app.flat_opening_diagnostic import select_panel, sac_gradients, project_gradient, compare_pair


def test_panel_controls_do_not_depend_on_profit_or_event_order():
    traces=[dict(timestamp=i,year=2021,position=0,signal=0) for i in range(10)]
    events=[dict(timestamp=3,isolated_position=0,outcome='harmful'),
            dict(timestamp=7,isolated_position=0,outcome='helpful')]
    a=select_panel(traces,events,{3,7,4})
    assert [x['timestamp'] for x in a]==[2,3,6,7]
    assert select_panel(traces,list(reversed(events)),{3,7,4})==a
    events[0]['outcome']='neutral'
    assert select_panel(traces,events,{3,7,4})==a
    assert len({x['timestamp'] for x in a})==len(a)


def test_gradient_matches_autograd_and_finite_difference():
    rng=np.random.default_rng(7)
    z=rng.normal(size=(8,3));q=rng.normal(size=(8,3));h=rng.normal(size=(8,4))
    g=sac_gradients(z,q,.04)
    t=torch.tensor(z,requires_grad=True);qt=torch.tensor(q)
    lp=t.log_softmax(-1);loss=(lp.exp()*(.04*lp-qt)).sum(-1).sum()
    loss.backward()
    assert np.allclose(g['total'],t.grad.numpy(),atol=1e-12)
    assert np.allclose(g['total'],g['q']+g['entropy'])
    d=project_gradient(g['total'],h,2,0,1)
    w=g['total'].T@h/len(h);b=g['total'].mean(0)
    eps=1e-6
    altered=z[2]-eps*(w@h[2]+b)
    assert d['shared']==pytest.approx(((altered[1]-altered[0])-(z[2,1]-z[2,0]))/eps,abs=1e-9)
    assert d['local']==pytest.approx(-(g['total'][2,1]-g['total'][2,0])*(h[2]@h[2]+1))


def test_zero_gradient_and_bad_shape():
    assert np.allclose(sac_gradients(np.zeros((4,3)),np.ones((4,3)),.1)['total'],0)
    with pytest.raises(ValueError):sac_gradients(np.zeros((3,2)),np.ones((3,2)),.1)


def test_state_flip_is_distinct_from_matching_state_critic_error():
    a=np.array([[1.,0.,-1.]]*3);b=np.array([[0.,1.,-1.]]*3)
    r=compare_pair([0.,.5,-1.],a,b,0,1)
    assert r['state_order_flip'] and r['flat_q_conflict'] and not r['calibration_q_conflict']
    r=compare_pair([0.,.5,-1.],a,a,0,1)
    assert not r['state_order_flip'] and r['calibration_q_conflict']


def test_native_two_real_histories_and_snapshot_immutability(curve_env):
    from tests.unit.test_double_tcn_probe import probe_recipe
    from app.three_signal_learning import synthetic_env
    from app.decision_attribution import continuation,state_view
    a,_=synthetic_env(probe_recipe(),curve_env.risk_config,17,1,.99)
    b,_=synthetic_env(probe_recipe(),curve_env.risk_config,17,1,.99)
    a.reset();b.reset()
    while not a.engine.entry_gate(a.bar)[0]:a.step(0);b.step(0)
    a.step(0);b.step(1)
    assert a.bar==b.bar and a.engine.pos is None and b.engine.pos is not None
    tape=np.zeros(len(a.inputs.valid),dtype=int);p=np.tile([.5,.25,.25],(len(tape),1))
    before=[state_view(x) for x in (a,b)]
    flat=continuation(a,tape,2,p,.02);held=continuation(b,tape,2,p,.02)
    assert flat['first_effect']['entries']==1
    assert held['first_effect']['closes']==1
    assert [state_view(x) for x in (a,b)]==before


from tests.unit.test_discrete_curve_native import curve_env
