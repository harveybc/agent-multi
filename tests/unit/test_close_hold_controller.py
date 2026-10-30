from dataclasses import replace
import json

import numpy as np
import pytest

from app.close_hold_controller import PositionView, PathReport, CloseController, CloseConfig, value_paths


def position(direction=1, sl=99, tp=106):
    return PositionView('p1',direction,100.,1.,sl,tp,2.,0.,0.,0.,0.,0.)


def report(prices=(100.,98.,104.), weights=None, tag='causal', time=10):
    a=np.asarray(prices,float)
    if a.ndim==1: a=a[None,:]
    paths=np.repeat(a[:,:,None],4,axis=2)
    return PathReport(time,time,time,1,time+a.shape[1],paths,np.zeros(a.shape),
                      np.ones(len(a))/len(a) if weights is None else np.array(weights),tag,np.zeros(len(a)))


def decide(p,r,cfg=None):
    return CloseController(cfg or CloseConfig()).decide(p,r,r.issued_at,
        np.zeros(r.paths.shape[1]),np.zeros(r.paths.shape[1]),expected_deadline=r.deadline)


@pytest.mark.parametrize('d',[1,-1])
def test_same_market_distinct_position_stops(d):
    r=report(np.array([100,98,104]) if d==1 else 200-np.array([100,98,104]))
    tight=position(d,100-d,100+6*d)
    wide=position(d,100-3*d,100+4*d)
    assert decide(tight,r)['action']=='CLOSE'
    assert decide(wide,r)['action']=='HOLD'


def test_tp_hold_and_gap_protection_tie():
    assert decide(position(),report([100,102,106]))['action']=='HOLD'
    assert decide(position(),report([98,97,96]))['action']=='HOLD'


def test_barrier_before_mean_and_one_common_wait_action():
    r=report([[100,98,104],[100,102,104]])
    values=value_paths(position(),r,np.zeros(3),np.zeros(3))
    assert values[:,2].tolist()==[-2.,4.]
    mean=report(np.mean(r.paths[:,: ,0],axis=0))
    assert value_paths(position(),mean,np.zeros(3),np.zeros(3))[0,2]==4.
    # E(max per path) is not max(E per available action).
    assert np.max(values,axis=1).mean() > np.max(values.mean(axis=0))


def test_uncertainty_penalty_can_block_marginal_close():
    r=report([[100,98],[100,101]],weights=[.75,.25])
    assert decide(position(),r,CloseConfig(uncertainty_penalty=0))['action']=='CLOSE'
    assert decide(position(),r,CloseConfig(uncertainty_penalty=2))['action']=='HOLD'


def test_persistence_urgency_gap_reset_and_serialization():
    c=CloseController(CloseConfig(persistence=2,urgent_risk=2))
    p=position()
    def call(t,prices=(100,98,97),pos=p):
        r=report(prices,time=t)
        return c.decide(pos,r,t,np.zeros(3),np.zeros(3),expected_deadline=t+3)
    assert call(10)['action']=='HOLD'
    restored=CloseController(c.config)
    restored.restore(json.loads(json.dumps(c.state())))
    r=report(time=11)
    assert c.decide(p,r,11,np.zeros(3),np.zeros(3),expected_deadline=14)==restored.decide(p,r,11,np.zeros(3),np.zeros(3),expected_deadline=14)
    assert call(12,(100,102,104))['action']=='HOLD'
    assert call(13)['action']=='HOLD'
    assert call(15)['action']=='HOLD'
    assert call(16,pos=replace(p,identity='p2'))['action']=='HOLD'
    assert decide(p,report(),CloseConfig(persistence=5,urgent_risk=.5))['action']=='CLOSE'


@pytest.mark.parametrize('field,value', [('issued_at',11),('observed_through',11),
    ('first_fill',11),('deadline',20),('provenance','synthetic_ideal')])
def test_bad_availability_support_and_oracle_plane_rejected(field,value):
    with pytest.raises(ValueError): decide(position(),replace(report(),**{field:value}))


def test_invalid_inputs_and_duplicate_tick():
    p=position();r=report();z=np.zeros(3)
    for bad in (replace(r,weights=np.array([0.])),replace(r,paths=r.paths*np.nan),
                replace(r,spreads=-np.ones((1,3)))):
        with pytest.raises(ValueError): decide(p,bad)
    for cfg in (dict(persistence=0),dict(uncertainty_penalty=-1),dict(margin_risk=float('nan'))):
        with pytest.raises(ValueError): CloseConfig(**cfg)
    c=CloseController(CloseConfig());c.decide(p,r,10,z,z,expected_deadline=13)
    with pytest.raises(ValueError,match='DUPLICATE'): c.decide(p,r,10,z,z,expected_deadline=13)


def test_fee_funding_units_and_scale_invariance():
    p=replace(position(),market_slip=.1,stop_slip=.2,entry_slip=.1,commission_roundtrip=.1)
    r=report([100,98,104]);opening=np.array([0.,.5,.5]);ending=np.array([.5,.5,.5])
    v=value_paths(p,r,opening,ending)
    assert v[0,0]==pytest.approx(-.3)
    assert v[0,-1]==pytest.approx(-2.9)
    a=CloseController(CloseConfig()).decide(p,r,10,opening,ending,expected_deadline=13)
    b=CloseController(CloseConfig()).decide(replace(p,units=10,risk=20),r,10,opening,ending,expected_deadline=13)
    assert a['advantage_risk']==pytest.approx(b['advantage_risk'])


def test_synthetic_value_cache_matches_report_and_cannot_bypass_causal_api():
    p=position(); r=report(tag='synthetic_ideal'); z=np.zeros(3)
    values=value_paths(p,r,z,z)
    a=CloseController(CloseConfig(),allow_synthetic=True).decide(p,r,10,z,z,expected_deadline=13)
    from app.close_hold_controller import bind_values
    b=CloseController(CloseConfig(),allow_synthetic=True).decide_synthetic_values(p.identity,10,1,values,r.weights,p.risk,report=r,expected_deadline=13,binding=bind_values(p,r,values))
    assert a['action']==b['action'] and a['advantage_risk']==b['advantage_risk']
    with pytest.raises(ValueError,match='SYNTHETIC'):
        CloseController().decide_synthetic_values(p.identity,10,1,values,r.weights,p.risk,report=r,expected_deadline=13)
