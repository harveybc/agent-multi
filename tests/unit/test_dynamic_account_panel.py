import json
import copy
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest

from app.zigzag_opportunities import engine
from app.dynamic_account_panel import World, BASE, evaluate, expected_duration, HOUR
from app.close_hold_controller import PositionView, PathReport


@pytest.fixture
def world():
    root=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root: pytest.skip('set STRATEGY_ENGINE_ROOT to the native engine checkout')
    engine(Path(root))
    config=json.loads((Path(__file__).resolve().parents[2]/
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
    return World(dict(seed=7,pattern='overlap',volatility='increasing',mirror=1),config)


def test_native_cache_cash_calendar_and_replay(world):
    a=world.run(BASE,verify=True,details=True)
    assert a==world.run(BASE,verify=True,details=True)
    assert a['orders']>0
    assert a['net_usd']==pytest.approx(sum(x['net'] for x in a['ledger']))
    assert all(12<=x['entry_bar']<108 and x['exit_bar']<=108 for x in a['ledger'])
    assert all(0<x['q']<=1 for x in a['ledger'])
    assert world.env.pos is None and not world.env.trades
    assert a['cache_max_error']<1e-10


def test_nop_and_size_only_preserve_actions(world):
    nop=evaluate([world],BASE,enabled=set(),verify=True)
    assert nop['objectives']==[0.,0.] and not nop['eligible']
    large=world.run(BASE,details=True)
    g=list(BASE);g[8]=.25
    small=world.run(g,details=True)
    assert large['trace']==small['trace']
    assert all(a['units']>b['units'] for a,b in zip(large['ledger'],small['ledger']))


def test_mirror_and_prefix_no_hidden_regime(world):
    spec=dict(world.spec,mirror=-1)
    other=World(spec,world.risk_config)
    assert np.allclose(world.env.c+other.env.c,2.2)
    assert np.allclose(world.report(20).paths[:,:,1]+other.report(20).paths[:,:,2],2.2)
    before=world.past_efficiency(20)
    old=world.env.c
    world.env.c=old.copy()
    world.env.c[21:]+=1
    assert world.past_efficiency(20)==before
    world.env.c=old


def test_duration_uses_first_passage_not_nominal_horizon():
    p=PositionView('x',1,1.,1.,.9,1.2,.1,0.,0.,0.,0.,0.)
    paths=np.array([[[1.,1.01,.85,.88],[.88,.89,.87,.88]],
                    [[1.,1.05,.99,1.02],[1.02,1.06,1.,1.04]]])
    r=PathReport(0,0,0,HOUR,2*HOUR,paths,np.zeros((2,2)),np.array([.5,.5]),'synthetic_ideal',np.zeros(2))
    assert expected_duration(p,r,2)==1.5


def test_warm_bid_cache_rejects_stale_and_short_report(world):
    world.candidate_bids(20)
    r=world.report(20)
    world.reports[20]=replace(r,issued_at=r.issued_at-HOUR)
    with pytest.raises(ValueError,match='STALE_OR_FUTURE'): world.candidate_bids(20)
    world.reports[20]=replace(r,paths=r.paths[:,:-1],spreads=r.spreads[:,:-1],deadline=r.deadline-HOUR)
    with pytest.raises(ValueError,match='EXTERNAL_DEADLINE'): world.candidate_bids(20)


def test_warm_close_cache_binds_geometry_report_and_cost(world):
    e=copy.deepcopy(world.env)
    while e.next_decision<11: e.step(e.next_decision,0)
    e.step(11,1,entry_plan=world.entry_plan(3,1,1.))
    assert e.pos is not None
    b=e.next_decision
    original=world.close_values(e,3,b)
    e.pos.entry_price+=.0001
    with pytest.raises(AssertionError,match='CACHE_GEOMETRY'): world.close_values(e,3,b)
    e.pos.entry_price-=.0001
    r=world.report(b)
    world.reports[b]=replace(r,paths=r.paths+.0001)
    changed=world.close_values(e,3,b)
    assert not np.array_equal(original[1],changed[1])
    f=world.env.cfg.profile
    world.env.cfg=replace(world.env.cfg,profile=replace(f,slippage_market=f.slippage_market+.00001))
    with pytest.raises(ValueError,match='CACHE_EXECUTION_CONFIG'): world.close_values(e,3,b)
    e.cfg=world.env.cfg
    cost=world.close_values(e,3,b)
    assert not np.array_equal(cost[1],changed[1])
    world.reports[b]=replace(r,issued_at=r.issued_at-HOUR)
    with pytest.raises(ValueError,match='STALE_OR_FUTURE'): world.close_values(e,3,b)


@pytest.mark.parametrize('verify',[False,True])
def test_full_panel_rejects_stale_reports_even_with_verification(world,monkeypatch,verify):
    original=world.report
    monkeypatch.setattr(world,'report',lambda b: original(b-1))
    with pytest.raises(ValueError,match='STALE_OR_FUTURE'): world.run(BASE,verify=verify)


@pytest.mark.parametrize('direction',[1,-1])
@pytest.mark.parametrize('mode',['gap_close','ambiguous','through','deadline_spread'])
def test_native_protection_precedence_and_deadline_spread(world,direction,mode):
    from app.absolute_exit_corpus import create_case, specs, _funding
    from app.close_hold_verification import build_report, position_view
    from app.close_hold_controller import value_paths
    spec=dict(next(specs()),direction=direction)
    e,risk=create_case(spec,world.risk_config)
    b=e.next_decision;k=b+1;p=e.pos;f=e.cfg.profile
    mid=p.entry_price
    for name in ('o','h','l','c'):
        values=getattr(e,name).copy();values[k:]=mid;setattr(e,name,values)
    if mode=='deadline_spread':
        f=replace(f,spread_schedule=((0,10,.00005),(10,24,.0004)))
        e.cfg=replace(e.cfg,profile=f)
    spread=f.spread_at(int(e.v[k]),int(e.mo[p.seg]))
    if mode=='gap_close':
        for name in ('o','h','l','c'): getattr(e,name)[k:]=p.sl-direction*.001+direction*spread/2
    elif mode=='ambiguous':
        e.h[k]=max(p.sl,p.tp)+.001+direction*spread/2
        e.l[k]=min(p.sl,p.tp)-.001+direction*spread/2
    elif mode=='through':
        edge=p.tp+direction*f.limit_through+direction*spread/2
        if direction==1: e.h[k]=edge
        else: e.l[k]=edge
    r=build_report(e,'ideal',0)
    times=r.first_fill+np.arange(len(r.paths[0]))*HOUR
    fo=np.array([_funding(e,p,int(t))/p.units for t in times])
    fe=np.array([_funding(e,p,int(t)+HOUR)/p.units for t in times])
    expected=value_paths(position_view(e,risk),r,fo,fe)[0,-1]
    e.step(b,3 if mode=='gap_close' else 0)
    while e.pos is not None: e.step(e.next_decision,0)
    reason={'gap_close':'SL','ambiguous':'SL','through':'TP','deadline_spread':'FORCED_FLAT'}[mode]
    assert e.trades[0]['close_reason']==reason
    assert e.trades[0]['net_usd']==pytest.approx(expected,abs=1e-9)
