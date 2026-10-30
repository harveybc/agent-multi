import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.pragmatic_strategy import (ControllerConfig,rank_openings,select_plan,
    PragmaticWorld,shadow_environment,rollout_plan,PragmaticController,run_account,defined_sharpe)


def bid(h,d,gain,duration=6):
    return dict(horizon=h,direction=d,gain_risk=gain,uncertainty_risk=.1,duration=duration)


@pytest.mark.parametrize('daily',[[100.,100.],[110.,121.],[100.]])
def test_constant_return_sharpe_is_undefined_not_zero(daily):
    assert defined_sharpe(0.,daily,100.) is None
    assert defined_sharpe(0.,[110.,99.],100.)==0.


def test_joint_ranking_ties_nop_and_time_units():
    c=ControllerConfig('simple',alpha=0)
    bids=[bid(3,1,1,3),bid(24,-1,2,24)]
    assert rank_openings(bids,c)[0]['direction']==-1
    assert rank_openings(bids[::-1],c)==rank_openings(bids,c)
    assert rank_openings(bids,replace(c,alpha=1))[0]['direction']==1
    assert rank_openings([bid(3,1,0)],c)==[]
    assert rank_openings([bid(3,1,1),bid(3,-1,1)],c)==[]
    with pytest.raises(ValueError): rank_openings([bid(3,1,float('nan'))],c)
    with pytest.raises(ValueError): ControllerConfig('simple',tau0=0)
    with pytest.raises(ValueError): ControllerConfig('simple',persistence=0)
    with pytest.raises(ValueError): ControllerConfig('unknown')


def test_common_plan_selection_not_per_sample_oracle():
    plans=[('BUY:3',3,'NOP'),('SELL:3',3,'NOP'),('NOP',None,'NOP')]
    # Each directional plan wins in a different future; neither beats cash in expectation.
    samples=[np.array([[10,1],[-12,2]]),np.array([[-12,2],[10,1]]),np.zeros((2,2))]
    choice=select_plan(plans,samples,np.array([.5,.5]),ControllerConfig('sequential'))
    assert choice['plan']==plans[2]
    assert choice['score']==0


@pytest.fixture
def risk():
    root=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root: pytest.skip('native engine required')
    from app.zigzag_opportunities import engine
    engine(Path(root))
    return json.loads((Path(__file__).resolve().parents[2]/
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())


def make_world(risk,mode='ideal'):
    from app.crossed_account_panel import panel_specs
    spec=dict(panel_specs('calibration',draws=1)[0],seed=930001)
    return PragmaticWorld(spec,risk,mode)


def test_shadow_native_parity_and_no_true_future(risk):
    from app.sequential_account_pareto import fixture,clone
    w,e,eq=fixture(risk,'reversal',1,1,steps=4)
    # Use the report wrapper on the fixture's exact native arrays.
    p=make_world(risk);p.env=e;p.end=w.end;p.reports.clear()
    b=e.next_decision;r=p.report(b)
    shadow=shadow_environment(e,r,0)
    assert np.array_equal(shadow.o[b+1:p.end],e.o[b+1:p.end])
    observed=rollout_plan(p,e,eq,r,('CLOSE',None,'SELL:6'))
    e.o=e.o.copy();e.h=e.h.copy();e.l=e.l.copy();e.c=e.c.copy()
    for a in (e.o,e.h,e.l,e.c): a[b+1:]+=10
    replay=rollout_plan(p,e,eq,r,('CLOSE',None,'SELL:6'))
    assert np.array_equal(observed,replay)
    assert e.pos is not None and not e.trades
    # Pure HOLD has the native result for the supplied ideal scenario.
    passive=clone(shadow);equity=list(eq)
    while passive.next_decision<passive.last:
        k=passive.next_decision;passive.step(k,0);equity.append(passive.equity_at(k+1))
    from app.dynamic_account_arbiter import objectives
    expected=objectives(equity,e.cfg.capital)
    assert rollout_plan(p,e,eq,r,('HOLD',None,'NOP'))[0]==pytest.approx(expected)


def test_causal_reports_and_decisions_prefix_invariant(risk):
    from app.sequential_account_pareto import clone
    a=make_world(risk,'causal');b=make_world(risk,'causal')
    tick=20
    for name in ('o','h','l','c'):
        x=getattr(b.env,name).copy();x[tick+1:]+=10;setattr(b.env,name,x)
    ra,rb=a.report(tick),b.report(tick)
    assert ra.provenance==rb.provenance=='causal'
    assert np.array_equal(ra.paths,rb.paths)
    for mode in ('simple','predictive_proxy','sequential'):
        c=ControllerConfig(mode,close_offsets=(3,))
        ea,eb=clone(a.env),clone(b.env);ea.reset();eb.reset()
        while ea.next_decision<tick:
            ea.step(ea.next_decision,0);eb.step(eb.next_decision,0)
        x=PragmaticController(c).decide(a,ea,[ea.cfg.capital],None)
        y=PragmaticController(c).decide(b,eb,[eb.cfg.capital],None)
        assert x==y


@pytest.mark.parametrize('side',[1,-1])
@pytest.mark.parametrize('favorable',[False,True])
def test_ideal_close_protects_both_directions(risk,side,favorable):
    from app.sequential_account_pareto import fixture
    w,e,eq=fixture(risk,'smooth',side if favorable else -side,side,steps=4)
    p=make_world(risk);p.env=e;p.end=w.end;p.reports.clear()
    c=PragmaticController(ControllerConfig('simple',persistence=2,urgent_risk=0))
    decision=c.decide(p,e,eq,12)
    assert decision['action']==('HOLD' if favorable else 'CLOSE')


def test_native_account_and_replay_no_state_observation(risk):
    w=make_world(risk)
    config=ControllerConfig('simple')
    a=run_account(w,config,diagnose=True)
    b=run_account(make_world(risk),config,diagnose=True)
    assert a==b
    assert a['orders']>0 and a['orders']==a['buys']+a['sells']
    assert a['market_weeks']==1 and a['orders_per_week']==a['orders']
    assert a['native_iso_orders_per_week']==a['orders']/2
    assert abs(a['cash_error'])<1e-7
    assert all(0<t['q']<=1 for t in a['ledger'])
    assert all(x['exit_bar']<y['entry_bar'] for x,y in zip(a['ledger'],a['ledger'][1:]))
    assert a['close_diagnostics']['harmful']==0


def test_runner_parity_with_existing_unchanged_controller(risk):
    from app.crossed_account_panel import CrossedWorld
    from app.dynamic_account_panel import BASE,HOUR
    from app.dynamic_account_arbiter import choose,decode
    world=make_world(risk,'noise_1')
    reference=CrossedWorld(world.spec,risk).run(BASE,details=True,verify=True)
    class ExistingController:
        def __init__(self): self.close=world.make_controller(decode(BASE))
        def decide(self,w,e,equity,horizon):
            b=e.next_decision
            if e.pos is None:
                bid=choose(CrossedWorld.candidate_bids(w,b),BASE,w.past_efficiency(b)) if e.entry_gate(b)[0] else None
                if bid: return dict(action='BUY' if bid['direction']==1 else 'SELL',horizon=bid['horizon'])
                return dict(action='NOP',horizon=horizon)
            if b+1>=w.end: return dict(action='HOLD',horizon=horizon)
            unit,values,_,_,binding=w.close_values(e,horizon,b);r=w.report(b)
            result=w.close_decision(self.close,unit,r,int(e.v[b])+HOUR,int(e.ff[e.pos.seg]),values,binding,b,horizon)
            return dict(action=result['action'],horizon=horizon)
    actual=run_account(world,ControllerConfig('simple'),controller=ExistingController())
    for name in ('return_pct','drawdown_pct','net_usd','orders','wins','buys','buy_wins','sell_wins','horizons','costs'):
        assert actual[name]==reference[name]
    assert actual['ledger']==reference['ledger']
    actions={'NOP':0,'HOLD':0,'BUY':1,'SELL':2,'CLOSE':3}
    assert [dict(t,action=actions[t['action']]) for t in actual['trace']]==reference['trace']


def test_cache_cannot_survive_execution_change(risk):
    world=make_world(risk,'causal');world.opening_rows(20)
    world.planned['sentinel']=np.zeros((1,2))
    world.env.cfg=replace(world.env.cfg,profile=replace(world.env.cfg.profile,commission_per_lot_side=7.))
    world.opening_rows(20)
    assert 'sentinel' not in world.planned


def test_two_orders_can_beat_a_long_hold_natively(risk):
    from app.sequential_account_pareto import fixture
    w,e,eq=fixture(risk,'smooth',1,0,steps=6)
    start=e.next_decision
    future=np.array([1.1000,1.1020,1.1020,1.1000,1.0985,1.0985])
    close=e.c.copy();close[start+1:w.end]=future
    op=e.o.copy();op[start+1:w.end]=np.r_[1.1,future[:-1]]
    hi=np.maximum(op,close)+.00001;lo=np.minimum(op,close)-.00001
    e.o,e.h,e.l,e.c=op,hi,lo,close
    p=make_world(risk);p.env=e;p.end=w.end;p.reports.clear()
    r=p.report(start)
    short=rollout_plan(p,e,eq,r,('BUY:3',2,'SELL:6'))
    long=rollout_plan(p,e,eq,r,('BUY:24',None,'NOP'))
    assert short[0,0]>long[0,0]
    selected=select_plan([('BUY:3',2,'SELL:6'),('BUY:24',None,'NOP'),('NOP',None,'NOP')],
        [short,long,np.zeros((1,2))],r.weights,ControllerConfig('sequential'))
    assert selected['plan']==('BUY:3',2,'SELL:6')
