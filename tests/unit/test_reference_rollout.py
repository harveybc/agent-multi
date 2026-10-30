import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.reference_rollout import (ContinuationWorld,ReferenceRolloutController,
    reference_rollout,select_reference,PassiveController)
from app.pragmatic_strategy import ControllerConfig,PragmaticController,PragmaticWorld,run_account


@pytest.fixture
def risk():
    native=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not native: pytest.skip('native engine required')
    from app.zigzag_opportunities import engine
    engine(Path(native))
    return json.loads((Path(__file__).resolve().parents[2]/
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())


def world(risk,reporter='ideal'):
    from app.crossed_account_panel import panel_specs
    return PragmaticWorld(dict(panel_specs('calibration',draws=1)[0],seed=1010001),risk,reporter)


def test_common_selection_cannot_choose_a_different_winner_per_sample():
    rows=[dict(action='BUY:3',reference='a',outcomes=[[10,0],[-12,0]]),
          dict(action='SELL:3',reference='a',outcomes=[[-12,0],[10,0]]),
          dict(action='NOP',reference='a',outcomes=[[0,0],[0,0]])]
    choice=select_reference(rows,np.array([.5,.5]),1.)
    assert choice['action']=='NOP' and choice['score']==0


def test_carried_forecast_preserves_ensemble_and_blocks_future_callback(risk):
    from app.pragmatic_strategy import shadow_environment
    w=world(risk,'causal');origin=20;r=w.report(origin)
    w.env.next_decision=origin
    view=ContinuationWorld(w,shadow_environment(w.env,r,0),r,origin)
    w.report=lambda b:pytest.fail('future real report requested')
    later=view.report(origin+2)
    assert np.array_equal(later.paths,r.paths[:,2:])
    assert np.array_equal(later.weights,r.weights)
    assert later.observed_through==r.observed_through
    assert later.first_fill==r.first_fill+2*r.step
    assert view.origin_issued_at==r.issued_at
    with pytest.raises(ValueError): view.report(origin-1)


@pytest.mark.parametrize('side',[1,-1])
def test_ideal_reference_continuation_is_native_and_isolated(risk,side):
    from app.sequential_account_pareto import fixture,clone
    from app.dynamic_account_arbiter import objectives
    from app.pragmatic_strategy import action_plan
    base,e,eq=fixture(risk,'reversal',side,side,steps=6)
    w=world(risk);w.env=e;w.end=base.end;w.reports.clear()
    cfg=ControllerConfig('simple',persistence=2);reference=PragmaticController(cfg)
    reference.decide(w,e,eq,12)
    saved=copy.deepcopy(reference.close.state());before=(e.balance,e.pos,tuple(e.trades),tuple(eq))
    values=reference_rollout(w,e,eq,12,'HOLD',reference,w.report(e.next_decision),{})
    direct=clone(e);history=list(eq);ref=copy.deepcopy(reference);horizon=12
    first=direct.next_decision;direct.step(first,0);history.append(direct.equity_at(first+1))
    while direct.next_decision<direct.last:
        b=direct.next_decision;decision=ref.decide(w,direct,history,horizon);horizon=decision['horizon']
        named=decision['action']+':'+str(horizon) if decision['action'] in ('BUY','SELL') else decision['action']
        a,p=action_plan(w,named);direct.step(b,a,entry_plan=p);history.append(direct.equity_at(b+1))
    assert values[0]==pytest.approx(objectives(history,direct.cfg.capital))
    assert before==(e.balance,e.pos,tuple(e.trades),tuple(eq))
    assert reference.close.state()==saved


def test_causal_root_action_independent_of_unobserved_actual_prices(risk):
    from app.sequential_account_pareto import clone
    a=world(risk,'causal');b=world(risk,'causal');tick=100
    for name in ('o','h','l','c'):
        values=getattr(b.env,name).copy();values[tick+1:]+=10;setattr(b.env,name,values)
    ea,eb=clone(a.env),clone(b.env);ea.reset();eb.reset()
    for e in (ea,eb):
        while e.next_decision<tick:e.step(e.next_decision,0)
    refs={'simple':ControllerConfig('simple')}
    ca=ReferenceRolloutController(refs);cb=ReferenceRolloutController(refs)
    da=ca.decide(a,ea,[ea.cfg.capital],None);db=cb.decide(b,eb,[eb.cfg.capital],None)
    assert da==db


def test_passive_has_no_hidden_entry_or_close():
    from types import SimpleNamespace
    p=PassiveController()
    assert p.decide(None,SimpleNamespace(pos=None),[],None)['action']=='NOP'
    assert p.decide(None,SimpleNamespace(pos=object()),[],12)['action']=='HOLD'


@pytest.mark.parametrize('side',[1,-1])
@pytest.mark.parametrize('initial',[-1,0,1])
def test_small_ideal_rollout_not_worse_than_included_reference(risk,side,initial):
    from app.sequential_account_pareto import fixture,clone
    from app.dynamic_account_arbiter import objectives
    from app.pragmatic_strategy import action_plan
    base,env,eq=fixture(risk,'oscillating',side,initial,steps=6)
    w=world(risk);w.env=env;w.end=base.end;w.reports.clear()
    cfg=ControllerConfig('simple',persistence=2)
    outcomes=[]
    for policy in (PragmaticController(cfg),ReferenceRolloutController({'simple':cfg})):
        e=clone(env);history=list(eq);h=12 if initial else None
        while e.next_decision<e.last:
            b=e.next_decision;d=policy.decide(w,e,history,h);h=d['horizon'];named=d['action']
            if named in ('BUY','SELL'): named+=':'+str(h)
            a,p=action_plan(w,named);e.step(b,a,entry_plan=p);history.append(e.equity_at(b+1))
        ret,dd=objectives(history,e.cfg.capital);outcomes.append(ret-dd)
    assert outcomes[1]>=outcomes[0]-1e-10


def test_snapshot_override_matches_cold_replay_and_keeps_memory(risk):
    from app.reference_rollout import CaptureCloses,resume_without_close
    from app.pragmatic_close_attribution import SuppressOneClose,targets
    w=world(risk);cfg=ControllerConfig('simple',persistence=2)
    observer=CaptureCloses(PragmaticController(cfg))
    baseline=run_account(w,cfg,diagnose=True,controller=observer)
    ts=targets(baseline);assert ts
    target=ts[0];snapshot=observer.snapshots[target['bar']]
    result=resume_without_close(w,snapshot)
    override=SuppressOneClose(PragmaticController(cfg),target)
    cold=run_account(w,cfg,controller=override);override.verify()
    for key in ('net_usd','return_pct','drawdown_pct','orders'): assert result[key]==cold[key]
    assert result['trace']==[r for r in cold['trace'] if r['bar']>=target['bar']]
    assert result['ledger']==[{k:v for k,v in r.items() if k!='horizon'} for r in cold['ledger']]
    assert resume_without_close(w,snapshot)==result


def test_reject_recursive_reference():
    with pytest.raises(ValueError,match='NONRECURSIVE'):
        ReferenceRolloutController({'bad':ControllerConfig('sequential')})


def test_reference_real_memory_advances_once_not_once_per_simulation(risk):
    from app.sequential_account_pareto import fixture
    base,e,eq=fixture(risk,'reversal',1,1,steps=6)
    w=world(risk);w.env=e;w.end=base.end;w.reports.clear()
    cfg=ControllerConfig('simple',persistence=2)
    direct=PragmaticController(cfg);direct.decide(w,e,eq,12)
    policy=ReferenceRolloutController({'simple':cfg});decision=policy.decide(w,e,eq,12)
    assert policy.references['simple'].close.state()==direct.close.state()
    assert {r['action'] for r in decision['plan']['candidates']}=={'HOLD','CLOSE'}
