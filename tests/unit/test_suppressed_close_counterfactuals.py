import json
import os
from pathlib import Path

import pytest

from app.suppressed_close_counterfactuals import fixed_position_exits, SuppressedCloseWorld, compare_intervention


@pytest.fixture
def risk():
    root=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root: pytest.skip('native engine required')
    from app.zigzag_opportunities import engine
    engine(Path(root))
    return json.loads((Path(__file__).resolve().parents[2]/
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())


@pytest.mark.parametrize('side',[1,-1])
@pytest.mark.parametrize('favorable',[True,False])
def test_fixed_order_optimum_native_and_isolated(risk,side,favorable):
    from app.sequential_account_pareto import fixture,clone
    mirror=side if favorable else -side
    w,e,_=fixture(risk,'smooth',mirror,side,steps=4)
    before=(e.balance,e.next_decision,e.pos,tuple(e.trades))
    p=e.pos;budget=w.position(12,side,p.entry_price,p.entry_bar,p.units).risk
    row=fixed_position_exits(e,budget)
    assert before==(e.balance,e.next_decision,e.pos,tuple(e.trades))
    assert row['optimal']>=max(row['immediate'],row['hold'])
    assert (row['delta_vs_hold']<0)==favorable
    assert row['regret_risk']>=0
    passive=clone(e)
    while passive.pos is not None: passive.step(passive.next_decision,0)
    assert row['hold']==passive.trades[-1]['net_usd']
    assert row['hold_reason']==passive.trades[-1]['close_reason']


def test_flat_rejected_and_gap_preempts_close(risk):
    from app.sequential_account_pareto import fixture
    _,flat,_=fixture(risk,'smooth',1,0,steps=3)
    with pytest.raises(ValueError,match='OCCUPIED'): fixed_position_exits(flat,1.)
    w,e,_=fixture(risk,'smooth',1,1,steps=3)
    e.o=e.o.copy();e.o[e.next_decision+1]=e.pos.sl-.001
    row=fixed_position_exits(e,10.)
    assert row['preempted'] and row['immediate_reason']=='SL'
    assert row['immediate']==row['hold']==row['optimal']
    with pytest.raises(ValueError,match='RISK'): fixed_position_exits(e,0.)


def test_one_intervention_and_full_continuation(risk):
    from app.causal_history_account import HistoryWorld,base_genes,specifications
    spec=specifications('confirmation',smoke=True)[0]
    g=base_genes(0.,0.);g[9]=100.
    w=SuppressedCloseWorld(spec,risk,'median')
    baseline=w.run(g,details=True)
    assert baseline==HistoryWorld(spec,risk,'median').run(g,details=True)
    occupied=[t for t in baseline['trace'] if t['action'] in (1,2)]
    assert occupied
    trade=next(t for t in baseline['ledger'] if t['exit_bar']>t['entry_bar']+1)
    target=dict(bar=trade['entry_bar'],entry_bar=trade['entry_bar'],direction=trade['direction'])
    w.intervention=target
    alternative=w.run(g,details=True,verify=True)
    assert w.intervention_hits==1
    row=compare_intervention(baseline,alternative,target)
    assert row['account_delta_usd']==pytest.approx(row['position_delta_usd']+row['continuation_delta_usd'])
    assert row['target_close_reason']=='AGENT_CLOSE'
    w.intervention=None
    assert w.run(g,details=True)==baseline
    w.intervention=dict(target,bar=1000)
    with pytest.raises(ValueError,match='INTERVENTION_TARGET'): w.run(g,details=True)


def test_continuation_decomposition_keeps_opportunity_separate():
    a=dict(net_usd=10.,drawdown_pct=2.,orders=1,trace=[dict(bar=0,action=0)],
           ledger=[dict(entry_bar=0,direction=1,net=10.,reason='TP')])
    b=dict(net_usd=14.,drawdown_pct=1.,orders=2,trace=[dict(bar=0,action=3)],
           ledger=[dict(entry_bar=0,direction=1,net=6.,reason='AGENT_CLOSE'),
                   dict(entry_bar=2,direction=-1,net=8.,reason='TP')])
    r=compare_intervention(a,b,dict(bar=0,entry_bar=0,direction=1))
    assert r['position_delta_usd']==-4 and r['continuation_delta_usd']==8
    assert r['account_delta_usd']==4


def test_resumable_unit_rejects_changed_identity_and_payload(risk,tmp_path):
    from app.causal_history_account import HistoryWorld,base_genes,specifications
    from tools.diagnose_suppressed_closes import unit
    g=base_genes(0.,0.);g[9]=100.
    baseline=HistoryWorld(specifications('confirmation',smoke=True)[0],risk,'median').run(g,details=True)
    job=dict(world=0,output=str(tmp_path),engine_root=os.environ['STRATEGY_ENGINE_ROOT'],
             config=risk,mode='median',identity='test',policies=[dict(key='control',genes=g,archived=baseline)])
    result=unit(job)
    assert unit(job)==result
    assert result['control']['pairs']==[]
    with pytest.raises(ValueError): unit(dict(job,identity='changed'))
    p=tmp_path/'WORLD_000.json';payload=json.loads(p.read_text())
    payload['data']['control']['baseline_orders']+=1
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError): unit(job)
