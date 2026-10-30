import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.pragmatic_close_attribution import (SuppressOneClose,compare_close,
    compact_account,targets,verify_baseline)
from app.pragmatic_strategy import ControllerConfig,PragmaticWorld,run_account


@pytest.mark.parametrize('side',[1,-1])
def test_exact_override_preserves_controller_state(side):
    class Policy:
        calls=0
        def decide(self,*args):
            self.calls+=1
            return dict(action='CLOSE',horizon=6,detail='unchanged')
    target=dict(bar=20,entry_bar=15,direction=side)
    p=Policy();wrapped=SuppressOneClose(p,target)
    e=SimpleNamespace(next_decision=20,pos=SimpleNamespace(entry_bar=15,direction=side))
    assert wrapped.decide(None,e,[],6)==dict(action='HOLD',horizon=6,detail='unchanged')
    assert p.calls==1 and wrapped.hits==1
    wrapped.verify()
    e.next_decision=21
    assert wrapped.decide(None,e,[],6)['action']=='CLOSE'
    assert p.calls==2 and wrapped.hits==1
    e.next_decision=20
    with pytest.raises(ValueError,match='DUPLICATE'): wrapped.decide(None,e,[],6)


def test_wrong_target_and_nonclose_rejected():
    p=SimpleNamespace(decide=lambda *a:dict(action='HOLD',horizon=3))
    e=SimpleNamespace(next_decision=20,pos=SimpleNamespace(entry_bar=15,direction=1))
    w=SuppressOneClose(p,dict(bar=20,entry_bar=15,direction=1))
    with pytest.raises(ValueError,match='NOT_CLOSE'): w.decide(None,e,[],3)
    w=SuppressOneClose(p,dict(bar=21,entry_bar=15,direction=1))
    with pytest.raises(ValueError,match='NOT_VISITED'): w.verify()
    w=SuppressOneClose(p,dict(bar=20,entry_bar=14,direction=1))
    with pytest.raises(ValueError,match='POSITION'): w.decide(None,e,[],3)


def account(net,trade_net,action,dd=2.):
    return dict(net_usd=net,drawdown_pct=dd,orders=2,return_pct=net/100,
        trace=[dict(bar=0,action='BUY'),dict(bar=1,action=action)],
        ledger=[dict(entry_bar=1,direction=1,net=trade_net,reason='AGENT_CLOSE',units=10,q=.5)])


def test_local_account_decomposition_and_prefix_refusal():
    a=account(20,5,'CLOSE');b=account(14,9,'HOLD',dd=3.)
    target=dict(bar=1,entry_bar=1,direction=1)
    result=compare_close(a,b,target)
    assert result['position_delta_usd']==-4
    assert result['continuation_delta_usd']==10
    assert result['account_delta_usd']==6
    assert result['drawdown_advantage_pp']==1
    assert result['pareto_relation']=='CLOSE_DOMINATES'
    b['trace'][0]['action']='SELL'
    with pytest.raises(ValueError,match='PREFIX'): compare_close(a,b,target)


def test_tradeoff_not_dominance_and_no_fake_aggregate_gain():
    a=account(20,5,'CLOSE',dd=4.);b=account(14,9,'HOLD',dd=3.)
    assert compare_close(a,b,dict(bar=1,entry_bar=1,direction=1))['pareto_relation']=='TRADEOFF'


def test_targets_exclude_preempted_and_baseline_tampering():
    a=dict(close_rows=[dict(bar=1,entry_bar=0,direction=1,preempted=False),
                       dict(bar=3,entry_bar=2,direction=-1,preempted=True)],
           close_reasons={'AGENT_CLOSE':1})
    assert targets(a)==[dict(bar=1,entry_bar=0,direction=1)]
    verify_baseline(a,copy.deepcopy(a))
    with pytest.raises(ValueError,match='BASELINE'): verify_baseline(a,dict(a,extra=1))
    with pytest.raises(ValueError,match='COUNT'): targets(dict(a,close_reasons={'AGENT_CLOSE':2}))


def test_receipt_authentication_and_completed_unit_reuse(tmp_path,monkeypatch):
    from tools import attribute_pragmatic_closes as runner
    from tools.run_crossed_account_queue import save
    from tools.optimize_dynamic_account import digest
    from app.zigzag_opportunities import sha
    case=tmp_path/'case.json';save(case,'case',{'net':2})
    original=sha(case)
    assert runner.authenticated(case,original)=={'net':2}
    job=dict(output=str(tmp_path),name='UNIT',manifest={},engine_root='unused')
    result=dict(cases={'case.json':original},rows=[])
    save(tmp_path/'UNIT.json',digest(job),result)
    monkeypatch.setattr(runner,'check_code',lambda *a:None)
    monkeypatch.setattr(runner,'run_account',lambda *a,**kw:pytest.fail('completed unit reran'))
    assert runner.unit(job)==result
    save(case,'case',{'net':3})
    with pytest.raises(ValueError,match='HASH'): runner.authenticated(case,original)
    with pytest.raises(ValueError,match='CASE_CHANGED'): runner.unit(job)


@pytest.fixture
def risk():
    root=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root: pytest.skip('native engine required')
    from app.zigzag_opportunities import engine
    engine(Path(root))
    return json.loads((Path(__file__).resolve().parents[2]/
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())


def test_native_observer_and_one_tick_continuation_cold_parity(risk):
    from app.crossed_account_panel import panel_specs
    from app.pragmatic_strategy import PragmaticController
    spec=dict(panel_specs('calibration',draws=1)[0],seed=950001)
    world=PragmaticWorld(spec,risk,'ideal');config=ControllerConfig('simple')
    baseline=run_account(world,config,diagnose=True)
    observer=SuppressOneClose(PragmaticController(config),None)
    assert run_account(world,config,diagnose=True,controller=observer)==baseline
    observer.verify()
    target=targets(baseline)[0]
    wrapper=SuppressOneClose(PragmaticController(config),target)
    alternate=run_account(world,config,controller=wrapper);wrapper.verify()
    comparison=compare_close(baseline,alternate,target)
    assert comparison['account_delta_usd']==pytest.approx(
        comparison['position_delta_usd']+comparison['continuation_delta_usd'])
    cold=SuppressOneClose(PragmaticController(config),target)
    replay=run_account(PragmaticWorld(spec,risk,'ideal'),config,controller=cold);cold.verify()
    assert replay==alternate
    compact=compact_account(alternate)
    assert compact['ledger']==alternate['ledger']
    assert all('planning' not in row for row in compact['trace'])
