import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.pragmatic_plan_gap import committed_replay,check_valuation,decompose,first_deviation


@pytest.fixture
def risk():
    root=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root: pytest.skip('native engine required')
    from app.zigzag_opportunities import engine
    engine(Path(root))
    return json.loads((Path(__file__).resolve().parents[2]/
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())


@pytest.mark.parametrize('side',[1,-1])
@pytest.mark.parametrize('first',['HOLD','CLOSE'])
def test_committed_native_matches_ideal_shadow_without_mutation(risk,side,first):
    from app.sequential_account_pareto import fixture
    from app.pragmatic_strategy import PragmaticWorld,rollout_plan
    from app.crossed_account_panel import panel_specs
    w,e,eq=fixture(risk,'reversal',side,side,steps=6)
    p=PragmaticWorld(dict(panel_specs('calibration',draws=1)[0],seed=980001),risk,'ideal')
    p.env=e;p.end=w.end;p.reports.clear()
    second='SELL:6' if side==1 else 'BUY:6';plan=(first,2,second)
    before=(e.balance,e.next_decision,e.pos,tuple(e.trades),tuple(eq))
    expected=rollout_plan(p,e,eq,p.report(e.next_decision),plan)[0]
    native=committed_replay(p,e,eq,plan)
    assert np.allclose([native['return_pct'],native['drawdown_pct']],expected,atol=1e-10,rtol=0)
    assert before==(e.balance,e.next_decision,e.pos,tuple(e.trades),tuple(eq))
    assert abs(native['cash_error'])<1e-7
    assert native['trace'] and native['flat']


def test_exact_decomposition_and_ranking_reversal():
    row=lambda ret,dd:dict(return_pct=ret,drawdown_pct=dd)
    result=decompose(row(5,1),row(4,1),row(3,1),row(6,1),1)
    assert result['planned_margin']==1
    assert result['actual_margin']==-3
    assert result['close_continuation_shift']==-2
    assert result['hold_continuation_shift']==2
    assert result['ranking_reversal'] and result['identity_error']==0


def test_wrong_valuation_and_semantic_trace_difference():
    row=dict(return_pct=2.,drawdown_pct=1.)
    stored=dict(mean_return_pct=2.,worst_dd_pct=1.,score=1.,std_return_pct=0.)
    assert check_valuation(row,stored,1.)==0
    with pytest.raises(ValueError,match='VALUATION'): check_valuation(dict(row,return_pct=2.01),stored,1.)
    assert first_deviation([dict(bar=1,action='HOLD',horizon=6)],
                           [dict(bar=1,action='NOP',horizon=6)]) is None
    mismatch=first_deviation([dict(bar=1,action='BUY',horizon=6)],
                            [dict(bar=1,action='BUY',horizon=12)])
    assert mismatch['bar']==1


def test_archive_seals_and_resumed_unit_identity(tmp_path,monkeypatch):
    from tools import diagnose_pragmatic_plan_gap as runner
    from tools.run_crossed_account_queue import save
    from tools.optimize_dynamic_account import digest
    from app.zigzag_opportunities import sha
    file=tmp_path/'payload.json';file.write_text('{}')
    save(tmp_path/'COMPLETE.json','source',dict(artifacts={'payload.json':sha(file)}))
    assert runner.checked_archive(tmp_path)[0]=='source'
    file.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='ARCHIVE'): runner.checked_archive(tmp_path)
    job=dict(output=str(tmp_path),name='WORLD',engine_root='unused',manifest={})
    save(tmp_path/'WORLD.json',digest(job),dict(rows=[],key='test'))
    monkeypatch.setattr(runner,'check_sources',lambda *args:None)
    monkeypatch.setattr(runner,'run_account',lambda *a,**kw:pytest.fail('resumed completed unit reran'))
    assert runner.unit(job)==dict(rows=[],key='test')
    with pytest.raises(ValueError): runner.unit(dict(job,manifest={'changed':1}))
