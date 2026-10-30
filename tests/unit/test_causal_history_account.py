from dataclasses import replace

import numpy as np
import pytest

from app.close_hold_controller import CloseController, bind_values, value_paths
from tests.unit.test_close_hold_controller import position, report


def test_causal_cache_equals_direct_and_margin_only_suppresses():
    p=position();r=report(tag='causal');f=np.zeros(3)
    v=value_paths(p,r,f,f);binding=bind_values(p,r,v)
    direct=CloseController().decide(p,r,10,f,f,expected_deadline=13)
    cached=CloseController().decide_causal_values(p, r, 10, v,
        expected_deadline=13,binding=binding,history_margin_risk=0.)
    assert cached==direct
    penalized=CloseController().decide_causal_values(p,r,10,v,
        expected_deadline=13,binding=binding,history_margin_risk=100.)
    assert penalized['action']=='HOLD'
    assert penalized['advantage_risk']==pytest.approx(direct['advantage_risk']-100)


@pytest.mark.parametrize('change',['report','values','risk','identity','oracle','negative_margin'])
def test_causal_cache_rejects_invalid_contract(change):
    p=position();r=report(tag='causal');v=value_paths(p,r,np.zeros(3),np.zeros(3))
    binding=bind_values(p,r,v);margin=0.
    if change=='report': r=replace(r,paths=r.paths+.1)
    if change=='values': v=v+1
    if change=='risk': p=replace(p,risk=p.risk*2)
    if change=='identity': p=replace(p,identity='other')
    if change=='oracle': r=replace(r,provenance='synthetic_ideal')
    if change=='negative_margin': margin=-1.
    with pytest.raises(ValueError):
        CloseController().decide_causal_values(p,r,10,v,expected_deadline=13,
            binding=binding,history_margin_risk=margin)


@pytest.fixture
def causal_world():
    from pathlib import Path
    import json,os
    from app.zigzag_opportunities import engine
    root=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root: pytest.skip('native engine required')
    engine(Path(root))
    from app.causal_history_account import HistoryWorld, specifications
    cfg=json.loads((Path(__file__).resolve().parents[2]/
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
    return HistoryWorld(specifications('calibration',smoke=True)[0],cfg,'median')


def test_future_mutation_does_not_change_report_or_matured_history(causal_world):
    from app.causal_history_account import HistoryWorld
    a=causal_world;b=HistoryWorld(a.spec,a.risk_config,'median')
    for name in ('o','h','l','c'):
        data=getattr(b.env,name).copy();data[51:]+=.1;setattr(b.env,name,data)
    assert np.array_equal(a.report(50).paths,b.report(50).paths)
    assert a.quality[50]==b.quality[50]
    assert a.candidate_bids(50)==b.candidate_bids(50)
    assert a.report(50).provenance=='causal'


def test_cold_account_and_direct_replay(causal_world):
    from app.causal_history_account import HistoryWorld, base_genes
    a=causal_world;g=base_genes(0.,0.)
    x=a.run(g,verify=True,details=True)
    y=HistoryWorld(a.spec,a.risk_config,'median').run(g,verify=True,details=True)
    assert x==y
    assert x['cache_max_error']<1e-10
    z=a.run(g,verify=True,details=True,voluntary_close=False)
    assert z['closes'].get('AGENT_CLOSE',0)==0


def test_missing_history_does_not_get_a_fake_zero(causal_world):
    from app.causal_history_account import base_genes
    w=causal_world;w.active_genes=base_genes(1.,1.)
    assert w.candidate_bids(12)==[]
    w.report(50)
    assert w.history_error(50,3) is not None


def test_strategy_history_is_separate_from_report(causal_world):
    from app.causal_history_account import base_genes
    w=causal_world;r=w.report(50)
    w.active_genes=base_genes(0.,0.);a=w.candidate_bids(50)
    w.active_genes=base_genes(1.,0.);b=w.candidate_bids(50)
    assert r is w.report(50)
    assert len(a)==len(b)==8
    assert all(x['gain_risk']>=y['gain_risk'] for x,y in zip(a,b))


def test_separate_banks_and_genome_bounds():
    from app.causal_history_account import specifications, decode_history, base_genes
    banks=[{s['seed'] for s in specifications(b,smoke=smoke)}
           for b in ('calibration','confirmation') for smoke in (False,True)]
    assert all(not a&b for i,a in enumerate(banks) for b in banks[i+1:])
    assert {s['seed'] for s in specifications('calibration')}=={350100}
    assert {s['seed'] for s in specifications('confirmation')}=={360100,360101,360102}
    g=decode_history(base_genes(1.,2.))
    assert g[8]==1 and g[11:13]==(1.,2.)
    assert len(g)==14
