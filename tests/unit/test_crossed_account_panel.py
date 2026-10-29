import json
import os
from pathlib import Path
import numpy as np
import pytest

from app.crossed_account_panel import CrossedWorld, panel_specs, market, QUALITIES
from app.dynamic_account_panel import BASE
from app.close_hold_controller import validate, validate_context
from app.zigzag_opportunities import engine


@pytest.fixture
def world():
    root=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root: pytest.skip('native engine required')
    engine(Path(root))
    config=json.loads((Path(__file__).resolve().parents[2]/
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
    return CrossedWorld(panel_specs('calibration')[0],config)


def test_full_grid_independence_units_and_disjoint_banks():
    a=panel_specs('calibration');b=panel_specs('confirmation')
    assert len(a)==len(b)==360
    assert len({(s['seed'],s['pattern']) for s in a})==30
    assert {s['seed'] for s in a}.isdisjoint({s['seed'] for s in b})
    assert len({tuple(sorted(s.items())) for s in a})==360


def test_cost_quality_and_mirror_metamorphisms():
    s=panel_specs('calibration')[0];x=market(s)
    assert np.array_equal(x,market(dict(s,spread_pips=3.)))
    assert np.array_equal(x,market(dict(s,quality='missing')))
    y=market(dict(s,mirror=-1))
    assert np.array_equal(y,2.2-x[:,[0,2,1,3]])
    assert not np.array_equal(x,market(dict(s,seed=s['seed']+1)))
    assert not np.array_equal(x,market(dict(s,volatility='increasing')))


@pytest.mark.parametrize('quality',QUALITIES)
def test_report_support_sample_validity_and_native_replay(world,quality):
    s=dict(panel_specs('calibration')[0],quality=quality)
    w=CrossedWorld(s,world.risk_config)
    b=next(i for i in range(15,30) if w.report_available(i))
    r=w.report(b);p=w.position(3,1,1.1,12)
    validate_context(r,int(w.env.v[b])+r.step,int(w.env.ff[0]))
    z=np.zeros(r.paths.shape[1]);validate(p,r,z,z)
    if quality!='ideal':
        assert not np.array_equal(r.paths.mean(axis=0),np.column_stack((w.env.o,w.env.h,w.env.l,w.env.c))[b+1:w.end])
    a=w.run(BASE,verify=True,details=True)
    assert a==w.run(BASE,verify=True,details=True)
    assert a['net_usd']==pytest.approx(sum(t['net'] for t in a['ledger']))
    assert len(w.cache)<=128
    if quality=='missing':
        assert all(w.report_available(t['entry_bar']-1) for t in a['ledger'])


def test_missing_all_reports_abstains(world,monkeypatch):
    w=CrossedWorld(panel_specs('calibration')[0],world.risk_config)
    monkeypatch.setattr(w,'report_available',lambda b:False)
    monkeypatch.setattr(w,'report',lambda b:pytest.fail('missing report consumed'))
    assert w.run(BASE,verify=True)['orders']==0


def test_error_streams_differ_between_draws_and_match_across_costs(world):
    s=panel_specs('calibration')[0]
    a=CrossedWorld(s,world.risk_config)
    b=CrossedWorld(dict(s,seed=s['seed']+1),world.risk_config)
    c=CrossedWorld(dict(s,spread_pips=3.),world.risk_config)
    assert not np.array_equal(a.latent,b.latent)
    assert np.array_equal(a.latent,c.latent)
    assert np.array_equal(a.report(20).paths,c.report(20).paths)
    assert not np.array_equal(a.report(20).spreads,c.report(20).spreads)


def test_sharded_native_results_are_exactly_serial(world):
    from app.crossed_account_workers import ShardPool
    from app.dynamic_account_panel import evaluate
    specs=panel_specs('calibration')[:2]
    serial=evaluate([CrossedWorld(s,world.risk_config) for s in specs],BASE,verify=True,details=True)
    pool=ShardPool(specs,world.risk_config,os.environ['STRATEGY_ENGINE_ROOT'],2)
    try:
        assert pool.evaluate(BASE,verify=True,details=True)==serial
        assert pool.evaluate(BASE,verify=True,details=True)==serial
    finally: pool.close()
