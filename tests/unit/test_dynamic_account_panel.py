import json
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
    r=PathReport(0,0,0,HOUR,2*HOUR,paths,np.zeros((2,2)),np.array([.5,.5]),'synthetic_ideal')
    assert expected_duration(p,r,2)==1.5
