import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def world():
    root=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root:
        pytest.skip('native engine required')
    from app.zigzag_opportunities import engine
    engine(Path(root))
    from app.dynamic_account_panel import World
    from app.train_native_pareto import TrainPrefixWorld
    risk=json.loads((Path(__file__).resolve().parents[2]/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
    ref=World(dict(seed=7,pattern='trend',volatility='decreasing',mirror=1),risk)
    bars=ref.env.bars.copy()
    bars['OPEN']=bars['CLOSE']=1.1
    bars['HIGH']=1.10001;bars['LOW']=1.09999
    origins=np.arange(len(bars))
    graph=dict(origin=origins,complete=np.ones(len(bars),bool),candidate_end=origins[:,None]+np.array([1,3,5]))
    return TrainPrefixWorld(bars,ref.env.cal.segments[0],0,{h:graph for h in (3,6,12,24)},risk,ref.env.cfg.profile)


@pytest.mark.parametrize('side,sign',[('BUY',1),('SELL',-1)])
def test_bid_preserved_next_open_pivot_exit_and_cash(world,side,sign):
    e=world.env
    assert e.quote_price(12,e.o[12],-1)==e.o[12]
    spread=e.cfg.profile.spread_at(int(e.v[12]),int(e.mo[0]))
    assert e.quote_price(12,e.o[12],1)==e.o[12]+spread
    result=world.replay([f'{side}:3:0','HOLD','PIVOT_CLOSE'])
    trade=result['trades'][0]
    assert trade['entry']==12 and trade['exit']==14 and trade['reason']=='AGENT_CLOSE'
    assert result['plans'][0]['pivot_bar']==13
    cost=2*e.cfg.profile.slippage_market+2*e.cfg.profile.commission_per_lot_side/e.cfg.profile.lot_units
    expected=trade['units']*(sign*(e.quote_price(14,e.o[14],-sign)-e.quote_price(12,e.o[12],sign))-cost)
    assert trade['net']==pytest.approx(expected,abs=1e-9)
    assert abs(result['cash_error'])<1e-7


def test_all_horizons_sides_endpoints_and_phase_rules(world):
    root=world.initial()
    assert len(world.choices(root))==25
    node=world.transition(root,'BUY:24:1')
    assert world.choices(node)==['HOLD','CLOSE']
    with pytest.raises(ValueError,match='ILLEGAL_TRAIN_ACTION'):
        world.transition(node,'SELL:3:0')
    result,leaves=world.exact(steps=2)
    from tools.calculate_sequential_pareto_reference import independent_check
    independent_check(leaves,result['frontier'])
    for row in result['frontier']:
        assert world.replay(row['trace'])['vector']==row['vector']


def test_protection_preempts_planned_pivot(world):
    root=world.initial()
    node=world.transition(root,'BUY:3:2')
    node.env.l=node.env.l.copy()
    node.env.l[13]=node.env.pos.sl-.001
    after=world.transition(node,'HOLD')
    assert after.env.pos is None
    assert after.env.trades[-1]['close_reason']=='SL'
    assert node.env.pos is not None


def test_calendar_sampling_requires_four_years(world):
    from app.train_native_pareto import calendar_sample
    with pytest.raises(ValueError,match='FOUR_COMPLETE_TRAIN_SESSIONS_REQUIRED'):
        calendar_sample(world.env.bars,world.env.cal)
