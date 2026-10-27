import numpy as np
import pytest
import torch

from agent_plugins.hybrid_orders import OrderDistribution
from app.strategy_scenario_matrix import cells, market_frame, check_trade, WARMUP


def test_full_cartesian_matrix_is_unique():
    rows=list(cells())
    assert len(rows)==5832
    assert len({tuple(r.items()) for r in rows})==5832
    assert {r['before'] for r in rows}=={-1,0,1}
    assert {r['after'] for r in rows}=={-1,0,1}
    assert {r['cost'] for r in rows}=={'configured','zero_cost'}


def test_generator_is_deterministic_valid_and_prefix_local():
    c=next(cells());a=market_frame(c);b=market_frame(c)
    np.testing.assert_array_equal(a.values,b.values)
    assert np.isfinite(a.values).all() and (a.values>0).all()
    assert (a.HIGH>=a[['OPEN','CLOSE']].max(axis=1)).all()
    assert (a.LOW<=a[['OPEN','CLOSE']].min(axis=1)).all()
    changed=market_frame({**c,'after':1,'vol_after':.0006})
    np.testing.assert_array_equal(a.values[:WARMUP+c['switch']],changed.values[:WARMUP+c['switch']])
    assert len(a)==WARMUP+120


@pytest.mark.parametrize('scale',[1e-4,1.,10.])
@pytest.mark.parametrize('scores,mask,expected',[
    ([0,1,2,9,9],[1,1,1,0,0],2),
    ([0,2,1,9,9],[1,1,1,0,0],1),
    ([2,1,1,9,9],[1,1,1,0,0],0),
    ([0,9,9,2,9],[1,0,0,1,0],3),
    ([0,9,9,9,2],[1,0,0,0,1],4),
    ([1,1,1,9,9],[1,1,1,0,0],0),
    ([0,1,1,9,9],[1,1,1,0,0],1),
])
def test_all_heads_compete_and_masked_heads_cannot_win(scores,mask,expected,scale):
    d=OrderDistribution(torch.tensor([scores],dtype=torch.float32)*scale,
        torch.zeros(1,2,6),torch.tensor([mask],dtype=torch.bool),False)
    assert int(d.get_actions(True)[0,0])==expected


def test_signed_profit_check_rejects_short_accounted_as_long():
    t=dict(direction=-1,units=100,entry_price=1.1,exit_price=1.0,
           stop_loss=1.2,take_profit=.9,gross_usd=10.,net_usd=8.,
           spread_usd=1.,slippage_usd=.5,commission_usd=.5,financing_usd=0.)
    check_trade(t)
    with pytest.raises(ValueError,match='SIGNED_GROSS'):
        check_trade({**t,'gross_usd':-10.})
