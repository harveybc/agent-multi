import json
import pytest

from app.rl_subset import digest
from tools.synthetic_output_calibration import assess, choose, checked


def wave(mode, side):
    return dict(mode=mode,metrics=dict(valid=True,orders_per_week=3,n_long=2,n_short=2,
        net_profit_usd=10 if mode=='ideal' else -10 if mode=='inverted' else 0,
        trades=0 if mode=='nop' else 4,sharpe_daily_sqrt365=2,max_drawdown_pct=1),
        trades=[dict(direction=side,close_reason='AGENT_CLOSE')])


def sample(q=1):
    waves=[wave(mode,side) for side in (1,-1) for mode in ('ideal','inverted','nop')]
    return assess(dict(tp_fraction=.5,sl_fraction=.5,order_fraction=q),waves)


def test_selection_and_tie():
    a,b=sample(1),sample(.2)
    assert choose([a,b])==b
    assert choose([b,a])==b
    assert a['eligible']


def test_requested_close_does_not_replace_executed_close():
    waves=[wave(mode,1) for side in (1,-1) for mode in ('ideal','inverted','nop')]
    row=assess(sample()['parameters'],waves)
    assert not row['eligible']
    assert 'NO_EXECUTED_CLOSE_SHORT' in row['reasons']
    assert choose([row]) is None


def test_negative_invalid_and_inactive_ideal_rejected():
    for key,value in [('net_profit_usd',-1),('valid',False),('orders_per_week',1)]:
        waves=[wave(mode,side) for side in (1,-1) for mode in ('ideal','inverted','nop')]
        waves[0]['metrics'][key]=value
        assert not assess(sample()['parameters'],waves)['eligible']


def test_resume_rejects_changed_identity_and_payload(tmp_path):
    identity={'scope':'synthetic'};payload={'value':1};path=tmp_path/'cell.json'
    cell=dict(identity_sha256=digest(identity),payload_sha256=digest(payload),payload=payload)
    path.write_text(json.dumps(cell))
    assert checked(path,identity)==payload
    with pytest.raises(ValueError,match='CELL_IDENTITY'):checked(path,{'scope':'changed'})
    cell['payload']['value']=2;path.write_text(json.dumps(cell))
    with pytest.raises(ValueError,match='CELL_PAYLOAD'):checked(path,identity)
