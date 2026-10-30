import copy
import json
from pathlib import Path

import pytest

from tools.run_pragmatic_contrast import configurations,freeze_selection,summarize,job_receipt


def config():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/pragmatic_controller_contrast.json').read_text())


def test_full_grid_variation_and_smoke_subset():
    rows=configurations(config(),False)
    assert len(rows)==24
    assert len({x['key'] for x in rows})==24
    for arm in ('simple','predictive_proxy','sequential'):
        selected=[r['config'] for r in rows if r['config']['variant']==arm]
        assert len(selected)==8
        assert {c['alpha'] for c in selected}=={0.,1.}
        assert {c['uncertainty'] for c in selected}=={0.,1.}
        assert {c['persistence'] for c in selected}=={1,2}
    assert len(configurations(config(),True))==3


def test_selection_activity_strict_and_no_confirmation_input():
    c=configurations(config(),True)
    metrics={r['key']:dict(mean_return_pct=i,worst_dd_pct=1.,orders_per_week=2.) for i,r in enumerate(c)}
    frozen=freeze_selection(c,metrics)
    assert all(not r['activity_eligible'] for r in frozen)
    assert len(frozen)==3
    for value in metrics.values(): value['orders_per_week']=2.01
    assert all(r['activity_eligible'] for r in freeze_selection(c,metrics))


def test_zero_orders_does_not_invent_win_rate():
    row=dict(return_pct=0.,drawdown_pct=0.,net_usd=0.,orders=0,orders_per_week=0.,wins=0,
        buys=0,sells=0,buy_wins=0,sell_wins=0,sharpe=None,close_reasons={},horizons={},costs={},
        close_rows=[],close_diagnostics=dict(measured=True))
    summary=summarize([row])
    assert summary['win_pct'] is None and summary['buy_win_pct'] is None
    assert summary['sell_win_pct'] is None and summary['mean_sharpe'] is None
    assert not summary['activity_eligible']


def test_unit_receipt_identity_and_tampering(tmp_path):
    p=tmp_path/'result.json';job=dict(identity='x',config={'alpha':0})
    assert job_receipt(p,job) is None
    job_receipt(p,job,{'net':3.})
    assert job_receipt(p,job)=={'net':3.}
    changed=copy.deepcopy(job);changed['config']['alpha']=1
    with pytest.raises(ValueError): job_receipt(p,changed)
    body=json.loads(p.read_text());body['data']['net']=5;p.write_text(json.dumps(body))
    with pytest.raises(ValueError): job_receipt(p,job)
