import json
from pathlib import Path

import pytest

from tools.run_reference_rollout import configs,fresh_specs,verify_case_files,remaining_seconds
from tools.run_crossed_account_queue import save,load
from tools.optimize_dynamic_account import digest

ROOT=Path(__file__).resolve().parents[2]


def configuration():
    return json.loads((ROOT/'configs/reference_rollout_contrast.json').read_text())


def test_fresh_banks_and_smoke_are_disjoint():
    source={b:[dict(seed=1,pattern='trend')] for b in ('calibration','confirmation')}
    rows=[fresh_specs(source,configuration(),b,s)[0] for b in source for s in (False,True)]
    assert len({r['seed'] for r in rows})==4
    bad=dict(configuration(),confirmation_seed=990000)
    with pytest.raises(ValueError,match='OVERLAP'): fresh_specs(source,bad,'calibration',False)


def test_four_controls_remain_byte_identical_and_three_reference_arms_are_explicit():
    from dataclasses import asdict
    from app.pragmatic_strategy import ControllerConfig
    rows=[dict(config=asdict(ControllerConfig(v,alpha=a))) for v,a in
          [('simple',0),('predictive_proxy',1),('sequential',0),('sequential',1)]]
    rows=[dict(row,key=digest(row['config'])) for row in rows]
    new=configs(dict(configurations=rows),rows,configuration())
    assert len(new)==7
    assert [{k:r[k] for k in ('key','config')} for r in new[:4]]==rows
    assert all(r['references'] is None for r in new[:4])
    assert set(new[-1]['references'])=={'simple','predictive_proxy'}


def test_tampered_and_cross_run_receipts_rejected(tmp_path):
    path=tmp_path/'unit.json';save(path,'a',dict(orders=3))
    assert load(path,'a')==dict(orders=3)
    with pytest.raises(ValueError): load(path,'b')
    row=json.loads(path.read_text());row['data']['orders']=30;path.write_text(json.dumps(row))
    with pytest.raises(ValueError): load(path,'a')


def test_cached_attribution_rechecks_all_child_artifacts(tmp_path):
    from app.zigzag_opportunities import sha
    case=tmp_path/'case.json';case.write_text('{}')
    result=dict(case_files={case.name:sha(case)})
    verify_case_files(tmp_path,result)
    case.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='CASE_MUTATED'): verify_case_files(tmp_path,result)


def test_no_eta_from_fast_controls_before_heavy_arm_measured():
    a=dict(name='a',policy=dict(arm='fast'));b=dict(name='b',policy=dict(arm='slow'))
    assert remaining_seconds([a,b],[(a,dict(seconds=1))],4) is None
    c=dict(name='c',policy=dict(arm='slow'))
    assert remaining_seconds([a,b,c],[(a,dict(seconds=1)),(b,dict(seconds=40))],4)==10
