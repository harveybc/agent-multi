import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.c4_learning_control import future_move, control_inputs, qualify
from app.frozen_recipe_inputs import FrozenRecipeInputs
from tests.unit.test_frozen_recipe_inputs import recipe
from app.rl_subset import seal_recipe


def test_future_move_uses_next_open_and_stays_inside_split():
    o=np.arange(12,dtype=float)+10;c=o+.5
    periods=np.array(['TRAIN']*6+['INNER_A']*6)
    x=future_move(o,c,periods,4)
    assert x[0]==3.5 and x[1]==3.5
    assert np.isnan(x[2:6]).all() and np.isnan(x[8:]).all()
    c2=c.copy();c2[6:]+=100
    np.testing.assert_array_equal(x[:6],future_move(o,c2,periods,4)[:6])


def fixture():
    r=recipe();r['tensor']['window'].update(observations=3,stride_rows=2)
    r['split']['periods']={'TRAIN':['2020-01-01','2020-01-03'],
                          'INNER_A':['2020-01-03','2020-01-05']}
    r['split']['test_closed_from']='2020-01-05';seal_recipe(r)
    ts=pd.date_range('2020-01-01',periods=96,freq='h',tz='UTC')
    f=pd.DataFrame(dict(timestamp=ts,x=np.sin(np.arange(96)),y=np.cos(np.arange(96))))
    inputs=FrozenRecipeInputs.from_frame(r,f)
    values=np.sin(np.arange(len(inputs.valid))*.7)
    values[[45,46,47,93,94,95]]=np.nan
    return inputs,values


def test_arms_share_support_and_only_one_channel_changes():
    inputs,values=fixture()
    arms={a:control_inputs(inputs,values,a)[0] for a in ('original','planted','permuted')}
    for a in arms.values():
        np.testing.assert_array_equal(a.valid,arms['original'].valid)
        np.testing.assert_array_equal(a.windows[:,:,1],inputs.windows[:,:,1])
        assert not a.windows.flags.writeable
    np.testing.assert_array_equal(arms['original'].windows,inputs.windows)
    assert not np.array_equal(arms['planted'].windows[:,:,0],arms['permuted'].windows[:,:,0])
    assert arms['planted'].recipe['list']['columns'][0]=='DIAGNOSTIC_FUTURE_MOVE_NOT_FINANCIAL'
    assert arms['planted'].recipe['identity']!=inputs.recipe['identity']


def test_train_scaler_and_train_permutation_do_not_depend_on_selection_values():
    inputs,values=fixture();changed=values.copy();changed[inputs.period=='INNER_A']*=100
    for arm in ('planted','permuted'):
        a,ma=control_inputs(inputs,values,arm);b,mb=control_inputs(inputs,changed,arm)
        assert ma['mean']==mb['mean'] and ma['std']==mb['std']
        np.testing.assert_array_equal(a.windows[inputs.period=='TRAIN'],b.windows[inputs.period=='TRAIN'])


def test_qualification_requires_all_seeds_and_executed_behavior():
    rows=[]
    for seed in (7,19,43):
        for arm in ('original','planted','permuted'):
            rows.append(dict(seed=seed,arm=arm,periods={p:dict(
                strict={'eligible':True},metrics={'sharpe_daily_sqrt365':1. if arm=='planted' else -.1})
                for p in ('TRAIN','INNER_A')}))
    assert qualify(rows)['state']=='POSITIVE_CONTROL_PASSED'
    bad=copy.deepcopy(rows);bad[1]['periods']['INNER_A']['strict']['eligible']=False
    assert qualify(bad)['state']=='LEARNING_QUALIFICATION_FAILED'
    with pytest.raises(ValueError,match='NINE_CONTROL_UNITS'):qualify(rows[:-1])


def test_queue_adoption_authenticates_artifacts_and_scope(tmp_path):
    from tools.c4_learning_queue import write, file_sha, verified_result
    write(tmp_path/'SPEC.private.json',{'diagnostic':True})
    unit=tmp_path/'unit';unit.mkdir()
    write(unit/'TRAINING.json',{'steps':102400})
    result=dict(schema='c4_diagnostic_result.v1',financial_admission=False,
        spec_sha256=file_sha(tmp_path/'SPEC.private.json'),
        artifacts={'TRAINING.json':file_sha(unit/'TRAINING.json')})
    write(unit/'RESULT.json',result)
    assert verified_result(tmp_path,unit/'RESULT.json')==result
    write(unit/'TRAINING.json',{'steps':1})
    with pytest.raises(ValueError,match='C4_ARTIFACT_HASH'):
        verified_result(tmp_path,unit/'RESULT.json')
    result['financial_admission']=True;write(unit/'RESULT.json',result)
    with pytest.raises(ValueError,match='C4_RESULT_SCOPE'):
        verified_result(tmp_path,unit/'RESULT.json')


def test_queue_resource_wait_does_not_start_work(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from tools import c4_learning_queue as queue
    monkeypatch.setattr(Path,'read_text',lambda self:'MemAvailable: 1024 kB\n')
    monkeypatch.setattr(queue.subprocess,'run',lambda *a,**kw:pytest.fail('GPU query not needed under RAM pressure'))
    spec={'minimum_available_ram_gib':5,'physical_gpu':0}
    assert queue.admission(spec)==(False,'RAM')
    monkeypatch.setattr(Path,'read_text',lambda self:'MemAvailable: 10000000 kB\n')
    monkeypatch.setattr(queue.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='123\n'))
    assert queue.admission(spec)==(False,'GPU_BUSY_OR_UNAVAILABLE')
    monkeypatch.setattr(queue.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout=''))
    assert queue.admission(spec)==(True,None)
