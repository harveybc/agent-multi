import copy

import numpy as np
import pandas as pd
import pytest

from app.critic_blocked_stopping import RegretStopper,bounded_snapshot,fold_indices
from app.decision_attribution import continuation,state_view
from env_plugins.three_signal_native import ThreeSignalEnv
from env_plugins.discrete_curve_native import HOUR
from tests.unit.test_discrete_curve_native import curve_env


def records():
    out=[]
    for year in (2020,2021,2022,2023):
        for role,day in (('fit',2),('monitor',3)):
            ts=pd.Timestamp(f'{year}-02-{day:02}',tz='UTC').value
            out.append(dict(year=year,original_role=role,timestamp=ts,window_start=ts-HOUR*24,
                terminal=pd.Timestamp(f'{year}-12-30',tz='UTC').value))
    return out


def test_fold_selection_never_uses_held_for_stopping():
    r=records();a=fold_indices(r,[2020],2021,2022)
    assert a['fit'].tolist()==[0] and a['stop'].tolist()==[2] and a['audit'].tolist()==[4,5]
    r[3]['anything']='irrelevant'
    assert fold_indices(r,[2020],2021,2022)['stop'].tolist()==[2]
    b=fold_indices(r,[2020,2021],2022,2023)
    assert b['fit'].tolist()==[0,2] and b['stop'].tolist()==[4]


def test_fold_rejects_label_leak_context_leak_duplicate_and_empty():
    for field,value in (('terminal',pd.Timestamp('2021-02-01',tz='UTC').value),
                        ('window_start',pd.Timestamp('2019-12-31',tz='UTC').value)):
        r=records();r[0][field]=value
        with pytest.raises(ValueError,match='LABEL_OR_CONTEXT'):fold_indices(r,[2020],2021,2022)
    r=records();r[2]['original_role']='monitor'
    with pytest.raises(ValueError,match='EMPTY'):fold_indices(r,[2020],2021,2022)
    with pytest.raises(ValueError,match='DUPLICATE'):fold_indices(records()+[records()[0]],[2020],2021,2022)


def test_stopper_best_so_far_tiny_delta_and_no_reopening():
    s=RegretStopper(patience=2)
    assert s.observe(64,.3)
    assert not s.observe(128,.3-1e-9) and s.bad==1
    assert s.observe(192,.29) and s.bad==0
    assert not s.observe(256,.3)
    assert not s.observe(320,.295) and s.stopped_at==320
    assert not s.observe(384,.1) and s.best_step==192 and s.best==.29
    with pytest.raises(ValueError):s.observe(384,.1)
    with pytest.raises(ValueError):s.observe(448,float('nan'))


def test_native_bounded_suffix_future_perturbation_and_source_parity(curve_env):
    e=ThreeSignalEnv(curve_env.engine,curve_env.inputs,curve_env.risk_config,6)
    e.reset()
    while not e.engine.entry_gate(e.bar)[0]:e.step(0)
    signals=np.zeros(len(e.inputs.row_ts_ns),dtype=np.int64)
    source=state_view(e);old_last=e.engine.last
    full=bounded_snapshot(e,int(e.engine.v[old_last])+HOUR+1)
    assert continuation(full,signals,1)==continuation(e,signals,1)
    cut=min(e.bar+12,old_last)
    short=bounded_snapshot(e,int(e.engine.v[cut])+HOUR+1)
    target=continuation(short,signals,1)
    changed=copy.copy(short);changed.engine=copy.copy(short.engine)
    for name in ('o','h','l','c','sigma','sigma_rule'):
        values=getattr(short.engine,name).copy();values[cut+1:]*=1.5
        setattr(changed.engine,name,values)
    assert continuation(changed,signals,1)==target
    assert state_view(e)==source and e.engine.last==old_last
    assert short.engine.last==cut
    with pytest.raises(ValueError):bounded_snapshot(e,int(e.engine.v[e.bar])+HOUR)
