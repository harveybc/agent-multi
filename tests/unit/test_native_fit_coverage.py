from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from app.native_fit_coverage import compact_support, validate_record
from app.rl_subset import digest


def support():
    return dict(old_count=3, rows=[10,20,30,40,50,60], folds={
        'A':dict(masks=dict(fit=[0],stop=[1],audit=[2]),expanded={'4':[0,3,4,5]})})


def test_compaction_preserves_old_order_and_held_panels():
    ix, folds = compact_support(support())
    assert ix.tolist() == [0,1,2,3,4,5]
    assert folds['A']['expanded'] == [0,3,4,5]
    assert folds['A']['masks'] == dict(fit=[0],stop=[1],audit=[2])


def test_compaction_rejects_old_held_duplicate_and_size():
    for values in ([0,1,4,5], [0,3,3,5], [0,3,4]):
        s=support();s['folds']['A']['expanded']['4']=values
        with pytest.raises(ValueError):compact_support(s)
    s=support();s['rows'][4]=s['rows'][3]
    with pytest.raises(ValueError):compact_support(s)


def record():
    ns=lambda s:pd.Timestamp(s,tz='UTC').value
    outcomes=[dict(parent_unchanged=True,reward_return=float(i)) for i in range(3)]
    return dict(year=2020,window_start=ns('2020-02-01'),timestamp=ns('2020-02-02'),
                terminal=ns('2020-12-31'),outcomes=outcomes,outcomes_sha256=digest(outcomes))


def test_native_target_finite_annual_and_authenticated():
    r=record();assert validate_record(r).tolist()==[0.,1.,2.]
    for key,value in [('terminal',pd.Timestamp('2021-01-01',tz='UTC').value),
                      ('window_start',pd.Timestamp('2019-12-31',tz='UTC').value),
                      ('outcomes_sha256','bad')]:
        bad=deepcopy(r);bad[key]=value
        with pytest.raises(ValueError):validate_record(bad)
    for value in (np.nan,np.inf):
        bad=deepcopy(r);bad['outcomes'][0]['reward_return']=value
        with pytest.raises(ValueError):validate_record(bad)
    bad=deepcopy(r);bad['outcomes'][1]['parent_unchanged']=False
    bad['outcomes_sha256']=digest(bad['outcomes'])
    with pytest.raises(ValueError):validate_record(bad)
