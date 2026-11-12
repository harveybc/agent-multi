import copy
import numpy as np
import pytest

from tools.reuse_stl_columns import reusable_columns


def fixture():
    cfg=dict(train_end=2,statsmodels='pinned',grid={'p':[6]},origins=16,
        forecast_horizon=24,period_clock='bars',columns=['a','b'],
        sources={'app/causal_stl_inputs.py':'sha'})
    before={'times':np.arange(3),'values':np.array([[1.,2.],[np.nan,3.],[4.,5.]])}
    after={'times':np.arange(3),'values':before['values'][:,::-1].copy()}
    return cfg,dict(cfg,columns=['b','a']),before,after


def test_reuse_follows_names_and_exact_values_not_position():
    a,b,x,y=fixture()
    assert reusable_columns(a,b,x,y)==[(1,0),(0,1)]
    y['values'][2,0]+=1e-12
    assert reusable_columns(a,b,x,y)==[(0,1)]


@pytest.mark.parametrize('field',['train_end','statsmodels','grid','origins','forecast_horizon','period_clock'])
def test_reuse_rejects_any_changed_tuning_scope(field):
    a,b,x,y=fixture();b[field]='different'
    with pytest.raises(ValueError,match='SCOPE'):
        reusable_columns(a,b,x,y)


def test_reuse_rejects_clock_and_algorithm_changes():
    a,b,x,y=fixture();y['times'][0]=-1
    with pytest.raises(ValueError,match='CLOCK'):reusable_columns(a,b,x,y)
    a,b,x,y=fixture();b=copy.deepcopy(b);b['sources']['app/causal_stl_inputs.py']='other'
    with pytest.raises(ValueError,match='ALGORITHM'):reusable_columns(a,b,x,y)
