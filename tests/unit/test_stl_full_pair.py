from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from app.stl_full_pair import paired_coverage,assert_declared_changes,copy_loss_weights


def test_same_columns_allow_missing_interior_month():
    t=pd.date_range('2020-01-01',periods=366,freq='D',tz='UTC').as_unit('ns').asi8
    r=np.ones(len(t),bool);v=r.copy();v[152:182]=False
    out=paired_coverage(t,r,v,{'TRAIN':['2020-01-01','2021-01-01']})
    assert out['TRAIN']['retained']==336
    assert out['TRAIN']['excluded_runs'][0]['rows']==30


def test_support_refuses_large_loss_and_extra_rows():
    t=pd.date_range('2020-01-01',periods=10,tz='UTC').as_unit('ns').asi8
    r=np.ones(10,bool);v=r.copy();v[:3]=False
    with pytest.raises(ValueError,match='INSUFFICIENT'):paired_coverage(t,r,v,{'TRAIN':['2020','2021']})
    with pytest.raises(ValueError,match='SHAPE'):paired_coverage(t,v,r,{'TRAIN':['2020','2021']})


def test_only_declared_source_columns_change():
    a=np.array([[1.,np.nan],[2.,3.]])
    b=a.copy();b[0,1]=4
    assert_declared_changes(a,b,['a','activity'],{'activity'})
    b[1,0]=99
    with pytest.raises(ValueError,match='UNDECLARED'):assert_declared_changes(a,b,['a','activity'],{'activity'})


def test_copies_all_weights_and_buffers_exactly():
    donor=SimpleNamespace(loss=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4)))
    model=SimpleNamespace(loss=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4)))
    assert copy_loss_weights(model,donor)==7
    assert all(torch.equal(v,model.loss.state_dict()[k]) for k,v in donor.loss.state_dict().items())
    bad=SimpleNamespace(loss=torch.nn.Linear(3,5))
    with pytest.raises(ValueError,match='SHAPE'):copy_loss_weights(bad,donor)
