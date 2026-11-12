import numpy as np
import pandas as pd
import pytest

from app.stl_availability_subset import availability_subset


def fixture():
    names=['a','b','cal_sin','cal_cos','rg.p0','rg.p1','act.p_hit']
    times=pd.to_datetime(['2020-01-02','2020-01-03','2020-02-02','2020-02-03',
                          '2021-01-02'],utc=True).as_unit('ns').asi8
    periods={'TRAIN':['2020-01-01','2020-03-01'],'VALIDATION':['2021-01-01','2021-02-01']}
    valid=np.ones((5,7),bool)
    valid[:2,1]=False;valid[2:4,2]=False;valid[:2,5]=False
    return names,times,periods,valid


def test_availability_reasons_and_atomic_groups():
    names,times,periods,valid=fixture()
    out=availability_subset(names,valid,times,periods,{'act.p_hit':'cross_fit_context'})
    assert out['selected']==['a']
    assert out['removed']['b']==['empty_TRAIN_month:2020-01']
    assert out['removed']['cal_cos']==['atomic_block:cal_sin,cal_cos']
    assert out['removed']['rg.p0']==['atomic_block:rg.p0,rg.p1']
    assert out['removed']['act.p_hit']==['cross_fit_context']


def test_validation_availability_does_not_choose_columns():
    names,times,periods,valid=fixture()
    first=availability_subset(names,valid,times,periods,{})
    valid[-1]=False
    assert availability_subset(names,valid,times,periods,{})==first


def test_absent_calendar_month_not_silently_ignored():
    names,times,periods,valid=fixture()
    periods['TRAIN'][1]='2020-04-01'
    with pytest.raises(ValueError,match='NO_AVAILABLE_COLUMNS'):
        availability_subset(names,valid,times,periods,{})


def test_joint_support_must_pass_not_only_each_column():
    names=['a','b'];times=pd.to_datetime(['2020-01-02','2020-01-03'],utc=True).as_unit('ns').asi8
    with pytest.raises(ValueError,match='JOINT_TRAIN_COVERAGE'):
        availability_subset(names,np.eye(2,dtype=bool),times,{'TRAIN':['2020-01-01','2020-02-01']},{})


def test_shape_and_duplicates_rejected():
    names,times,periods,valid=fixture()
    with pytest.raises(ValueError,match='INPUT'):
        availability_subset(names,valid[:2],times,periods,{})
    names[-1]='a'
    with pytest.raises(ValueError,match='INPUT'):
        availability_subset(names,valid,times,periods,{})


def test_strict_reference_drops_partial_loss_not_just_empty_month():
    names=['a','b'];times=pd.to_datetime(['2020-01-02','2020-01-03','2021-01-02'],utc=True).as_unit('ns').asi8
    periods={'TRAIN':['2020-01-01','2020-02-01'],'VALIDATION':['2021-01-01','2021-02-01']}
    valid=np.ones((3,2),bool);valid[0,1]=False;reference=np.ones(3,bool)
    out=availability_subset(names,valid,times,periods,{},reference)
    assert out['selected']==['a']
    assert out['removed']['b']==['lost_reference_TRAIN_windows:1']
    valid[-1]=False
    assert availability_subset(names,valid,times,periods,{},reference)==out


def test_reference_cannot_claim_missing_rows_are_present():
    names,times,periods,valid=fixture()
    with pytest.raises(ValueError,match='REFERENCE'):
        availability_subset(names,valid,times,periods,{},np.ones(2,bool))
