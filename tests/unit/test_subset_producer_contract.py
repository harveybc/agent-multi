import numpy as np
import pytest
from app.subset_producer_contract import window_mask,assert_fit_rows,fixed_scaler


def test_window_exclusion():
    assert window_mask([0,1,1,0,1,1,1],3).tolist()==[False]*6+[True]


def test_label_crosses_hole_rejected():
    a=np.array([1,1,0,1,1],bool);h=np.array([0,1,0,0,1.]);r=np.array([3,1,2,3,4])
    with pytest.raises(ValueError,match='FIT_LABEL_SCOPE'):assert_fit_rows([0,1],a,h,r)


def test_valid_labels():
    assert_fit_rows([1,2],np.ones(5,bool),np.array([0.,0,1,1,0]),np.arange(5),2)


def test_input_before_subset_rejected():
    with pytest.raises(ValueError,match='FIT_INPUT_SCOPE'):assert_fit_rows([1],np.array([0,1,1],bool),lookback=2)


def test_scaler_future_perturbation_and_constant():
    x=np.array([[1.,7],[3,7],[99,1]]);mask=np.array([1,1,0],bool)
    a=fixed_scaler(x,mask);x[-1]=[-10000,500];b=fixed_scaler(x,mask)
    assert all(np.array_equal(u,v) for u,v in zip(a,b))
    assert a[2]==[1];assert a[1][1]==1


def test_empty_scaler_fails():
    with pytest.raises(ValueError,match='SCALER_ROWS'):fixed_scaler(np.ones((3,2)),[0,0,0])
