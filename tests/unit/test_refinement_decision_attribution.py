import numpy as np
import pytest
from tools.analyze_robust_refinement_decisions import decompose


def test_level_shift_does_not_change_gap_error():
    q=np.array([[[1.,2.,3.]],[[3.,2.,1.]]]);y=np.zeros((1,3))
    a=decompose(q,y);b=decompose(q+7,y)
    assert a['gap']==b['gap'] and b['level']>a['level']
    assert abs(a['total']-a['level']-a['gap'])<1e-12


def test_pure_centered_error_and_exact_target():
    y=np.array([[1.,2.,3.]])
    assert decompose(np.stack([y,y]),y)==dict(total=0.,level=0.,gap=0.)
    v=decompose(np.stack([y,y])+np.array([-1.,0.,1.]),y)
    assert v['level']==0 and v['gap']==v['total']


def test_nonfinite_or_wrong_shape_refused():
    for q in (np.zeros((1,1,3)),np.full((2,1,3),np.nan)):
        with pytest.raises(ValueError):decompose(q,np.zeros((1,3)))
