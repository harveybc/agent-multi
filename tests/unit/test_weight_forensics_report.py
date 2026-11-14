import pytest
from tools.report_saved_weight_forensics import compare_common


def test_only_declared_device_witnesses_are_omitted():
    a={'row':7,'cost':.1,'cuda':{'gradient':2.},'event':[True,None,'fit']}
    b={'row':7,'cost':.1+1e-12,'cuda':None,'event':[True,None,'fit']}
    d=compare_common(a,b)
    assert len(d)==1 and d[0]['path']=='cases/cost'
    for key,value in [('row',8),('cost',.11),('event',[False,None,'fit'])]:
        with pytest.raises(ValueError):compare_common(a,dict(b,**{key:value}))


def test_shape_type_missing_and_nonfinite_rejected():
    for a,b in [([1],[1,2]),({'x':1},{}),(True,1),(float('nan'),float('nan')),(.1,1)]:
        with pytest.raises(ValueError):compare_common(a,b)
