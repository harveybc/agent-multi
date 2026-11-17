import numpy as np
import pandas as pd
import pytest
from app.nested_history_preflight import contained,supports


def fixture():
    t=pd.date_range('2020-01-01','2024-12-31',freq='D',tz='UTC').as_unit('ns').asi8
    slots=np.stack([t-24*3600*10**9,t],axis=1)
    return t,slots,np.ones(len(t),bool)


def test_nested_and_window_boundary():
    result=supports(*fixture())
    assert [r['months'] for r in result]==[6,12,24,48]
    assert result[0]['first_close'].startswith('2023-07-02')
    assert len({r['validation_support_sha256'] for r in result})==1
    assert all(a['rows']<b['rows'] for a,b in zip(result,result[1:]))


@pytest.mark.parametrize('fit,expected',[(None,False),(['2020-01-01','2023-07-01'],False),
    (['2023-07-01','2024-01-01'],True),(['2023-07-01','2024-01-02'],False)])
def test_containment(fit,expected):
    assert contained(fit,['2023-07-01','2024-01-01']) is expected


def test_test_data_forbidden():
    t,s,v=fixture();t[-1]=pd.Timestamp('2025-01-01',tz='UTC').value;s[-1,-1]=t[-1]
    with pytest.raises(ValueError,match='TEST_FORBIDDEN'):supports(t,s,v)


def test_empty_not_zero_success():
    t,s,v=fixture();v[:]=False
    with pytest.raises(ValueError,match='EMPTY_VALIDATION'):supports(t,s,v)
