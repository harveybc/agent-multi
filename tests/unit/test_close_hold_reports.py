import numpy as np
import pytest

from app.close_hold_reports import prefix_report


def test_prefix_report_uses_completed_bars_only_and_keeps_joint_samples():
    prefix=np.repeat(np.array([1.,1.01,1.02,1.025])[:,None],4,axis=1)
    a=prefix_report(prefix,100,10,140,.0002)
    b=prefix_report(prefix.copy(),100,10,140,.0002)
    assert np.array_equal(a.paths,b.paths)
    assert a.paths.shape==(3,4,4)
    assert a.observed_through==a.issued_at==100
    assert a.provenance=='causal'
    assert np.all(a.paths[:,0,0]==prefix[-1,3])
    assert not np.array_equal(a.paths[0],a.paths[2])


@pytest.mark.parametrize('prefix',[np.ones((1,4)),np.ones((3,3)),np.full((3,4),np.nan)])
def test_invalid_prefix_refused(prefix):
    with pytest.raises(ValueError): prefix_report(prefix,100,10,140,.0002)
