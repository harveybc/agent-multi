import numpy as np
import pytest

from app.policy_margin_trace import policy_delta


def test_full_support_changes_and_unchanged_logits():
    a=np.array([[1.,0.,-1.],[0.,1.,-1.],[np.nan]*3])
    periods=np.array(['TRAIN','VALIDATION','TRAIN'])
    r=policy_delta(a,a.copy(),periods)
    assert r['TRAIN']['changed_signals']==0 and r['VALIDATION']['mean_kl_current_to_source']==0
    b=a.copy();b[1]=[2.,1.,-1.]
    r=policy_delta(a,b,periods)
    assert r['VALIDATION']['changed_signals']==1
    assert r['VALIDATION']['transitions']['1->0']==1
    assert r['VALIDATION']['median_source_margin_changed']==1
    assert r['VALIDATION']['mean_kl_current_to_source']>0
    b[2]=0
    with pytest.raises(ValueError,match='SUPPORT'):policy_delta(a,b,periods)
