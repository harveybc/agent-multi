import numpy as np
import pytest


def test_soft_return_excludes_current_entropy_and_respects_duration():
    from app.policy_return_calibration import discounted_returns
    rewards=np.array([1.,2.]);duration=np.array([2,1]);surprise=np.array([999.,2.])
    assert discounted_returns(rewards,duration,surprise,.5,.1)==pytest.approx([1.55,2.])
    assert discounted_returns(rewards,duration,surprise,.5,0.)==pytest.approx([1.5,2.])
    assert discounted_returns(np.array([3.]),np.array([40]),np.array([100.]),.99,.1)==pytest.approx([3.])


@pytest.mark.parametrize('durations',[[0,1],[1.5,1],[-1,1]])
def test_invalid_duration_rejected(durations):
    from app.policy_return_calibration import discounted_returns
    with pytest.raises(ValueError,match='RETURN_INPUT'):
        discounted_returns(np.ones(2),np.array(durations),np.zeros(2),.99,.1)


def test_categorical_sampler_matches_softmax_and_is_reproducible():
    import torch
    from app.policy_return_calibration import policy_signals
    logits=np.tile(np.log([.2,.3,.5]),(100000,1)).astype(np.float32)
    a,p=policy_signals(logits,7)
    again,_=policy_signals(logits,7)
    np.testing.assert_array_equal(a[0],again[0])
    np.testing.assert_allclose(p[0],torch.softmax(torch.tensor(logits[0]),0).numpy(),atol=1e-7)
    np.testing.assert_allclose(np.bincount(a[0],minlength=3)/len(logits),[.2,.3,.5],atol=.005)


def test_greedy_tie_is_nop_and_invalid_rows_stay_invalid():
    from app.policy_return_calibration import policy_signals
    logits=np.array([[0.,2.,2.],[0.,1.,2.],[np.nan,np.nan,np.nan]],dtype=np.float32)
    prepared,p=policy_signals(logits,None)
    assert prepared[0].tolist()==[0,2,0]
    assert prepared[3].tolist()==[True,True,False]
    assert np.isnan(p[2]).all()


def test_calibration_statistics_known_errors():
    from app.policy_return_calibration import calibration
    result=calibration(np.array([1.,3.]),np.array([0.,1.]))
    assert result['bias']==pytest.approx(1.5)
    assert result['mae']==pytest.approx(1.5)
    assert result['rmse']==pytest.approx(np.sqrt(2.5))
    assert sum(b['n'] for b in result['bins'])==2


def test_account_trace_mapping_and_greedy_scope(tmp_path):
    import json
    from types import SimpleNamespace
    from tools.audit_frozen_policy_returns import account_calibration,metric_text
    trace=[dict(timestamp=i,signal=1,position=0,reward=float(i),duration=1) for i in (1,2)]
    (tmp_path/'ACTIONS.json').write_text(json.dumps(trace))
    inputs=SimpleNamespace(row_ts_ns=np.array([1,2]),valid=np.ones(2,dtype=bool))
    q=np.zeros((2,2,3));p=np.full((2,3),1/3)
    result=account_calibration(tmp_path,inputs,q,p,.5,.1,True)
    assert result['rows']==2
    assert result['min_q']['return_mean']==pytest.approx(2.+.025*np.log(3))
    greedy=account_calibration(tmp_path,inputs,q,p,.5,.1,False)
    assert greedy['min_q']['return_mean']==pytest.approx(2.)
    assert 'not on-policy' in greedy['scope']
    assert metric_text(None)=='NA'
    inputs.valid[0]=False
    with pytest.raises(ValueError,match='CALIBRATION_TRACE_SUPPORT'):
        account_calibration(tmp_path,inputs,q,p,.5,.1,True)
