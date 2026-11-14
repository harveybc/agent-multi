"""Verifier rejects weakened guards, broken trace chains and harmful decisions."""
from copy import deepcopy
import numpy as np
import pytest
from tools.verify_optimal_protection_continuation import score,verify_trace


def fixture():
    before={'numpy':dict(cost=2.,regret=1.)}
    after=dict(cost=1.,regret=.5,passed=True,lost_optimal=0)
    trace=[dict(offset=1,accepted=True,details=dict(baseline=before,
        assessment=dict(passed=True,original_passed=True,motion=.1,metrics={'numpy':after})))]
    return before,trace


def test_trace_valid():
    before,trace=fixture()
    accepted,last=verify_trace(trace,before)
    assert accepted==1 and last['numpy']['regret']==.5


@pytest.mark.parametrize('key,value',[('cost',2.),('regret',1.1),('lost_optimal',1),('passed',False)])
def test_trace_rejects_bad_quality(key,value):
    before,trace=fixture();trace[0]['details']['assessment']['metrics']['numpy'][key]=value
    with pytest.raises(ValueError,match='QUALITY'):verify_trace(trace,before)


def test_trace_rejects_chain_break():
    before,trace=fixture();trace[0]['details']['baseline']=deepcopy(before)
    trace[0]['details']['baseline']['numpy']['cost']=3.
    with pytest.raises(ValueError,match='CHAIN'):verify_trace(trace,before)


def test_score_tied_optimal_and_wrong():
    q=np.array([[[0.,2.,1.],[2.,0.,1.]]]*2);y=np.array([[0.,1.,1.],[0.,1.,2.]])
    got=score(q,y,.5)
    assert got['regret']==1.
    assert got['cost']>0
