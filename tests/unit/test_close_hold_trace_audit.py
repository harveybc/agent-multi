import pytest

from tools.audit_close_hold_prefix_trace import audit


def test_late_signal_not_counted_as_executable_close():
    row=dict(arm='prefix_guarded',spec=dict(id='x'),metrics=dict(early_close=False),
             trace=[dict(decision_bar=10,advantage_risk=0.,expected_close_net=1.,best_expected_wait_net=2.),
                    dict(decision_bar=11,advantage_risk=.1,expected_close_net=3.,best_expected_wait_net=2.)])
    out=audit([row],[dict(spec=dict(id='x'),fill_bars=[11])])
    assert all(v['first_positive_requests']==1 and v['executable_early_requests']==0 for v in out.values())
    out=audit([row],[dict(spec=dict(id='x'),fill_bars=[11,12])])
    assert all(v['executable_early_requests']==1 for v in out.values())


def test_incomplete_trace_cannot_be_reused_as_passive():
    row=dict(arm='prefix_guarded',spec=dict(id='x'),metrics=dict(early_close=True))
    with pytest.raises(ValueError,match='COMPLETE_PASSIVE'):
        audit([row],[dict(spec=dict(id='x'),fill_bars=[11])])
    with pytest.raises(ValueError,match='SUPPORT'):
        audit([], [dict(spec=dict(id='x'),fill_bars=[])])
