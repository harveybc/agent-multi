import numpy as np
import pytest

from app.anchored_action_attribution import paired_event_metrics
from tests.unit.test_anchored_action_attribution import fixture


def test_both_critics_use_calibrated_policy_gap_not_account_policy_gap():
    e=fixture()
    e.update(calibration_old=dict(reward_return=.4),calibration_new=dict(reward_return=.1),
             isolated_state=dict(position=dict(direction=-1)))
    parent=np.array([.01,-1.,0.]);candidate=np.array([0.,-1.,.01])
    old=np.array([[0.,0.,.2],[.1,.1,.3]])
    new=np.array([[0.,0.,-.25],[.1,.1,-.15]])
    r=paired_event_metrics(e,parent,candidate,old,new,.005)
    assert r['calibration_reward_gap']==pytest.approx(-.3)
    assert r['isolated_reward_gap']==pytest.approx(.1)
    assert r['baseline_q_gap']==pytest.approx(.2)
    assert r['q_gap']==pytest.approx(-.25)
    assert r['baseline_calibration_conflict'] and not r['candidate_calibration_conflict']
    assert r['candidate_gap_error']<r['baseline_gap_error']
    assert r['actor_change_against_candidate_q']
    assert r['isolated_position']==-1 and r['position']==0


def test_invalid_calibration_gap_refuses():
    e=fixture();e.update(calibration_old=dict(reward_return=0.),
        calibration_new=dict(reward_return=float('nan')),isolated_state=dict(position=None))
    with pytest.raises(ValueError,match='CALIBRATION_GAP'):
        paired_event_metrics(e,np.array([1.,0.,0.]),np.array([0.,0.,1.]),
                             np.zeros((2,3)),np.zeros((2,3)),.005)
