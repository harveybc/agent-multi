import numpy as np
import pytest

from app.account_counterexamples import new_constraints, account_qualifies


def event(stamp=10, delta=-2.):
    return dict(timestamp=stamp,old_signal=0,new_signal=2,contribution_usd=delta)


def test_constraints_only_from_negative_training_account_effects():
    rows=new_constraints([event(),event(delta=3.),event(20,0.)],np.array([10,20]),
        np.array(['TRAIN','TRAIN']),set())
    assert rows==[(0,0,2)]
    assert new_constraints([event()],np.array([10]),np.array(['TRAIN']),{(0,0,2)})==[]


def test_validation_or_unmapped_timestamp_is_rejected():
    with pytest.raises(ValueError,match='TRAIN_ONLY'):
        new_constraints([event()],np.array([10]),np.array(['VALIDATION']),set())
    with pytest.raises(ValueError,match='TIMESTAMP'):
        new_constraints([event()],np.array([11]),np.array(['TRAIN']),set())
    with pytest.raises(ValueError,match='NONFINITE'):
        new_constraints([event(delta=float('nan'))],np.array([10]),np.array(['TRAIN']),set())


def test_positive_total_cannot_hide_harmful_changed_decision():
    reference={'net_profit_usd':10.,'orders_per_week':3.}
    candidate={'net_profit_usd':15.,'orders_per_week':3.}
    assert account_qualifies(reference,candidate,[event(delta=5.)])
    assert not account_qualifies(reference,candidate,[event(delta=-1.),event(delta=6.)])
    assert not account_qualifies(reference,reference,[])
    assert not account_qualifies(reference,dict(candidate,orders_per_week=0.),[])
