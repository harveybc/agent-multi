import numpy as np
import pytest
from app.matched_divergence import locate_events, factorial_effects


def test_first_disagreement_and_control_acquisition():
    y=np.array([[0.,2.,1.]])
    c=np.array([[0],[0],[1],[1]])
    t=np.array([[0],[2],[2],[2]])
    result=locate_events(c,t,y,[0])
    assert result==[dict(row=0,control_action=1,treatment_action=2,control_advantage=1.,first_disagreement=1,control_acquisition=2)]


def test_tie_is_not_a_deficit():
    assert locate_events(np.array([[0],[1]]),np.array([[0],[2]]),np.array([[0.,1.,1.]]),[0])==[]


def test_already_superior_at_source_is_not_an_update():
    r=locate_events(np.array([[1],[1]]),np.array([[0],[0]]),np.array([[0.,2.,1.]]),[0])
    assert r[0]['control_acquisition'] is None
    assert r[0]['first_disagreement']==0


def test_bad_actions_or_shapes_rejected():
    for c,t in [(np.array([[3],[0]]),np.array([[0],[0]])),(np.array([[0]]),np.array([[0],[0]]))]:
        with pytest.raises(ValueError):locate_events(c,t,np.zeros((1,3)),[0])


def test_factorial_known_effects():
    r=factorial_effects({'CC':1.,'CT':3.,'TC':4.,'TT':8.})
    assert r==dict(optimizer_at_control_weights=2.,optimizer_at_treatment_weights=4.,weights_at_control_optimizer=3.,weights_at_treatment_optimizer=5.,interaction=2.)


@pytest.mark.parametrize('cells',[{'CC':0.},{'CC':0.,'CT':1.,'TC':2.,'TT':np.nan}])
def test_factorial_invalid(cells):
    with pytest.raises(ValueError):factorial_effects(cells)
