import numpy as np
import pytest
from app.close_surface import coordinates, weights, fit, scores

def test_direction_time_gap_and_tie():
    long = coordinates(1, 101, 98, 106, 100, 12)
    short = coordinates(-1, 99, 102, 94, 100, 12)
    np.testing.assert_array_equal(long, short)
    bounds = np.array([[0,0,0],[2,2,24]])
    for hours in (0,12,24):
        w = weights(coordinates(1, 95,98,106,100,hours), bounds)
        assert np.isclose(w.sum(),1) and (w>=0).all()
        assert np.isclose(w @ np.tile([0,24],4),hours)
        assert np.isclose(w[4:].sum(),0)
    with pytest.raises(ValueError):
        coordinates(1,100,98,106,100,-1)
    assert not (np.zeros(8) @ weights(long,bounds) > 0)

def test_same_market_can_require_opposite_actions():
    x=np.zeros((160,2))
    q=np.tile([[0,2,0],[2,0,0]],(80,1))
    y=np.tile([-1.,1.],80)
    model=fit(x,q,y)
    a,b=scores(model,x,q)
    assert np.all(a==a[0])
    np.testing.assert_array_equal(b>0,y>0)
