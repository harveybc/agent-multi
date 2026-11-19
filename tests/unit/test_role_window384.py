import numpy as np
import pytest
from app.role_window384 import windows

def test_owner_case_inclusive_edges_mirror_and_censoring():
    pivots=[(0,-1,1),(8,1,1),(20,-1,1),(36,1,1),(48,-1,1)]
    out,ok=windows(50,pivots)
    np.testing.assert_array_equal(np.flatnonzero(out[:,0]),np.arange(17,29))
    assert not ok[:8].any() and not ok[37:].any() and ok[8:37].all()
    rev,_=windows(50,[(i,-s,v) for i,s,v in pivots])
    np.testing.assert_array_equal(out,rev[:,::-1])
    assert not windows(8,[])[1].any()
    with pytest.raises(ValueError):windows(50,pivots,-.1,.5)
