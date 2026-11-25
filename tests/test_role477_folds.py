import numpy as np
from tools.role477_train import masks


def test_purge_labels_at_each_boundary_and_preserve_missing_targets():
    hour=3600*10**9
    t=np.arange(40)*hour
    ok=np.ones(40,bool);ok[12]=False
    train,monitor,test=masks(t,ok,3,hour,10*hour,20*hour,30*hour)
    assert np.flatnonzero(train).tolist()==list(range(1,7))
    assert np.flatnonzero(monitor).tolist()==[10,11,13,14,15,16]
    assert np.flatnonzero(test).tolist()==list(range(20,27))
    assert not (train & monitor | train & test | monitor & test).any()
