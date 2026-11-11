import numpy as np
import pandas as pd
import pytest
from tools.rebuild_recovered_fx_features import COLS,install_changes,same


def test_finite_masks_must_match():
    assert not same([np.nan,1],[0,1])
    assert same([np.nan,1],[np.nan,1])


def test_only_declared_dependency_cells_change():
    s=pd.DataFrame({c:[1.,2.,3.] for c in COLS});s['other']=[4.,5.,6.]
    b=s[COLS].copy();n=b.copy();n.iloc[1]=7.
    f=install_changes(s,b,n,np.array([False,True,False]))
    assert f['other'].equals(s['other'])
    assert f.iloc[0].equals(s.iloc[0])
    assert s.iloc[1,0]==2.
    n.iloc[0]=8.
    with pytest.raises(ValueError,match='OUTSIDE'):
        install_changes(s,b,n,np.array([False,True,False]))


def test_old_producer_and_column_identity_required():
    s=pd.DataFrame({c:[1.,2.] for c in COLS});b=s.copy();b.iloc[0,0]=9
    with pytest.raises(ValueError,match='BASELINE'):
        install_changes(s,b,s,np.array([True,True]))
    with pytest.raises(ValueError,match='ORDER'):
        install_changes(s,s,s[COLS[::-1]],np.array([True,True]))
