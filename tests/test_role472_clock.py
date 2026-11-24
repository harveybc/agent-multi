import numpy as np
import pytest
from tools.role472_sell_clock import NS, MON, WEEK, month_permutations, replay


def test_schedule_skips_exit_bar_and_compounds():
    ii=np.arange(4);pnl=np.full((4,6),100.)
    ends=np.repeat(np.array([1,1,3,3])[:,None],6,axis=1)
    panel=(ii,None,pnl,ends)
    r=replay([(0,5,1.),(1,5,1.),(2,5,.5)],panel,ii*NS)
    assert r['n']==2 and r['skips']==1 and r['q_bars']==3
    assert r['ret']==pytest.approx(1.505)


def test_week_permutation_preserves_utc_slot_and_sizes():
    ns=MON+np.array([1,2])*WEEK+12*NS
    ii=np.arange(2);end=np.tile(ii[:,None],(1,6))
    opts=month_permutations((ii,None,None,end),ns,[(0,5,.3)])
    assert len(opts)==1
    assert sorted((s[0][0],s[0][2],moved) for s,moved in opts[0])==[(0,.3,0),(1,.3,1)]
