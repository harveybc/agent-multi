import numpy as np
import pandas as pd

from tools.role482_net_gate import proposals


def test_gate_is_not_occupancy_or_hindsight_horizon():
    ts=np.arange(1,7)
    d=np.tile([3,0,0,2,0,0],(6,1))
    d[4]=0  # even a positive score cannot break a gross-side tie
    d[5,3]=4
    net=np.array([[-3,-2,-1,7,7,7],[1,15,4,99,0,0],[4,3,2,0,0,0],
                  [-2,-3,0,0,0,0],[9,0,0,0,0,0],[0,0,0,-3,-4,-2]])
    pnl=np.zeros((6,6))
    pnl[0,2],pnl[1,1],pnl[5,5]=8,11,-9
    book=pd.DataFrame([dict(entry_ns=2,exit_bar_open_ns=4,hours=24,side='buy',
                            edge_bp=15.,q=1.,balance=10011.,net_usd=11.)])
    r=proposals(ts,d,net,pnl,book)
    assert list(r.ts)==[1,2,3,4,6]
    assert list(r.approved)==[False,True,True,False,False]
    assert list(r.busy)==[False,False,True,True,False]
    assert list(r.h)==[48,24,6,48,48]
    assert list(r.net_usd_q1)==[8,11,0,0,-9]
