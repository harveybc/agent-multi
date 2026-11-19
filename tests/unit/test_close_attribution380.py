import pytest
from docs.work_plan.generalization.close_attribution380 import partition

def trade(bar,pnl,size=10):
    return dict(entry_bar_abs=bar,direction=1,net_usd=pnl,units=size)

def test_partition_and_reverse():
    a=[trade(1,-4),trade(3,5),trade(5,-6)]
    b=[trade(1,1),trade(3,3,6),trade(6,4)]
    expected=dict(current=5,matched=-2,changed_entries=10)
    assert partition(a,b,(1,1,10))==expected
    assert partition(b,a,(1,1,10))=={k:-v for k,v in expected.items()}
    assert partition(a[:1],a[:1],(1,1,10))==dict.fromkeys(expected,0)

def test_fail_closed():
    for a in ([],[trade(2,0)],[trade(1,0,9)],[trade(1,0),trade(2,0),trade(2,1)]):
        with pytest.raises(ValueError):
            partition(a,[trade(1,0)],(1,1,10))
