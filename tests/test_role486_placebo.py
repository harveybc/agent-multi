import numpy as np
import pytest
from tools.role486_placebo import HOUR, schedule


@pytest.mark.parametrize('nop_only,expected',[(False,{(0,3),(0,4),(1,3),(1,4),(2,4)}),
                                          (True,{(1,3),(1,4),(2,4)})])
def test_conditional_schedule_uniform_and_flat(nop_only,expected):
    # Without a pause at 0 the accepted order lasts through bar 0.
    t=np.arange(6)*HOUR
    trade,p,jump,count=schedule(t,t,np.array([True,False,False,False,False,False]),
                                t[[1,3]],t[[2,4]],5*HOUR,False,nop_only)
    paths={}
    def visit(i,chosen,prob):
        if i==6:
            paths[chosen]=prob
            return
        j=len(chosen)
        v=p[j,i]
        if v:visit(jump[j,i],chosen+(i,),prob*v)
        if v<1:visit(trade[i],chosen,prob*(1-v))
    visit(0,(),1.)
    # The second pause can also start at 2 when the first starts at 0.
    expected=expected|({(0,2)} if not nop_only else set())
    assert set(paths)==expected
    assert count==len(paths)
    np.testing.assert_allclose(list(paths.values()),1/count)


def test_moving_pause_cannot_interrupt_open_trade_or_cross_deadline():
    t=np.arange(6)*HOUR;ex=t.copy();ex[0]=t[3]
    _,p,_,ways=schedule(t,ex,np.array([True]+[False]*5),t[[4]],t[[5]],t[5],False,True)
    assert ways==1 and p[0,4]==1 and p[0,0]==0 and p[0,5]==0
