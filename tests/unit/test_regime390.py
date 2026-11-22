import numpy as np
import pandas as pd
from tools.regime390 import gate_table, generate, macroblocks, transport_basis


def fixture():
    t = pd.date_range('2021-01-01', periods=257, freq='h', tz='UTC')
    c = np.exp(np.cumsum(np.random.default_rng(8).normal(0,.001,len(t))))
    o = np.r_[c[0],c[:-1]]
    prices = np.column_stack([o,np.maximum(o,c)*1.001,np.minimum(o,c)*.999,c])
    a = np.arange(0,256,2)
    cat = pd.DataFrame(dict(start=a,end=a+2,retrace=.001,displacement=.002))
    blocks = macroblocks(cat,prices,t,4)
    labels = np.arange(len(blocks))%2
    fx = pd.DataFrame({'f':c*1.7},index=t)
    context = np.sin(np.arange(len(t))[:,None]*.1)
    return blocks,labels,t,prices,fx,context


def test_price_only_is_blind_to_macro_and_preserves_bar_geometry():
    data = fixture()
    x = generate(*data,7,False,False)
    y = generate(*data[:-1],np.full_like(data[-1],np.nan),7,False,False)
    for one,two in zip(x,y):
        np.testing.assert_array_equal(one,two)
    p,fx,trace = x
    np.testing.assert_allclose(p[:,3]*1.7,fx[:,0])
    assert (p[:,1]>=p[:,2]).all() and len(set(trace[:,-1]))>2
    assert (trace[1:,0]==trace[:-1,1]).all()
    np.testing.assert_allclose(p[1:,0],p[:-1,3])


def test_donors_do_not_cross_calendar_gaps():
    _,_,ts,p,_,_ = fixture()
    cat = pd.DataFrame(dict(start=[0,2,6,8,10,12],end=[2,4,8,10,12,14],
                            retrace=.1,displacement=.2))
    assert macroblocks(cat,p,ts,4)[:,:2].tolist()==[[6,14]]


def test_subset_gate_uses_names_not_positions():
    table = gate_table(pd.Series({'b':3.,'a':1.}),pd.Series({'a':0.,'b':0.,'c':0.}),
                       pd.Series({'c':1.,'a':.5,'b':4.}))
    assert table['pass'].to_dict()=={'b':True,'a':False}


def test_fx_basis_transports_observed_residual_not_an_exact_zero():
    c = np.array([1.02,1.03])
    fx = pd.DataFrame(dict(usdjpy=[120.,121.], gbpusd=[1.2,1.3]))
    fx['eurjpy'] = c*fx.usdjpy*[1.0001,.9998]
    fx['eurgbp'] = c/fx.gbpusd*[.9997,1.0002]
    series = fx.to_numpy().copy()*1.04
    path, donor = c*.97,np.array([1,0])
    transport_basis(c,fx,path,series,donor)
    np.testing.assert_allclose(series[:,2]/(path*series[:,0]),[.9998,1.0001])
    np.testing.assert_allclose(series[:,3]*series[:,1]/path,[1.0002,.9997])
