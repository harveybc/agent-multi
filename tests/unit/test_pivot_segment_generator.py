import numpy as np
import pytest
from app.pivot_segment_generator import controls, draw, match


def test_controls_steer_independently_and_retain_randomness():
    rng=np.random.default_rng(31);y=rng.normal(size=21)
    noise=rng.normal(size=(2,len(y))); base=controls(y)
    for target in (base,base+[.12,0,0],base+[0,.3,0],base+[0,0,.02]):
        z=draw(y,noise,target)
        np.testing.assert_allclose(controls(z),target,atol=1e-12)
        assert not np.allclose(z,y)
        np.testing.assert_array_equal(z,draw(y,noise,target))
    with pytest.raises(ValueError):draw(y,noise,[0,0,1])


def test_pivots_not_double_counted_or_matched_across_gaps():
    assert match([[(3,1),(5,1)],[(1,-1)]],[[(4,1)],[(1,1)]],1)==1


def test_joint_generator_protects_pivots_gaps_and_ohlc():
    import pandas as pd
    from app.pivot_segment_generator import generate
    clock=pd.date_range('2021-03-01',periods=82,freq='h',tz='UTC').delete([41,42])
    rng=np.random.default_rng(23)
    c=np.exp(np.cumsum(rng.normal(0,.001,len(clock))));o=np.r_[c[0],c[:-1]]
    bars=pd.DataFrame(dict(OPEN=o,HIGH=np.maximum(c,o),LOW=np.minimum(c,o),CLOSE=c),index=clock)
    fx=pd.DataFrame({'cross':c*2},index=clock)
    fx.iloc[54]=np.nan
    pairs=[[(2,-1),(13,1),(22,-1),(39,1)],[(43,1),(66,-1)]]
    gen,x,mask,rows=generate(bars,fx,pairs,3,warmup=1)
    np.testing.assert_allclose(rows[:,2:5],rows[:,5:8],atol=1e-14)
    assert not mask[40] and not mask[41]
    assert x.isna().equals(fx.isna())
    assert (gen.LOW<=gen[['OPEN','CLOSE']].min(axis=1)).all()
    assert (gen.HIGH>=gen[['OPEN','CLOSE']].max(axis=1)).all()
    assert np.any(abs(gen.CLOSE-bars.CLOSE)>1e-8)
    pd.testing.assert_frame_equal(gen.iloc[[2,13,22,39,43,66]],bars.iloc[[2,13,22,39,43,66]])
    np.testing.assert_allclose(x.iloc[:40,0],gen.CLOSE.iloc[:40]*2,atol=1e-13)
