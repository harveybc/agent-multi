"""Prefix-only resampled-path baseline, not a learned forecaster or input scaler."""
import numpy as np

from app.close_hold_controller import PathReport


def sampled_prefix_report(prefix,*,now,step,deadline,spreads,terminal_spread,
                          seed,samples=32,lookback=64,drift='median'):
    x=np.asarray(prefix,float)
    if (x.ndim!=2 or x.shape[1]!=4 or len(x)<2 or not np.isfinite(x).all()
            or (x<=0).any() or (x[:,1]<x.max(axis=1)).any()
            or (x[:,2]>x.min(axis=1)).any() or type(samples) is not int or samples<32
            or type(lookback) is not int or lookback<2 or drift not in ('median','zero')
            or any(type(t) is not int for t in (now,step,deadline))
            or step<=0 or deadline<=now or (deadline-now)%step):
        raise ValueError('CAUSAL_BASELINE_CONTRACT')
    h=(deadline-now)//step
    spread=np.asarray(spreads,float)
    if (spread.shape!=(h,) or not np.isfinite(spread).all() or (spread<0).any()
            or not np.isfinite(terminal_spread) or terminal_spread<0):
        raise ValueError('CAUSAL_BASELINE_SPREAD')
    x=x[-lookback:]
    increments=np.diff(x[:,3])
    location=float(np.median(increments)) if drift=='median' else 0.
    residuals=increments-np.median(increments)
    rng=np.random.default_rng(seed)
    selected=rng.integers(0,len(increments),size=(samples,h))
    close=x[-1,3]+np.cumsum(location+residuals[selected],axis=1)
    opening=np.column_stack((np.full(samples,x[-1,3]),close[:,:-1]))
    wick_hi=x[1:,1]-np.maximum(x[1:,0],x[1:,3])
    wick_lo=np.minimum(x[1:,0],x[1:,3])-x[1:,2]
    paths=np.stack((opening,np.maximum(opening,close)+wick_hi[selected],
                    np.minimum(opening,close)-wick_lo[selected],close),axis=2)
    if (paths-spread[None,:,None]/2<=0).any() or (close[:,-1]-terminal_spread/2<=0).any():
        raise ValueError('CAUSAL_BASELINE_NONPOSITIVE_PRICE')
    return PathReport(now,now,now,step,deadline,paths,
        np.broadcast_to(spread,(samples,h)).copy(),np.full(samples,1/samples),
        'causal',np.full(samples,terminal_spread))
