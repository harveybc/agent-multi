"""Small fit-boundary primitives shared by subset rebuild and its tests."""
import numpy as np


def window_mask(allowed,lookback):
    allowed=np.asarray(allowed,bool)
    if type(lookback) is not int or lookback<1:raise ValueError('LOOKBACK')
    bad=np.r_[0,np.cumsum(~allowed)];rows=np.arange(len(allowed));lo=np.maximum(0,rows-lookback+1)
    return allowed & (rows>=lookback-1) & ((bad[rows+1]-bad[lo])==0)


def assert_fit_rows(rows,allowed,hit=None,res=None,lookback=1):
    rows=np.asarray(rows,int);allowed=np.asarray(allowed,bool)
    if not len(rows) or np.any(rows<0) or np.any(rows>=len(allowed)):
        raise ValueError('FIT_ROWS')
    if not window_mask(allowed,lookback)[rows].all():raise ValueError('FIT_INPUT_SCOPE')
    if hit is not None:
        if not np.isfinite(hit[rows]).all() or len(np.unique(hit[rows]))!=2:
            raise ValueError('FIT_LABEL_CLASSES')
        if np.any(res[rows]<rows) or np.any(res[rows]>=len(allowed)):raise ValueError('FIT_LABEL_RESOLUTION')
        bad=np.r_[0,np.cumsum(~allowed)]
        if np.any(bad[res[rows]+1]-bad[rows]):raise ValueError('FIT_LABEL_SCOPE')


def fixed_scaler(values,allowed):
    values=np.asarray(values,float);mask=np.asarray(allowed,bool)&np.isfinite(values).all(axis=1)
    if mask.sum()<2:raise ValueError('SCALER_ROWS')
    mean=values[mask].mean(axis=0);std=values[mask].std(axis=0)
    constant=np.flatnonzero(std==0).tolist()
    return mean,np.where(std>0,std,1.),constant,np.flatnonzero(mask)
