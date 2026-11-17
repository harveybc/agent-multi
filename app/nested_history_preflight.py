"""Exact nested score supports; no preprocessing fit or training authorization."""
import hashlib
import numpy as np
import pandas as pd


def contained(fit, interval):
    if fit is None:
        return False
    a,b=map(pd.Timestamp,fit);start,end=map(pd.Timestamp,interval)
    if not a<b or not start<end:raise ValueError('INVALID_FIT_INTERVAL')
    return start<=a and b<=end


def supports(row_ts,slot_ts,valid,months=(6,12,24,48)):
    row_ts=np.asarray(row_ts);slot_ts=np.asarray(slot_ts);valid=np.asarray(valid,bool)
    if slot_ts.ndim!=2 or len(row_ts)!=len(valid) or slot_ts.shape[0]!=len(valid):
        raise ValueError('SUPPORT_SHAPE')
    if len(row_ts)==0 or np.any(np.diff(row_ts)<=0) or np.any(slot_ts[:,-1]!=row_ts):
        raise ValueError('SUPPORT_CLOCK')
    end=pd.Timestamp('2024-01-01',tz='UTC');test=pd.Timestamp('2025-01-01',tz='UTC')
    if np.any(row_ts>=test.value):raise ValueError('TEST_FORBIDDEN')
    validation=np.flatnonzero(valid & (row_ts>=end.value) & (row_ts<test.value))
    if not len(validation):raise ValueError('EMPTY_VALIDATION')
    result=[];last=set()
    for month in sorted(set(months)):
        if type(month) is not int or month<=0:raise ValueError('INVALID_MONTHS')
        start=end-pd.DateOffset(months=month)
        fit=np.flatnonzero(valid & (row_ts>=start.value) & (row_ts<end.value)
                           & (slot_ts[:,0]>=start.value))
        if not len(fit):raise ValueError('EMPTY_FIT')
        if not last.issubset(set(fit)):raise ValueError('NON_NESTED_SUPPORT')
        last=set(fit)
        result.append(dict(months=month,interval=[start.date().isoformat(),end.date().isoformat()],
            rows=len(fit),first_close=pd.Timestamp(row_ts[fit[0]],tz='UTC').isoformat(),
            last_close=pd.Timestamp(row_ts[fit[-1]],tz='UTC').isoformat(),
            support_sha256=hashlib.sha256(row_ts[fit].astype('<i8').tobytes()).hexdigest(),
            validation_rows=len(validation),
            validation_support_sha256=hashlib.sha256(row_ts[validation].astype('<i8').tobytes()).hexdigest()))
    return result
