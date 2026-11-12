"""TRAIN-only availability pruning for a separately identified STL contrast."""
import numpy as np
import pandas as pd

from app.rl_subset import coherent_blocks


def availability_subset(names, valid, times, periods, forbidden, reference_valid=None):
    valid=np.asarray(valid);times=pd.to_datetime(times,utc=True)
    if (valid.dtype!=bool or valid.shape!=(len(times),len(names)) or
            len(set(names))!=len(names) or not len(names) or
            not times.is_monotonic_increasing or not times.is_unique):
        raise ValueError('STL_AVAILABILITY_INPUT')
    start,end=(pd.Timestamp(t,tz='UTC') for t in periods['TRAIN'])
    if reference_valid is not None:
        reference_valid=np.asarray(reference_valid)
        if reference_valid.dtype!=bool or reference_valid.shape!=(len(times),):
            raise ValueError('STL_AVAILABILITY_REFERENCE')
        reference_train=reference_valid & (times>=start) & (times<end)
        if not reference_train.any():raise ValueError('STL_AVAILABILITY_NO_REFERENCE_TRAIN')
    months=pd.date_range(start.normalize().replace(day=1),end,freq='MS',inclusive='left')
    masks={m.strftime('%Y-%m'):(times>=max(start,m)) &
           (times<min(end,m+pd.offsets.MonthBegin(1))) for m in months}
    removed={};coverage={}
    for j,name in enumerate(names):
        coverage[name]={m:int(valid[mask,j].sum()) for m,mask in masks.items()}
        reasons=['empty_TRAIN_month:'+m for m,n in coverage[name].items() if not n]
        if reference_valid is not None:
            lost=int((reference_train & ~valid[:,j]).sum())
            if lost:reasons.append('lost_reference_TRAIN_windows:'+str(lost))
        if name in forbidden:reasons.append(forbidden[name])
        if reasons:removed[name]=reasons
    for block in coherent_blocks(names):
        if any(name in removed for name in block):
            for name in block:removed.setdefault(name,['atomic_block:'+','.join(block)])
    selected=[name for name in names if name not in removed]
    if not selected:raise ValueError('STL_NO_AVAILABLE_COLUMNS')
    joint=valid[:,[names.index(n) for n in selected]].all(1)
    if not masks or any(not joint[mask].any() for mask in masks.values()):
        raise ValueError('STL_JOINT_TRAIN_COVERAGE')
    return dict(selected=selected,removed=removed,train_monthly_coverage=coverage,
                preserve_full_reference_train=reference_valid is not None,
                selection_uses_validation=False,selection_uses_trading_metrics=False)
