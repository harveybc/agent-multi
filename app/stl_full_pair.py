"""Explicit temporal-support relaxation without feature pruning."""
import numpy as np
import pandas as pd
import torch


def paired_coverage(times, reference, common, periods, minimum_fraction=0.8):
    times=np.asarray(times);reference=np.asarray(reference,dtype=bool);common=np.asarray(common,dtype=bool)
    if (times.ndim!=1 or reference.shape!=times.shape or common.shape!=times.shape
            or (np.diff(times)<=0).any() or (common & ~reference).any()):
        raise ValueError('FULL71_SUPPORT_SHAPE')
    report={}
    for name,(start,end) in periods.items():
        inside=(times>=pd.Timestamp(start,tz='UTC').value)&(times<pd.Timestamp(end,tz='UTC').value)
        n=int((reference&inside).sum());kept=int((common&inside).sum())
        if not n or kept/n<minimum_fraction:raise ValueError('FULL71_INSUFFICIENT_SUPPORT:'+name)
        missing=np.flatnonzero(reference&inside&~common);runs=[]
        for block in np.split(missing,np.flatnonzero(np.diff(missing)>1)+1):
            if len(block):runs.append(dict(first=pd.Timestamp(times[block[0]],tz='UTC').isoformat(),
                last=pd.Timestamp(times[block[-1]],tz='UTC').isoformat(),rows=len(block)))
        valid=np.flatnonzero(common&inside)
        report[name]=dict(reference=n,retained=kept,lost=n-kept,fraction=kept/n,
            first=pd.Timestamp(times[valid[0]],tz='UTC').isoformat(),
            last=pd.Timestamp(times[valid[-1]],tz='UTC').isoformat(),excluded_runs=runs)
    return report


def assert_declared_changes(before, after, columns, allowed):
    if before.shape!=after.shape or before.shape[1]!=len(columns):raise ValueError('FULL71_RAW_SHAPE')
    untouched=[j for j,n in enumerate(columns) if n not in allowed]
    if not np.array_equal(before[:,untouched],after[:,untouched],equal_nan=True):
        raise ValueError('FULL71_UNDECLARED_CHANGE')


def copy_loss_weights(model, donor):
    """Transfer values only; each arm retains its fresh optimizer/replay."""
    source=donor.loss.state_dict();target=model.loss.state_dict()
    if set(source)!=set(target) or any(source[k].shape!=target[k].shape for k in source):
        raise ValueError('FULL71_WEIGHT_SHAPE')
    model.loss.load_state_dict(source,strict=True)
    if any(not torch.equal(v.detach().cpu(),model.loss.state_dict()[k].detach().cpu()) for k,v in source.items()):
        raise ValueError('FULL71_WEIGHT_TRANSFER')
    return len(source)
