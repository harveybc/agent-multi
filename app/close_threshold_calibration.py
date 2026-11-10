"""Exact empirical fit-only threshold calibration for frozen close scores."""
import numpy as np

BASELINE=1e-8
TIE_TOLERANCE=1e-12


def fit_threshold(scores,target,banks):
    z=np.asarray(scores,dtype=float);y=np.asarray(target,dtype=float)
    if (z.ndim!=1 or not len(z) or y.shape!=z.shape or not np.isfinite([z,y]).all()
            or not banks or any(not len(b) for b in banks)
            or sorted(i for b in banks for i in b)!=list(range(len(z)))):
        raise ValueError('CLOSE_THRESHOLD_SUPPORT')
    weight=np.zeros(len(z))
    for b in banks:weight[b]=1/(len(banks)*len(b))
    order=np.argsort(z,kind='stable');unique,counts=np.unique(z,return_counts=True)
    stops=np.cumsum(counts);wy=(weight*y)[order].astype(np.longdouble)
    all_close=np.sum((weight*np.maximum(y,0)).astype(np.longdouble))-np.sum(wy)
    regrets=np.r_[all_close,all_close+np.cumsum(wy)[stops-1]].astype(float)
    thresholds=[]
    for j in range(len(unique)+1):
        lo=unique[j-1] if j else None;hi=unique[j] if j<len(unique) else None
        if lo is not None and BASELINE<lo:t=lo
        elif hi is not None and BASELINE>=hi:t=np.nextafter(hi,-np.inf)
        else:t=BASELINE
        if not np.isfinite(t):raise ValueError('CLOSE_THRESHOLD_UNREPRESENTABLE')
        thresholds.append(float(t))
    minimum=float(regrets.min());eligible=np.flatnonzero(regrets<=minimum+TIE_TOLERANCE)
    selected=min(eligible,key=lambda i:(abs(thresholds[i]-BASELINE),thresholds[i]))
    threshold=thresholds[selected]
    def measure(t):
        error=np.maximum(y,0)-(z>t)*y
        return float(np.mean([error[b].mean() for b in banks]))
    actual=measure(threshold)
    if abs(actual-regrets[selected])>TIE_TOLERANCE:raise ValueError('CLOSE_THRESHOLD_PREFIX_PARITY')
    return dict(threshold=threshold,fit_regret=actual,minimum_fit_regret=minimum,
        baseline_fit_regret=measure(BASELINE),n=len(z),banks=len(banks),
        fit_requested_closes=int((z>threshold).sum()),tie_tolerance=TIE_TOLERANCE,
        partitions=len(thresholds),candidates=[dict(threshold=t,fit_regret=float(v),
            requested_closes=int(len(z)-(stops[i-1] if i else 0))) for i,(t,v) in enumerate(zip(thresholds,regrets))])


def fit_thresholds(scores,records,banks,mode):
    z=np.asarray(scores)
    if z.shape!=(len(records),) or mode not in ('common','directional'):
        raise ValueError('CLOSE_THRESHOLD_MODE')
    if sorted(i for b in banks for i in b)!=list(range(len(records))):
        raise ValueError('CLOSE_THRESHOLD_PANEL_COVERAGE')
    if any(len({records[i]['role'] for i in b})!=1 for b in banks):
        raise ValueError('CLOSE_THRESHOLD_MIXED_ROLE')
    result={}
    for direction in (('all',) if mode=='common' else ('long','short')):
        chosen=[[i for i in b if records[i]['role']=='fit' and
                 (direction=='all' or records[i]['direction']==direction)] for b in banks]
        chosen=[b for b in chosen if b];ids=[i for b in chosen for i in b];lookup={i:j for j,i in enumerate(ids)}
        result[direction]=fit_threshold(z[ids],np.array([records[i]['gap'] for i in ids]),
                                       [[lookup[i] for i in b] for b in chosen])
    return dict(mode=mode,thresholds=result,comparison='float64_score_strictly_greater_than_threshold')


def apply_thresholds(scores,directions,calibration):
    z=np.asarray(scores,dtype=float);directions=np.asarray(directions)
    if z.shape!=directions.shape or z.ndim!=1 or not np.isfinite(z).all() or not np.isin(directions,['long','short']).all():
        raise ValueError('CLOSE_THRESHOLD_QUERY')
    if calibration['mode']=='common':threshold=calibration['thresholds']['all']['threshold']
    elif calibration['mode']=='directional':
        threshold=np.array([calibration['thresholds'][d]['threshold'] for d in directions])
    else:raise ValueError('CLOSE_THRESHOLD_MODE')
    if not np.isfinite(threshold).all():raise ValueError('CLOSE_THRESHOLD_NONFINITE')
    return z>threshold
