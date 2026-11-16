"""Full-support function changes between sealed and current actor outputs."""
import numpy as np

from app.frozen_signal_masks import prepared_signals


def policy_delta(before, after, periods):
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 3 or len(periods) != len(before):
        raise ValueError('POLICY_DELTA_SHAPE')
    a,b=prepared_signals(before),prepared_signals(after)
    if not np.array_equal(a[3],b[3]):raise ValueError('POLICY_DELTA_SUPPORT')
    result={}
    for period in ('TRAIN','VALIDATION'):
        valid=a[3]&(periods==period);changed=valid&(a[0]!=b[0])
        old=np.sort(before[changed].astype(float),axis=1)
        x,y=before[valid].astype(float),after[valid].astype(float)
        if len(x)==0:raise ValueError('POLICY_DELTA_EMPTY_PERIOD')
        def logprob(z):
            z=z-z.max(axis=1,keepdims=True)
            return z-np.log(np.exp(z).sum(axis=1,keepdims=True))
        lp,lq=logprob(x),logprob(y)
        result[period]=dict(valid_windows=int(valid.sum()),changed_signals=int(changed.sum()),
            mean_kl_current_to_source=float((np.exp(lq)*(lq-lp)).sum(axis=1).mean()),
            median_source_margin_changed=None if len(old)==0 else float(np.median(old[:,-1]-old[:,-2])),
            transitions={f'{i}->{j}':int((valid&(a[0]==i)&(b[0]==j)).sum()) for i in range(3) for j in range(3)})
    return result
