"""Support and equivalence gates for TRAIN-only opening counterfactuals."""
import numpy as np

BOUNDARY=1672531200000000000
END=1704067200000000000


def emit_actor_reference(model,inputs,indices,trainability):
    """Compare actor outputs under identical kernel-dispatch conditions."""
    from app.three_signal_learning import emit
    params=dict(model.loss.named_parameters())
    if params.keys()!=trainability.keys():raise ValueError('OPENING_REFERENCE_PARAMETERS')
    previous={n:p.requires_grad for n,p in params.items()}
    try:
        for n,p in params.items():p.requires_grad_(trainability[n])
        return emit(model,inputs,indices=indices)
    finally:
        for n,p in params.items():p.requires_grad_(previous[n])


def validate_labels(records):
    if not records or {r['role'] for r in records}!={'fit','monitor'}:raise ValueError('OPENING_LABEL_ROLES')
    for r in records:
        end=BOUNDARY if r['role']=='fit' else END
        if r['position']!=0 or not r['timestamp']<r['terminal']<=end:raise ValueError('OPENING_LABEL_SUPPORT')
        if r['role']=='monitor' and r['window_start']<BOUNDARY:raise ValueError('OPENING_MONITOR_CONTEXT')
        y=np.array([[a['soft_return'] for a in s['actions']] for s in r['suffixes']])
        if y.shape!=(3,3) or not np.isfinite(y).all():raise ValueError('OPENING_LABEL_VALUES')
        if r['forbidden'] and not np.allclose(y,y[:,0,None],atol=1e-10,rtol=0):
            raise ValueError('OPENING_EQUIVALENT_ACTIONS')
    return True
