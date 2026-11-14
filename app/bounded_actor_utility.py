"""Native critic transplant and terminal decision for the single350 actor trial."""
import math
import torch


def install_native_heads(model,state):
    expected={f'twins.{t}.0.{layer}.{kind}' for t in range(2) for layer in (0,2) for kind in ('weight','bias')}
    if set(state)!=expected:raise ValueError('ACTOR_UTILITY_DONOR_KEYS')
    destinations=[]
    for layer in (0,2):
        for kind in ('weight','bias'):
            key=('module','head',str(layer),kind)
            dest=model.loss.qvalue_network_params[key]
            values=[state[f'twins.{t}.0.{layer}.{kind}'] for t in range(2)]
            if (dest.shape[0]!=2 or any(v.shape!=dest.shape[1:] or v.dtype!=dest.dtype or not torch.isfinite(v).all() for v in values)):
                raise ValueError('ACTOR_UTILITY_DONOR_SHAPE')
            destinations.append((dest,torch.stack(values).to(dest)))
    with torch.no_grad():
        for dest,value in destinations:dest.copy_(value)


def choose_fold(support):
    folds=support['folds']
    if not folds or any(not f['expanded'] for f in folds.values()):raise ValueError('ACTOR_UTILITY_FOLD')
    return min(folds,key=lambda k:(-len(folds[k]['expanded']),k))


def qualify(metrics,reference,accepted):
    reasons=[];strict=False
    if accepted!=8:reasons.append('INCOMPLETE_BUDGET')
    for p in ('TRAIN','VALIDATION'):
        m,b=metrics[p],reference[p]
        values=[m['return_pct'],m['orders_per_week'],m['cash_error'],b['return_pct']]
        if not all(math.isfinite(v) for v in values):
            reasons.append(p+'_NONFINITE');continue
        if m['orders_per_week']<2:reasons.append(p+'_ACTIVITY')
        if abs(m['cash_error'])>1e-7:reasons.append(p+'_CASH')
        if m['return_pct']<0:reasons.append(p+'_NEGATIVE')
        if m['return_pct']<b['return_pct']:reasons.append(p+'_REGRESSION')
        strict|=m['return_pct']>b['return_pct']
    if not strict:reasons.append('NO_STRICT_IMPROVEMENT')
    return dict(qualified=not reasons,reasons=reasons)


def winner(results):
    if len(results)!=2 or {r['seed'] for r in results}!={19,43}:raise ValueError('ACTOR_UTILITY_SEEDS')
    valid=[r for r in results if r['qualification']['qualified']]
    if not valid:return None
    return max(valid,key=lambda r:(r['metrics']['VALIDATION']['return_pct'],r['metrics']['TRAIN']['return_pct'],-r['seed']))['seed']
