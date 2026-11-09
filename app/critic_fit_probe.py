"""TRAIN panel audit and bounded critic fitting diagnostics, not policy selection."""
from collections import Counter

import numpy as np
import torch

from app.action_ranking_contrast import ranking_metrics, differentiable_q


def classify_anchor(record):
    anchor=record['anchor'];outcomes=record['results']['greedy']
    if (len(outcomes)!=3 or anchor['position'] not in (-1,0,1)
            or anchor['year'] not in range(2020,2024) or anchor['role'] not in ('fit','monitor')):
        raise ValueError('FIT_PROBE_ANCHOR')
    targets=np.array([r['reward_return'] for r in outcomes],dtype=float)
    if (not np.isfinite(targets).all()
            or any(not r['parent_unchanged'] or r['first_effect']['duration']<=0 for r in outcomes)
            or any(not np.isfinite(r['soft_return']) or not np.isfinite(r['net_usd']) for r in outcomes)
            or any(abs(r['soft_return']-r['reward_return'])>1e-8 for r in outcomes)):
        raise ValueError('FIT_PROBE_LABEL_OR_DURATION')
    pos=anchor['position'];effects=[r['first_effect'] for r in outcomes]
    if pos:
        hold=1 if pos==1 else 2
        if abs(targets[0]-targets[hold])>1e-8 or abs(outcomes[0]['net_usd']-outcomes[hold]['net_usd'])>1e-6:
            raise ValueError('FIT_PROBE_HOLD_EQUIVALENCE')
    kind=('open_long' if pos==1 else 'open_short') if pos else (
        'flat_both_allowed' if effects[1]['entries']>0 and effects[2]['entries']>0 else 'flat_restricted')
    gaps=np.array([targets[a]-targets[b] for a,b in ((0,1),(0,2),(1,2))])
    reasons=Counter(k for e in effects for k,v in e['count_changes'].items() if k.startswith('refused_') and v)
    return dict(kind=kind,targets=targets.tolist(),informative_pairs=int((np.abs(gaps)>1e-8).sum()),
        tied_pairs=int((np.abs(gaps)<=1e-8).sum()),refusals=dict(reasons),
        durations=[e['duration'] for e in effects])


def choose_small_set(records,seed):
    rng=np.random.default_rng(seed);chosen=[]
    for year in range(2020,2024):
        eligible=[]
        for i,r in enumerate(records):
            a=r['anchor'];c=classify_anchor(r)
            if a['year']==year and a['role']=='fit' and c['kind']=='flat_both_allowed' and c['informative_pairs']:
                eligible.append(i)
        if len(eligible)<8:raise ValueError('FIT_PROBE_YEAR_COVERAGE')
        chosen.extend(int(i) for i in rng.permutation(eligible)[:8])
    return sorted(chosen)


def fit_metrics(twins,target):
    q=np.asarray(twins,dtype=float);target=np.asarray(target,dtype=float)
    if (q.ndim!=3 or q.shape[0]!=2 or q.shape[1:]!=target.shape or target.shape[-1]!=3
            or not q.shape[1] or not np.isfinite(q).all() or not np.isfinite(target).all()):
        raise ValueError('FIT_PROBE_Q_SHAPE')
    result={}
    truth=np.stack([target[:,a]-target[:,b] for a,b in ((0,1),(0,2),(1,2))],-1)
    for name,values in (('twin0',q[0]),('twin1',q[1]),('minimum',q.min(0))):
        gaps=np.stack([values[:,a]-values[:,b] for a,b in ((0,1),(0,2),(1,2))],-1)
        result[name]=dict(ranking_metrics(values,target),
            gap_rmse=float(np.sqrt(np.mean((gaps-truth)**2))),
            level_rmse=float(np.sqrt(np.mean((values.mean(-1)-target.mean(-1))**2))))
    return result


def fit_passed(metrics):
    return all(m['pairwise_accuracy'] is not None and m['pairwise_accuracy']>=1.-1e-12
               and m['gap_rmse']<=.05 and m['maximum_regret']<=1e-8 for m in metrics.values())


def set_scope(model,scope):
    if scope not in ('heads','critic_representation'):raise ValueError('FIT_PROBE_SCOPE')
    for name,param in model.loss.named_parameters():
        param.requires_grad_(name.startswith('qvalue_network_params.') and
            ('.head.' in name or scope=='critic_representation'))
    for group in model.optimizer.param_groups:group['lr']=model.effective['learning_rate']


def probe_q(model,features):
    # Functional twin parameters can trigger the inference-only fused attention
    # path even when encoder gradients are required. Limit its bypass to this
    # critic forward and restore it before any actor or ordinary inference call.
    encoder_grad=any(p.requires_grad and n.startswith('qvalue_network_params.') and '.encoder.' in n
                     for n,p in model.loss.named_parameters())
    previous=torch.backends.mha.get_fastpath_enabled()
    try:
        if encoder_grad:torch.backends.mha.set_fastpath_enabled(False)
        return differentiable_q(model,features)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
