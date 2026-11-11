"""Outcome-independent TRAIN panel and greedy-continuation diagnostic gates."""
from collections import defaultdict

import numpy as np

START = 1577836800000000000
BOUNDARY = 1672531200000000000
END = 1704067200000000000


def qualification_identity(config):
    from app.rl_subset import digest
    if config.get('schema')!='one_update_actor_qualification.v1':
        raise ValueError('ALL_STATE_QUALIFICATION_SCHEMA')
    return digest(config)


def choose_panel(rows, seed=281):
    groups = defaultdict(list)
    seen = set()
    for r in rows:
        if r['timestamp'] in seen or r['year'] not in range(2020,2024) or r['position'] not in (-1,0,1):
            raise ValueError('ALL_STATE_PANEL_TRACE')
        seen.add(r['timestamp'])
        groups[r['year'],r['position']].append(r)
    rng=np.random.default_rng(seed)
    out=[]
    for (_,position),group in sorted(groups.items()):
        count=8 if position==0 else 4
        out.extend(group[int(i)] for i in rng.permutation(len(group))[:count])
    return sorted(out,key=lambda r:r['timestamp'])


def validate_panel(rows):
    if not rows or {r['role'] for r in rows}!={'fit','monitor'}:
        raise ValueError('ALL_STATE_PANEL_ROLES')
    for r in rows:
        end=BOUNDARY if r['role']=='fit' else END
        start=START if r['role']=='fit' else BOUNDARY
        if (r['position'] not in (-1,0,1) or not start<=r['timestamp']<r['terminal']<=end
                or r['window_start']>r['timestamp'] or (r['role']=='monitor' and r['window_start']<start)):
            raise ValueError('ALL_STATE_PANEL_SUPPORT')
        v=r['outcomes']
        if (len(v)!=3 or not np.isfinite([[x[k] for k in ('reward_return','soft_return','net_usd')] for x in v]).all()
                or any(not x['parent_unchanged'] or x['first_effect']['duration']<1
                       or abs(x['reward_return']-x['soft_return'])>1e-8 for x in v)):
            raise ValueError('ALL_STATE_PANEL_VALUES')
        if r['position']:
            hold=1 if r['position']==1 else 2
            if (abs(v[0]['reward_return']-v[hold]['reward_return'])>1e-8
                    or abs(v[0]['net_usd']-v[hold]['net_usd'])>1e-6):
                raise ValueError('ALL_STATE_HOLD_EQUIVALENCE')
    return True


def policy_effect(before, after, target):
    a,b,y=[np.asarray(v,dtype=float) for v in (before,after,target)]
    if (a.ndim!=2 or a.shape[-1]!=3 or not len(a) or a.shape!=b.shape or a.shape!=y.shape
            or not np.isfinite([a,b,y]).all()):
        raise ValueError('ALL_STATE_POLICY_SHAPE')
    def prob(x):
        p=np.exp(x-x.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
    ix=np.arange(len(y));pa,pb=prob(a),prob(b)
    return dict(n=len(y),expected_return_delta=float(((pb-pa)*y).sum(-1).mean()),
                greedy_regret_before=float((y.max(-1)-y[ix,a.argmax(-1)]).mean()),
                greedy_regret_after=float((y.max(-1)-y[ix,b.argmax(-1)]).mean()),
                greedy_changes=int((a.argmax(-1)!=b.argmax(-1)).sum()))


def constant_controls(target):
    y=np.asarray(target,dtype=float)
    if y.ndim!=2 or y.shape[-1]!=3 or not len(y) or not np.isfinite(y).all():
        raise ValueError('ALL_STATE_CONSTANT_TARGET')
    return {name:dict(mean_regret=float((y.max(-1)-y[:,a]).mean()),
                      maximum_regret=float((y.max(-1)-y[:,a]).max()))
            for a,name in enumerate(('NOP','BUY','SELL'))}
