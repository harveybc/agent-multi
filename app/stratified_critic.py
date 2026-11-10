"""Sampling and temporal-support gates for fixed-actor critic diagnostics."""
import numpy as np

from app.critic_fit_probe import fit_metrics

QUOTAS={'flat_both_allowed':32,'open_long':32,'open_short':32,'flat_restricted':8}


def select_strata(rows,seed):
    if any(r['kind'] not in QUOTAS for r in rows):
        raise ValueError('STRATIFIED_UNKNOWN_STATE')
    rng=np.random.default_rng(seed);selected=[];counts={}
    for kind,quota in QUOTAS.items():
        eligible=[i for i,r in enumerate(rows) if r['kind']==kind]
        chosen=rng.permutation(eligible)[:quota].tolist()
        selected.extend(chosen)
        counts[kind]=dict(available=len(eligible),selected=len(chosen),requested=quota)
    return sorted(selected),counts


def validate_support(rows,boundary):
    for r in rows:
        if not r['window_start']<=r['timestamp']<r['terminal']:
            raise ValueError('STRATIFIED_WINDOW_OR_TERMINAL')
        if r['role']=='fit':
            if r['terminal']>=boundary:raise ValueError('STRATIFIED_FIT_CROSSES_MONITOR')
        elif r['role']=='monitor':
            if r['window_start']<boundary:raise ValueError('STRATIFIED_MONITOR_WINDOW')
        else:raise ValueError('STRATIFIED_ROLE')
    if {r['role'] for r in rows}!={'fit','monitor'}:
        raise ValueError('STRATIFIED_ROLE_COVERAGE')


def grouped_metrics(q,target,rows):
    result={}
    for role in ('fit','monitor'):
        result[role]={}
        for kind in ('all',*QUOTAS):
            ix=np.array([i for i,r in enumerate(rows) if r['role']==role and
                         (kind=='all' or r['kind']==kind)],dtype=int)
            result[role][kind]=fit_metrics(q[:,ix],target[ix]) if len(ix) else None
    return result
