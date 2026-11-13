"""Year-bounded native targets and FIT-only temporal stopping decisions."""
import math

import numpy as np
import pandas as pd

from app.decision_attribution import fork_env, state_view
from env_plugins.discrete_curve_native import HOUR


def bounded_snapshot(snapshot, end_ns):
    before=state_view(snapshot)
    last=int(np.searchsorted(snapshot.engine.v+HOUR,end_ns,side='left'))-1
    if snapshot.finished or not snapshot.bar<last<=snapshot.engine.last:
        raise ValueError('BLOCKED_TERMINAL')
    result=fork_env(snapshot)
    result.engine.last=last
    result.engine.split=(snapshot.engine.split[0],pd.Timestamp(end_ns-HOUR,tz='UTC'))
    if state_view(snapshot)!=before:raise ValueError('BLOCKED_PARENT_CHANGED')
    return result


def fold_indices(records, fit_years, stop_year, audit_year):
    if not fit_years or max(fit_years)>=stop_year or stop_year>=audit_year:
        raise ValueError('BLOCKED_YEAR_ORDER')
    groups={'fit':[], 'stop':[], 'audit':[]}
    for i,r in enumerate(records):
        a=pd.Timestamp(f"{r['year']}-01-01",tz='UTC').value
        b=pd.Timestamp(f"{r['year']+1}-01-01",tz='UTC').value
        if not a<=r['window_start']<=r['timestamp']<r['terminal']<b:
            raise ValueError('BLOCKED_LABEL_OR_CONTEXT')
        if r['original_role']=='fit':
            if r['year'] in fit_years:groups['fit'].append(i)
            if r['year']==stop_year:groups['stop'].append(i)
        if r['year']==audit_year:groups['audit'].append(i)
    if any(not v for v in groups.values()):raise ValueError('BLOCKED_EMPTY_SPLIT')
    for left,right in (('fit','stop'),('stop','audit')):
        if max(records[i]['terminal'] for i in groups[left])>=min(records[i]['window_start'] for i in groups[right]):
            raise ValueError('BLOCKED_OVERLAP')
    if len({r['timestamp'] for r in records})!=len(records):raise ValueError('BLOCKED_DUPLICATE')
    return {k:np.array(v,dtype=np.int64) for k,v in groups.items()}


class RegretStopper:
    def __init__(self, patience=20, delta=1e-8):
        if type(patience) is not int or patience<1 or not math.isfinite(delta) or delta<0:
            raise ValueError('BLOCKED_STOP_CONFIG')
        self.patience,self.delta=patience,delta
        self.best=None;self.best_step=None;self.bad=0;self.stopped_at=None;self.last_step=0

    def observe(self, step, score):
        if type(step) is not int or step<=self.last_step or not math.isfinite(score):
            raise ValueError('BLOCKED_STOP_OBSERVATION')
        self.last_step=step
        if self.stopped_at is not None:return False
        improved=self.best is None or self.best-score>self.delta
        if improved:self.best,self.best_step,self.bad=float(score),step,0
        else:self.bad+=1
        if self.bad>=self.patience:self.stopped_at=step
        return improved
