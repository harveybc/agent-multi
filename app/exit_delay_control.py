"""TRAIN-selected deterministic stopping clock for synthetic exit qualification."""
import numpy as np

from app.frozen_recipe_inputs import require
from app.optimal_exit import outcome
from app.rl_subset import digest


def _summary(cases, delay):
    require(bool(cases) and {c['direction'] for c in cases}=={1,-1},'DELAY_CONTROL_BOTH_DIRECTIONS')
    rows=[]
    for case in cases:
        actions=np.zeros(len(case['close_net']))
        if 0<=delay<len(actions):actions[delay]=1
        rows.append(outcome(case,actions))
    return dict(mean_regret_risk=float(np.mean([r['regret_risk'] for r in rows])),
        mean_net_usd=float(np.mean([r['net_usd'] for r in rows])),
        early_close_share=float(np.mean([r['early_close'] for r in rows])),
        by_direction={str(d):dict(mean_regret_risk=float(np.mean(
            [r['regret_risk'] for r in rows if r['direction']==d]))) for d in (1,-1)})


def fit_delay(train_cases):
    require(bool(train_cases),'DELAY_CONTROL_EMPTY_TRAIN')
    candidates=list(range(-1,max(len(c['close_net']) for c in train_cases)))
    scores=[(k,_summary(train_cases,k)) for k in candidates]
    delay,summary=min(scores,key=lambda pair:(pair[1]['mean_regret_risk'],pair[0]))
    fields=('close_net','terminal_net','risk_usd','direction')
    control=dict(schema='exit_fixed_delay.v1',selection_split='TRAIN',delay=delay,
        candidates=candidates,criterion='mean regret / risk; ties use smaller delay; -1 means passive',
        train_case_digest=digest([{k:c[k] for k in fields} for c in train_cases]),
        train_summary=summary)
    return dict(**control,identity=digest(control))


def evaluate_delay(cases, control):
    payload={k:v for k,v in control.items() if k!='identity'}
    require(control.get('identity')==digest(payload),'DELAY_CONTROL_IDENTITY')
    require(control.get('schema')=='exit_fixed_delay.v1' and control.get('selection_split')=='TRAIN',
            'DELAY_CONTROL_TRAIN_ONLY')
    require(type(control['delay']) is int and control['delay'] in control['candidates'],
            'DELAY_CONTROL_CANDIDATE')
    return _summary(cases,control['delay'])
