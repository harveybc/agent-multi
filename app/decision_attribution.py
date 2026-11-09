"""Retrospective same-state interventions, never policy input or training."""
import copy
from dataclasses import asdict

import numpy as np

from app.sequential_account_pareto import clone
from env_plugins.discrete_curve_native import HOUR


def state_view(env):
    e=env.engine
    return dict(bar=env.bar,timestamp=int(e.v[env.bar])+HOUR,balance=float(e.balance),
        position=None if e.pos is None else asdict(e.pos),
        entries_day=sorted((str(k),v) for k,v in e.entries_day.items()),
        counts=dict(e.counts),trades=len(e.trades),auto_holds=env.auto_holds,
        next_decision=e.next_decision,finished=env.finished)


def fork_env(env):
    out=copy.copy(env);out.engine=clone(env.engine)
    out.engine.pos=copy.deepcopy(env.engine.pos)
    out.engine.preparation=copy.deepcopy(env.engine.preparation)
    out.equity_trace=list(env.equity_trace)
    out.last_observation=copy.deepcopy(env.last_observation)
    return out


def continuation(snapshot,signals,first_action,probabilities=None,alpha=0.):
    if snapshot.finished:raise ValueError('ATTRIBUTION_TERMINAL_START')
    if first_action not in (0,1,2):raise ValueError('ATTRIBUTION_ACTION')
    before=state_view(snapshot);env=fork_env(snapshot);first=True
    discount=1.;soft=0.;first_effect=None
    while not env.finished:
        row=int(env.rows[env.bar]);action=int(first_action if first else signals[row])
        if not first and probabilities is not None:
            p=float(probabilities[row,action])
            if not 0<p<=1:raise ValueError('ATTRIBUTION_PROBABILITY')
            soft+=discount*alpha*(-np.log(p))
        old_entries=env.engine.counts['entries'];old_closes=env.engine.counts['closes']
        old_counts=dict(env.engine.counts)
        _,reward,_,duration=env.step(action)
        soft+=discount*reward;discount*=env.gamma**duration
        if first:
            first_effect=dict(command=env.last_command,reward=float(reward),duration=duration,
                entries=env.engine.counts['entries']-old_entries,
                closes=env.engine.counts['closes']-old_closes,
                count_changes={k:v-old_counts[k] for k,v in env.engine.counts.items() if v!=old_counts[k]},
                position_after=0 if env.engine.pos is None else env.engine.pos.direction)
        first=False
    if before!=state_view(snapshot):raise ValueError('ATTRIBUTION_PARENT_MUTATED')
    e=env.engine;net=float(e.balance-e.cfg.capital)
    if abs(net-sum(t['net_usd'] for t in e.trades))>1e-6:
        raise ValueError('ATTRIBUTION_CASH_IDENTITY')
    return dict(balance=float(e.balance),net_usd=net,trades=len(e.trades),
        soft_return=float(soft),first_effect=first_effect,parent_unchanged=True)


def reconcile(contributions,reference,candidate,tolerance=1e-6):
    error=float(sum(contributions)-(candidate-reference))
    if abs(error)>tolerance:raise ValueError('ATTRIBUTION_TELESCOPE')
    return dict(error_usd=error,delta_usd=float(candidate-reference),
                chronological=True,isolated_sum_is_account_effect=False)
