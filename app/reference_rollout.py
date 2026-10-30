"""Finite-horizon native rollout with frozen, nonrecursive reference policies."""
from collections import OrderedDict
import copy
from dataclasses import replace

import numpy as np

from app.dynamic_account_arbiter import HORIZONS, objectives
from app.pragmatic_strategy import PragmaticWorld, PragmaticController, action_plan, shadow_environment
from app.sequential_account_pareto import clone


class ContinuationWorld(PragmaticWorld):
    """Carry one whole forecast ensemble forward, never fetch a future report."""
    def __init__(self, parent, env, report, origin, shared=None):
        self.env=env;self.end=parent.end;self.risk_config=parent.risk_config
        self.references=parent.references;self.rolls=parent.rolls
        self.spec=parent.spec;self.reporter=parent.reporter
        self.origin=origin;self.origin_issued_at=report.issued_at;self.frozen_report=report
        shared={} if shared is None else shared
        self.reports=shared.setdefault('reports',{})
        self.openings=shared.setdefault('openings',{})
        self.cache=shared.setdefault('values',OrderedDict())
        self.bids={};self.planned=OrderedDict();self._cache_context=None

    def report_available(self,b):
        return self.origin<=b and b+1<self.end

    def report(self,b):
        if not self.report_available(b): raise ValueError('FROZEN_REPORT_SUPPORT')
        if b not in self.reports:
            r=self.frozen_report;k=b-self.origin
            self.reports[b]=replace(r,issued_at=r.issued_at+k*r.step,
                first_fill=r.first_fill+k*r.step,paths=r.paths[:,k:],spreads=r.spreads[:,k:])
        return self.reports[b]


class PassiveController:
    def decide(self,w,e,equity,horizon):
        return dict(action='HOLD' if e.pos is not None else 'NOP',horizon=horizon)


def select_reference(rows,weights,risk_weight):
    w=np.asarray(weights,float)
    if (not rows or w.ndim!=1 or not np.isfinite(w).all() or (w<0).any()
            or not np.isclose(w.sum(),1) or not np.isfinite(risk_weight) or risk_weight<0):
        raise ValueError('REFERENCE_SELECTION')
    ranked=[]
    for row in rows:
        x=np.asarray(row['outcomes'],float)
        if x.shape!=(len(w),2) or not np.isfinite(x).all() or (x[:,1]<0).any():
            raise ValueError('REFERENCE_OUTCOMES')
        mean=float(w@x[:,0]);dd=float(x[:,1].max())
        ranked.append(dict(action=row['action'],reference=row['reference'],
            mean_return_pct=mean,worst_dd_pct=dd,score=mean-risk_weight*dd))
    best=max(r['score'] for r in ranked)
    tied=[r for r in ranked if abs(r['score']-best)<=1e-12]
    quiet=[r for r in tied if r['action'] in ('NOP','HOLD')]
    # An active tie is deterministic, not dependent on enumeration order.
    winner=min(quiet or tied,key=lambda r:(r['action'],r['reference']))
    return dict(winner,candidates=ranked)


def reference_rollout(world,env,equity,horizon,action,reference,report,shared=None):
    origin=env.next_decision;rows=[]
    for sample in range(len(report.weights)):
        e=shadow_environment(env,report,sample);history=list(equity)
        view=ContinuationWorld(world,e,report,origin,shared)
        policy=copy.deepcopy(reference);h=horizon
        while e.next_decision<e.last:
            b=e.next_decision
            if b==origin:
                named=action
                if ':' in named: h=int(named.split(':')[1])
            else:
                decision=policy.decide(view,e,history,h);h=decision['horizon']
                named=decision['action']
                if named in ('BUY','SELL'): named+=':'+str(h)
            a,p=action_plan(view,named);info=e.step(b,a,entry_plan=p)
            if info['refused']: raise AssertionError(('REFERENCE_ORDER_REFUSED',info))
            history.append(float(e.equity_at(b+1)))
        if e.pos is not None or abs(e.balance-e.cfg.capital-sum(t['net_usd'] for t in e.trades))>1e-7:
            raise AssertionError('REFERENCE_CASH_CONSERVATION')
        rows.append(objectives(history,e.cfg.capital))
    return np.asarray(rows)


class ReferenceRolloutController:
    def __init__(self,references,risk_weight=1.):
        if (not references or any(c.variant not in ('simple','predictive_proxy') for c in references.values())
                or 'passive' in references or not np.isfinite(risk_weight) or risk_weight<0):
            raise ValueError('NONRECURSIVE_REFERENCE_REQUIRED')
        self.references={name:PragmaticController(c) for name,c in references.items()}
        self.references['passive']=PassiveController()
        self.risk_weight=risk_weight

    def decide(self,w,e,equity,horizon):
        b=e.next_decision
        # Advance each reference's memory exactly once on the real trajectory.
        for reference in self.references.values(): reference.decide(w,e,equity,horizon)
        if not w.report_available(b): return dict(action='HOLD' if e.pos else 'NOP',horizon=horizon)
        if e.pos is None and not e.entry_gate(b)[0]: return dict(action='NOP',horizon=None)
        actions=['HOLD','CLOSE'] if e.pos is not None else ['NOP']+[
            f'{side}:{h}' for h in HORIZONS for side in ('BUY','SELL')]
        r=w.report(b);rows=[];shared={}
        for action in actions:
            for name,reference in self.references.items():
                values=reference_rollout(w,e,equity,horizon,action,reference,r,shared)
                rows.append(dict(action=action,reference=name,outcomes=values))
        chosen=select_reference(rows,r.weights,self.risk_weight);action=chosen['action']
        h=int(action.split(':')[1]) if ':' in action else horizon
        return dict(action=action.split(':')[0],horizon=h,plan=dict(chosen,
            valuation='FROZEN_REFERENCE_NOT_ADAPTIVE_POLICY',origin_issued_at=r.issued_at,
            observed_through=r.observed_through))


class CaptureCloses:
    """Snapshot AFTER observing this tick, BEFORE executing its CLOSE."""
    def __init__(self,controller):
        self.controller=controller;self.snapshots={}

    def decide(self,w,e,equity,horizon):
        decision=self.controller.decide(w,e,equity,horizon)
        if decision['action']=='CLOSE':
            self.snapshots[e.next_decision]=(clone(e),list(equity),horizon,copy.deepcopy(self.controller))
        return decision


def resume_without_close(world,snapshot):
    env,history,horizon,controller=snapshot
    e=clone(env);equity=list(history);policy=copy.deepcopy(controller);origin=e.next_decision
    trace=[]
    while e.next_decision<e.last:
        b=e.next_decision
        if b==origin:
            action='HOLD'
        else:
            decision=policy.decide(world,e,equity,horizon)
            action=decision['action'];horizon=decision['horizon']
        named=action+':'+str(horizon) if action in ('BUY','SELL') else action
        a,p=action_plan(world,named);info=e.step(b,a,entry_plan=p)
        if info['refused']: raise AssertionError(('SNAPSHOT_ORDER_REFUSED',info))
        equity.append(float(e.equity_at(b+1)))
        trace.append(dict(bar=b,action=action,horizon=horizon,close=info['close']))
    ret,dd=objectives(equity,e.cfg.capital);net=sum(t['net_usd'] for t in e.trades)
    if e.pos is not None or abs(e.balance-e.cfg.capital-net)>1e-7:
        raise AssertionError('SNAPSHOT_CASH_CONSERVATION')
    return dict(return_pct=ret,drawdown_pct=dd,net_usd=net,orders=len(e.trades),trace=trace,
        ledger=[dict(direction=t['direction'],entry_bar=t['entry_bar_abs'],exit_bar=t['exit_bar_abs'],
                     net=t['net_usd'],units=t['units'],q=t['percentual_order_size'],reason=t['close_reason'])
                for t in e.trades])
