"""Frozen one-tick close interventions; retrospective values never enter the policy."""
from tools.optimize_dynamic_account import digest


class SuppressOneClose:
    def __init__(self,controller,target):
        self.controller=controller
        self.target=target
        self.hits=0

    def decide(self,world,env,equity,horizon):
        decision=self.controller.decide(world,env,equity,horizon)
        if self.target is None or env.next_decision!=self.target['bar']:
            return decision
        if self.hits: raise ValueError('CLOSE_INTERVENTION_DUPLICATE')
        if (env.pos is None or env.pos.entry_bar!=self.target['entry_bar']
                or env.pos.direction!=self.target['direction']):
            raise ValueError('CLOSE_INTERVENTION_POSITION')
        if decision['action']!='CLOSE': raise ValueError('CLOSE_INTERVENTION_NOT_CLOSE')
        self.hits+=1
        return dict(decision,action='HOLD')

    def verify(self):
        if self.hits!=(0 if self.target is None else 1):
            raise ValueError('CLOSE_INTERVENTION_NOT_VISITED')


def verify_baseline(actual,archived):
    if digest(actual)!=digest(archived): raise ValueError('CLOSE_ATTRIBUTION_BASELINE_MISMATCH')


def targets(account):
    rows=[{k:c[k] for k in ('bar','entry_bar','direction')}
          for c in account['close_rows'] if not c['preempted']]
    if (len(rows)!=account['close_reasons'].get('AGENT_CLOSE',0)
            or len({digest(r) for r in rows})!=len(rows)):
        raise ValueError('CLOSE_ATTRIBUTION_COUNT')
    return rows


def compare_close(baseline,alternative,target):
    before=lambda a:[t for t in a['trace'] if t['bar']<target['bar']]
    if digest(before(baseline))!=digest(before(alternative)):
        raise ValueError('CLOSE_ATTRIBUTION_PREFIX_CHANGED')
    for account,action in ((baseline,'CLOSE'),(alternative,'HOLD')):
        events=[t for t in account['trace'] if t['bar']==target['bar']]
        if len(events)!=1 or events[0]['action']!=action:
            raise ValueError('CLOSE_ATTRIBUTION_ACTION')
    def position(account):
        rows=[t for t in account['ledger'] if t['entry_bar']==target['entry_bar']
              and t['direction']==target['direction']]
        if len(rows)!=1: raise ValueError('CLOSE_ATTRIBUTION_POSITION')
        return rows[0]
    closed,held=position(baseline),position(alternative)
    if closed['reason']!='AGENT_CLOSE' or any(closed[k]!=held[k] for k in ('units','q')):
        raise ValueError('CLOSE_ATTRIBUTION_ORIGINAL_ORDER')
    local=closed['net']-held['net'];total=baseline['net_usd']-alternative['net_usd']
    dd=alternative['drawdown_pct']-baseline['drawdown_pct'];eps=1e-9
    if abs(total)<=eps and abs(dd)<=eps: relation='EQUAL'
    elif total>=-eps and dd>=-eps: relation='CLOSE_DOMINATES'
    elif total<=eps and dd<=eps: relation='HOLD_DOMINATES'
    else: relation='TRADEOFF'
    return dict(position_delta_usd=local,account_delta_usd=total,
        continuation_delta_usd=total-local,drawdown_advantage_pp=dd,
        return_delta_pp=baseline['return_pct']-alternative['return_pct'],
        order_count_delta=baseline['orders']-alternative['orders'],
        hold_branch_exit_reason=held['reason'],pareto_relation=relation)


def compact_account(account):
    result={k:v for k,v in account.items() if k not in ('trace','close_rows')}
    result['full_trace_digest']=digest(account['trace'])
    result['trace']=[{k:v for k,v in row.items() if k!='planning'} for row in account['trace']]
    return result
