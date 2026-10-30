"""Native retrospective diagnostics; no future value is returned to the policy."""
import math

from app.frozen_policy_diagnostics import DiagnosticWorld,close_stages
from app.sequential_account_pareto import clone


def fixed_position_exits(env,risk):
    if env.pos is None: raise ValueError('COUNTERFACTUAL_OCCUPIED_REQUIRED')
    if not math.isfinite(risk) or risk<=0: raise ValueError('COUNTERFACTUAL_RISK')
    hold=clone(env);alternatives=[]
    while hold.pos is not None:
        b=hold.next_decision
        candidate=clone(hold);candidate.step(b,3)
        if candidate.pos is not None: raise AssertionError('CLOSE_NOT_TERMINAL')
        trade=candidate.trades[-1]
        alternatives.append(dict(decision_bar=b,exit_bar=trade['exit_bar_abs'],
                                 net=trade['net_usd'],reason=trade['close_reason']))
        hold.step(b,0)
    passive=hold.trades[-1];now=alternatives[0]
    options=alternatives+[dict(decision_bar=None,exit_bar=passive['exit_bar_abs'],
                              net=passive['net_usd'],reason=passive['close_reason'])]
    best=max(options,key=lambda x:x['net'])
    return dict(bar=env.next_decision,entry_bar=env.pos.entry_bar,direction=env.pos.direction,
        units=env.pos.units,risk=risk,immediate=now['net'],immediate_reason=now['reason'],
        preempted=now['reason']!='AGENT_CLOSE',hold=passive['net_usd'],
        hold_reason=passive['close_reason'],optimal=best['net'],optimal_exit_bar=best['exit_bar'],
        delta_vs_hold=now['net']-passive['net_usd'],
        delta_vs_hold_risk=(now['net']-passive['net_usd'])/risk,
        regret_risk=(best['net']-now['net'])/risk,
        passive_regret_risk=(best['net']-passive['net_usd'])/risk,alternatives=options)


class SuppressedCloseWorld(DiagnosticWorld):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.intervention=None
        self.collect_suppressed=False
        self.proposals=[]
        self.intervention_hits=0
        self.verifying_direct=False

    def run(self,*args,**kwargs):
        self.proposals=[];self.intervention_hits=0
        result=super().run(*args,**kwargs)
        if self.intervention is not None and self.intervention_hits!=1:
            raise ValueError('INTERVENTION_TARGET_NOT_VISITED_ONCE')
        return result

    def close_values(self,e,h,b):
        self.visited_env=e
        return super().close_values(e,h,b)

    def verify_close_decision(self,*args):
        self.verifying_direct=True
        try:
            return super().verify_close_decision(*args)
        finally:
            self.verifying_direct=False

    def close_decision(self,controller,unit,r,clock,deadline,values,binding,b,horizon):
        result=super().close_decision(controller,unit,r,clock,deadline,values,binding,b,horizon)
        e=self.visited_env
        target=dict(bar=b,entry_bar=e.pos.entry_bar,direction=e.pos.direction)
        if self.intervention==target:
            self.intervention_hits+=not self.verifying_direct
            return dict(result,action='CLOSE')
        if self.collect_suppressed and not self.verifying_direct and self.intervention is None and result['action']=='HOLD':
            stages=close_stages(values,r.weights,unit.risk,0.,0.,self.active_genes[9])
            if stages['raw_pass']:
                row=fixed_position_exits(e,unit.risk*e.pos.units)
                row.update(horizon=horizon,raw_advantage_risk=stages['raw'],
                           final_advantage_risk=result['advantage_risk'])
                self.proposals.append(row)
        return result


def compare_intervention(baseline,alternative,target):
    before=lambda account:[t for t in account['trace'] if t['bar']<target['bar']]
    if before(baseline)!=before(alternative): raise ValueError('INTERVENTION_PREFIX_CHANGED')
    event=next(t for t in alternative['trace'] if t['bar']==target['bar'])
    if event['action']!=3: raise ValueError('INTERVENTION_CLOSE_MISSING')
    def trade(account):
        rows=[t for t in account['ledger'] if t['entry_bar']==target['entry_bar']
              and t['direction']==target['direction']]
        if len(rows)!=1: raise ValueError('INTERVENTION_TRADE_IDENTITY')
        return rows[0]
    old,new=trade(baseline),trade(alternative)
    local=new['net']-old['net'];total=alternative['net_usd']-baseline['net_usd']
    return dict(position_delta_usd=local,account_delta_usd=total,
                continuation_delta_usd=total-local,
                drawdown_delta_pp=alternative['drawdown_pct']-baseline['drawdown_pct'],
                order_count_delta=alternative['orders']-baseline['orders'],
                target_close_reason=new['reason'])
