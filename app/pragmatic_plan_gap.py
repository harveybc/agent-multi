"""Independent native committed-plan replay and frozen continuation accounting."""
import math

from app.dynamic_account_arbiter import objectives
from app.pragmatic_strategy import action_plan
from app.sequential_account_pareto import clone


def committed_replay(world,env,equity,plan):
    first,delay,second=plan
    e=clone(env);history=list(equity);origin=e.next_decision;entered_second=False;trace=[]
    while e.next_decision<e.last:
        bar=e.next_decision
        if bar==origin:
            decision=first
        elif entered_second:
            decision='HOLD' if e.pos else 'NOP'
        elif e.pos is None:
            decision=second if second!='NOP' and e.entry_gate(bar)[0] else 'NOP'
            entered_second=decision!='NOP'
        elif delay is not None and bar-origin>=delay:
            decision='CLOSE'
        else:
            decision='HOLD'
        action,entry=action_plan(world,decision)
        info=e.step(bar,action,entry_plan=entry)
        if info['refused']: raise ValueError('COMMITTED_NATIVE_REFUSAL')
        history.append(float(e.equity_at(bar+1)))
        trace.append(dict(bar=bar,action=decision.split(':')[0],
            horizon=int(decision.split(':')[1]) if ':' in decision else None,
            close=info['close']))
    ret,dd=objectives(history,e.cfg.capital)
    net=sum(t['net_usd'] for t in e.trades);cash=e.balance-e.cfg.capital-net
    if e.pos is not None or abs(cash)>1e-7: raise ValueError('COMMITTED_ACCOUNT_INTEGRITY')
    return dict(return_pct=ret,drawdown_pct=dd,net_usd=net,cash_error=cash,flat=e.pos is None,
        orders=len(e.trades),trace=trace,ledger=[dict(entry_bar=t['entry_bar_abs'],exit_bar=t['exit_bar_abs'],
            direction=t['direction'],net=t['net_usd'],reason=t['close_reason'],units=t['units']) for t in e.trades])


def utility(account,weight):
    return account['return_pct']-weight*account['drawdown_pct']


def check_valuation(native,stored,weight):
    differences=[abs(native['return_pct']-stored['mean_return_pct']),
        abs(native['drawdown_pct']-stored['worst_dd_pct']),
        abs(utility(native,weight)-stored['score']),abs(stored['std_return_pct'])]
    if not all(math.isfinite(v) and v<=1e-8 for v in differences):
        raise ValueError(('IDEAL_COMMITTED_VALUATION',differences))
    return max(differences)


def decompose(planned_close,planned_hold,actual_close,actual_hold,weight):
    pc,ph,ac,ah=[utility(a,weight) for a in (planned_close,planned_hold,actual_close,actual_hold)]
    margin=ac-ah;prediction=pc-ph;cshift=ac-pc;hshift=ah-ph
    error=abs(margin-(prediction+cshift-hshift))
    if error>1e-10: raise ValueError('CONTINUATION_DECOMPOSITION')
    return dict(planned_margin=prediction,actual_margin=margin,
        close_continuation_shift=cshift,hold_continuation_shift=hshift,identity_error=error,
        ranking_reversal=prediction>1e-9 and margin< -1e-9,
        account_profit_delta_usd=actual_close['net_usd']-actual_hold['net_usd']
            if 'net_usd' in actual_close else None,
        drawdown_advantage_pp=actual_hold['drawdown_pct']-actual_close['drawdown_pct'])


def first_deviation(planned,actual):
    if not planned: return None
    start=planned[0]['bar'];other=[r for r in actual if r['bar']>=start]
    if [r['bar'] for r in planned]!=[r['bar'] for r in other]:
        raise ValueError('CONTINUATION_TRACE_SUPPORT')
    def action(r):
        a=r['action']
        return (a,r['horizon']) if a in ('BUY','SELL') else ('IDLE' if a in ('NOP','HOLD') else a,None)
    for p,a in zip(planned,other):
        if action(p)!=action(a): return dict(bar=p['bar'],planned=action(p),actual=action(a))
    return None
