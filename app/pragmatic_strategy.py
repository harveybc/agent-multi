"""Paired engineering controllers; reports, risk geometry and execution stay separate."""
from collections import Counter,OrderedDict
from dataclasses import dataclass,replace,asdict
import math

import numpy as np

from app.close_hold_controller import CloseConfig,CloseController,value_paths,validate_context
from app.close_hold_reports import prefix_report
from app.crossed_account_panel import CrossedWorld
from app.dynamic_account_arbiter import HORIZONS,objectives,frontier
from app.dynamic_account_panel import HOUR,expected_duration
from app.sequential_account_pareto import clone
from app.suppressed_close_counterfactuals import fixed_position_exits


@dataclass(frozen=True)
class ControllerConfig:
    variant: str
    alpha: float=0.
    uncertainty: float=0.
    persistence: int=1
    tau0: float=1.
    risk_penalty: float=0.
    entry_threshold: float=0.
    close_margin: float=0.
    urgent_risk: float=.25
    account_risk_weight: float=1.
    shortlist: int=2
    close_offsets: tuple=(3,6,12,24)

    def __post_init__(self):
        numbers=(self.alpha,self.uncertainty,self.tau0,self.risk_penalty,
                 self.entry_threshold,self.close_margin,self.urgent_risk,self.account_risk_weight)
        if (self.variant not in ('simple','predictive_proxy','sequential')
                or not all(math.isfinite(x) and x>=0 for x in numbers)
                or self.tau0<=0 or self.alpha>1 or self.urgent_risk<self.close_margin
                or type(self.persistence) is not int or self.persistence<1
                or type(self.shortlist) is not int or not 1<=self.shortlist<=8
                or not self.close_offsets or any(type(x) is not int or x<1 for x in self.close_offsets)):
            raise ValueError('PRAGMATIC_CONFIG')


def rank_openings(bids,c):
    ranked=[]
    for b in bids:
        if (b['horizon'] not in HORIZONS or b['direction'] not in (-1,1)
                or not all(math.isfinite(b[k]) for k in ('gain_risk','uncertainty_risk','duration'))
                or min(b['duration'],b['uncertainty_risk'])<0):
            raise ValueError('PRAGMATIC_BID')
        score=(b['gain_risk']-c.risk_penalty-c.uncertainty*b['uncertainty_risk'])/(c.tau0+b['duration'])**c.alpha
        if score>c.entry_threshold+1e-12: ranked.append(dict(b,score=score))
    ranked.sort(key=lambda b:(-b['score'],b['horizon'],b['direction']))
    if len(ranked)>1 and abs(ranked[0]['score']-ranked[1]['score'])<=1e-12:
        return []
    return ranked


def defined_sharpe(native_value,daily_equity,initial):
    values=np.r_[initial,np.asarray(daily_equity,float)]
    if not np.isfinite(values).all() or initial<=0:
        raise ValueError('SHARPE_EQUITY')
    if len(values)<3 or np.any(values[:-1]==0) or not math.isfinite(native_value):
        return None
    returns=np.diff(values)/values[:-1]
    return float(native_value) if float(np.std(returns,ddof=1))>0 else None


def select_plan(plans,samples,weights,c):
    if len(plans)!=len(samples) or not plans: raise ValueError('PLAN_SET')
    w=np.asarray(weights,float)
    if w.ndim!=1 or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1):
        raise ValueError('PLAN_WEIGHTS')
    vectors={};rows=[]
    for i,(plan,x) in enumerate(zip(plans,samples)):
        x=np.asarray(x,float)
        if x.shape!=(len(w),2) or not np.isfinite(x).all() or (x[:,1]<0).any():
            raise ValueError('PLAN_OUTCOMES')
        mean=float(w@x[:,0]);sd=float(np.sqrt(w@((x[:,0]-mean)**2)))
        robust=mean-c.uncertainty*sd;dd=float(x[:,1].max())
        vectors[str(i)]=(robust,-dd)
        rows.append(dict(plan=plan,mean_return_pct=mean,std_return_pct=sd,worst_dd_pct=dd,
                         robust_return_pct=robust,score=robust-c.account_risk_weight*dd))
    indices=[int(i) for i in frontier(vectors)]
    best=max(rows[i]['score'] for i in indices)
    tied=[i for i in indices if abs(rows[i]['score']-best)<=1e-12]
    conservative=[i for i in tied if rows[i]['plan'][0] in ('NOP','HOLD')]
    if conservative: chosen=conservative[0]
    elif len({rows[i]['plan'][0] for i in tied})>1:
        chosen=next(i for i,r in enumerate(rows) if r['plan'][0] in ('NOP','HOLD') and r['plan'][2]=='NOP')
    else: chosen=tied[0]
    return dict(rows[chosen],frontier=indices,plans=rows)


class PragmaticWorld(CrossedWorld):
    def __init__(self,spec,risk,reporter):
        if reporter not in ('ideal','noise_0.5','noise_1','noise_2','causal'):
            raise ValueError('PRAGMATIC_REPORTER')
        self.reporter=reporter
        super().__init__(dict(spec,quality='ideal' if reporter=='ideal' else 'correlated'),risk)
        self.openings={};self.planned=OrderedDict()

    def _ensure_cache_context(self):
        old=self._cache_context
        super()._ensure_cache_context()
        if old is not None and old!=self._cache_context:
            self.openings.clear();self.planned.clear()

    def report_available(self,b):
        return b>=2 and b+1<self.end

    def report(self,b):
        self._ensure_cache_context()
        if b in self.reports: return self.reports[b]
        if not self.report_available(b): raise ValueError('REPORT_SUPPORT')
        e=self.env
        if self.reporter=='causal':
            f=e.cfg.profile;clock=int(e.v[b])+HOUR;opening=int(e.mo[0])
            r=prefix_report(np.column_stack((e.o,e.h,e.l,e.c))[:b+1],clock,HOUR,int(e.ff[0]),
                f.spread_at(clock,opening),terminal_spread=f.spread_at(int(e.ff[0]),opening))
            spreads=np.array([f.spread_at(int(t),opening) for t in e.v[b+1:self.end]])
            r=replace(r,spreads=np.broadcast_to(spreads,r.spreads.shape).copy())
        else:
            r=super().report(b)
            if self.reporter!='ideal':
                scale=float(self.reporter.split('_')[1])
                actual=np.column_stack((e.o,e.h,e.l,e.c))[b+1:self.end]
                paths=actual[None]+scale*(r.paths-actual[None])
                paths[:,:,1]=np.maximum(paths[:,:,1],np.maximum(paths[:,:,0],paths[:,:,3]))
                paths[:,:,2]=np.minimum(paths[:,:,2],np.minimum(paths[:,:,0],paths[:,:,3]))
                r=replace(r,paths=paths)
        self.reports[b]=r
        return r

    def opening_rows(self,b):
        self._ensure_cache_context()
        if b in self.openings: return self.openings[b]
        r=self.report(b);e=self.env
        validate_context(r,int(e.v[b])+HOUR,int(e.ff[0]))
        times=r.first_fill+np.arange(r.paths.shape[1])*HOUR
        rows=[]
        for h in HORIZONS:
            j=min(h,r.paths.shape[1])
            for d in (1,-1):
                entry=float(r.weights@(r.paths[:,0,0]+d*r.spreads[:,0]/2))
                p=self.position(h,d,entry,b+1)
                fo=self.funding(int(e.v[b+1]),d,times);fe=self.funding(int(e.v[b+1]),d,times+HOUR)
                v=value_paths(p,r,fo,fe)[:,j];mean=float(r.weights@v)
                rows.append(dict(horizon=h,direction=d,gain_risk=mean/p.risk,
                    uncertainty_risk=float(np.sqrt(r.weights@((v-mean)**2)))/p.risk,
                    duration=expected_duration(p,r,j)))
        self.openings[b]=rows
        return rows


def shadow_environment(env,r,sample):
    b=env.next_decision;start=b+1;end=start+r.paths.shape[1]
    validate_context(r,int(env.v[b])+HOUR,int(env.ff[env.seg_of[b]]))
    f=env.cfg.profile;opening=int(env.mo[env.seg_of[b]])
    expected=np.array([f.spread_at(int(t),opening) for t in env.v[start:end]])
    if (not np.array_equal(r.spreads[sample],expected)
            or r.terminal_spreads[sample]!=f.spread_at(r.deadline,opening)):
        raise ValueError('SHADOW_NATIVE_SPREAD_MISMATCH')
    out=clone(env);out.bars=env.bars.copy()
    for k,name in enumerate(('o','h','l','c')):
        values=getattr(env,name).copy();values[start:end]=r.paths[sample,:,k]
        # No observed suffix survives after the report's native weekly flat.
        values[end:]=r.paths[sample,-1,3]
        setattr(out,name,values)
        out.bars[('OPEN','HIGH','LOW','CLOSE')[k]]=values
    return out


def action_plan(world,action):
    if ':' in action:
        side,h=action.split(':');d=1 if side=='BUY' else -1
        return (1 if d==1 else 2),world.entry_plan(int(h),d,1.)
    if action not in ('NOP','HOLD','CLOSE'): raise ValueError('PLAN_ACTION')
    return (3 if action=='CLOSE' else 0),None


def rollout_plan(world,env,equity,r,plan):
    first,delay,second=plan;origin=env.next_decision;rows=[]
    for sample in range(len(r.weights)):
        e=shadow_environment(env,r,sample);eq=list(equity);second_opened=False
        while e.next_decision<e.last:
            b=e.next_decision;action='HOLD' if e.pos is not None else 'NOP'
            if b==origin: action=first
            elif not second_opened:
                if e.pos is None and second!='NOP' and e.entry_gate(b)[0]:
                    action=second;second_opened=True
                elif e.pos is not None and delay is not None and b>=origin+delay:
                    action='CLOSE'
            a,entry=action_plan(world,action)
            info=e.step(b,a,entry_plan=entry)
            if info['refused']: raise AssertionError(('SHADOW_ORDER_REFUSED',info))
            eq.append(float(e.equity_at(b+1)))
        if e.pos is not None or abs(e.balance-e.cfg.capital-sum(t['net_usd'] for t in e.trades))>1e-7:
            raise AssertionError('SHADOW_NATIVE_CASH')
        rows.append(objectives(eq,e.cfg.capital))
    return np.asarray(rows)


class PragmaticController:
    def __init__(self,config):
        self.config=config
        self.close=CloseController(CloseConfig(config.uncertainty,config.close_margin,
                                              config.persistence,config.urgent_risk),allow_synthetic=True)

    def decide(self,w,e,equity,horizon):
        b=e.next_decision;c=self.config
        if not w.report_available(b): return dict(action='HOLD' if e.pos else 'NOP',horizon=horizon)
        if e.pos is None and not e.entry_gate(b)[0]: return dict(action='NOP',horizon=None)
        ranked=rank_openings(w.opening_rows(b),c) if (e.pos is None or c.variant=='sequential') else []
        if c.variant=='sequential': return self.sequential(w,e,equity,horizon,ranked)
        if e.pos is None:
            if not ranked: return dict(action='NOP',horizon=None)
            bid=ranked[0]
            return dict(action='BUY' if bid['direction']==1 else 'SELL',horizon=bid['horizon'],score=bid['score'])
        unit,values,_,_,_=w.close_values(e,horizon,b);r=w.report(b)
        if c.variant=='predictive_proxy':
            n=r.paths.shape[1]
            columns=sorted({0,n,*[min(x,n) for x in (1,2,3,horizon)]})
            values=values[:,columns]
        result=self.close._decide_values(unit.identity,int(e.v[b])+HOUR,HOUR,
                                         values,r.weights,unit.risk,r.provenance)
        return dict(action=result['action'],horizon=horizon,close=result)

    def sequential(self,w,e,equity,horizon,ranked):
        b=e.next_decision;c=self.config;r=w.report(b)
        entries=[('BUY' if row['direction']==1 else 'SELL')+':'+str(row['horizon']) for row in ranked[:c.shortlist]]
        passive='HOLD' if e.pos is not None else 'NOP'
        roots=['HOLD','CLOSE'] if e.pos is not None else ['NOP']+entries
        plans=[(passive,None,'NOP')]
        for first in roots:
            if first in ('CLOSE','NOP'):
                plans.extend((first,None,next_) for next_ in ['NOP']+entries)
            else:
                plans.append((first,None,'NOP'))
                plans.extend((first,d,next_) for d in c.close_offsets
                             for next_ in ['NOP']+entries)
        plans=list(dict.fromkeys(plans))
        # Account state is controller-side only. The plan cache never shares unequal states.
        state=(b,e.balance,repr(e.pos),tuple(equity))
        samples=[]
        for plan in plans:
            key=(state,plan)
            if key not in w.planned:
                w.planned[key]=rollout_plan(w,e,equity,r,plan)
                if len(w.planned)>4096: w.planned.popitem(last=False)
            samples.append(w.planned[key])
        chosen=select_plan(plans,samples,r.weights,c);first=chosen['plan'][0]
        if e.pos is not None:
            best_hold=max(p['score'] for p in chosen['plans'] if p['plan'][0]=='HOLD')
            best_close=max(p['score'] for p in chosen['plans'] if p['plan'][0]=='CLOSE')
            unit=w.position(horizon,e.pos.direction,e.pos.entry_price,e.pos.entry_bar,e.pos.units)
            advantage=(best_close-best_hold)*e.cfg.capital/100/unit.risk
            now=int(e.v[b])+HOUR
            qualifies=advantage>max(c.close_margin,1e-12)
            contiguous=self.close.position_id==unit.identity and self.close.last_tick==now-HOUR
            self.close.streak=(self.close.streak if contiguous else 0)+1 if qualifies else 0
            self.close.position_id=unit.identity;self.close.last_tick=now
            close=first=='CLOSE' and qualifies and (self.close.streak>=c.persistence or advantage>=c.urgent_risk)
            first='CLOSE' if close else 'HOLD'
        action=first.split(':')[0]
        selected_h=int(first.split(':')[1]) if ':' in first else horizon
        return dict(action=action,horizon=selected_h,plan=chosen)


def run_account(world,config,*,diagnose=False,controller=None):
    e=clone(world.env);e.reset();equity=[e.cfg.capital]
    controller=PragmaticController(config) if controller is None else controller
    horizon=None;used={};trace=[];closes=[]
    while e.next_decision<e.last:
        b=e.next_decision;decision=controller.decide(world,e,equity,horizon)
        action=decision['action'];horizon=decision['horizon']
        if diagnose and action=='CLOSE' and e.pos is not None:
            risk=world.position(horizon,e.pos.direction,e.pos.entry_price,e.pos.entry_bar,e.pos.units).risk
            closes.append(fixed_position_exits(e,risk))
        named=action+':'+str(horizon) if action in ('BUY','SELL') else action
        a,plan=action_plan(world,named);info=e.step(b,a,entry_plan=plan)
        if info['refused']: raise AssertionError(('ACCOUNT_ORDER_REFUSED',info))
        if info['entry']: used[b+1]=horizon
        equity.append(float(e.equity_at(b+1)))
        event=dict(bar=b,action=action,horizon=horizon,close=info['close'])
        if 'plan' in decision: event['planning']=decision['plan']
        trace.append(event)
    if e.pos is not None: raise AssertionError('WEEKLY_POSITION_REMAINS')
    result=e.result();ret,dd=objectives(equity,e.cfg.capital)
    net=sum(t['net_usd'] for t in e.trades);cash=e.balance-e.cfg.capital-net
    if abs(cash)>1e-7 or abs(result['net_profit_usd']-net)>1e-7 or abs(result['max_drawdown_pct']-dd)>1e-7:
        raise AssertionError('PRAGMATIC_NATIVE_METRICS')
    trades=e.trades;buys=[t for t in trades if t['direction']==1];sells=[t for t in trades if t['direction']==-1]
    executed=[x for x in closes if not x['preempted']]
    sharpe=defined_sharpe(float(result['sharpe_daily_sqrt365']),result['daily_equity'],e.cfg.capital)
    market_weeks=len(e.cal.segments)
    if market_weeks!=1: raise ValueError('PRAGMATIC_FULL_SINGLE_SESSION_REQUIRED')
    return dict(spec=world.spec,reporter=world.reporter,config=asdict(config),return_pct=ret,drawdown_pct=dd,
        net_usd=net,sharpe=sharpe,cash_error=cash,
        orders=len(trades),orders_per_week=len(trades)/market_weeks,market_weeks=market_weeks,
        native_iso_orders_per_week=result['orders_per_week'],wins=sum(t['net_usd']>0 for t in trades),
        buys=len(buys),sells=len(sells),buy_wins=sum(t['net_usd']>0 for t in buys),sell_wins=sum(t['net_usd']>0 for t in sells),
        horizons=dict(Counter(str(used[t['entry_bar_abs']]) for t in trades)),
        close_reasons=dict(Counter(t['close_reason'] for t in trades)),
        costs={k:sum(t[k] for t in trades) for k in ('spread_usd','slippage_usd','commission_usd','financing_usd')},
        close_diagnostics=dict(measured=diagnose,executed=len(executed),
            helpful=sum(x['delta_vs_hold']>1e-9 for x in executed),harmful=sum(x['delta_vs_hold']< -1e-9 for x in executed),
            mean_regret_risk=float(np.mean([x['regret_risk'] for x in executed])) if executed else None),
        close_rows=closes,trace=trace,ledger=[dict(direction=t['direction'],horizon=used[t['entry_bar_abs']],
            entry_bar=t['entry_bar_abs'],exit_bar=t['exit_bar_abs'],net=t['net_usd'],units=t['units'],
            q=t['percentual_order_size'],reason=t['close_reason']) for t in trades])
