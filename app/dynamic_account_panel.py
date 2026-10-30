"""Explicitly synthetic, full-account allocation experiments on the native engine."""
from collections import OrderedDict, Counter
import copy
from dataclasses import replace

import numpy as np
import pandas as pd

from app.close_hold_controller import PositionView, PathReport, CloseConfig, CloseController, value_paths, validate_context, report_identity, bind_values
from app.dynamic_account_arbiter import HORIZONS, choose, decode, objectives
from app.horizon_risk_rules import levels, propose
from app.protection_reference import protection_exit

HOUR=3600*10**9
BASE=(1.,1.,1.,1.,0.,0.,0.,0.,1.,0.,1.)


def expected_duration(position,report,choice):
    """Hours occupied, not nominal exit horizon; protection can end a path first."""
    count=report.paths.shape[1]
    if not 1<=choice<=count: raise ValueError('ENTRY_EXIT_CHOICE')
    lengths=[]
    for path,spread in zip(report.paths,report.spreads):
        q=path-position.direction*spread[:,None]/2
        at,_,reason=protection_exit(position.direction,position.tp,position.sl,*q.T,position.limit_through)
        at_open=reason!='DEADLINE' and (position.direction*(q[at,0]-position.sl)<=0
                                      or position.direction*(q[at,0]-position.tp)>=0)
        lengths.append(min(choice,at if at_open else at+1))
    return float(report.weights@np.asarray(lengths))*report.step/HOUR


def panel_specs(seed):
    return [dict(seed=seed,pattern=p,volatility=v,mirror=m)
            for p in ('trend','reversal','overlap')
            for v in ('increasing','decreasing') for m in (1,-1)]


class World:
    """Fixed exogenous randomness; no generated input depends on policy actions."""
    def __init__(self,spec,risk_config):
        from app.short_horizon import execution as X
        from app.short_horizon.expert_guided.absolute import AbsoluteBarEnv
        from app.short_horizon.expert_guided.env import EnvConfig
        from app.short_horizon.expert_guided.data import calendar_from_closures
        self.spec=spec;self.risk_config=risk_config
        p=('trend','reversal','overlap').index(spec['pattern'])
        rng=np.random.default_rng(np.random.SeedSequence([spec['seed'],p,51]))
        t=np.arange(120);turn=int(rng.integers(35,75));phase=rng.uniform(0,2*np.pi)
        if p==0:
            signal=.000045*t+.0005*np.sin(t/5+phase)
        elif p==1:
            signal=.00009*np.minimum(t,turn)-.000085*np.maximum(0,t-turn)
        else:
            signal=.0018*np.sin(t/16+phase)+.0008*np.sin(t/4+.7*phase)
        envelope=np.linspace(.3,1.8,120)
        if spec['volatility']=='decreasing': envelope=envelope[::-1]
        close=1.1+signal+np.cumsum(rng.normal(0,.00016,120)*envelope)
        op=np.r_[close[0],close[:-1]]
        if p!=0:
            gap=int(rng.integers(50,95));close[gap:]-=.0012;op[gap:]-=.0012
        width=rng.uniform(.000025,.00012,120)*envelope
        high=np.maximum(op,close)+width;low=np.minimum(op,close)-width
        if spec['mirror']==-1:
            op,high,low,close=2.2-op,2.2-low,2.2-high,2.2-close
        ts=pd.date_range('2020-01-05T22:00Z',periods=120,freq='h')
        bars=pd.DataFrame(dict(OPEN=op,HIGH=high,LOW=low,CLOSE=close),index=ts)
        cal=calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],
                                   ['2020-01-10T22:00Z','2020-01-12T22:00Z']],cutoff=0)
        spread=.00005 if spec['volatility']=='decreasing' else .0003
        profile=replace(X.RealisticProfile(),spread_schedule=((0,24,spread),),weekly_open_spread=spread)
        cfg=EnvConfig(profile=profile,sizing=X.Sizing(rel_volume=.1,leverage=50),
                      execution_price_mode='bid_ask_v3',max_entries_per_day=None,min_orders_per_week=3.)
        self.env=AbsoluteBarEnv(bars,cal,np.full(120,.001),np.full(120,.002),
            ts[0],ts[-1]+pd.Timedelta(hours=1),cfg,tick=.00001)
        self.end=int(np.searchsorted(self.env.v,self.env.ff[0]))
        self.reports={};self.bids={};self.cache=OrderedDict()
        self._cache_context=None
        self.references={(r['horizon'],1 if r['direction']=='buy' else -1):r['reference']['median']
                         for r in risk_config['rows']}
        self.rolls=[(x.value,3 if i==profile.triple_swap_rollover_index else 1)
                    for i,x in enumerate(cal.segments[0].rollovers(),1)]

    def funding(self,entry,d,times,positive_only=False):
        f=self.env.cfg.profile
        rate=-(f.swap_long_per_lot if d==1 else f.swap_short_per_lot)/f.lot_units
        if positive_only: rate=max(0.,rate)
        a=np.asarray(times,dtype=np.int64)
        return sum((rate*n*(a>=t) for t,n in self.rolls if entry<t),np.zeros(a.shape))

    def _ensure_cache_context(self):
        context=repr((self.env.cfg,self.risk_config,self.references,self.rolls))
        if self._cache_context is not None and context!=self._cache_context:
            self.reports.clear();self.bids.clear();self.cache.clear()
        self._cache_context=context

    def report(self,b):
        self._ensure_cache_context()
        if b not in self.reports:
            e=self.env;future=np.column_stack((e.o,e.h,e.l,e.c))[b+1:self.end]
            rng=np.random.default_rng(np.random.SeedSequence([self.spec['seed'],b,901]))
            bias=np.cumsum(rng.normal(.000005,.00006,len(future)))
            noise=np.cumsum(rng.normal(0,.0001,len(future)))
            error=(bias[None,:]+np.array([-1.,0.,1.])[:,None]*noise)*self.spec['mirror']
            paths=future[None,:,:]+error[:,:,None]
            spreads=np.full(paths.shape[:2],e.cfg.profile.spread_at(int(e.v[b]),int(e.mo[0])))
            first=int(e.v[b])+HOUR
            self.reports[b]=PathReport(first,first,first,HOUR,int(e.ff[0]),paths,spreads,
                                      np.full(3,1/3),'synthetic_noisy',
                                      np.full(3,e.cfg.profile.spread_at(int(e.ff[0]),int(e.mo[0]))))
        return self.reports[b]

    def report_available(self,b):
        return True

    def geometry(self,h,d,entry):
        ref=self.references[h,d]
        return levels(d,entry,ref['tp_pips']*.0001,ref['mae_pips']*.0001,
                      .001,self.risk_config['baseline'],.00001)

    def position(self,h,d,entry,entry_bar,units=1.,risk=None):
        f=self.env.cfg.profile;lv=self.geometry(h,d,entry)
        tp,sl=lv['tp'],lv['sl']
        reserve=float(self.funding(int(self.env.v[entry_bar]),d,self.env.ff[0],True))
        perunit=d*(entry-sl)+f.slippage_market+f.slippage_stop+2*f.commission_per_lot_side/f.lot_units+reserve
        return PositionView(f'{h}:{d}:{entry_bar}',d,entry,units,sl,tp,perunit*units if risk is None else risk,
                            f.slippage_market,f.slippage_market,f.slippage_stop,
                            2*f.commission_per_lot_side/f.lot_units,f.limit_through)

    def candidate_bids(self,b):
        r=self.report(b)
        validate_context(r,int(self.env.v[b])+HOUR,int(self.env.ff[0]))
        key=(b,report_identity(r))
        if key not in self.bids:
            rows=[];entry_bar=b+1
            times=r.first_fill+np.arange(r.paths.shape[1])*HOUR
            for h in HORIZONS:
                for d in (1,-1):
                    entry=float(r.weights@(r.paths[:,0,0]+d*r.spreads[:,0]/2))
                    p=self.position(h,d,entry,entry_bar)
                    fo=self.funding(int(self.env.v[entry_bar]),d,times)
                    fe=self.funding(int(self.env.v[entry_bar]),d,times+HOUR)
                    # Entry-bar voluntary exit is not executable. Barriers still are.
                    values=value_paths(p,r,fo,fe)[:,1:]
                    means=r.weights@values;j=int(np.argmax(means))
                    uncertainty=float(np.sqrt(r.weights@((values[:,j]-means[j])**2)))
                    duration=expected_duration(p,r,j+1)
                    rows.append(dict(horizon=h,direction=d,gain_risk=float(means[j])/p.risk,
                        uncertainty_risk=uncertainty/p.risk,duration=duration))
            self.bids[key]=rows
        return self.bids[key]

    def past_efficiency(self,b):
        x=np.diff(self.env.c[max(0,b-6):b+1]);den=float(np.abs(x).sum())
        return float(x.sum()/den) if den else 0.

    def entry_plan(self,h,d,quality):
        from app.short_horizon.expert_guided.absolute import AbsoluteOrder
        e=self.env;ref=self.references[h,d];f=e.cfg.profile
        def build(ctx):
            reserve=float(self.funding(ctx.fill_ns,d,e.ff[0],True))
            result=propose(direction=d,entry_quote=ctx.entry_quote,liquidation_quote=ctx.liquidation_quote,
                balance=ctx.balance,tp_reference=ref['tp_pips']*.0001,mae_reference=ref['mae_pips']*.0001,
                sigma=.001,genes=self.risk_config['baseline'],sizing=e.cfg.sizing,tick=e.tick,
                min_distance=e.cfg.rules.min_distance,quality=quality,
                loss_cost_per_unit=f.slippage_market+f.slippage_stop+2*f.commission_per_lot_side/f.lot_units,
                funding_reserve_per_unit=reserve)
            if result['status']!='ORDER': raise ValueError(('PANEL_ORDER_REFUSED',result))
            return AbsoluteOrder(result['tp'],result['sl'],result['q'],result['risk_budget'],reserve)
        return build

    def close_values(self,e,h,b):
        if e.cfg!=self.env.cfg: raise ValueError('CACHE_EXECUTION_CONFIG')
        p=e.pos;r=self.report(b)
        validate_context(r,int(e.v[b])+HOUR,int(e.ff[p.seg]))
        unit=self.position(h,p.direction,p.entry_price,p.entry_bar)
        if (unit.sl,unit.tp)!=(p.sl,p.tp): raise AssertionError('CACHE_GEOMETRY')
        times=r.first_fill+np.arange(r.paths.shape[1])*HOUR
        fo=self.funding(int(e.v[p.entry_bar]),p.direction,times)
        fe=self.funding(int(e.v[p.entry_bar]),p.direction,times+HOUR)
        key=(unit,report_identity(r),fo.tobytes(),fe.tobytes())
        if key not in self.cache:
            values=value_paths(unit,r,fo,fe)
            self.cache[key]=(unit,values,fo,fe,bind_values(unit,r,values))
            if len(self.cache)>512: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def run(self,genes,enabled=None,verify=False,details=False,*,voluntary_close=True):
        genes=decode(genes);e=copy.copy(self.env);e.reset();equity=[e.cfg.capital]
        controller=CloseController(CloseConfig(genes[5],genes[9],genes[10],.25),allow_synthetic=True)
        direct=CloseController(controller.config,allow_synthetic=True) if verify else None
        horizon=None;trace=[];used={};occupied=0;max_error=0.
        while e.next_decision<e.last:
            b=e.next_decision;action=0;plan=None;selected=None
            if e.pos is not None:
                occupied+=1
                if voluntary_close and b+1<self.end and self.report_available(b):
                    unit,values,fo,fe,binding=self.close_values(e,horizon,b);r=self.report(b)
                    clock=int(e.v[b])+HOUR;deadline=int(e.ff[e.pos.seg])
                    result=controller.decide_synthetic_values(unit.identity,clock,HOUR,
                        values,r.weights,unit.risk,report=r,expected_deadline=deadline,binding=binding)
                    if verify:
                        actual=replace(unit,units=e.pos.units,risk=unit.risk*e.pos.units)
                        other=direct.decide(actual,r,clock,fo,fe,expected_deadline=deadline)
                        max_error=max(max_error,abs(other['advantage_risk']-result['advantage_risk']))
                        if other['action']!=result['action'] or max_error>1e-10:
                            raise AssertionError('CACHE_DIRECT_PARITY')
                    action=3 if result['action']=='CLOSE' else 0
            elif e.entry_gate(b)[0] and self.report_available(b):
                bids=self.candidate_bids(b)
                if enabled is not None: bids=[x for x in bids if (x['horizon'],x['direction']) in enabled]
                selected=choose(bids,genes,self.past_efficiency(b))
                if selected:
                    horizon=selected['horizon'];d=selected['direction'];action=1 if d==1 else 2
                    plan=self.entry_plan(horizon,d,genes[8])
            info=e.step(b,action,entry_plan=plan)
            if info['entry']: used[b+1]=horizon
            if info['refused']: raise AssertionError(('UNEXPECTED_REFUSAL',info))
            equity.append(float(e.equity_at(b+1)))
            if details: trace.append(dict(bar=b,action=action,horizon=horizon,close=info['close']))
        if e.pos is not None: raise AssertionError('POSITION_AFTER_WEEKLY_FLAT')
        net=sum(t['net_usd'] for t in e.trades)
        if abs(e.balance-e.cfg.capital-net)>1e-7: raise AssertionError('CASH_CONSERVATION')
        for left,right in zip(e.trades,e.trades[1:]):
            if right['entry_bar_abs']<=left['exit_bar_abs']: raise AssertionError('OVERLAPPING_TRADES')
        ret,dd=objectives(equity,e.cfg.capital)
        out=dict(spec=self.spec,return_pct=ret,drawdown_pct=dd,net_usd=net,orders=len(e.trades),
            wins=sum(t['net_usd']>0 for t in e.trades),buys=sum(t['direction']==1 for t in e.trades),
            buy_wins=sum(t['direction']==1 and t['net_usd']>0 for t in e.trades),
            sell_wins=sum(t['direction']==-1 and t['net_usd']>0 for t in e.trades),
            horizons=dict(Counter(str(used[t['entry_bar_abs']]) for t in e.trades)),
            closes=dict(Counter(t['close_reason'] for t in e.trades)),exposure=occupied/(e.last-e.first),
            costs={k:sum(t[k] for t in e.trades) for k in ('spread_usd','slippage_usd','commission_usd','financing_usd')})
        if verify:
            m=e.result()
            if abs(m['net_profit_usd']-net)>1e-7 or abs(m['max_drawdown_pct']-dd)>1e-7:
                raise AssertionError(('NATIVE_ACCOUNT_PARITY',net,dd,m))
            out['native_sharpe']=m['sharpe_daily_sqrt365'];out['cache_max_error']=max_error
        if details:
            out['trace']=trace
            out['ledger']=[dict(horizon=used[t['entry_bar_abs']],direction=t['direction'],
                entry_bar=t['entry_bar_abs'],exit_bar=t['exit_bar_abs'],net=t['net_usd'],
                q=t['percentual_order_size'],units=t['units'],reason=t['close_reason']) for t in e.trades]
        return out


def evaluate(worlds,genes,enabled=None,verify=False,details=False):
    rows=[w.run(genes,enabled,verify,details) for w in worlds]
    mean_orders=float(np.mean([r['orders'] for r in rows]))
    return dict(objectives=[float(np.mean([r['return_pct'] for r in rows])),
                            -max(r['drawdown_pct'] for r in rows)],
                eligible=mean_orders>=3,mean_orders=mean_orders,worlds=rows)
