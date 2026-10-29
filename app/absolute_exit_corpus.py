"""Synthetic position counterfactuals executed by the native absolute adapter."""
import copy
from dataclasses import replace
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from app.early_close_metrics import evaluate_exit, tick_metrics
from app.horizon_risk_rules import propose
from app.protection_reference import protection_exit


def specs():
    axes=itertools.product((3,6,12,24),(1,-1),('gain','loss','tp'),(2,6,12),
                           ('constant','increasing','decreasing'),(.5,3.),(12,36),(7,19))
    for i,values in enumerate(axes):
        yield dict(id=f'matrix_{i:04d}',**dict(zip(
            ('horizon','direction','mode','peak','volatility','spread_pips','entry_hour','seed'),values)))
    for i,(h,d,spread) in enumerate(itertools.product((3,6,12,24),(1,-1),(.5,3.))):
        yield dict(id=f'gap_{i:02d}',horizon=h,direction=d,mode='gap',peak=2,
                   volatility='constant',spread_pips=spread,entry_hour=12,seed=7)


def _funding(e,p,when):
    profile=e.cfg.profile
    rate=profile.swap_long_per_lot if p.direction==1 else profile.swap_short_per_lot
    return sum(-rate*p.units/profile.lot_units*(3 if i==profile.triple_swap_rollover_index else 1)
               for i,t in enumerate(e.cal.segments[p.seg].rollovers(),1)
               if e.v[p.entry_bar] < t.value <= when)


def independent_net(e,p,quote,when,reason):
    profile=e.cfg.profile
    exit_slip=profile.slippage_stop if reason in ('SL','MARGIN_CLOSEOUT') else 0. if reason=='TP' else profile.slippage_market
    fees=p.units*(profile.slippage_market+exit_slip+2*profile.commission_per_lot_side/profile.lot_units)
    return float(p.direction*p.units*(quote-p.entry_price)-fees-_funding(e,p,when))


def create_case(spec,config):
    from app.short_horizon import execution as X
    from app.short_horizon.expert_guided.absolute import AbsoluteBarEnv, AbsoluteOrder
    from app.short_horizon.expert_guided.env import EnvConfig
    from app.short_horizon.expert_guided.data import calendar_from_closures

    d=spec['direction']; spread=spec['spread_pips']*.0001; entry=spec['entry_hour']
    ref=next(r for r in config['rows'] if r['horizon']==spec['horizon'] and r['direction']==('buy' if d==1 else 'sell'))['reference']['median']
    ts=pd.date_range('2020-01-05T22:00Z',periods=120,freq='h')
    bars=pd.DataFrame(dict(OPEN=1.1,HIGH=1.100001,LOW=1.099999,CLOSE=1.1),index=ts)
    cal=calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],
                               ['2020-01-10T22:00Z','2020-01-12T22:00Z']],cutoff=0)
    profile=replace(X.RealisticProfile(),spread_schedule=((0,24,spread),),weekly_open_spread=spread)
    sizing=X.Sizing(rel_volume=.1,leverage=50)
    cfg=EnvConfig(profile=profile,sizing=sizing,execution_price_mode='bid_ask_v3',
                  max_entries_per_day=None,min_orders_per_week=0)
    # Known calendar charges, reserved conservatively through the forced-flat deadline.
    rate=profile.swap_long_per_lot if d==1 else profile.swap_short_per_lot
    reserve=sum(max(0.,-rate/profile.lot_units)*(3 if i==profile.triple_swap_rollover_index else 1)
                for i,t in enumerate(cal.segments[0].rollovers(),1)
                if ts[entry].value<t.value<=cal.segments[0].force_flat.value)

    def build(context):
        result=propose(direction=context.direction,entry_quote=context.entry_quote,
            liquidation_quote=context.liquidation_quote,balance=context.balance,
            tp_reference=ref['tp_pips']*.0001,mae_reference=ref['mae_pips']*.0001,
            sigma=.001,genes=config['baseline'],sizing=sizing,tick=.00001,min_distance=cfg.rules.min_distance,
            quality=1.,loss_cost_per_unit=profile.slippage_market+profile.slippage_stop+2*profile.commission_per_lot_side/profile.lot_units,
            funding_reserve_per_unit=reserve)
        if result['status']!='ORDER':
            raise AssertionError('SYNTHETIC_ENTRY_REFUSED')
        return AbsoluteOrder(result['tp'],result['sl'],result['q'],result['risk_budget'],reserve)

    # Generate market before replay; neither the callback nor the policy receives future bars.
    from app.short_horizon.expert_guided.absolute import FillContext
    quote=1.1+d*spread/2
    initial=build(FillContext(d,10000.,quote,1.1-d*spread/2,int(ts[entry].value),int(ts[entry].value)))
    sl_distance=d*(quote-initial.sl); tp_distance=d*(initial.tp-quote)
    rng=np.random.default_rng(spec['seed']); peak=spec['peak']
    for j in range(entry+1,len(ts)):
        t=j-entry
        if spec['mode']=='gain':
            exc=.6*tp_distance*t/peak if t<=peak else .6*tp_distance-(.6*tp_distance+1.5*sl_distance)*min(1.,(t-peak)/5)
        elif spec['mode']=='loss':
            exc=(-.65+.5*t/peak)*sl_distance if t<=peak else (-.15-1.35*min(1.,(t-peak)/5))*sl_distance
        elif spec['mode']=='tp':
            exc=min(1.3,1.15*t/peak)*tp_distance
        else:
            exc=-sl_distance-.002
        if spec['mode']!='gap' and t!=peak:
            exc+=rng.uniform(-.01,.01)*min(sl_distance,tp_distance)
        frac=min(1.,t/(peak+5))
        vol={'constant':1.,'increasing':.25+1.5*frac,'decreasing':1.75-1.5*frac}[spec['volatility']]
        width=.02*sl_distance*vol
        mid=quote+d*exc+d*spread/2
        bars.iloc[j]=[mid,mid+width,mid-width,mid]
    e=AbsoluteBarEnv(bars,cal,np.full(120,.001),np.full(120,.002),ts[0],ts[-1]+pd.Timedelta(hours=1),cfg,tick=.00001)
    while e.next_decision<entry-1: e.step(e.next_decision,0)
    info=e.step(entry-1,1 if d==1 else 2,entry_plan=build)
    if not info['entry'] or e.pos is None:
        raise AssertionError('INITIAL_POSITION_NOT_SURVIVING_ENTRY_BAR')
    return e,info['absolute_order']['modeled_loss']


def evaluate_case(spec,config):
    e,risk=create_case(spec,config)
    p=copy.deepcopy(e.pos); start=e.next_decision+1
    end=min(e.last,int(np.searchsorted(e.v,e.ff[p.seg]))-1)
    profile=e.cfg.profile
    def spread(k):
        return profile.spread_at(int(e.v[k]),int(e.mo[p.seg]))
    half=np.array([spread(k)/2 for k in range(start,end+1)])
    quotes=[a[start:end+1]-p.direction*half for a in (e.o,e.h,e.l,e.c)]
    offset,terminal_quote,reason=protection_exit(p.direction,p.tp,p.sl,*quotes,profile.limit_through)
    expected_reason='TP' if spec['mode']=='tp' else 'SL'
    if reason!=expected_reason:
        raise AssertionError(('SCENARIO_PASSIVE_REASON',spec,reason))
    terminal_bar=start+offset
    at_open=(p.direction*(quotes[0][offset]-p.sl)<=0 or p.direction*(quotes[0][offset]-p.tp)>=0)
    terminal_ns=int(e.v[terminal_bar])+(0 if at_open else 3600*10**9)
    terminal_net=independent_net(e,p,terminal_quote,terminal_ns,reason)
    sig=dict(entry=p.entry_price,side=p.direction,units=p.units,sl=p.sl,tp=p.tp,bar=p.entry_bar,risk=risk)
    signature=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest()
    menu=[]; native=[]; fill_bars=[]; worst=0.
    while e.pos is not None:
        b=e.next_decision; k=b+1
        quote=float(e.o[k])-p.direction*spread(k)/2
        if p.direction*(quote-p.sl)>0 and p.direction*(p.tp-quote)>0:
            value=independent_net(e,p,quote,int(e.v[k]),'AGENT_CLOSE')
            trial=copy.deepcopy(e)
            if (trial.pos.entry_price,trial.pos.units,trial.pos.sl,trial.pos.tp)!=(p.entry_price,p.units,p.sl,p.tp):
                raise AssertionError('COUNTERFACTUAL_POSITION_CHANGED')
            info=trial.step(b,3)
            if info['close']!='AGENT_CLOSE' or len(trial.trades)!=1:
                raise AssertionError('EARLY_CLOSE_EXECUTION')
            measured=float(trial.trades[0]['net_usd']); worst=max(worst,abs(measured-value))
            if abs(measured-value)>1e-7:
                raise AssertionError('INDEPENDENT_CLOSE_CASH')
            menu.append(value); native.append(measured); fill_bars.append(k)
        e.step(b,0)
    trade=e.trades[0]
    if (trade['close_reason']!=reason or trade['exit_bar_abs']!=terminal_bar
            or abs(trade['exit_price']-terminal_quote)>1e-12 or abs(trade['net_usd']-terminal_net)>1e-7):
        raise AssertionError('INDEPENDENT_PASSIVE_EXECUTION')
    worst=max(worst,abs(trade['net_usd']-terminal_net))
    if spec['mode']=='gap':
        if menu: raise AssertionError('GAP_HAS_NO_EXECUTABLE_CLOSE')
    elif not menu:
        raise AssertionError('NONVACUOUS_CLOSE_MENU')
    best=max([terminal_net]+menu)
    oracle=next((i for i,v in enumerate(menu) if v>=best),None)
    if (spec['mode'] in ('gain','loss') and oracle is None) or (spec['mode']=='tp' and oracle is not None):
        raise AssertionError(('PREDECLARED_EARLY_BEHAVIOR',spec))
    policies={}
    for name,index in dict(oracle=oracle,close_first=0 if menu else None,never_close=None).items():
        actual=native[index] if index is not None else trade['net_usd']
        policies[name]=evaluate_exit(menu,terminal_net,risk,index,actual_net=actual)
    if policies['oracle']['regret_risk']>1e-7:
        raise AssertionError('ORACLE_REGRET')
    if abs(e.balance-10000.-trade['net_usd'])>1e-7:
        raise AssertionError('PASSIVE_CASH_BALANCE')
    return dict(spec=spec,position= sig,position_sha256=signature,
                passive_reason=reason,passive_exit_bar=terminal_bar,
                fill_bars=fill_bars,tick_metrics=tick_metrics(menu,terminal_net,risk),
                policies=policies,native_close_checks=len(menu),max_cash_error=worst,
                oracle_exit_bar=fill_bars[oracle] if oracle is not None else terminal_bar,
                native_passive_net=float(trade['net_usd']))
