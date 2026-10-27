"""Full synthetic quote-engine episodes with a separate forward-path reference.

Engineering oracle only: it deliberately sees synthetic future quotes. No policy
fit, financial features, optimization superiority or learning qualification.
"""
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from app.frozen_recipe_inputs import require
from app.protection_reference import protection_exit, entry_command, prefer_close
from app.stochastic_strategy_paths import market_path
from app.strategy_scenario_matrix import WARMUP, check_trade
from app.directional_risk_calibration import DirectionalRisk, DirectionalOrders


def expected_pnl(e,d,units,entry,k_entry,tp,sl,start,end,immediate=False):
    """Independent quote-side path/fee arithmetic, never calls engine.step/_protect."""
    p=e.cfg.profile
    def spread(k,close=False):
        return p.spread_at(int(e.v[k])+(3600*10**9 if close else 0),int(e.mo[e.seg_of[k]]))
    ix=np.arange(start,end+1)
    half=np.array([spread(int(k))/2 for k in ix])
    arrays=[v[ix]-d*half for v in (e.o,e.h,e.l,e.c)]
    if immediate:
        offset,price,reason=0,float(arrays[0][0]),'MARKET'
        open_exit=True
    else:
        # A deadline liquidation uses its known close-time spread, unlike intrabar fills.
        offset,price,reason=protection_exit(d,tp,sl,*arrays,p.limit_through)
        if reason=='DEADLINE':price=float(e.c[end]-d*spread(end,True)/2)
        open_exit=(reason=='SL' and d*(arrays[0][offset]-sl)<=0) or (
                   reason=='TP' and d*(arrays[0][offset]-tp)>=0)
    k=start+offset
    exit_ns=int(e.v[k])+(0 if open_exit else 3600*10**9)
    entry_ns=int(e.v[k_entry]);si=int(e.seg_of[k_entry])
    rate=p.swap_long_per_lot if d==1 else p.swap_short_per_lot
    funding=sum(-rate*units/p.lot_units*(3 if j==p.triple_swap_rollover_index else 1)
        for j,t in enumerate(e.cal.rollover_ns(si),1) if entry_ns<t<=exit_ns)
    slip=p.slippage_stop if reason=='SL' else 0. if reason=='TP' else p.slippage_market
    fees=units*(p.slippage_market+slip)+2*units/p.lot_units*p.commission_per_lot_side+funding
    return float(d*units*(price-entry)-fees),dict(reason=reason,exit_bar=k,exit_quote=price)


def ideal_command(env,horizon,minimum_gain_risk=1.):
    """Generate diagnostic desired outputs from the same frozen synthetic future."""
    e=env.engine;b=env.bar;k=b+1
    end=min(k+horizon-1,e.last,int(np.searchsorted(e.v,e.ff[e.seg_of[k]],side='left'))-1)
    end=max(k,end)
    if e.pos is not None:
        p=e.pos
        spread=e.cfg.profile.spread_at(int(e.v[k]),int(e.mo[e.seg_of[k]]))
        liquidation=e.o[k]-p.direction*spread/2
        if p.direction*(liquidation-p.sl)<=0 or p.direction*(liquidation-p.tp)>=0:
            return 0,dict(reason='BROKER_OPEN_PROTECTION_PRIORITY')
        hold,ref=expected_pnl(e,p.direction,p.units,p.entry_price,p.entry_bar,p.tp,p.sl,k,end)
        close,_=expected_pnl(e,p.direction,p.units,p.entry_price,p.entry_bar,p.tp,p.sl,k,k,True)
        chosen=(3 if p.direction==1 else 4) if prefer_close(close,hold,0.) else 0
        return chosen,dict(close_net=close,hold_net=hold,reference=ref)
    if not env.mask()[1]:return 0,dict(reason='BUSINESS_MASK')
    gains=[];risks=[];details=[]
    for cmd,d in ((1,1),(2,-1)):
        ft,fs,q=env.parameters.for_command(cmd)
        bounds=env.feasible_bounds()
        tm,sm=[a+f*(z-a) for (a,z),f in zip(bounds,(ft,fs))]
        # Levels follow the documented production formula; path/fill evaluation is independent.
        sigma=e.decision_sigma(b)
        tp=e.c[b]+d*tm*sigma;sl=e.c[b]-d*sm*sigma
        spread=e.cfg.profile.spread_at(int(e.v[k]),int(e.mo[e.seg_of[k]]))
        entry=e.o[k]+d*spread/2;liquidation=e.o[k]-d*spread/2
        factor=e.cfg.sizing.leverage if e.cfg.sizing.basis=='margin' else 1.
        units=math.floor(q*math.floor(e.balance*e.cfg.sizing.rel_volume*factor/entry))
        feasible=(units>=e.cfg.sizing.min_units and d*(tp-entry)>0 and d*(entry-sl)>0
            and d*(tp-liquidation)>0 and d*(liquidation-sl)>0
            and min(abs(tp-entry),abs(sl-entry),abs(tp-liquidation),abs(sl-liquidation))>=e.cfg.rules.min_distance)
        if not feasible:
            gains.append(-1.);risks.append(1.);details.append(dict(feasible=False));continue
        gain,ref=expected_pnl(e,d,units,entry,k,tp,sl,k,end)
        p=e.cfg.profile
        risk=units*(abs(entry-sl)+p.slippage_market+p.slippage_stop+
                    2*p.commission_per_lot_side/p.lot_units)
        gains.append(gain);risks.append(risk);details.append(dict(feasible=True,reference=ref))
    chosen=entry_command(gains,risks,minimum_gain_risk)
    return chosen,dict(gains=gains,losses_at_stop=risks,details=details,
                       minimum_gain_risk=minimum_gain_risk)


def run_episode(spec,config,shape,genes=(.5,.5,.2)*2,horizon=24,mode='ideal',spread_factor=1.,minimum_gain_risk=1.):
    from app.short_horizon.expert_guided.env import BarEnv
    from app.short_horizon.expert_guided.data import calendar_from_closures
    from app.short_horizon.zigzag import causal_sigma,causal_range_sigma
    from tools.native_strategy_contract_study import synthetic_inputs
    from tools.systematic_strategy_matrix import action_from_head,check_costs
    from tools.recipe_sac_queue import clean
    require(mode in ('ideal','nop','inverted','position_long','position_short'),'REPAIR_OUTPUT_MODE')
    require(type(horizon) is int and 0<horizon<=72,'REPAIR_HORIZON')
    require(spread_factor in (1.,3.),'REPAIR_SPREAD_FACTOR')
    require(math.isfinite(minimum_gain_risk) and minimum_gain_risk>=0,'REPAIR_GAIN_RISK')
    bars=market_path(spec);ts=bars.index
    starts=np.flatnonzero(np.r_[True,np.diff(ts.asi8)>3600*10**9])
    closures=[[(ts[0]-pd.Timedelta(hours=48)).isoformat(),ts[0].isoformat()]]
    closures.extend([(ts[i-1]+pd.Timedelta(hours=1)).isoformat(),ts[i].isoformat()] for i in starts[1:])
    closures.append([(ts[-1]+pd.Timedelta(hours=1)).isoformat(),(ts[-1]+pd.Timedelta(hours=49)).isoformat()])
    cal=calendar_from_closures(closures,delay=12,flat=12,cutoff=0,calendar_id='SYNTHETIC_QUOTE_REPAIR')
    prof=config.profile
    prof=replace(prof,spread_schedule=tuple((a,b,v*spread_factor) for a,b,v in prof.spread_schedule),
                 weekly_open_spread=prof.weekly_open_spread*spread_factor)
    cfg=replace(config,profile=prof,execution_price_mode='bid_ask_v3')
    engine=BarEnv(bars,cal,causal_sigma(bars),causal_range_sigma(bars),ts[WARMUP],
                  ts[-1]+pd.Timedelta(hours=1),cfg)
    env=DirectionalOrders(engine,synthetic_inputs(ts,shape),DirectionalRisk(tuple(genes)))
    env.reset();decisions=[];probe_entry=False
    while not env.finished:
        cmd,expected=ideal_command(env,horizon,minimum_gain_risk)
        # Forced synthetic entry isolates position management from oracle entry selection.
        if mode in ('position_long','position_short') and not probe_entry:
            opening=1 if mode=='position_long' else 2
            if env.mask()[opening]:
                cmd=opening
                expected=dict(reason='FORCED_SYNTHETIC_POSITION_PROBE')
        if mode=='nop':cmd=0
        elif mode=='inverted':
            if engine.pos is None:cmd={0:0,1:2,2:1}[cmd]
            else:cmd=(3 if engine.pos.direction==1 else 4) if cmd==0 else 0
        scores=np.full(5,-1.);scores[cmd]=1.
        action=action_from_head(None,env._observe(),scores)
        require(int(action[0])==cmd,'REPAIR_HEAD_ROUTING')
        b=env.bar
        _,_,_,_,info=env.step(action)
        if mode in ('position_long','position_short') and info['engine_last_event'].get('entry'):
            probe_entry=True
        decisions.append(dict(bar=b,command=cmd,expected=expected,event=clean(info['engine_last_event'])))
    result=engine.result();balance=cfg.capital
    for t in engine.trades:
        check_trade(t);check_costs(t,engine)
        q=genes[2 if t['direction']==1 else 5]
        factor=cfg.sizing.leverage if cfg.sizing.basis=='margin' else 1.
        require(t['units']==math.floor(q*math.floor(balance*cfg.sizing.rel_volume*factor/t['entry_price'])),'REPAIR_SIZE')
        balance+=t['net_usd']
    require(abs(balance-engine.balance)<1e-7,'REPAIR_CASH')
    require(all(n<=cfg.max_entries_per_day for n in engine.entries_day.values()),'REPAIR_DAILY_CAP')
    return dict(scope='SYNTHETIC_ORACLE_ENGINEERING_ONLY',spec=spec,genes=list(genes),horizon=horizon,
        mode=mode,spread_factor=spread_factor,minimum_gain_risk=minimum_gain_risk,
        position_probe_entry=probe_entry if mode.startswith('position_') else None,
        oracle_version='synthetic_protection_reference.v1',decisions=decisions,trades=clean(engine.trades),
        metrics=clean({k:v for k,v in result.items() if k not in ('equity','daily_equity')}),
        financial_admission=False,learning_qualified=False)
