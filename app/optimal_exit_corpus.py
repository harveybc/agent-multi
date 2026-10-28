"""Native-engine synthetic exit menus; each feasible close is actually executed."""
from dataclasses import replace
import itertools

import numpy as np

from app.frozen_recipe_inputs import require
from app.optimal_exit import targets


def specs(split):
    require(split in ('train','confirmation'),'EXIT_CORPUS_SPLIT')
    peaks=[4,8,12,20] if split=='train' else [6,10,16,24]
    gains=[.25,.5,.8] if split=='train' else [.35,.65]
    offsets=[-.25,0.,.25] if split=='train' else [-.15,.15]
    for i,(side,peak,gain,offset,spread) in enumerate(itertools.product(
            (1,-1),peaks,gains,offsets,(1.,3.))):
        yield dict(id=f'{split}_{i:04d}',split=split,direction=side,peak=peak,
                   gain=gain,offset=offset,spread=spread)


def causal_observation(e,b,shape):
    require(shape[0]==13 and shape[1]>=3,'EXIT_DIAGNOSTIC_SHAPE')
    ix=np.arange(b-24,b+1,2)
    require(ix.min()>=0,'EXIT_OBSERVATION_HISTORY')
    x=np.zeros(shape,np.float32)
    # Fixed physical scales. No order state, future samples or rolling z-score.
    x[:,0]=(e.c[ix]-1.1)/.001
    x[:,1]=(e.c[ix]-e.o[ix])/.001
    x[:,2]=(e.h[ix]-e.l[ix])/.001
    return x


def build_case(spec,cfg,shape):
    from tools.systematic_strategy_matrix import environment,clone,check_costs
    from app.multiscale_strategy_search import panel
    from app.directional_risk_calibration import DirectionalRisk
    from app.strategy_repair_episode import expected_pnl
    from app.native_optimal_exit import optimal_position_command
    from app.strategy_scenario_matrix import check_trade
    cell=panel('calibration')[0]
    cell.update(drift_before=0.,drift_after=0.,vol_before=0.,vol_after=0.)
    factor=spec['spread'];profile=cfg.profile
    cfg=replace(cfg,execution_price_mode='bid_ask_v3',profile=replace(profile,
        spread_schedule=tuple((a,b,v*factor) for a,b,v in profile.spread_schedule),
        weekly_open_spread=profile.weekly_open_spread*factor))
    env=environment(cell,cfg,shape,constant_sigma=True,parameters=DirectionalRisk((.5,.5,.2)*2))
    d=spec['direction'];opening=1 if d==1 else 2
    while not env.finished and not env.mask()[opening]:env.step(np.array([0,.5,.5,.5]))
    env.step(np.array([opening,.5,.5,.5]))
    e=env.engine;p=e.pos
    require(p is not None and not e.trades,'EXIT_SEEDED_POSITION')
    start=env.bar
    for name in ('o','h','l','c'):setattr(e,name,getattr(e,name).copy())
    profit_distance=abs(p.tp-p.entry_price);loss_distance=abs(p.sl-p.entry_price)
    amplitude=spec['gain']*profit_distance;offset=spec['offset']*loss_distance
    def price(t):
        return p.entry_price+d*(offset+amplitude*(1-((t-spec['peak'])/spec['peak'])**2))
    # Artificial initial marked position; still inside both native barriers.
    e.c[start]=price(0)
    e.h[start]=max(e.o[start],e.c[start])+1e-6
    e.l[start]=min(e.o[start],e.c[start])-1e-6
    for j in range(start+1,e.last+1):
        e.o[j]=price(j-start-1);e.c[j]=price(j-start)
        e.h[j]=max(e.o[j],e.c[j])+1e-6;e.l[j]=min(e.o[j],e.c[j])-1e-6
    risk=p.units*(loss_distance+profile.slippage_market+profile.slippage_stop+
                  2*profile.commission_per_lot_side/profile.lot_units)
    close=[];obs=[];bars=[];receipts=[];oracle_commands=[]
    while e.pos is not None and not env.finished:
        b=env.bar;k=b+1
        spread=e.cfg.profile.spread_at(int(e.v[k]),int(e.mo[e.seg_of[k]]))
        quote=e.o[k]-d*spread/2
        if d*(quote-p.sl)>0 and d*(quote-p.tp)<0:
            command,_=optimal_position_command(env)
            oracle_commands.append(command)
            trial=clone(env)
            trial.step(np.array([3 if d==1 else 4,.5,.5,.5]))
            require(len(trial.engine.trades)==1,'EXIT_SINGLE_ORDER')
            trade=trial.engine.trades[0]
            require(trade['close_reason']=='AGENT_CLOSE','EXIT_EXECUTABLE_CLOSE')
            expected,_=expected_pnl(e,d,p.units,p.entry_price,p.entry_bar,p.tp,p.sl,k,k,True)
            require(abs(expected-trade['net_usd'])<1e-7,'EXIT_INDEPENDENT_CLOSE_NET')
            check_trade(trade);check_costs(trade,trial.engine)
            close.append(float(trade['net_usd']));obs.append(causal_observation(e,b,shape));bars.append(b)
            receipts.append(dict(bar=b,exit_bar=trade['exit_bar_abs'],net=trade['net_usd'],
                                 reason=trade['close_reason'],direction=d))
        env.step(np.array([0,.5,.5,.5]))
    require(len(e.trades)==1 and len(close)>1,'EXIT_TERMINAL_AND_MENU')
    terminal=e.trades[0];check_trade(terminal);check_costs(terminal,e)
    t=targets(close,terminal['net_usd'],risk)
    require(np.array_equal(np.array(oracle_commands),np.where(t['oracle_close'],3 if d==1 else 4,0)),
            'NATIVE_ORACLE_MATCHES_ALL_EXECUTED_COUNTERFACTUALS')
    return dict(**spec,close_net=close,terminal_net=float(terminal['net_usd']),
        terminal_reason=terminal['close_reason'],risk_usd=float(risk),bars=bars,
        best_net=t['best_net'],oracle_has_early_exit=bool(t['oracle_close'].any()),
        receipts=receipts,scope='SYNTHETIC_EXIT_QUALIFICATION_ONLY',
        observation_semantics='three causal price channels, remaining channels zero; not financial features',
        financial_admission=False),np.stack(obs)
