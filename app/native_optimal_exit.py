"""Synthetic full-position optimal stopping oracle, independent of native step."""
import numpy as np

from app.frozen_recipe_inputs import require
from app.optimal_exit import targets


def optimal_position_command(env):
    from app.strategy_repair_episode import expected_pnl
    e=env.engine;p=e.pos;b=env.bar;k=b+1
    require(p is not None,'OPTIMAL_EXIT_POSITION')
    key=(p.entry_bar,p.direction,p.entry_price,p.tp,p.sl,p.units)
    cache=getattr(env,'_synthetic_optimal_exit_cache',None)
    if cache is None or cache['key']!=key:
        end=min(e.last,int(np.searchsorted(e.v,e.ff[e.seg_of[k]],side='left'))-1)
        require(end>=k,'OPTIMAL_EXIT_BEFORE_DEADLINE')
        passive,exit_=expected_pnl(e,p.direction,p.units,p.entry_price,p.entry_bar,p.tp,p.sl,k,end)
        bars=[];net=[]
        for j in range(k,exit_['exit_bar']+1):
            spread=e.cfg.profile.spread_at(int(e.v[j]),int(e.mo[e.seg_of[j]]))
            quote=e.o[j]-p.direction*spread/2
            if p.direction*(quote-p.sl)<=0 or p.direction*(quote-p.tp)>=0:break
            value,_=expected_pnl(e,p.direction,p.units,p.entry_price,p.entry_bar,p.tp,p.sl,j,j,True)
            bars.append(j);net.append(value)
        risk=p.units*(abs(p.entry_price-p.sl)+e.cfg.profile.slippage_market+
                      e.cfg.profile.slippage_stop+2*e.cfg.profile.commission_per_lot_side/e.cfg.profile.lot_units)
        cache=dict(key=key,bars=bars,net=net,passive=passive,exit=exit_,risk=risk,
                   target=targets(net,passive,risk) if net else None)
        env._synthetic_optimal_exit_cache=cache
    if k not in cache['bars']:
        return 0,dict(reason='BROKER_OPEN_PROTECTION_PRIORITY')
    i=cache['bars'].index(k);t=cache['target']
    close=float(t['close_net'][i]);hold=float(t['hold_value'][i])
    cmd=(3 if p.direction==1 else 4) if close>=hold else 0
    return cmd,dict(close_net=close,hold_net=hold,reference=cache['exit'],
        comparator='BEST_FEASIBLE_FUTURE_CLOSE_OR_PASSIVE_EXIT',tie_policy='EARLIEST_CLOSE',
        passive_net=cache['passive'],global_best_net=t['best_net'],
        close_regret_risk=float(t['regret_close'][i]))
