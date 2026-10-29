"""Native synthetic controller harness. Oracle reports stay in this test plane."""
from dataclasses import asdict
import hashlib
import json

import numpy as np

from app.absolute_exit_corpus import create_case, _funding
from app.close_hold_controller import PositionView, PathReport, CloseConfig, CloseController
from app.close_hold_reports import prefix_report
from app.early_close_metrics import evaluate_exit

HOUR=3600*10**9


def arms():
    immediate=asdict(CloseConfig())
    guarded=asdict(CloseConfig(uncertainty_penalty=1.,margin_risk=.02,persistence=2,urgent_risk=.25))
    return [dict(name='ideal_immediate',report='ideal',config=immediate),
            dict(name='ideal_guarded',report='ideal',config=guarded),
            dict(name='noisy_immediate',report='noisy',config=immediate),
            dict(name='noisy_guarded',report='noisy',config=guarded),
            dict(name='prefix_guarded',report='prefix',config=guarded)]


def position_view(e,risk):
    p=e.pos; f=e.cfg.profile
    identity=hashlib.sha256(json.dumps([int(e.v[p.entry_bar]),p.direction,p.entry_price,p.units,p.sl,p.tp]).encode()).hexdigest()
    return PositionView(identity,p.direction,p.entry_price,p.units,p.sl,p.tp,risk,
                        f.slippage_market,f.slippage_market,f.slippage_stop,
                        2*f.commission_per_lot_side/f.lot_units,f.limit_through)


def build_report(e,kind,seed):
    b=e.next_decision; first=int(e.v[b])+HOUR; deadline=int(e.ff[e.pos.seg])
    terminal=e.cfg.profile.spread_at(deadline,int(e.mo[e.pos.seg]))
    if first>=deadline:
        return None
    if kind=='prefix':
        # Only the completed prefix crosses the reporter boundary. Future quote
        # observations, labels and position state are not arguments.
        prefix=np.column_stack([v[max(0,b-7):b+1] for v in (e.o,e.h,e.l,e.c)])
        return prefix_report(prefix,first,HOUR,deadline,e.cfg.profile.spread_at(int(e.v[b]),int(e.mo[e.pos.seg])),
                             terminal_spread=terminal)
    end=int(np.searchsorted(e.v,deadline))
    future=np.column_stack([v[b+1:end] for v in (e.o,e.h,e.l,e.c)])
    spread=np.array([e.cfg.profile.spread_at(int(t),int(e.mo[e.pos.seg])) for t in e.v[b+1:end]])
    if kind=='ideal':
        paths=future[None,:,:]; spreads=spread[None,:]; weights=np.ones(1)
    elif kind=='noisy':
        # Antithetic seeded errors preserve the mean curve, not barrier outcomes.
        rng=np.random.default_rng(np.random.SeedSequence([seed,b,2301]))
        errors=np.cumsum(rng.normal(0,.0001,len(future)))
        paths=future[None,:,:]+np.array([-1.,0.,1.])[:,None,None]*errors[None,:,None]
        spreads=np.repeat(spread[None,:],3,axis=0); weights=np.full(3,1/3)
    else:
        raise ValueError('REPORT_KIND')
    return PathReport(first,first,first,HOUR,deadline,paths,spreads,weights,
                      'synthetic_ideal' if kind=='ideal' else 'synthetic_noisy',np.full(len(paths),terminal))


def run_case(spec,config,reference,arm):
    e,risk=create_case(spec,config)
    controller=CloseController(CloseConfig(**arm['config']),allow_synthetic=arm['report']!='prefix')
    p=position_view(e,risk); trace=[]
    while e.pos is not None:
        report=build_report(e,arm['report'],spec['seed'])
        b=e.next_decision
        if report is None:
            result=dict(action='HOLD',reason='MANDATORY_DEADLINE')
        else:
            times=report.first_fill+np.arange(report.paths.shape[1])*report.step
            funding_open=np.array([_funding(e,e.pos,int(t))/e.pos.units for t in times])
            funding_end=np.array([_funding(e,e.pos,int(t)+HOUR)/e.pos.units for t in times])
            result=controller.decide(p,report,int(e.v[b])+HOUR,funding_open,funding_end,
                                     expected_deadline=int(e.ff[e.pos.seg]))
        native=e.step(b,3 if result['action']=='CLOSE' else 0)
        trace.append(dict(decision_bar=b,**result,native_close=native['close']))
    trade=e.trades[0]
    ix=reference['fill_bars'].index(trade['exit_bar_abs']) if trade['close_reason']=='AGENT_CLOSE' else None
    metrics=evaluate_exit([v['close_net'] for v in reference['tick_metrics']],
                          reference['native_passive_net'],risk,ix,actual_net=trade['net_usd'])
    if arm['name']=='ideal_immediate' and metrics['regret_risk']>1e-10:
        raise AssertionError(('IDEAL_REPORT_CONTROLLER_MISSES_ORACLE',spec['id'],metrics))
    return dict(spec=spec,arm=arm['name'],trace=trace,metrics=metrics,
                native_reason=trade['close_reason'],exit_bar=trade['exit_bar_abs'])


def prefix_future_witness(spec,config):
    import copy
    e,risk=create_case(spec,config)
    a=build_report(e,'prefix',0)
    altered=copy.deepcopy(e)
    # Deliberately change every unrealized OHLC value; the observed prefix stays.
    for name in ('o','h','l','c'):
        values=getattr(altered,name).copy(); values[e.next_decision+1:]*=1.1
        setattr(altered,name,values)
    b=build_report(altered,'prefix',999)
    if not np.array_equal(a.paths,b.paths) or not np.array_equal(a.spreads,b.spreads):
        raise AssertionError('PREFIX_FUTURE_LEAK')
    p=position_view(e,risk)
    times=a.first_fill+np.arange(a.paths.shape[1])*HOUR
    fo=np.array([_funding(e,e.pos,int(t))/e.pos.units for t in times])
    fe=np.array([_funding(e,e.pos,int(t)+HOUR)/e.pos.units for t in times])
    clock=int(e.v[e.next_decision])+HOUR; deadline=int(e.ff[e.pos.seg])
    ca=CloseController(CloseConfig()).decide(p,a,clock,fo,fe,expected_deadline=deadline)
    cb=CloseController(CloseConfig()).decide(p,b,clock,fo,fe,expected_deadline=deadline)
    if ca!=cb:
        raise AssertionError('DECISION_FUTURE_LEAK')
    return dict(case=spec['id'],report_equal=True,decision_equal=True)
