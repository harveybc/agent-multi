"""Fixed synthetic head tapes and native-ledger scoring for strategy calibration."""
from collections import Counter
import math

import numpy as np

from app.frozen_recipe_inputs import require
from app.rl_subset import digest

RUNTIME=None


def panel(split):
    require(split in ('calibration','confirmation'),'PANEL_SPLIT')
    patterns=((1,1),(-1,-1),(1,-1),(-1,1),(0,1),(0,-1),(1,0),(-1,0))
    confirmation=split=='confirmation'
    strengths=(.00008,.00022,.00038) if confirmation else (.00005,.00015,.00045)
    vols=(.00004,.00025,.00050) if confirmation else (.00002,.00015,.00060)
    periods=(12,36,60) if confirmation else (8,24,72)
    horizons=(2,8,18,36,60,70) if confirmation else (1,4,12,24,48,72)
    speeds=(0,6,10) if confirmation else (0,4,12)
    return [dict(drift_before=a*strength,drift_after=b*strength,
        vol_before=vols[(i+j)%3],vol_after=vols[(i+2*j+1)%3],
        transition_bars=speeds[(i+j)%3],period=periods[(i+2*j)%3],
        horizon=horizons[(3*i+j)%6],switch=32,phase=17 if confirmation else 0,
        target_bars=240,cost='configured')
        for i,(a,b) in enumerate(patterns) for j,strength in enumerate(strengths)]


def score(name,components,scales):
    keys={'sharpe':('sharpe',),'sharpe_return':('sharpe','return_fraction'),
          'sharpe_return_activity':('sharpe','return_fraction','orders_per_day')}
    require(name in keys,'OBJECTIVE_NAME')
    for k in keys[name]:
        require(type(components[k]) in (float,int) and math.isfinite(components[k]) and
                math.isfinite(scales[k]) and scales[k]>0,'FINITE_OBJECTIVE')
    return sum(components[k]/scales[k] for k in keys[name])


def finalists(searches):
    return [min([r for r in searches if r['objective']==name],
        key=lambda r:(-r['fitness'],r['report']['mean_drawdown_pct'],r['genes']))
        for name in ('sharpe','sharpe_return','sharpe_return_activity')]


def require_final_evidence(searches,confirmation,matrices):
    require(len(searches)==9 and len({r['id'] for r in searches})==9,'NINE_SEARCHES')
    require({r['id'] for r in searches}<={r['id'] for r in confirmation},'ALL_CONFIRMATIONS')
    expected={digest(r['genes']) for r in finalists(searches)}
    require(expected=={r['genome_sha256'] for r in matrices},'ALL_FINALIST_MATRICES')
    require(all(r['terminal'] and r['cells']==47628 for r in matrices),'FULL_FINALIST_MATRICES')


def initialize(recipe,engine):
    import torch
    from tools.systematic_strategy_matrix import setup
    global RUNTIME
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available(),'CPU_ONLY')
    r,c=setup(recipe,engine)
    RUNTIME=dict(recipe=r,config=c,templates={})


def output_tape(engine,horizon,mode):
    """Only synthetic future prices determine scores; no candidate gene is read."""
    import torch
    from agent_plugins.hybrid_orders import OrderDistribution
    n=len(engine.c);b=np.arange(n);k=np.minimum(b+1,n-1)
    end=np.searchsorted(engine.v,engine.ff[engine.seg_of],side='left')-1
    future=np.maximum(k,np.minimum(np.minimum(k+horizon-1,end),n-1))
    delta=engine.c[future]-engine.o[k]
    if mode=='inverted':delta=-delta
    cost=engine.cfg.profile.round_trip_price()
    tapes=[]
    for position,mask in ((0,[1,1,1,0,0]),(1,[1,0,0,1,0]),(-1,[1,0,0,0,1])):
        logits=np.full((n,5),-1e6,np.float32)
        logits[:,0]=0.
        if position==0:
            logits[:,1]=delta-cost;logits[:,2]=-delta-cost
        else:
            logits[:,0]=position*delta-cost*.5;logits[:,3 if position==1 else 4]=0.
        if mode=='nop':logits[:,0]=1e6
        masks=torch.tensor(np.broadcast_to(mask,(n,5)).copy(),dtype=torch.bool)
        dist=OrderDistribution(torch.from_numpy(logits),torch.zeros(n,2,6),masks,False)
        actions=dist.get_actions(True).numpy()
        require(np.array_equal(actions[:,0],np.argmax(np.where(masks.numpy(),logits,-np.inf),axis=1)),
                'TAPE_CATEGORICAL_PARITY')
        tapes.append(actions)
    return np.stack(tapes)


def episode(task):
    from app.directional_risk_calibration import DirectionalRisk
    from app.strategy_scenario_matrix import check_trade
    from tools.systematic_strategy_matrix import environment,clone,check_costs
    from tools.recipe_sac_queue import clean
    cell,genes,mode=task
    require(mode in ('ideal','inverted','nop'),'OUTPUT_MODE')
    key=digest(dict(cell=cell,mode=mode))
    if key not in RUNTIME['templates']:
        r=RUNTIME['recipe'];shape=(r['tensor']['window']['observations'],r['list']['n'])
        env=environment(cell,RUNTIME['config'],shape,parameters=DirectionalRisk((.5,.5,1)*2))
        tape=output_tape(env.engine,cell['horizon'],mode)
        RUNTIME['templates'][key]=(env,tape)
    template,tape=RUNTIME['templates'][key]
    env=clone(template);env.parameters=DirectionalRisk(tuple(genes))
    while not env.finished:
        direction=env.engine.pos.direction if env.engine.pos is not None else 0
        route=0 if direction==0 else 1 if direction==1 else 2
        action=tape[route,env.bar].copy()
        if not env.mask()[int(action[0])]:action[0]=0
        env.step(action)
    e=env.engine;m=e.result();capital=e.cfg.capital
    balance=capital
    for t in e.trades:
        check_trade(t);check_costs(t,e)
        q=genes[2 if t['direction']==1 else 5]
        factor=e.cfg.sizing.leverage if e.cfg.sizing.basis=='margin' else 1.
        expected=math.floor(q*math.floor(balance*e.cfg.sizing.rel_volume*factor/t['entry_price']))
        require(t['units']==expected and t['units']>=e.cfg.sizing.min_units,'GENOME_UNITS')
        balance+=t['net_usd']
    require(math.isclose(balance,e.balance,abs_tol=1e-7,rel_tol=1e-10),'GENOME_CASH')
    require(all(v<=e.cfg.max_entries_per_day for v in e.entries_day.values()),'GENOME_DAILY_CAP')
    daily=m['daily_equity'].to_numpy(float)
    require(np.isfinite(daily).all() and (daily>0).all(),'FINITE_POSITIVE_EQUITY')
    returns=np.diff(np.r_[capital,daily])/np.r_[capital,daily[:-1]]
    rr=[];breakeven=[];cost_pips=[]
    for t in e.trades:
        tp=abs(t['take_profit']-t['entry_price']);sl=abs(t['entry_price']-t['stop_loss'])
        fee=sum(t[k] for k in ('spread_usd','slippage_usd','commission_usd','financing_usd'))/t['units']
        rr.append(tp/sl);breakeven.append((sl+fee)/(tp+sl));cost_pips.append(fee/1e-4)
    return dict(cell=cell,genes=genes,mode=mode,tape_sha256=__import__('hashlib').sha256(tape.tobytes()).hexdigest(),
        daily_returns=returns.tolist(),net_profit_usd=e.balance-capital,return_fraction=(e.balance-capital)/capital,
        capital=capital,trades=clean(e.trades),orders=dict(env.order_counts),
        max_drawdown_pct=float(m['max_drawdown_pct']),ruined=bool(m['ruined']),
        weeks=cell['target_bars']/120,business_days=cell['target_bars']/24,
        effective_rr=rr,barrier_breakeven=breakeven,cost_pips=cost_pips)


def aggregate(rows):
    require(bool(rows),'NONEMPTY_PANEL')
    returns=np.asarray([v for r in rows for v in r['daily_returns']],float)
    require(np.isfinite(returns).all(),'FINITE_DAILY_RETURNS')
    std=float(np.std(returns,ddof=1)) if len(returns)>1 else 0.
    sharpe=float(np.mean(returns)/std*np.sqrt(365)) if std>0 else None
    trades=[t for r in rows for t in r['trades']]
    closes=Counter(t['close_reason'] for t in trades)
    sides={str(d):[t for t in trades if t['direction']==d] for d in (1,-1)}
    mean=lambda values:float(np.mean(values)) if len(values) else None
    components=dict(sharpe=sharpe,return_fraction=mean([r['return_fraction'] for r in rows]),
        orders_per_day=len(trades)/sum(r['business_days'] for r in rows))
    reasons=[]
    if sharpe is None:reasons.append('UNDEFINED_SHARPE')
    if len(trades)/sum(r['weeks'] for r in rows)<2:reasons.append('ACTIVITY_BELOW_TWO')
    for d in (1,-1):
        if not sides[str(d)]:reasons.append('NO_SIDE_'+str(d))
        if not any(t['close_reason']=='AGENT_CLOSE' for t in sides[str(d)]):reasons.append('NO_CLOSE_'+str(d))
    if any(r['ruined'] for r in rows):reasons.append('RUIN')
    return dict(components=components,eligible=not reasons,reasons=reasons,trades=len(trades),
        orders_per_week=len(trades)/sum(r['weeks'] for r in rows),
        mean_profit_usd=mean([r['net_profit_usd'] for r in rows]),
        mean_drawdown_pct=mean([r['max_drawdown_pct'] for r in rows]),
        worst_drawdown_pct=max(r['max_drawdown_pct'] for r in rows),
        win_rate=mean([t['net_usd']>0 for t in trades]),
        sides={d:dict(trades=len(ts),win_rate=mean([t['net_usd']>0 for t in ts])) for d,ts in sides.items()},
        close_reasons=dict(closes),
        costs={k:sum(t[k] for t in trades) for k in ('spread_usd','slippage_usd','commission_usd','financing_usd')},
        mean_effective_rr=mean([v for r in rows for v in r['effective_rr']]),
        mean_barrier_breakeven=mean([v for r in rows for v in r['barrier_breakeven']]),
        mean_cost_pips=mean([v for r in rows for v in r['cost_pips']]))
