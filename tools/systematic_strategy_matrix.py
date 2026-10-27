"""CPU-only finite market/state/action contract matrix; no fitting or financial release."""
import argparse
import copy
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np
import pandas as pd
import torch

from app.frozen_recipe_inputs import file_sha, require
from app.rl_subset import digest
from app.strategy_scenario_matrix import cells, market_frame, check_trade, WARMUP
from tools.recipe_sac_queue import read, write, now, source_pins
from tools.native_strategy_contract_study import synthetic_inputs, witnesses


ROOT=Path(__file__).resolve().parents[1]
WORKER=None


def initialize_worker(recipe_path,engine_root,parameters):
    global WORKER
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available(),'CPU_ONLY_WORKER')
    recipe,config=setup(recipe_path,engine_root)
    WORKER=(recipe,config,parameters)


def execute_worker(cell):
    recipe,config,parameters=WORKER
    try:
        return dict(state='PASS',**run_cell(cell,config,recipe,None,parameters=parameters))
    except Exception as error:
        return dict(state='FAIL',cell=cell,error_type=type(error).__name__,error=str(error))


def setup(recipe_path,engine_root):
    from app.recipe_runtime import attach_engine
    attach_engine(engine_root)
    from app.short_horizon.expert_guided.env import EnvConfig
    from app.short_horizon import execution as X
    recipe=read(recipe_path)
    cfg=copy.deepcopy(recipe['business']['evaluation_and_policy'])
    cfg['sizing']=X.Sizing(**cfg['sizing']);cfg['profile']=X.RealisticProfile(**cfg['profile'])
    cfg['rules']=X.OrderRules(**cfg['rules'])
    return recipe,EnvConfig(**cfg)


def environment(cell,config,shape,constant_sigma=False,parameters=None):
    from app.short_horizon.expert_guided.env import BarEnv
    from app.short_horizon.expert_guided.data import calendar_from_closures
    from app.short_horizon.zigzag import causal_sigma,causal_range_sigma
    from env_plugins.hybrid_orders import HybridOrderEnv
    bars=market_frame(cell);ts=bars.index
    cal=calendar_from_closures([[str(pd.Timestamp('2020-01-03T22:00Z')+pd.Timedelta(weeks=w)),
        str(pd.Timestamp('2020-01-05T22:00Z')+pd.Timedelta(weeks=w))] for w in range(-5,cell.get('target_bars',120)//120+1)],delay=12,flat=12,cutoff=0,
        calendar_id='SYSTEMATIC_SYNTHETIC')
    sigma=np.full(len(ts),.001) if constant_sigma else causal_sigma(bars)
    rule=np.full(len(ts),.002) if constant_sigma else causal_range_sigma(bars)
    require(np.isfinite(sigma[WARMUP:]).all() and np.isfinite(rule[WARMUP:]).all(),'SYNTHETIC_WARMUP')
    engine=BarEnv(bars,cal,sigma,rule,ts[WARMUP],ts[-1]+pd.Timedelta(hours=1),
                  replace(config,cost_mode=cell['cost']))
    inputs=synthetic_inputs(ts,shape)
    if parameters is None:
        env=HybridOrderEnv(engine,inputs,learned_risk=False)
    else:
        from app.directional_risk_calibration import DirectionalOrders
        env=DirectionalOrders(engine,inputs,parameters)
    env.reset();return env


def step(env,command):
    return env.step(np.array([command,.5,.5,.5],np.float32))


def action_from_head(model,obs,values,scale=1.):
    # Only this isolated diagnostic model is mutated. Encoder and output codec
    # remain the actual production classes; values are synthetic head scores.
    if model is None:
        from agent_plugins.hybrid_orders import OrderDistribution
        dist=OrderDistribution(torch.as_tensor(np.asarray(values)[None,:],dtype=torch.float32)*scale,
            torch.zeros(1,2,6),torch.tensor(obs['action_mask'][None,:],dtype=torch.bool),False)
        action=dist.get_actions(True)[0].numpy()
    else:
        with torch.no_grad():
            model.policy.commands.weight.zero_()
            model.policy.commands.bias.copy_(torch.tensor(values,dtype=torch.float32)*scale)
        action=model.predict(obs,deterministic=True)[0]
    expected=int(np.argmax(np.where(obs['action_mask'].astype(bool),values,-np.inf)))
    require(int(action[0])==expected,'HEAD_ARGMAX')
    return action


def clone(source):
    # Synthetic inputs are immutable. Never duplicate the large zero tensor for
    # each counterfactual; mutable engine, masks and accounting remain isolated.
    return copy.deepcopy(source,{id(source.inputs):source.inputs})


def check_costs(trade,engine):
    """Independent event-price accounting, without engine cost helper calls."""
    t=trade;p=engine.cfg.profile;u=t['units'];hour=3600*10**9
    opened=int(engine.mo[t['segment']])
    def spread(when):
        elapsed=(when-opened)/hour
        if elapsed<1:return p.weekly_open_spread
        matches=[v for a,b,v in p.spread_schedule if a<=elapsed%24<b]
        require(len(matches)==1,'INDEPENDENT_SPREAD_SCHEDULE')
        return matches[0]
    entry=int(t['entry_time'].value);exit_=int(t['exit_time'].value)
    at_open=t['exit_clock'] in ('OPEN','INTRABAR_END_PROXY_COST_AT_OPEN')
    cost_clock=int(engine.v[t['exit_bar_abs']]) if at_open else exit_
    costs=dict(spread_usd=u*(spread(entry)+spread(cost_clock))/2,
        commission_usd=2*u/p.lot_units*p.commission_per_lot_side,
        slippage_usd=u*(p.slippage_market+(0 if t['close_reason']=='TP' else
            p.slippage_stop if t['close_reason'] in ('SL','MARGIN_CLOSEOUT') else p.slippage_market)))
    rate=p.swap_long_per_lot if t['direction']==1 else p.swap_short_per_lot
    costs['financing_usd']=sum(-rate*u/p.lot_units*(3 if i==p.triple_swap_rollover_index else 1)
        for i in range(1,5) if entry<opened+i*24*hour<=exit_)
    for name,value in costs.items():
        require(math.isclose(t[name],value,rel_tol=1e-10,abs_tol=1e-8),'INDEPENDENT_'+name)


def outcome(source,command,horizon=4,holding_clock=False):
    env=clone(source);start=env.bar
    step(env,command)
    end=start+horizon+(1 if holding_clock else 0)
    while not env.finished and env.bar<end:
        close=3 if env.engine.pos and env.engine.pos.direction>0 else 4
        cmd=close if env.engine.pos is not None and env.bar==end-1 else 0
        step(env,cmd)
    require(env.engine.pos is None,'BOUNDED_HORIZON_FLAT')
    for t in env.engine.trades:
        check_trade(t)
        if holding_clock:check_costs(t,env.engine)
        require(t['entry_bar_abs']<=t['exit_bar_abs'],'TRADE_TIME')
        require(t['entry_price']==env.engine.o[t['entry_bar_abs']],'NEXT_OPEN_PRICE')
        require(t['entry_time'].value>=env.engine.op[t['segment']],'ENTRY_DELAY')
        require(t['exit_time'].value<=env.engine.ff[t['segment']],'FORCED_FLAT_DEADLINE')
    net=sum(t['net_usd'] for t in env.engine.trades)
    require(np.isclose(env.engine.balance,env.engine.cfg.capital+net,rtol=1e-10,atol=1e-8),'BALANCE')
    require(all(n<=env.engine.cfg.max_entries_per_day for n in env.engine.entries_day.values()),'DAILY_CAP')
    return float(env.engine.balance-source.engine.balance),env


def compare(source,model,commands,label,horizon=4,holding_clock=False):
    values={};results={}
    for command in commands:
        values[command],results[command]=outcome(source,command,horizon,holding_clock)
    logits=np.full(5,-1e5,dtype=float)
    # Positive affine transform: both BUY and SELL can be positive while the
    # better one must still win. NOP is index zero and wins exact flat ties.
    low=min(values.values());span=max(max(values.values())-low,1.)
    for command,value in values.items():logits[command]=1+(value-low)/span
    obs=source._observe();chosen=None
    for scale in (1e-4,1.,10.):
        action=action_from_head(model,obs,logits,scale)
        selected=int(action[0])
        require(values[selected]>=max(values.values())-1e-6,'LOWER_VALUE_CHOSEN')
        if chosen is not None:require(chosen==selected,'SCORE_SCALE_CHANGED_ACTION')
        chosen=selected
    # Execute the real policy output, not just a separately decoded integer.
    execution=clone(source);execution.step(action)
    expected=results[chosen]
    for t in execution.engine.trades:check_trade(t)
    return dict(state=label,bar=source.bar,horizon=horizon,values={str(k):v for k,v in values.items()},
        chosen=chosen,score_scales=[1e-4,1.,10.],
        winning_trades=[dict(direction=t['direction'],reason=t['close_reason'],net=t['net_usd'])
                        for t in expected.engine.trades])


def prescribed_head_checks():
    fixtures=[([0,1,2,9,9],[1,1,1,0,0],2),([0,2,1,9,9],[1,1,1,0,0],1),
              ([2,1,1,9,9],[1,1,1,0,0],0),([1,1,1,9,9],[1,1,1,0,0],0),
              ([0,9,9,2,9],[1,0,0,1,0],3),([0,9,9,9,2],[1,0,0,0,1],4),
              ([2,9,9,1,9],[1,0,0,1,0],0)]
    for scores,mask,expected in fixtures:
        for scale in (1e-4,1.,10.):
            actual=action_from_head(None,{'action_mask':np.array(mask,np.float32)},scores,scale)
            require(int(actual[0])==expected,'PRESCRIBED_HEAD_COMMAND')
    return len(fixtures)*3


def prescribed_execution(source,commands,parameters=None):
    """Literal commands and independently calculated protection/size expectations."""
    checks=[]
    for command in commands:
        env=clone(source);e=env.engine;b=env.bar
        scores=np.ones(5);scores[command]=2
        action=action_from_head(None,env._observe(),scores)
        require(int(action[0])==command,'PRESCRIBED_EXECUTION_COMMAND')
        if command in (1,2):
            side=1 if command==1 else -1
            sigma=float(e.sigma[b+1]);sr=float(e.sigma_rule[b+1]);cfg=e.cfg
            lo=cfg.rules.min_distance*(1+2e-9)/sigma
            hi=cfg.rules.effectiveness_c*sr*(1-2e-9)/sigma
            ft,fs,q=(.5,.5,1.) if parameters is None else parameters.for_command(command)
            bounds=[(max(a,lo),min(z,hi)) for a,z in (cfg.tp_bounds,cfg.sl_bounds)]
            multiples=[(a+z)/2 if f==.5 else a+f*(z-a) for (a,z),f in zip(bounds,(ft,fs))]
            tp=e.c[b]+side*multiples[0]*sigma;sl=e.c[b]-side*multiples[1]*sigma
            factor=cfg.sizing.leverage if cfg.sizing.basis=='margin' else 1.
            capacity=math.floor(e.balance*cfg.sizing.rel_volume*factor/e.o[b+1])
            units=math.floor(q*capacity)
        before=e.pos;previous_trades=len(e.trades)
        _,_,_,_,info=env.step(action);event=info['engine_last_event']
        if command in (1,2):
            if q==0:
                require(not event['entry'] and e.pos is None,'ZERO_ALLOCATION_NOP')
            elif units<cfg.sizing.min_units:
                require(event['refused']=='UNITS' and not event['entry'],'NO_MINIMUM_RISK_INCREASE')
            else:
                require(event['entry'] and event['refused'] is None,'PRESCRIBED_ENTRY')
                p=e.pos
                t=e.trades[-1] if p is None else None
                actual=(p.tp,p.sl,p.units,p.direction,p.entry_bar) if p is not None else (
                    t['take_profit'],t['stop_loss'],t['units'],t['direction'],t['entry_bar_abs'])
                require(np.allclose(actual[:2],(tp,sl),rtol=1e-12,atol=1e-12),'PRESCRIBED_LEVELS')
                require(actual[2:]==(units,side,b+1),'PRESCRIBED_SIZE_SIDE_FILL')
        elif command in (3,4):
            require(before is not None and e.pos is None and len(e.trades)==previous_trades+1,'PRESCRIBED_CLOSE')
            t=e.trades[-1]
            require(t['close_reason'] in ('AGENT_CLOSE','SL','TP','MARGIN_CLOSEOUT'),'CLOSE_PRIORITY')
            if t['close_reason']=='AGENT_CLOSE':
                require(t['exit_bar_abs']==b+1 and t['exit_price']==e.o[b+1],'EARLY_CLOSE_NEXT_OPEN')
        elif before is None:
            require(e.pos is None and len(e.trades)==previous_trades,'PRESCRIBED_NOP')
        for t in e.trades:
            check_trade(t);check_costs(t,e)
        checks.append(dict(command=command,event=event))
    return checks


def run_cell(cell,config,recipe,model,constant_sigma=False,parameters=None):
    shape=(recipe['tensor']['window']['observations'],recipe['list']['n'])
    results=[];absent=[];prescribed=[]
    horizon=cell.get('horizon',4);extended='horizon' in cell
    for anchor in (WARMUP+cell['switch']-2,WARMUP+cell['switch']+cell.get('transition_bars',0)+2):
        source=environment(cell,config,shape,constant_sigma,parameters)
        while source.bar<anchor:step(source,0)
        mask=source.mask()
        require(mask[1]==mask[2],'ASYMMETRIC_ENTRY_MASK')
        if not mask[1]:
            absent.append(dict(bar=anchor,state='FLAT_OPEN',reason='ENTRY_UNAVAILABLE',
                               entry_gate=source.engine.entry_gate(anchor),bounds=source.feasible_bounds()))
            results.append(compare(source,model,[0],'FLAT_NOP_ONLY',horizon,extended));continue
        if extended:prescribed.extend(prescribed_execution(source,[0,1,2],parameters))
        results.append(compare(source,model,[0,1,2],'FLAT',horizon,extended))
        for command,label,close in ((1,'LONG',3),(2,'SHORT',4)):
            positioned=clone(source);_,_,_,_,event=step(positioned,command)
            if positioned.engine.pos is None:
                absent.append(dict(bar=anchor,state=label,reason='PROTECTION_OR_ENTRY_REFUSAL',
                                   last_event=event['engine_last_event']));continue
            require(positioned.engine.pos.direction==(1 if command==1 else -1),'SIDE_MAPPING')
            p=positioned.engine.pos;cfg=positioned.engine.cfg
            cap=np.floor(positioned.engine.balance*cfg.sizing.rel_volume*cfg.sizing.leverage/p.entry_price)
            require(0<p.units<=cap,'ORDER_VOLUME_CAP')
            require(p.entry_bar==anchor+1,'NEXT_BAR_EXECUTION')
            if extended:prescribed.extend(prescribed_execution(positioned,[0,close],parameters))
            results.append(compare(positioned,model,[0,close],label,horizon,extended))
    increments=source.engine.c-source.engine.o
    switch=WARMUP+cell['switch'];transition=cell.get('transition_bars',0)
    return dict(cell=cell,comparisons=results,unreachable=absent,prescribed=prescribed,
        measured_increment_std=dict(before=float(np.std(increments[WARMUP:switch])),
            after=float(np.std(increments[switch+transition:]))))


def run(a):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available(),'CPU_ONLY_ENV_REQUIRED')
    parent=read(a.source_spec)
    require(file_sha(a.recipe)==parent['loading']['recipe_sha256'],'PARENT_RECIPE_PIN')
    for name,sha in parent['code']['engine'].items():
        require(file_sha(a.engine_root/name)==sha,'ENGINE_SOURCE_PIN:'+name)
    recipe,config=setup(a.recipe,a.engine_root)
    parameters=None
    if a.strategy_config is not None:
        from app.directional_risk_calibration import DirectionalRisk
        parameters=DirectionalRisk.from_json(a.strategy_config)
    design=read(a.design) if a.design else None
    require(type(a.workers) is int and 1<=a.workers<=8 and (design is not None or a.workers==1),'CPU_WORKER_COUNT')
    matrix=list(cells(design));expected_cells=len(matrix)
    source=dict(agent=source_pins(ROOT,('app','agent_plugins','env_plugins','tools')),
                engine=source_pins(a.engine_root,('app',)))
    identity=dict(schema='systematic_strategy_matrix.v1',recipe_sha256=file_sha(a.recipe),
        sources=source,expected_cells=expected_cells,limit=a.limit,financial_admission=False,
        design=design,design_sha256=file_sha(a.design) if a.design else None,
        cpu_workers=a.workers,
        head_scope='production_distribution_only' if design else 'full_injected_policy',
        actual_training_steps=0,oracle=('prescribed command/levels/size/cost oracles plus separately labelled simulator ranking at each horizon'
            if design else 'four-bar finite alternatives, diagnostic lookahead'),
        strategy_config_sha256=file_sha(a.strategy_config) if a.strategy_config else None,
        strategy_genes=list(parameters.genes) if parameters else None)
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.output/'IDENTITY.json').exists():require(read(a.output/'IDENTITY.json')==identity,'MATRIX_IDENTITY')
    else:write(a.output/'IDENTITY.json',identity)
    if (a.output/'CLOSE.json').exists():
        close=read(a.output/'CLOSE.json')
        for name,sha in close['artifacts'].items():
            path=(a.output/name).resolve()
            require(path.is_relative_to(a.output.resolve()) and file_sha(path)==sha,'MATRIX_CLOSE_HASH')
        print(json.dumps(close['summary']),flush=True)
        if close['summary']['state']=='FAIL':raise SystemExit(2)
        return
    from tools.rl_subset_bridge import build_model
    shape=(recipe['tensor']['window']['observations'],recipe['list']['n'])
    model=build_model(environment(next(cells()),config,shape),recipe,7,'cpu',
                       dict(n_steps=8,batch_size=8,n_epochs=1))
    monotonic=[]
    for side in (-1,1):
        cell=dict(before=side,after=side,vol_before=0.,vol_after=0.,switch=60,period=24,phase=0,cost='zero_cost')
        result=run_cell(cell,config,recipe,model,constant_sigma=True,parameters=parameters)
        flats=[r for r in result['comparisons'] if r['state']=='FLAT']
        require(len(flats)==2 and all(r['chosen']==(1 if side==1 else 2) for r in flats),'MONOTONIC_DIRECTION')
        require(all(r['values'][str(1 if side==1 else 2)]>0 and
                    r['values'][str(2 if side==1 else 1)]<0 for r in flats),'MONOTONIC_PROFIT_SIGN')
        monotonic.append(result)
    write(a.output/'MONOTONIC.json',monotonic)
    factory=None
    if parameters is not None:
        from app.directional_risk_calibration import DirectionalOrders
        factory=lambda engine,inputs:DirectionalOrders(engine,inputs,parameters)
    write(a.output/'BROKER_EVENTS.json',witnesses(config,shape,factory))
    write(a.output/'PRESCRIBED_HEADS.json',dict(checks=prescribed_head_checks(),failures=0))
    rows=[];started=time.monotonic()
    if a.limit is not None:
        require(0<a.limit<=len(matrix),'SMOKE_LIMIT')
        matrix=[matrix[i] for i in np.linspace(0,len(matrix)-1,a.limit,dtype=int)]
    pool=None;pending=None
    if a.workers>1:
        unfinished=[c for c in matrix if not (a.output/'cells'/(digest(c)+'.json')).exists()]
        pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(a.recipe,a.engine_root,parameters))
        pending=iter(pool.map(execute_worker,unfinished,chunksize=8))
    for i,cell in enumerate(matrix):
        path=a.output/'cells'/(digest(cell)+'.json')
        if path.exists():
            r=read(path);require(r['identity_sha256']==digest(identity) and r['cell']==cell,'CELL_IDENTITY')
        else:
            if pending is not None:
                r=next(pending);require(r['cell']==cell,'WORKER_CELL_ORDER')
            else:
                try:
                    r=dict(state='PASS',**run_cell(cell,config,recipe,None if design else model,parameters=parameters))
                except Exception as error:
                    r=dict(state='FAIL',cell=cell,error_type=type(error).__name__,error=str(error))
            r.update(identity_sha256=digest(identity));write(path,r)
        rows.append(r)
        if i%25==0:
            write(a.output/'STATUS.json',dict(state='RUNNING',complete=len(rows),total=a.limit or expected_cells,
                failed=sum(x['state']=='FAIL' for x in rows),seconds=time.monotonic()-started,at=now()))
    if pool is not None:pool.shutdown(wait=True)
    failures=[r for r in rows if r['state']=='FAIL']
    counts={s:sum(x['state']==s for r in rows for x in r.get('comparisons',[])) for s in ('FLAT','FLAT_NOP_ONLY','LONG','SHORT')}
    wins={str(cmd):sum(x['chosen']==cmd for r in rows for x in r.get('comparisons',[])) for cmd in range(5)}
    full=len(rows)==expected_cells and a.limit is None
    coverage=all(counts[s]>0 for s in ('FLAT','LONG','SHORT')) and all(wins[str(cmd)]>0 for cmd in range(5))
    state='PASS_DECLARED_GRID' if full and not failures and coverage else 'SMOKE_ONLY' if not full and not failures else 'FAIL'
    summary=dict(state=state,cells=len(rows),failures=len(failures),comparisons=counts,chosen=wins,
        prescribed_checks=sum(len(r.get('prescribed',[])) for r in rows),
        axes=design['axes'] if design else None,
        unreachable=sum(len(r.get('unreachable',[])) for r in rows),seconds=time.monotonic()-started,
        financial_admission=False,training_steps=0,finished_at=now())
    write(a.output/'SUMMARY.json',summary);write(a.output/'FAILURES.json',failures)
    write(a.output/'CLOSE.json',dict(summary=summary,artifacts={str(p.relative_to(a.output)):file_sha(p)
        for p in a.output.rglob('*') if p.is_file() and p.name not in ('CLOSE.json','STATUS.json','RUN.lock')}))
    write(a.output/'STATUS.json',summary);print(json.dumps(summary),flush=True)
    if state=='FAIL':raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--source-spec',type=Path,required=True)
    p.add_argument('--strategy-config',type=Path)
    p.add_argument('--design',type=Path)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int)
    run(p.parse_args())
