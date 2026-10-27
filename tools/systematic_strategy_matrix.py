"""CPU-only finite market/state/action contract matrix; no fitting or financial release."""
import argparse
import copy
from dataclasses import replace
import fcntl
import json
from pathlib import Path
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


def environment(cell,config,shape,constant_sigma=False):
    from app.short_horizon.expert_guided.env import BarEnv
    from app.short_horizon.expert_guided.data import calendar_from_closures
    from app.short_horizon.zigzag import causal_sigma,causal_range_sigma
    from env_plugins.hybrid_orders import HybridOrderEnv
    bars=market_frame(cell);ts=bars.index
    cal=calendar_from_closures([[str(pd.Timestamp('2020-01-03T22:00Z')+pd.Timedelta(weeks=w)),
        str(pd.Timestamp('2020-01-05T22:00Z')+pd.Timedelta(weeks=w))] for w in range(-5,2)],delay=12,flat=12,cutoff=0,
        calendar_id='SYSTEMATIC_SYNTHETIC')
    sigma=np.full(len(ts),.001) if constant_sigma else causal_sigma(bars)
    rule=np.full(len(ts),.002) if constant_sigma else causal_range_sigma(bars)
    require(np.isfinite(sigma[WARMUP:]).all() and np.isfinite(rule[WARMUP:]).all(),'SYNTHETIC_WARMUP')
    engine=BarEnv(bars,cal,sigma,rule,ts[WARMUP],ts[-1]+pd.Timedelta(hours=1),
                  replace(config,cost_mode=cell['cost']))
    env=HybridOrderEnv(engine,synthetic_inputs(ts,shape),learned_risk=False)
    env.reset();return env


def step(env,command):
    return env.step(np.array([command,.5,.5,.5],np.float32))


def action_from_head(model,obs,values,scale=1.):
    # Only this isolated diagnostic model is mutated. Encoder and output codec
    # remain the actual production classes; values are synthetic head scores.
    with torch.no_grad():
        model.policy.commands.weight.zero_()
        model.policy.commands.bias.copy_(torch.tensor(values,dtype=torch.float32)*scale)
    action=model.predict(obs,deterministic=True)[0]
    expected=int(np.argmax(np.where(obs['action_mask'].astype(bool),values,-np.inf)))
    require(int(action[0])==expected,'HEAD_ARGMAX')
    return action


def outcome(source,command,horizon=4):
    env=copy.deepcopy(source);start=env.bar
    step(env,command)
    while not env.finished and env.bar<start+horizon:
        close=3 if env.engine.pos and env.engine.pos.direction>0 else 4
        cmd=close if env.engine.pos is not None and env.bar==start+horizon-1 else 0
        step(env,cmd)
    require(env.engine.pos is None,'BOUNDED_HORIZON_FLAT')
    for t in env.engine.trades:
        check_trade(t)
        require(t['entry_bar_abs']<=t['exit_bar_abs'],'TRADE_TIME')
        require(t['entry_price']==env.engine.o[t['entry_bar_abs']],'NEXT_OPEN_PRICE')
        require(t['entry_time'].value>=env.engine.op[t['segment']],'ENTRY_DELAY')
        require(t['exit_time'].value<=env.engine.ff[t['segment']],'FORCED_FLAT_DEADLINE')
    net=sum(t['net_usd'] for t in env.engine.trades)
    require(np.isclose(env.engine.balance,env.engine.cfg.capital+net,rtol=1e-10,atol=1e-8),'BALANCE')
    require(all(n<=env.engine.cfg.max_entries_per_day for n in env.engine.entries_day.values()),'DAILY_CAP')
    return float(env.engine.balance-source.engine.balance),env


def compare(source,model,commands,label):
    values={};results={}
    for command in commands:
        values[command],results[command]=outcome(source,command)
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
    execution=copy.deepcopy(source);execution.step(action)
    expected=results[chosen]
    for t in execution.engine.trades:check_trade(t)
    return dict(state=label,bar=source.bar,values={str(k):v for k,v in values.items()},
        chosen=chosen,score_scales=[1e-4,1.,10.],
        winning_trades=[dict(direction=t['direction'],reason=t['close_reason'],net=t['net_usd'])
                        for t in expected.engine.trades])


def run_cell(cell,config,recipe,model,constant_sigma=False):
    shape=(recipe['tensor']['window']['observations'],recipe['list']['n'])
    results=[];absent=[]
    for anchor in (WARMUP+cell['switch']-2,WARMUP+cell['switch']+2):
        source=environment(cell,config,shape,constant_sigma)
        while source.bar<anchor:step(source,0)
        mask=source.mask()
        require(mask[1]==mask[2],'ASYMMETRIC_ENTRY_MASK')
        if not mask[1]:
            absent.append(dict(bar=anchor,state='FLAT_OPEN',reason='ENTRY_UNAVAILABLE',
                               entry_gate=source.engine.entry_gate(anchor),bounds=source.feasible_bounds()))
            results.append(compare(source,model,[0],'FLAT_NOP_ONLY'));continue
        results.append(compare(source,model,[0,1,2],'FLAT'))
        for command,label,close in ((1,'LONG',3),(2,'SHORT',4)):
            positioned=copy.deepcopy(source);_,_,_,_,event=step(positioned,command)
            if positioned.engine.pos is None:
                absent.append(dict(bar=anchor,state=label,reason='PROTECTION_OR_ENTRY_REFUSAL',
                                   last_event=event['engine_last_event']));continue
            require(positioned.engine.pos.direction==(1 if command==1 else -1),'SIDE_MAPPING')
            p=positioned.engine.pos;cfg=positioned.engine.cfg
            cap=np.floor(positioned.engine.balance*cfg.sizing.rel_volume*cfg.sizing.leverage/p.entry_price)
            require(0<p.units<=cap,'ORDER_VOLUME_CAP')
            require(p.entry_bar==anchor+1,'NEXT_BAR_EXECUTION')
            results.append(compare(positioned,model,[0,close],label))
    return dict(cell=cell,comparisons=results,unreachable=absent)


def run(a):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available(),'CPU_ONLY_ENV_REQUIRED')
    parent=read(a.source_spec)
    require(file_sha(a.recipe)==parent['loading']['recipe_sha256'],'PARENT_RECIPE_PIN')
    for name,sha in parent['code']['engine'].items():
        require(file_sha(a.engine_root/name)==sha,'ENGINE_SOURCE_PIN:'+name)
    recipe,config=setup(a.recipe,a.engine_root)
    source=dict(agent=source_pins(ROOT,('app','agent_plugins','env_plugins','tools')),
                engine=source_pins(a.engine_root,('app',)))
    identity=dict(schema='systematic_strategy_matrix.v1',recipe_sha256=file_sha(a.recipe),
        sources=source,expected_cells=5832,limit=a.limit,financial_admission=False,
        actual_training_steps=0,oracle='four-bar finite alternatives, diagnostic lookahead')
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.output/'IDENTITY.json').exists():require(read(a.output/'IDENTITY.json')==identity,'MATRIX_IDENTITY')
    else:write(a.output/'IDENTITY.json',identity)
    from tools.rl_subset_bridge import build_model
    shape=(recipe['tensor']['window']['observations'],recipe['list']['n'])
    model=build_model(environment(next(cells()),config,shape),recipe,7,'cpu',
                       dict(n_steps=8,batch_size=8,n_epochs=1))
    monotonic=[]
    for side in (-1,1):
        cell=dict(before=side,after=side,vol_before=0.,vol_after=0.,switch=60,period=24,phase=0,cost='zero_cost')
        result=run_cell(cell,config,recipe,model,constant_sigma=True)
        flats=[r for r in result['comparisons'] if r['state']=='FLAT']
        require(len(flats)==2 and all(r['chosen']==(1 if side==1 else 2) for r in flats),'MONOTONIC_DIRECTION')
        require(all(r['values'][str(1 if side==1 else 2)]>0 and
                    r['values'][str(2 if side==1 else 1)]<0 for r in flats),'MONOTONIC_PROFIT_SIGN')
        monotonic.append(result)
    write(a.output/'MONOTONIC.json',monotonic)
    write(a.output/'BROKER_EVENTS.json',witnesses(config,shape))
    rows=[];started=time.monotonic();matrix=list(cells())
    if a.limit is not None:
        require(0<a.limit<=len(matrix),'SMOKE_LIMIT')
        matrix=[matrix[i] for i in np.linspace(0,len(matrix)-1,a.limit,dtype=int)]
    for i,cell in enumerate(matrix):
        path=a.output/'cells'/(digest(cell)+'.json')
        if path.exists():
            r=read(path);require(r['identity_sha256']==digest(identity) and r['cell']==cell,'CELL_IDENTITY')
        else:
            try:
                r=dict(state='PASS',**run_cell(cell,config,recipe,model))
            except Exception as error:
                r=dict(state='FAIL',cell=cell,error_type=type(error).__name__,error=str(error))
            r.update(identity_sha256=digest(identity));write(path,r)
        rows.append(r)
        if i%25==0:
            write(a.output/'STATUS.json',dict(state='RUNNING',complete=len(rows),total=a.limit or 5832,
                failed=sum(x['state']=='FAIL' for x in rows),seconds=time.monotonic()-started,at=now()))
    failures=[r for r in rows if r['state']=='FAIL']
    counts={s:sum(x['state']==s for r in rows for x in r.get('comparisons',[])) for s in ('FLAT','FLAT_NOP_ONLY','LONG','SHORT')}
    wins={str(cmd):sum(x['chosen']==cmd for r in rows for x in r.get('comparisons',[])) for cmd in range(5)}
    full=len(rows)==5832 and a.limit is None
    coverage=all(counts[s]>0 for s in ('FLAT','LONG','SHORT')) and all(wins[str(cmd)]>0 for cmd in range(5))
    state='PASS_DECLARED_GRID' if full and not failures and coverage else 'SMOKE_ONLY' if not full and not failures else 'FAIL'
    summary=dict(state=state,cells=len(rows),failures=len(failures),comparisons=counts,chosen=wins,
        unreachable=sum(len(r.get('unreachable',[])) for r in rows),seconds=time.monotonic()-started,
        financial_admission=False,training_steps=0,finished_at=now())
    write(a.output/'SUMMARY.json',summary);write(a.output/'FAILURES.json',failures)
    write(a.output/'CLOSE.json',dict(summary=summary,artifacts={str(p.relative_to(a.output)):file_sha(p)
        for p in a.output.rglob('*') if p.is_file() and p.name not in ('CLOSE.json','STATUS.json','RUN.lock')}))
    write(a.output/'STATUS.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--source-spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int)
    run(p.parse_args())
