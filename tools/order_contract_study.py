"""Business alpha and frozen-policy diagnostics, prerequisites to the order contrast."""
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from app.frozen_recipe_inputs import file_sha, require
from tools.recipe_sac_queue import (authenticated_result, clean, now, preserve_rng, read,
                                    tensor_hash, verify, write)


def witness(engine_root):
    import pandas as pd
    from app.recipe_runtime import attach_engine
    attach_engine(engine_root)
    from app.short_horizon.expert_guided.env import BarEnv, EnvConfig
    from app.short_horizon.expert_guided.data import calendar_from_closures
    from app.short_horizon.execution import Sizing
    from env_plugins.order_contract import OrderContractEnv
    ts = pd.date_range('2020-01-05T22:00Z', periods=120, freq='h')
    cal = calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],
                                 ['2020-01-10T22:00Z','2020-01-12T22:00Z']])
    inputs = SimpleNamespace(row_ts_ns=(ts + pd.Timedelta(hours=1)).as_unit('ns').asi8,
                             valid=np.ones(120,bool), windows=np.ones((120,3,2),np.float32),
                             observation=lambda row:{'features': np.full((3,2),row,np.float32)})
    cfg = EnvConfig(max_entries_per_day=1,min_orders_per_week=2,
                    sizing=Sizing(rel_volume=.1,leverage=50))
    cases=[]
    for direction in (1,-1):
        for scenario in ('close','daily_cap','SL','TP','both','gap','forced','invalid_hold'):
            bars=pd.DataFrame({'OPEN':1.1,'HIGH':1.10001,'LOW':1.09999,'CLOSE':1.1},index=ts)
            engine=BarEnv(bars,cal,np.full(120,.001),np.full(120,.002),ts[0],
                          ts[-1]+pd.Timedelta(hours=1),cfg)
            for name in ('o','h','l','c'):
                setattr(engine,name,getattr(engine,name).copy())
            env=OrderContractEnv(engine,inputs,1.,{'mode':'commands5','threshold':None})
            obs,_=env.reset()
            total=0.
            def step(value):
                nonlocal total
                result=env.step(np.array([value,0.,0.],np.float32))
                total+=result[1]
                require(set(result[0])=={'features'},'ALPHA_FEEDBACK')
                return result
            step(.4*direction)
            require(engine.pos is None and engine.counts['refused_BEFORE_OPERATIONAL_START']==1,
                    'ALPHA_REOPEN_DELAY')
            while env.bar<12:
                step(0.)
            step(.4*direction)
            p=engine.pos
            require(p is not None and p.direction==direction and p.entry_bar==13,'ALPHA_NEXT_OPEN')
            require(p.units==cfg.sizing.order_units(10000,1.1,1.) and p.sl!=p.tp,
                    'ALPHA_SIZE_PROTECTION')
            require((p.sl<1.1<p.tp) if direction==1 else (p.tp<1.1<p.sl),'ALPHA_LEVEL_SIDE')
            before=env._observe()['features'].copy()
            engine.pos=None
            np.testing.assert_array_equal(before,env._observe()['features'])
            engine.pos=p
            if scenario in ('SL','TP','both','gap'):
                k=env.bar+1
                if scenario=='gap':
                    engine.o[k]=p.sl-direction*.001
                    engine.h[k]=engine.l[k]=engine.c[k]=engine.o[k]
                else:
                    if scenario in ('SL','both'):
                        if direction==1: engine.l[k]=p.sl-.0001
                        else: engine.h[k]=p.sl+.0001
                    if scenario in ('TP','both'):
                        if direction==1: engine.h[k]=p.tp+.0001
                        else: engine.l[k]=p.tp-.0001
                step(0.)
                expected='TP' if scenario=='TP' else 'SL'
                require(engine.pos is None and engine.trades[-1]['close_reason']==expected,'ALPHA_PROTECT')
                if scenario=='gap':
                    require(engine.trades[-1]['exit_price']==engine.o[k],'ALPHA_STOP_GAP_OPEN')
                if scenario=='both':
                    require(engine.trades[-1]['ambiguous'],'ALPHA_STOP_FIRST')
            elif scenario in ('close','daily_cap'):
                step(.8*direction)
                require(engine.pos is None and engine.trades[-1]['exit_bar_abs']==14
                        and engine.trades[-1]['close_reason']=='AGENT_CLOSE','ALPHA_CLOSE')
                if scenario=='daily_cap':
                    step(.4*direction)
                    require(engine.pos is None and engine.counts['refused_DAILY_CAP']==1,'ALPHA_DAY_CAP')
            elif scenario=='invalid_hold':
                step(-.4*direction)
                require(engine.pos is p and env.last_order['invalid'],'ALPHA_INVALID_HOLD')
            while not env.finished:
                step(0.)
            require(engine.pos is None,'ALPHA_FLAT_TERMINAL')
            if scenario in ('forced','invalid_hold'):
                require(engine.trades[-1]['close_reason']=='FORCED_FLAT'
                        and engine.trades[-1]['exit_time'].value<=engine.ff[p.seg], 'ALPHA_WEEKEND')
            net=sum(t['net_usd'] for t in engine.trades)
            require(abs(total*cfg.capital/100-net)<1e-8 and abs(engine.balance-10000-net)<1e-8,
                    'ALPHA_REWARD_ACCOUNTING')
            require(all(t['commission_usd']>0 and t['spread_usd']>0 and t['margin_usd']>0
                        for t in engine.trades),'ALPHA_COSTS')
            cases.append(dict(direction=direction,scenario=scenario,net_usd=net,
                              reward_usd=total*cfg.capital/100,orders=len(engine.trades)))
    return dict(state='PASS',scope='SYNTHETIC_BUSINESS_CONTRACT',cases=cases,
                no_feedback=True,market_inputs_synthetic=True,scientific_fits=0)


def action_summary(actions):
    x=np.asarray(actions)
    return {'count':len(x),'quantiles':np.quantile(x,[0,.05,.25,.5,.75,.95,1],axis=0).tolist(),
            'target_flat_fraction':float((np.abs(x[:,0])<=1/3).mean())}


def diagnose(root):
    import torch
    from stable_baselines3 import SAC
    from app.recipe_runtime import RecipeMarket
    spec=verify(root)
    parent=Path(spec['order_parent']['root'])
    parent_sha=spec['order_parent']['spec_sha256']
    require(file_sha(parent/'SPEC.private.json')==parent_sha,'PARENT_SPEC')
    parent_spec=read(parent/'SPEC.private.json')
    require(parent_spec['recipe_sha256']==spec['recipe_sha256'] and parent_spec['inputs']==spec['inputs'],
            'PARENT_INPUTS')
    result_path=parent/spec['order_parent']['result']
    require(file_sha(result_path)==spec['order_parent']['result_sha256'],'PARENT_RESULT_PIN')
    authenticated_result(result_path,parent_sha)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    market=RecipeMarket(spec,{'mode':'target','threshold':1/3})
    from app.short_horizon.expert_guided.env import summary
    output=root/'diagnostic'
    output.mkdir(exist_ok=True)
    records=[]
    for checkpoint in (25000,50000,75000,100000):
        model_path=result_path.parent/f'step-{checkpoint}'/'model.zip'
        model=SAC.load(model_path,device='cuda')
        digest=tensor_hash(model)
        for period in ('TRAIN','INNER_A'):
            write(root/'PROGRESS.json',dict(state='DIAGNOSING',checkpoint=checkpoint,
                period=period,heartbeat_at=now(),spec_sha256=file_sha(root/'SPEC.private.json')))
            source=read(model_path.parent/(period+'.json'))
            require(source['policy_tensor_sha256']==digest,'PARENT_POLICY')
            env=market.env(period)
            obs,_=env.reset()
            bars=[]; deterministic=[]; stochastic=[]
            started=time.monotonic()
            with preserve_rng():
                torch.manual_seed(7000+checkpoint)
                while True:
                    bars.append(env.bar)
                    a,_=model.predict(obs,deterministic=True)
                    deterministic.append(a.copy())
                    b,_=model.predict(obs,deterministic=False)
                    stochastic.append(b.copy())
                    obs,_,done,_,_=env.step(a)
                    if done: break
            require(tensor_hash(model)==digest,'DIAGNOSTIC_WEIGHTS')
            np.savez_compressed(output/f'{checkpoint}-{period}-actions.npz',
                                bars=np.array(bars),deterministic=deterministic,stochastic=stochastic)
            baseline=summary(env.engine.result())
            comparable={k:v for k,v in source.items() if k in baseline}
            differences={k:[clean(v),clean(baseline[k])] for k,v in comparable.items()
                         if clean(v)!=clean(baseline[k])}
            write(output/f'{checkpoint}-{period}-REPLAY.json',dict(exact=not differences,
                differences=differences,source_policy_sha256=digest,seconds=time.monotonic()-started))
            require(not differences,'PARENT_METRIC_REPLAY')
            for threshold in (.15,.25,1/3,.45):
                market.order_contract={'mode':'target','threshold':threshold}
                e=market.env(period)
                e.reset()
                for i,a in enumerate(deterministic):
                    require(e.bar==bars[i],'DIAGNOSTIC_SUPPORT')
                    _,_,done,_,_=e.step(a)
                    require(done==(i==len(bars)-1),'DIAGNOSTIC_DONE')
                native=e.engine.result()
                durations=[(t['exit_time']-t['entry_time']).total_seconds()/3600 for t in e.engine.trades]
                row=clean(dict(checkpoint=checkpoint,period=period,threshold=threshold,
                    metrics=summary(native),policy_tensor_sha256=digest,order_counts=e.order_counts,
                    duration_hours_quantiles=np.quantile(durations,[0,.5,.9,1]).tolist() if durations else None,
                    deterministic=action_summary(deterministic),stochastic=action_summary(stochastic),
                    stochastic_scope='ACTION_DISTRIBUTION_ONLY_NOT_TRADING_SCORE',
                    support_sha256=__import__('hashlib').sha256(np.array(bars,dtype=np.int64).tobytes()).hexdigest()))
                records.append(row)
                write(output/f'{checkpoint}-{period}-{threshold:.6f}.json',row)
                write(output/f'{checkpoint}-{period}-{threshold:.6f}-trades.json',clean(e.engine.trades))
                e.close()
            market.order_contract={'mode':'target','threshold':1/3}
            env.close()
        del model
    report=dict(state='PASS',spec_sha256=file_sha(root/'SPEC.private.json'),
                parent_sha256=parent_sha,checkpoint_period_replays=8,records=records,
                selection='NONE_FIXED_CONTRAST_FOLLOWS',finished_at=now(),
                artifacts={p.name:file_sha(p) for p in output.iterdir() if p.is_file()})
    write(root/'DIAGNOSTIC.json',report)
    return 0
