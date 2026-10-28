"""Sequential actual-PPO learning qualification with isolated future-signal controls."""
import argparse
import copy
from dataclasses import replace
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from app.c4_learning_control import future_move, control_inputs, qualify
from app.frozen_recipe_inputs import file_sha, require
from app.rl_subset import seal_recipe, digest
from app.native_strategy_contract import executed_behavior
from tools.recipe_sac_queue import read, write, now, source_pins, software, tensor_hash, evaluate


ROOT=Path(__file__).resolve().parents[1]
ARMS=('planted','permuted','original')
SEEDS=(7,19,43)


def verify(root):
    spec=read(root/'SPEC.private.json')
    require(file_sha(root/'SPEC.private.json')==read(root/'IDENTITY.json')['spec_sha256'],'C4_SPEC_PIN')
    require(spec['schema']=='c4_learning_diagnostic.v1' and spec['financial_admission'] is False,'C4_SCOPE')
    require(spec['software']==software(),'C4_SOFTWARE')
    for role,base in [('agent',ROOT),('engine',Path(spec['loading']['engine_root']))]:
        for name,sha in spec['code'][role].items():require(file_sha(base/name)==sha,'C4_CODE:'+role+':'+name)
    return spec


def prepare(a):
    require(not a.root.exists(),'C4_NEW_ROOT_REQUIRED')
    parent=read(a.parent/'SPEC.private.json');recipe=read(a.parent/'RECIPE.json')
    require(file_sha(a.parent/'SPEC.private.json')==read(a.parent/'IDENTITY.json')['spec_sha256'],'C4_PARENT_PIN')
    for name,sha in parent['code']['engine'].items():require(file_sha(a.engine_root/name)==sha,'C4_ENGINE_PIN')
    # This loader cannot materialize OBS_2024 or TEST; the original recipe is untouched.
    recipe['split']['periods']={p:recipe['split']['periods'][p] for p in ('TRAIN','INNER_A')}
    recipe['split']['test_closed_from']=recipe['split']['periods']['INNER_A'][1]
    recipe['labels']['c4_diagnostic']=dict(deliberate_lookahead=True,financial_admission=False)
    seal_recipe(recipe)
    write(a.root/'LOADING_RECIPE.json',recipe)
    loading=copy.deepcopy(parent);loading['engine_root']=str(a.engine_root.resolve())
    loading['inputs'].update(recipe=str((a.root/'LOADING_RECIPE.json').resolve()),
        store=str(a.data_root/'full_pool_s_a1/full_pool_eurusd_h1.parquet'),
        manifest=str(a.data_root/'full_pool_s_a1/MANIFEST.json'),
        bars=str(a.data_root/'admitted/eurusd.parquet'),
        closures=str(a.data_root/'admitted/CLOSURE_INTERVALS.json'))
    loading['recipe_sha256']=file_sha(a.root/'LOADING_RECIPE.json')
    spec=dict(schema='c4_learning_diagnostic.v1',financial_admission=False,
        parent_spec_sha256=file_sha(a.parent/'SPEC.private.json'),loading=loading,budget=parent['budget'],
        seeds=list(SEEDS),arms=list(ARMS),software=software(),
        units=[dict(arm=arm,seed=seed,id=f'{arm}_s{seed}') for seed in SEEDS for arm in ARMS],
        code=dict(agent=source_pins(ROOT,('app','agent_plugins','env_plugins','tools')),
                  engine=parent['code']['engine']),
        physical_gpu=a.physical_gpu,minimum_available_ram_gib=5,
        timeout_seconds=2700,max_attempts=2,created_at=now())
    require(spec['budget']['total_timesteps']==102400 and not spec['budget']['early_stopping'],'C4_FROZEN_BUDGET')
    write(a.root/'SPEC.private.json',spec)
    write(a.root/'IDENTITY.json',dict(spec_sha256=file_sha(a.root/'SPEC.private.json')))
    write(a.root/'STATUS.json',dict(state='PREPARED',units=9,financial_admission=False,at=now()))


def market_for(spec,arm):
    from app.recipe_runtime import RecipeMarket
    from tools.rl_subset_bridge import SubsetMarket
    base=RecipeMarket(spec['loading'])
    stamps=base.bars.index.asi8+3600*10**9
    periods=np.full(len(stamps),'',object)
    for name,(lo,hi) in base.recipe['split']['periods'].items():
        periods[(stamps>=pd.Timestamp(lo,tz='UTC').value)&(stamps<pd.Timestamp(hi,tz='UTC').value)]=name
    series=future_move(base.bars.OPEN.to_numpy(),base.bars.CLOSE.to_numpy(),periods)
    idx=np.searchsorted(stamps,base.inputs.row_ts_ns)
    safe=np.minimum(idx,len(stamps)-1)
    matched=(idx<len(stamps))&(stamps[safe]==base.inputs.row_ts_ns)
    signal=np.full(len(idx),np.nan);signal[matched]=series[safe[matched]]
    inputs,metadata=control_inputs(base.inputs,signal,arm)
    market=object.__new__(SubsetMarket)
    market.base=base;market.inputs=inputs;market.recipe=inputs.recipe;market.mode='configured'
    require(np.isfinite(inputs.windows[inputs.valid]).all(),'C4_FINITE_INPUTS')
    metadata.update(recipe_identity=inputs.recipe['identity'],shape=list(inputs.windows.shape[1:]),
                    source_bar_end=str(base.bars.index[-1]),column_replaced=0)
    return market,metadata


def preflight(root):
    import torch
    from tools.rl_subset_bridge import build_model
    torch.set_num_threads(1)
    spec=verify(root);reports=[];support=None
    for arm in ARMS:
        market,meta=market_for(spec,arm)
        if support is None:support=meta['common_support_hash']
        require(meta['common_support_hash']==support,'C4_PAIRED_SUPPORT')
        env=market.env('TRAIN');obs,_=env.reset()
        model=build_model(env,market.recipe,7,'cpu',spec['budget'])
        action=model.predict(obs,deterministic=True)[0]
        require(action[0]==int(action[0]) and obs['action_mask'][int(action[0])]==1,'C4_ACTION')
        meta['parameters']=sum(p.numel() for p in model.policy.parameters());reports.append(meta)
        del model,env,market
    require(len({r['parameters'] for r in reports})==1,'C4_IDENTICAL_ARCHITECTURE')
    value=dict(state='PASS',spec_sha256=file_sha(root/'SPEC.private.json'),arms=reports,
               training_steps=0,device='cpu',finished_at=now())
    write(root/'PREFLIGHT.json',value);print(value,flush=True)


def verified_result(root,path):
    r=read(path)
    require(r['schema']=='c4_diagnostic_result.v1' and r['financial_admission'] is False,'C4_RESULT_SCOPE')
    require(r['spec_sha256']==file_sha(root/'SPEC.private.json'),'C4_RESULT_PIN')
    for name,sha in r['artifacts'].items():
        p=(path.parent/name).resolve();require(p.is_relative_to(path.parent.resolve()),'C4_ARTIFACT_PATH')
        require(file_sha(p)==sha,'C4_ARTIFACT_HASH')
    return r


def worker(a):
    import torch
    from stable_baselines3 import PPO
    from tools.recipe_hybrid_queue import configure
    from tools.rl_subset_bridge import build_model,progress_callback
    from tools.native_wrapper_audit import ledger_checks
    configure();spec=verify(a.root)
    require(torch.cuda.is_available(),'C4_CUDA')
    unit=next(u for u in spec['units'] if u['id']==a.unit)
    folder=a.root/'units'/a.unit/f'attempt-{a.attempt}'
    require(not folder.exists(),'C4_FRESH_ATTEMPT');folder.mkdir(parents=True)
    market,metadata=market_for(spec,unit['arm'])
    pf=read(a.root/'PREFLIGHT.json')
    require(pf['spec_sha256']==file_sha(a.root/'SPEC.private.json') and pf['state']=='PASS','C4_PREFLIGHT_PIN')
    require(metadata['common_support_hash']==pf['arms'][0]['common_support_hash'],'C4_SUPPORT_PIN')
    write(folder/'DIAGNOSTIC_INPUTS.json',metadata);write(folder/'RECIPE.json',market.recipe)
    env=market.env('TRAIN');started=time.monotonic()
    model=build_model(env,market.recipe,unit['seed'],'cuda',spec['budget'])
    before=tensor_hash(model)
    model.learn(spec['budget']['total_timesteps'],callback=progress_callback(a.root,a.unit,spec['budget']['total_timesteps']))
    require(model.num_timesteps==102400 and tensor_hash(model)!=before,'C4_WEIGHT_UPDATES')
    model.c4_diagnostic_only=True
    model.save(folder/'DIAGNOSTIC_ONLY.model.zip')
    loaded=PPO.load(folder/'DIAGNOSTIC_ONLY.model.zip',device='cuda')
    require(loaded.c4_diagnostic_only and tensor_hash(loaded)==tensor_hash(model),'C4_RELOAD')
    obs,_=env.reset()
    require(np.array_equal(loaded.predict(obs,deterministic=True)[0],model.predict(obs,deterministic=True)[0]),'C4_RELOAD_ACTION')
    del loaded
    periods={}
    for period in ('TRAIN','INNER_A'):
        write(a.root/'PROGRESS.json',dict(state='EVALUATING',unit=a.unit,period=period,
              elapsed_seconds=time.monotonic()-started,heartbeat_at=now()))
        report=evaluate(model,market,period,folder)
        trades=read(folder/(period+'.trades.json'));counts=read(folder/(period+'.orders.json'))['counts']
        ledger_checks(report,trades)
        periods[period]=dict(metrics=report,strict=executed_behavior(trades,counts,report))
    write(folder/'TRAINING.json',dict(initial_weights=before,final_weights=tensor_hash(model),
          steps=model.num_timesteps,updates=model._n_updates,gpu_name=torch.cuda.get_device_name(),
          gpu_peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
          budget=spec['budget'],seed=unit['seed'],new_optimizer=True,financial_admission=False))
    r=dict(schema='c4_diagnostic_result.v1',spec_sha256=file_sha(a.root/'SPEC.private.json'),
        arm=unit['arm'],seed=unit['seed'],periods=periods,financial_admission=False,
        seconds=time.monotonic()-started,finished_at=now(),
        artifacts={p.name:file_sha(p) for p in folder.iterdir() if p.is_file()})
    write(folder/'RESULT.json',r);print(unit['id'],'COMPLETE',r['seconds'],flush=True)


def admission(spec):
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))
    if available<spec['minimum_available_ram_gib']*1024**3:return False,'RAM'
    query=subprocess.run(['nvidia-smi','-i',str(spec['physical_gpu']),
        '--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    if query.returncode or query.stdout.strip():return False,'GPU_BUSY_OR_UNAVAILABLE'
    return True,None


def run(root):
    spec=verify(root)
    with (root/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'CLOSE.json').exists():
            for name,sha in read(root/'CLOSE.json')['artifacts'].items():
                require(file_sha(root/name)==sha,'C4_CLOSE_ARTIFACT')
            print('C4_COMPLETE_VERIFIED',flush=True);return
        pf=read(root/'PREFLIGHT.json')
        require(pf['state']=='PASS' and pf['spec_sha256']==file_sha(root/'SPEC.private.json'),'C4_PREFLIGHT_PIN')
        rows=[]
        for unit in spec['units']:
            base=root/'units'/unit['id'];done=False
            for path in sorted(base.glob('attempt-*/RESULT.json')):
                r=verified_result(root,path)
                require((r['arm'],r['seed'])==(unit['arm'],unit['seed']),'C4_UNIT_IDENTITY')
                rows.append(r);done=True;break
            if done:continue
            existing=list(base.glob('attempt-*'));attempt=len(existing)+1
            while attempt<=spec['max_attempts']:
                ready,reason=admission(spec)
                if not ready:
                    write(root/'STATUS.json',dict(state='WAIT_RESOURCES',unit=unit['id'],reason=reason,
                          completed=len(rows),total=9,heartbeat_at=now()))
                    time.sleep(30);continue
                write(root/'STATUS.json',dict(state='RUNNING',unit=unit['id'],attempt=attempt,
                      completed=len(rows),total=9,financial_admission=False,heartbeat_at=now()))
                with (root/(unit['id']+f'-{attempt}.log')).open('a') as log:
                    try:
                        process=subprocess.run([sys.executable,'-m','tools.c4_learning_queue','worker',
                            '--root',str(root),'--unit',unit['id'],'--attempt',str(attempt)],
                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=spec['timeout_seconds'])
                        code=process.returncode
                    except subprocess.TimeoutExpired:code=124
                path=base/f'attempt-{attempt}'/'RESULT.json'
                write(root/(unit['id']+f'-{attempt}.exit.json'),dict(returncode=code,finished_at=now()))
                if code==0 and path.exists():
                    rows.append(verified_result(root,path));done=True;break
                attempt+=1;time.sleep(30)
            if not done:
                write(root/'STATUS.json',dict(state='FAILED_AFTER_TWO_ATTEMPTS',unit=unit['id'],
                    completed=len(rows),total=9,finished_at=now()))
                raise RuntimeError('C4_UNIT_FAILED:'+unit['id'])
        decision=qualify(rows)
        write(root/'RESULTS.json',rows);write(root/'DECISION.json',decision)
        lines=['# C4 diagnostic learning results','',
               'Deliberate future information. No financial admission. INNER_A is selection, not confirmation.','',
               '| Arm | Seed | Period | Sharpe | Profit USD | Orders | Buy / Sell | Win % | Max DD % | Strict |',
               '|---|---:|---|---:|---:|---:|---|---:|---:|---|']
        for r in rows:
            for period,p in r['periods'].items():
                m=p['metrics'];lines.append(f'| {r["arm"]} | {r["seed"]} | {period} | '
                    f'{m["sharpe_daily_sqrt365"]} | {m["net_profit_usd"]} | {m["trades"]} | '
                    f'{m["n_long"]} / {m["n_short"]} | {m["win_rate"]*100 if m["win_rate"] is not None else None} | '
                    f'{m["max_drawdown_pct"]} | {p["strict"]["eligible"]} |')
        lines+=['',decision['state'],'',*decision['failures']]
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        write(root/'CLOSE.json',dict(state=decision['state'],finished_at=now(),financial_admission=False,
              artifacts={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()
                         and p.name not in ('QUEUE.lock','STATUS.json','PROGRESS.json','SUPERVISOR.log')
                         and not p.name.endswith('.log')}))
        write(root/'STATUS.json',dict(state=decision['state'],completed=9,total=9,finished_at=now()))
        print(decision,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','preflight','run','worker'])
    p.add_argument('--root',type=Path,required=True)
    for name in ('parent','engine-root','data-root'):p.add_argument('--'+name,type=Path)
    p.add_argument('--physical-gpu',type=int,default=0);p.add_argument('--unit');p.add_argument('--attempt',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='preflight':preflight(a.root)
    elif a.mode=='worker':worker(a)
    else:run(a.root)


if __name__=='__main__':main()
