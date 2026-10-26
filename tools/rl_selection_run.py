"""Sequential, resumable native wrapper phase; upstream selection is not replaced."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from app.frozen_recipe_inputs import file_sha, require
from app.rl_subset import (digest, seal_recipe, refit_train_scaler, support_hash,
                          cost_profile, groups_for_selector, coherent_blocks)
from tools.recipe_sac_queue import ROOT, read, write, now, source_pins, software

BUDGET = dict(total_timesteps=102400, n_steps=512, batch_size=256, n_epochs=5,
    learning_rate=.0003, gamma=.99, gae_lambda=.95, ent_coef=.01, clip_range=.2,
    vf_coef=.5, max_grad_norm=.5, early_stopping=False, learned_risk=False,
    reward='equity', fusion_units=8, head_units=16, dropout_effective=0.)
SEARCH = dict(min_features=20, max_evaluations=128, tolerance=0.)
CONSTRAINTS = dict(agent_state='none', scaling='fixed_train', clipping=False,
                   fit_scope='train_only', test_access=False)
BUSINESS_GATE = 'both_directions_nop_hold_and_voluntary_close_each_side_v1'


def prepare(a):
    import pandas as pd
    from app.recipe_runtime import RecipeMarket
    from feature_selector.contracts import validate_dataset
    require(not a.root.exists(), 'NEW_CAMPAIGN_REQUIRED')
    original = read(a.recipe)
    require(original['tensor']['window']['observations'] == 13, 'WINDOW_PIN')
    # The feature rows are bar starts. Never load TEST even for fitting constants.
    end = pd.Timestamp(original['split']['test_closed_from'], tz='UTC') - pd.Timedelta(hours=1)
    frame = pd.read_parquet(a.store, columns=['timestamp'] + original['list']['columns'],
                           filters=[('timestamp', '<', end)])
    recipe = refit_train_scaler(original, frame)
    recipe['business']['evaluation_and_policy']['cost_mode'] = 'configured'
    recipe['labels'] = dict(mode='none', expert_used=False, inherited_metadata_not_consumed=True)
    recipe['provenance'] = dict(parent_sha256=file_sha(a.recipe),
        scope='new native wrapper; upstream selection retained, not independently revalidated',
        internal_selection_bias='candidate pool used 2023; INNER_A is development, not independent confirmation')
    seal_recipe(recipe)
    write(a.root / 'RECIPE.json', recipe)
    spec = dict(schema='native_rl_selection.v1', inputs={k: str(getattr(a,k).resolve())
        for k in ('store','manifest','bars','closures')}, engine_root=str(a.engine_root.resolve()),
        agent_root=str(ROOT), selector_root=str(a.selector_root.resolve()), device='cuda',
        budget=BUDGET, search=SEARCH, seeds=[7,19,43], software=software(), max_attempts=2,
        business_gate=BUSINESS_GATE,
        task_timeout_seconds=43200, created_at=now(),
        code=dict(agent=source_pins(ROOT,['app','env_plugins','agent_plugins','tools']),
                  engine=source_pins(a.engine_root,['app']),
                  selector=source_pins(a.selector_root,['src'])))
    spec['inputs']['recipe'] = str((a.root / 'RECIPE.json').resolve())
    spec['recipe_sha256'] = file_sha(a.root / 'RECIPE.json')
    market = RecipeMarket(spec)
    dataset = dict(schema='feature_selector.dataset.v1', scope='development',
        features=recipe['list']['columns'], groups=groups_for_selector(recipe),
        blocks=coherent_blocks(recipe['list']['columns']), support_hash=support_hash(market.inputs),
        scaler_hash=digest(recipe['scaler']), representation=recipe['representation'],
        splits=dict(train=recipe['split']['periods']['TRAIN'], selection=recipe['split']['periods']['INNER_A'],
                    test_start=recipe['split']['test_closed_from']), constraints=CONSTRAINTS)
    validate_dataset(dataset)
    spec['dataset'] = dataset
    spec['cost_profile'] = cost_profile(recipe['business']['evaluation_and_policy']['profile'])
    spec['available_rows'] = {k: int(((market.inputs.period == k) & market.inputs.valid).sum())
                              for k in recipe['split']['periods']}
    require(spec['available_rows']['TRAIN'] > 1000 and spec['available_rows']['INNER_A'] > 100, 'SUPPORT')
    write(a.root / 'SPEC.private.json', spec)
    write(a.root / 'IDENTITY.json', dict(spec_sha256=file_sha(a.root / 'SPEC.private.json')))
    write(a.root / 'STATUS.json', dict(state='PREPARED', available_rows=spec['available_rows'],
        features=len(dataset['features']), branches=len(dataset['groups']), test_access=False))
    print(json.dumps(read(a.root / 'STATUS.json')))


def verify(root):
    spec = read(root / 'SPEC.private.json')
    require(file_sha(root / 'SPEC.private.json') == read(root / 'IDENTITY.json')['spec_sha256'], 'SPEC_PIN')
    require(spec['schema'] == 'native_rl_selection.v1' and spec['business_gate'] == BUSINESS_GATE and spec['budget'] == BUDGET
            and spec['search'] == SEARCH and spec['software'] == software(), 'RUNTIME_PIN')
    for role in ('agent','engine','selector'):
        base = Path(spec[role + '_root'])
        for name, sha in spec['code'][role].items():
            require(file_sha(base / name) == sha, 'SOURCE_PIN:' + role + ':' + name)
    require(file_sha(spec['inputs']['recipe']) == spec['recipe_sha256'], 'RECIPE_PIN')
    return spec


def params_for(spec, mode, family):
    from feature_selector.evaluators import RLModelEvaluator
    p = {**RLModelEvaluator.plugin_params, 'model_id': family,
        'model_sha256':digest(spec['code']['agent']), 'engine_sha256':digest(spec['code']['engine']),
        'training_budget':spec['budget'], 'cost_mode':mode, 'cost_profile':spec['cost_profile'],
        'selection_use':'cost_aware' if mode == 'configured' else 'diagnostic'}
    RLModelEvaluator().validate(p)
    return p


def task_for(spec, columns, seed, mode, family, cross):
    from feature_selector.contracts import subset_view
    d = spec['dataset']
    request = {**subset_view(d, columns), **{k:d[k] for k in ('support_hash','scaler_hash','splits','constraints')},
        'seed':seed, 'unit':digest([columns,seed,mode,family])[:20],
        'campaign':digest(spec), 'scope':'development'}
    return dict(request=request, params=params_for(spec,mode,family), cross=cross)


def authenticate(path, task):
    from feature_selector.evaluators import RLModelEvaluator
    receipt = read(path)
    score = RLModelEvaluator().verify(receipt,task['request'],task['params'],path.parent)
    declared = {str(Path(a['path']).resolve()) for a in receipt['artifacts']}
    counts = {}
    for split, period in [('train','TRAIN'),('selection','INNER_A')]:
        orders = path.parent / (period + '.orders.json')
        require(str(orders.resolve()) in declared, 'BEHAVIOR_RECEIPT_REQUIRED')
        counts[split] = read(orders)['counts']
    return apply_business_gate(score, counts)


def apply_business_gate(score, counts):
    from tools.recipe_hybrid_queue import business_behavior
    behavior = {}
    for split in ('train','selection'):
        metrics = score['native']['metrics'][split]
        behavior[split] = business_behavior(
            dict(n_long=metrics['buy_orders'],n_short=metrics['sell_orders']),counts[split])
    return {**score, 'business_behavior':behavior,
            'eligible':score['eligible'] and all(v['business_behavior_eligible'] for v in behavior.values())}


def execute_task(root, spec, task):
    name = digest(task)
    path = root / 'tasks' / (name + '.json')
    if path.exists():
        require(read(path) == task, 'TASK_COLLISION')
    else:
        write(path,task)
    for attempt in range(1,spec['max_attempts']+1):
        folder = path.parent / (name + f'-attempt-{attempt}')
        result = folder / 'RESULT.json'
        if result.exists():
            return authenticate(result,task)
        ledger = path.parent / (name + f'-attempt-{attempt}.json')
        if ledger.exists():
            # An interrupted attempt is retained; never overwrite its artifacts.
            continue
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                             if x.startswith('MemAvailable:'))) * 1024
        while available < 5 * 1024**3:
            write(root / 'STATUS.json',dict(state='WAIT_RESOURCES',unit=task['request']['unit'],heartbeat_at=now()))
            time.sleep(60)
            available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                                 if x.startswith('MemAvailable:'))) * 1024
        write(ledger,dict(state='STARTED',started_at=now(),attempt=attempt,task_sha256=name))
        write(root / 'STATUS.json',dict(state='RUNNING',unit=task['request']['unit'],features=len(task['request']['features']),
            seed=task['request']['seed'],cost_mode=task['params']['cost_mode'],family=task['params']['model_id'],
            attempt=attempt,heartbeat_at=now()))
        try:
            with (path.parent / (name + f'-attempt-{attempt}.log')).open('ab') as log:
                result_code = subprocess.run([sys.executable,'-m','tools.rl_subset_bridge','--root',str(root),
                    '--task',str(path),'--attempt',str(attempt)],cwd=spec['agent_root'],stdout=log,stderr=subprocess.STDOUT,
                    timeout=spec['task_timeout_seconds']).returncode
        except subprocess.TimeoutExpired:
            result_code = 124
        write(ledger,dict(state='COMPLETE' if result_code == 0 else 'FAILED',returncode=result_code,
                          attempt=attempt,task_sha256=name,finished_at=now()))
        if result_code == 0:
            return authenticate(result,task)
        time.sleep(30 * attempt)
    raise RuntimeError('TASK_FAILED_AFTER_BOUNDED_RETRIES:' + name)


def aggregate(rows):
    import statistics
    measured = all(r['score'] is not None and r['gap'] is not None for r in rows)
    return dict(score=statistics.mean(r['score'] for r in rows) if measured else None,
        gap=statistics.mean(r['gap'] for r in rows) if measured else None,
        eligible=measured and all(r['eligible'] for r in rows),
        seeds=[dict(seed=r['native']['seed'],score=r['score'],train_score=r['train_score'],gap=r['gap'],
                    eligible=r['eligible'],business_behavior=r.get('business_behavior'),
                    metrics=r['native']['metrics'],request_sha256=r['native']['request_sha256']) for r in rows])


def admit(root, spec):
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from tools.recipe_hybrid_queue import configure
    from tools.rl_subset_bridge import build_model,progress_callback
    configure()
    require(torch.cuda.is_available(),'CUDA_REQUIRED')
    r=read(spec['inputs']['recipe']); shape=(r['tensor']['window']['observations'],r['list']['n'])
    class Synthetic(gym.Env):
        observation_space=gym.spaces.Dict({'features':gym.spaces.Box(-np.inf,np.inf,shape,np.float32),
            'action_mask':gym.spaces.Box(0,1,(5,),np.float32)})
        action_space=gym.spaces.Box(np.array([0,0,0,0],np.float32),np.array([4,1,1,1],np.float32))
        def obs(self):
            return dict(features=np.ones(shape,np.float32),action_mask=np.array([1,1,1,0,0],np.float32))
        def reset(self,seed=None,options=None):
            super().reset(seed=seed);self.n=0
            return self.obs(),{}
        def step(self,a):
            require(a[0]==int(a[0]) and int(a[0]) in (0,1,2),'CATEGORICAL_ACTION')
            self.n+=1
            return self.obs(),float(a[0]==1),self.n==8,False,{'elapsed_bars':1}
    proofs=[]
    for family in ('tcn','cnn'):
        model=build_model(Synthetic(),r,7,'cuda',{**BUDGET,'n_steps':32,'batch_size':32,'n_epochs':1},family)
        model.learn(64,callback=progress_callback(root,'admission_'+family,64))
        require(all(torch.isfinite(p).all() for p in model.policy.parameters()),'FINITE_WEIGHTS')
        target=root / ('ADMISSION_'+family+'.zip');model.save(target)
        loaded=PPO.load(target,device='cuda')
        require(np.array_equal(model.predict(Synthetic().obs(),deterministic=True)[0],
                               loaded.predict(Synthetic().obs(),deterministic=True)[0]),'GPU_RELOAD')
        proofs.append(dict(family=family,sha256=file_sha(target),updates=model._n_updates))
        del model,loaded
    write(root/'ADMISSION.json',dict(state='PASS',scope='SYNTHETIC_ENGINEERING_ONLY',models=proofs,
        spec_sha256=file_sha(root/'SPEC.private.json'),gpu_peak_bytes=torch.cuda.max_memory_allocated()))


def supervise(root):
    from feature_selector.search import FloatingBackward
    spec=verify(root)
    lock=(root/'RUN.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'CLOSE.json').exists():
        require(read(root/'CLOSE.json')['spec_sha256']==file_sha(root/'SPEC.private.json'),'CLOSE_PIN')
        return
    if not (root/'ADMISSION.json').exists():
        with (root/'admission.log').open('ab') as log:
            subprocess.run([sys.executable,'-m','tools.rl_selection_run','admit','--root',str(root)],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    require(read(root/'ADMISSION.json')['state']=='PASS' and
            read(root/'ADMISSION.json')['spec_sha256']==file_sha(root/'SPEC.private.json'),'ADMISSION_PIN')
    full=spec['dataset']['features']
    controls={}
    for family,mode in [('tcn','configured'),('tcn','zero_cost'),('cnn','configured')]:
        rows=[execute_task(root,spec,task_for(spec,full,s,mode,family,True)) for s in spec['seeds']]
        controls[family+'_'+mode]=aggregate(rows)
        write(root/'CONTROLS.json',controls)
    def evaluate(columns):
        rows=[execute_task(root,spec,task_for(spec,columns,s,'configured','tcn',columns==full)) for s in spec['seeds']]
        result=aggregate(rows)
        write(root/'subsets'/(digest(columns)+'.json'),dict(features=columns,**result))
        return result
    result=FloatingBackward().select(spec['dataset'],evaluate,spec['search'])
    write(root/'SEARCH.json',result)
    state='BUDGET_REACHED_PROVISIONAL' if result['termination']=='BUDGET' else 'WRAPPER_PHASE_COMPLETE'
    write(root/'CLOSE.json',dict(state=state,termination=result['termination'],evaluations=result['evaluations'],
        selected=result['selected'],finished_at=now(),spec_sha256=file_sha(root/'SPEC.private.json'),
        global_release=False,test_access=False,upstream_methodology='retained; not replaced by wrapper',
        limitations=['selection internal to development','upstream pool used selection-year rows',
                     'cost assumptions not broker calibrated','no independent confirmation or TEST read']))
    write(root/'STATUS.json',read(root/'CLOSE.json'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','admit','run','status'])
    p.add_argument('--root',type=Path,required=True)
    for k in ('recipe','store','manifest','bars','closures','engine-root','selector-root'):
        p.add_argument('--'+k,type=Path)
    a=p.parse_args()
    if a.command=='prepare': prepare(a)
    elif a.command=='admit': admit(a.root,verify(a.root))
    elif a.command=='status':
        print(json.dumps({k:read(a.root/(k+'.json')) for k in ('STATUS','PROGRESS','CONTROLS','CLOSE')
                          if (a.root/(k+'.json')).exists()},indent=2))
    else:
        try: supervise(a.root)
        except Exception as exc:
            write(a.root/'STATUS.json',dict(state='FAILED',error=type(exc).__name__,detail=str(exc),finished_at=now()))
            raise


if __name__=='__main__':
    main()
