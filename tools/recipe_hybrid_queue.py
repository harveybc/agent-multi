"""Finite, pinned categorical-output and learning-reward queue; no expert labels."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from app.frozen_recipe_inputs import file_sha, require
from tools.recipe_sac_queue import (ROOT, read, write, now, clean, software, source_pins,
                                    admission, evaluate, tensor_hash, preserve_rng)

DESIGN=dict(total_timesteps=102400,n_steps=512,batch_size=256,n_epochs=5,
            learning_rate=.0003,gamma=.99,gae_lambda=.95,ent_coef=.01,
            clip_range=.2,vf_coef=.5,max_grad_norm=.5,early_stopping=False,
            encoder_dropout_effective=0.,agent_feedback='ROUTING_MASK_ONLY',expert=False,
            policy_family='grouped_sequence_v1',fusion_units=8,
            representation='selected branches; causal fusion; full sequence; separate pi/vf heads')
QUALIFICATION=dict(steps=131072,n_steps=64,batch_size=64,
                   cases=[dict(seed=s,lag=0) for s in (7,19,43)]+[dict(seed=7,lag=6)],
                   minimum_expected_probability=.6,scope='SYNTHETIC_NOT_FINANCIAL')
ARMS={
    'CAT_FIXED':dict(learned_risk=False,reward_mode='equity',penalty=0.),
    'HYBRID_EQUITY':dict(learned_risk=True,reward_mode='equity',penalty=0.),
    'HYBRID_LOG':dict(learned_risk=True,reward_mode='log',penalty=0.),
    'HYBRID_DD025':dict(learned_risk=True,reward_mode='drawdown',penalty=.25),
    'HYBRID_DD100':dict(learned_risk=True,reward_mode='drawdown',penalty=1.),
    'HYBRID_DD400':dict(learned_risk=True,reward_mode='drawdown',penalty=4.),
    'HYBRID_DSR':dict(learned_risk=True,reward_mode='dsr',penalty=0.),
    'HYBRID_MIXED':dict(learned_risk=True,reward_mode='mixed',penalty=0.),
}
SEEDS=[7,19,43]


def prepare(root,parent):
    require(not root.exists(),'NEW_CAMPAIGN_REQUIRED')
    p=read(parent/'SPEC.private.json')
    sha=file_sha(parent/'SPEC.private.json')
    require(read(parent/'CLOSE.json')['state']=='COMPLETE'
            and read(parent/'IDENTITY.json')['spec_sha256']==sha,'PARENT_COMPLETE')
    require(p['software']==software(),'PARENT_SOFTWARE')
    spec=dict(schema='recipe_hybrid_queue.v2',inputs=p['inputs'],recipe_sha256=p['recipe_sha256'],
        engine_root=p['engine_root'],agent_root=str(ROOT),software=software(),design=DESIGN,
        arms=ARMS,seeds=SEEDS,units=[f'{a}_s{s}' for a in ARMS for s in SEEDS],qualification=QUALIFICATION,
        source_parent_sha256=sha,device='cuda',max_attempts=2,task_timeout_seconds=43200,
        code={'agent':source_pins(ROOT,['app','env_plugins','agent_plugins','tools']),
              'engine':p['code']['engine']},
        reward_parameters=dict(eta=.01,variance_floor=1e-6,warmup=32,mixed_weights=[.5,.5],
            scaling='fixed declared engineering units; no online normalization'),
        contrasts='new PPO algorithm control, not a one-variable SAC contrast')
    write(root/'SPEC.private.json',spec)
    write(root/'IDENTITY.json',dict(spec_sha256=file_sha(root/'SPEC.private.json'),created_at=now()))
    print(json.dumps({'state':'PREPARED','units':len(spec['units']),
                      'spec_sha256':file_sha(root/'SPEC.private.json')}))


def verify(root):
    p=read(root/'SPEC.private.json')
    require(file_sha(root/'SPEC.private.json')==read(root/'IDENTITY.json')['spec_sha256'],'SPEC_PIN')
    require(p['schema']=='recipe_hybrid_queue.v2' and p['qualification']==QUALIFICATION
            and p['design']==DESIGN and p['arms']==ARMS
            and p['seeds']==SEEDS and p['units']==[f'{a}_s{s}' for a in ARMS for s in SEEDS], 'DESIGN_PIN')
    require(p['software']==software(),'SOFTWARE_PIN')
    for role,base in [('agent',p['agent_root']),('engine',p['engine_root'])]:
        for name,digest in p['code'][role].items():
            require(file_sha(Path(base)/name)==digest,'CODE_PIN')
    return p


class Market:
    def __init__(self,spec,arm):
        from app.recipe_runtime import RecipeMarket
        self.base=RecipeMarket(spec)
        self.recipe=self.base.recipe
        self.arm=arm

    def env(self,period):
        from env_plugins.hybrid_orders import HybridOrderEnv
        native=self.base.env(period)
        return HybridOrderEnv(native.engine,native.inputs,gamma=DESIGN['gamma'],**ARMS[self.arm])


def make_model(env,recipe,seed,device,learned_risk,n_steps=512,n_epochs=5,policy_family='legacy'):
    from stable_baselines3 import PPO
    from agent_plugins.hybrid_orders import OrderPolicy,ElapsedRolloutBuffer
    from agent_plugins.qualified_orders import QualifiedOrderPolicy
    require(policy_family in ('legacy','grouped_sequence_v1'),'POLICY_FAMILY')
    policy=OrderPolicy if policy_family=='legacy' else QualifiedOrderPolicy
    return PPO(policy,env,policy_kwargs=dict(recipe=recipe,learned_risk=learned_risk),
        rollout_buffer_class=ElapsedRolloutBuffer,device=device,seed=seed,
        n_steps=n_steps,n_epochs=n_epochs,batch_size=min(DESIGN['batch_size'],n_steps),
        learning_rate=DESIGN['learning_rate'],gamma=DESIGN['gamma'],gae_lambda=DESIGN['gae_lambda'],
        ent_coef=DESIGN['ent_coef'],clip_range=DESIGN['clip_range'],vf_coef=DESIGN['vf_coef'],
        max_grad_norm=DESIGN['max_grad_norm'],verbose=0)


def callback(root,unit,start):
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback
    class Progress(BaseCallback):
        def _on_step(self):
            self.model.rollout_buffer.pending_duration=np.asarray(
                [i['elapsed_bars'] for i in self.locals['infos']],dtype=np.int32)
            if root is not None and self.num_timesteps%512==0:
                import torch
                write(root/'PROGRESS.json',dict(state='TRAINING',unit=unit,step=self.num_timesteps,
                    budget=DESIGN['total_timesteps'],updates=self.model._n_updates,heartbeat_at=now(),
                    elapsed_seconds=time.monotonic()-start,gpu_peak_bytes=torch.cuda.max_memory_allocated()))
            return True
    return Progress()


def configure():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def admit(root):
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    spec=verify(root)
    configure()
    rec=read(spec['inputs']['recipe'])
    require(torch.cuda.is_available(),'CUDA_REQUIRED')
    class Synthetic(gym.Env):
        observation_space=gym.spaces.Dict({
            'features':gym.spaces.Box(-np.inf,np.inf,(rec['tensor']['window']['observations'],rec['list']['n']),np.float32),
            'action_mask':gym.spaces.Box(0,1,(5,),np.float32)})
        action_space=gym.spaces.Box(np.array([0,0,0,0],np.float32),np.array([4,1,1,1],np.float32))
        def observation(self):
            return {'features':np.ones(self.observation_space['features'].shape,np.float32),
                    'action_mask':np.array([1,1,1,0,0],np.float32)}
        def reset(self,seed=None,options=None):
            super().reset(seed=seed);self.n=0
            return self.observation(),{}
        def step(self,a):
            require(a[0]==int(a[0]) and int(a[0]) in (0,1,2),'SYNTHETIC_COMMAND')
            self.n+=1
            return self.observation(),float(a[0]==1),self.n==20,False,{'elapsed_bars':1}
    results=[]
    for learned in (False,True):
        env=Synthetic()
        model=make_model(env,rec,7,'cuda',learned,n_steps=32,n_epochs=2,policy_family=DESIGN['policy_family'])
        model.learn(64,callback=callback(None,'synthetic',time.monotonic()))
        require(all(torch.isfinite(p).all() for p in model.policy.parameters()),'GPU_FINITE_WEIGHTS')
        path=root/f'GPU_ADMISSION_{int(learned)}.zip'
        model.save(path)
        loaded=PPO.load(path,device='cuda')
        obs,_=env.reset()
        require(np.array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0]),'GPU_RELOAD')
        results.append(dict(learned_risk=learned,updates=model._n_updates,policy=tensor_hash(model)))
        del model,loaded
    write(root/'GPU_ADMISSION.json',dict(state='PASS',spec_sha256=file_sha(root/'SPEC.private.json'),
          scope='ENGINEERING_SYNTHETIC_NOT_LEARNING_EFFICACY',models=results,finished_at=now()))


def worker(root,unit,attempt):
    import numpy as np
    from stable_baselines3 import PPO
    spec=verify(root);configure()
    require(unit in spec['units'],'UNIT')
    arm,seed=unit.rsplit('_s',1);seed=int(seed)
    folder=root/unit/f'attempt-{attempt}'
    folder.mkdir(parents=True,exist_ok=False)
    market=Market(spec,arm)
    env=market.env('TRAIN')
    qualification=root/'LEARNING_QUALIFICATION.json'
    require(qualification.exists() and read(qualification)['state']=='PASS'
            and read(qualification)['spec_sha256']==file_sha(root/'SPEC.private.json'),'LEARNING_QUALIFICATION_REQUIRED')
    require(qualify(root,spec,file_sha(root/'SPEC.private.json')),'LEARNING_QUALIFICATION_REQUIRED')
    model=make_model(env,market.recipe,seed,'cuda',ARMS[arm]['learned_risk'],policy_family=DESIGN['policy_family'])
    if arm!='CAT_FIXED':
        state=read(root/'STATUS.json')
        control=root/state['completed'][f'CAT_FIXED_s{seed}']
        authenticated(control,file_sha(root/'SPEC.private.json'))
        require(read(control.parent/'EFFECTIVE_MODEL.json')['initial_policy_tensor_sha256']==tensor_hash(model),
                'INITIAL_TENSOR_PARITY')
    write(folder/'EFFECTIVE_MODEL.json',dict(design=DESIGN,arm=ARMS[arm],seed=seed,
        policy_class=type(model.policy).__name__,parameters=sum(p.numel() for p in model.policy.parameters()),
        recipe_representation_overridden=True,qualification_sha256=file_sha(qualification),
        initial_policy_tensor_sha256=tensor_hash(model),reward_parameters=spec['reward_parameters'],
        storage_codec='exact integer categorical id plus three Beta parameters; no continuous binning'))
    start=time.monotonic()
    for step in range(25600,102401,25600):
        model.learn(25600,reset_num_timesteps=(step==25600),callback=callback(root,unit,start))
        require(model.num_timesteps==step,'STEP_COUNT')
        model.save(folder/f'model-{step}.zip')
        write(folder/f'TRAINING_{step}.json',clean(dict(step=step,elapsed_seconds=time.monotonic()-start,
              metrics=dict(model.logger.name_to_value),policy_tensor_sha256=tensor_hash(model))))
    path=folder/'model-102400.zip'
    with preserve_rng():
        obs,_=market.env('TRAIN').reset()
        loaded=PPO.load(path,device='cuda')
        require(np.array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0]),'RELOAD')
        del loaded
    reports=[]
    for period in ('TRAIN','INNER_A','OBS_2024'):
        write(root/'PROGRESS.json',dict(state='EVALUATING',unit=unit,period=period,
            step=102400,heartbeat_at=now(),elapsed_seconds=time.monotonic()-start))
        report=evaluate(model,market,period,folder)
        # Count the same scheduled business-day identifiers as the engine,
        # including every day without an entry, not only active trading days.
        calendar_env=market.env(period)
        business_days={calendar_env.engine.business_day(k) for k in range(calendar_env.engine.first+1,calendar_env.engine.last+1)}
        trades=read(folder/(period+'.trades.json'))
        counts=read(folder/(period+'.orders.json'))['counts']
        behavior=business_behavior(report,counts)
        report.update(behavior)
        write(folder/(period+'.behavior.json'),dict(**behavior,action_counts=counts,
            scope='necessary behavior coverage, not evidence of optimal timing'))
        report.update(arm=arm,seed=seed,step=102400,objective=ARMS[arm]['reward_mode'])
        report.update(average_entries_per_business_day=len(trades)/len(business_days),
                      scheduled_business_days=len(business_days))
        import pandas as pd
        daily=pd.read_csv(folder/(period+'.daily_equity.csv')).iloc[:,-1].to_numpy(float)
        annual_growth=(daily[-1]/report['initial_capital'])**(365/len(daily))-1 if (daily>0).all() else None
        dd=report['max_drawdown_pct']/100
        report['calmar_cagr']=annual_growth/dd if annual_growth is not None and dd>0 else None
        write(folder/(period+'.json'),report)
        reports.append(report)
    write(folder/'RESULT.json',dict(state='COMPLETE',unit=unit,arm=arm,seed=seed,
        spec_sha256=file_sha(root/'SPEC.private.json'),reports=reports,finished_at=now(),
        seconds=time.monotonic()-start,
        artifacts={p.name:file_sha(p) for p in folder.iterdir() if p.is_file()}))


def authenticated(path,sha):
    r=read(path)
    require(r['state']=='COMPLETE' and r['spec_sha256']==sha and len(r['reports'])==3,'RESULT_IDENTITY')
    require({v['period'] for v in r['reports']}=={'TRAIN','INNER_A','OBS_2024'},'RESULT_PERIODS')
    require({'model-102400.zip','EFFECTIVE_MODEL.json','TRAIN.json','INNER_A.json','OBS_2024.json'}<=set(r['artifacts']),
            'RESULT_REQUIRED_ARTIFACTS')
    for name,digest in r['artifacts'].items():
        p=(path.parent/name).resolve()
        require(p.parent==path.parent.resolve() and file_sha(p)==digest,'RESULT_ARTIFACT')
    for report in r['reports']:
        require(report==read(path.parent/(report['period']+'.json')),'NATIVE_REPORT_BINDING')
    return r


def business_behavior(report,counts):
    reasons=[]
    if report.get('n_long',0)==0: reasons.append('NO_LONG_TRADES')
    if report.get('n_short',0)==0: reasons.append('NO_SHORT_TRADES')
    for name in ('NOP','CLOSE_LONG','CLOSE_SHORT','HOLD_LONG','HOLD_SHORT'):
        if counts.get(name,0)==0: reasons.append('NO_'+name)
    return dict(business_behavior_eligible=not reasons,behavior_failures=reasons,
                voluntary_close_long=counts.get('CLOSE_LONG',0),
                voluntary_close_short=counts.get('CLOSE_SHORT',0))


def qualify(root,spec,sha):
    """Finite cases, resumable after interruption; failures never start a fit."""
    receipt=root/'LEARNING_QUALIFICATION.json'
    if receipt.exists():
        r=read(receipt)
        require(r['spec_sha256']==sha,'QUALIFICATION_IDENTITY')
        require([(c['seed'],c['lag']) for c in r['cases']]==
                [(c['seed'],c['lag']) for c in spec['qualification']['cases']],'QUALIFICATION_CASES')
        for name,digest in r['artifacts'].items():
            require(Path(name).name==name,'QUALIFICATION_PATH')
            require(file_sha(root/name)==digest,'QUALIFICATION_ARTIFACT')
        from app.order_qualification import passes
        require(r['state']==('PASS' if all(passes(c['rows']) and c['reload_exact'] for c in r['cases'])
                             else 'FAILED'),'QUALIFICATION_VERDICT')
        return r['state']=='PASS'
    rows=[]
    for case in spec['qualification']['cases']:
        name=f"qualification-s{case['seed']}-lag{case['lag']}"
        path=root/(name+'.json')
        if not path.exists():
            write(root/'PROGRESS.json',dict(state='QUALIFYING',unit=name,heartbeat_at=now(),
                                          financial_fits=0))
            with (root/(name+'.log')).open('a') as log:
                child=subprocess.run([sys.executable,'-m','tools.qualify_order_routes',
                    '--recipe',spec['inputs']['recipe'],'--output',str(path),'--device','cuda',
                    '--seed',str(case['seed']),'--lag',str(case['lag']),
                    '--steps',str(spec['qualification']['steps'])],cwd=ROOT,stdout=log,
                    stderr=subprocess.STDOUT,timeout=7200)
            require(child.returncode in (0,2) and path.exists(),'QUALIFICATION_PROCESS_FAILED')
        r=read(path)
        require(r['recipe_sha256']==spec['recipe_sha256'] and r['seed']==case['seed']
                and r['lag']==case['lag'] and r['steps']==spec['qualification']['steps']
                and r['device']=='cuda','QUALIFICATION_CASE_IDENTITY')
        require(file_sha(path.with_suffix('.zip'))==r['model_sha256'],'QUALIFICATION_MODEL')
        from app.order_qualification import passes
        require(r['passed']==(passes(r['rows']) and r['reload_exact']),'QUALIFICATION_VERDICT')
        rows.append(r)
    write(receipt,dict(state='PASS' if all(r['passed'] for r in rows) else 'FAILED',
        spec_sha256=sha,scope='SYNTHETIC_NOT_FINANCIAL',cases=rows,finished_at=now(),
        artifacts={p.name:file_sha(p) for p in root.glob('qualification-*') if p.suffix in ('.json','.zip')}))
    return read(receipt)['state']=='PASS'


def run(root):
    spec=verify(root);sha=file_sha(root/'SPEC.private.json')
    with (root/'QUEUE.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return 0
        state=read(root/'STATUS.json') if (root/'STATUS.json').exists() else dict(
            spec_sha256=sha,state='PREPARED',attempts={},completed={})
        require(state['spec_sha256']==sha,'QUEUE_IDENTITY')
        if spec.get('qualification'):
            if not admission(root):
                state.update(state='WAIT_RESOURCES',updated_at=now());write(root/'STATUS.json',state);return 75
            state.update(state='QUALIFYING',updated_at=now());write(root/'STATUS.json',state)
            if not qualify(root,spec,sha):
                state.update(state='QUALIFICATION_FAILED',finished_at=now())
                write(root/'STATUS.json',state);write(root/'CLOSE.json',state);return 0
        for unit in spec['units']:
            if unit in state['completed']:
                authenticated(root/state['completed'][unit],sha);continue
            attempts=state['attempts'].setdefault(unit,[])
            for row in attempts:
                result=root/unit/f'attempt-{row["attempt"]}'/'RESULT.json'
                if result.exists():
                    authenticated(result,sha)
                    state['completed'][unit]=str(result.relative_to(root));break
                if row['state']=='RUNNING': row['state']='INTERRUPTED'
            if unit in state['completed']: continue
            arm,seed=unit.rsplit('_s',1)
            if arm!='CAT_FIXED' and f'CAT_FIXED_s{seed}' not in state['completed']:
                state.setdefault('dependency_failures',{})[unit]='CATEGORICAL_CONTROL_MISSING'
                write(root/'STATUS.json',state)
                continue
            while len(attempts)<2:
                if not admission(root):
                    state.update(state='WAIT_RESOURCES',unit=unit,updated_at=now())
                    write(root/'STATUS.json',state);return 75
                if not (root/'GPU_ADMISSION.json').exists():
                    with (root/'ADMISSION.log').open('a') as log:
                        result=subprocess.run([sys.executable,'-m','tools.recipe_hybrid_queue','admit','--root',str(root)],
                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                    require(result.returncode==0,'GPU_ADMISSION_FAILED')
                require(read(root/'GPU_ADMISSION.json')['spec_sha256']==sha
                        and read(root/'GPU_ADMISSION.json')['state']=='PASS','ADMISSION_PIN')
                number=len(attempts)+1
                row=dict(attempt=number,state='RUNNING',started_at=now());attempts.append(row)
                state.update(state='RUNNING',unit=unit,updated_at=now());write(root/'STATUS.json',state)
                with (root/f'{unit}-{number}.log').open('x') as log:
                    try:
                        child=subprocess.run([sys.executable,'-m','tools.recipe_hybrid_queue','worker','--root',str(root),
                            '--unit',unit,'--attempt',str(number)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                            timeout=spec['task_timeout_seconds'])
                        rc=child.returncode
                    except subprocess.TimeoutExpired: rc=124
                row.update(state='FAILED',returncode=rc,finished_at=now())
                path=root/unit/f'attempt-{number}'/'RESULT.json'
                if rc==0 and path.exists():
                    authenticated(path,sha);row['state']='COMPLETE';state['completed'][unit]=str(path.relative_to(root))
                write(root/'STATUS.json',state)
                if row['state']=='COMPLETE': break
                time.sleep(30)
        rows=[]
        for path in state['completed'].values(): rows.extend(authenticated(root/path,sha)['reports'])
        from app.hybrid_fitness import compare
        write(root/'COMPARISON.json',clean(compare(rows)))
        state.update(state='COMPLETE' if len(state['completed'])==len(spec['units']) else 'COMPLETE_WITH_FAILURES',finished_at=now())
        write(root/'STATUS.json',state);write(root/'CLOSE.json',state)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run','worker','admit','status'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path)
    p.add_argument('--unit');p.add_argument('--attempt',type=int)
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='prepare': prepare(root,a.parent.resolve())
    elif a.mode=='run': return run(root)
    elif a.mode=='worker': worker(root,a.unit,a.attempt)
    elif a.mode=='admit': admit(root)
    else: print(json.dumps({n:read(root/n) for n in ('STATUS.json','PROGRESS.json','CLOSE.json') if (root/n).exists()},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
