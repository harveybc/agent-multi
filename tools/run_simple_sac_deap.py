"""Resumable paired-seed DEAP for the selected native three-signal SAC."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

from deap import base, creator, tools as dt
import numpy as np
import torch

from app.simple_sac_search import GENES, decode, baseline_genome, make_model, assert_consumed
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from app.recipe_runtime import RecipeMarket, attach_engine
from app.three_signal_learning import Patience, synthetic_env, environment, emit, evaluate
from tools.run_discrete_curve_probe import read, write, now, pins, configure
from tools.run_three_signal_baseline import (train_loop, monitor, synthetic_controls,
    assess_synthetic, validate_monitor_scope)

ROOT = Path(__file__).resolve().parents[1]


def check_spec(root):
    s = read(root/'SPEC.json')
    if s['identity'] != digest({k:v for k,v in s.items() if k!='identity'}):
        raise ValueError('SEARCH_SPEC_IDENTITY')
    for name,sha in s['sources'].items():
        if file_sha(ROOT/name)!=sha: raise ValueError('SEARCH_SOURCE_PIN:'+name)
    for key,path in s['parent']['inputs'].items():
        if file_sha(path)!=s['parent']['input_sha256'][key]: raise ValueError('SEARCH_DATA_PIN')
    for name,sha in s['parent']['native'].items():
        if file_sha(Path(s['parent']['engine_root'])/name)!=sha: raise ValueError('SEARCH_ENGINE_PIN')
    for name,sha in s['evidence'].items():
        if file_sha(root/name)!=sha: raise ValueError('SEARCH_EVIDENCE_PIN')
    return s


def prepare(args):
    if args.root.exists(): raise ValueError('NEW_SEARCH_ROOT_REQUIRED')
    parent = read(args.parent/'SPEC.private.json')
    if parent['identity']!=digest({k:v for k,v in parent.items() if k!='identity'}):
        raise ValueError('PARENT_IDENTITY')
    baseline = read(args.baseline/'SPEC.private.json')
    risk = read(args.baseline/'RISK.json')
    if file_sha(args.baseline/'RISK.json')!=baseline['risk_sha256']: raise ValueError('RISK_PIN')
    d = copy.deepcopy(baseline['design'])
    d.update(seeds=[7,19],steps=1048576,patience=10)
    recipe = read(parent['inputs']['recipe']); validate_monitor_scope(d,recipe)
    wavelet = read(args.wavelet_audit)
    if not (wavelet['passed'] and wavelet['store_sha256']==recipe['store']['sha256']
            and wavelet['recipe_sha256']==parent['recipe_sha256']):
        raise ValueError('WAVELET_AUDIT_PIN')
    attach_engine(parent['engine_root'])
    controls = synthetic_controls(recipe,risk,d)
    if not controls['passed']: raise ValueError('NATIVE_POSITIVE_CONTROLS')
    args.root.mkdir(parents=True)
    for name,value in [('RISK.json',risk),('WAVELET_AUDIT.json',wavelet),('CONTROLS.json',controls)]:
        write(args.root/name,value)
    s = dict(schema='simple_sac_deap.v1',parent=parent,design=d,created_at=now(),
        search=dict(population=20,max_generations=2000,patience=10,min_delta=1e-5,
                    seed=20261102,elite=2,tournament=3,crossover=.7,mutation=.4,
                    genes=GENES,fitness='paired mean of selected combined TRAIN/INNER_A monitor'),
        sources=pins(ROOT,['app','agent_plugins','env_plugins','tools']),
        evidence={n:file_sha(args.root/n) for n in ('RISK.json','WAVELET_AUDIT.json','CONTROLS.json')},
        phase2_auto_launch=False)
    s['identity']=digest(s); write(args.root/'SPEC.json',s)
    check_spec(args.root)
    print(json.dumps(dict(identity=s['identity'],population=20,max_generations=2000)),flush=True)


def verified(path, identity, config, seed):
    if not path.exists(): return None
    r=read(path)
    if r['identity']!=identity or r['config']!=config or r['seed']!=seed:
        raise ValueError('CANDIDATE_RECEIPT_IDENTITY')
    for name,sha in r['artifacts'].items():
        p=(path.parent/name).resolve()
        if not p.is_relative_to(path.parent.resolve()) or file_sha(p)!=sha:
            raise ValueError('CANDIDATE_ARTIFACT_PIN')
    return r


def finish(out,s,config,seed,**fields):
    row=dict(identity=s['identity'],config=config,seed=seed,finished_at=now(),**fields)
    row['artifacts']={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*')
        if p.is_file() and p.name not in ('RESULT.json','PROGRESS.json') and not p.name.endswith('.tmp')}
    write(out/'RESULT.json',row)


def worker(args):
    s=check_spec(args.root); task=read(args.task); c=decode(task['genome']); seed=task['seed']
    if seed not in s['design']['seeds']: raise ValueError('UNKNOWN_SEARCH_SEED')
    key=digest(dict(config=c,seed=seed)); folder=args.root/'runs'/key
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for attempt in (1,2):
            out=folder/f'attempt{attempt}'
            if verified(out/'RESULT.json',s['identity'],c,seed): return
            if out.exists(): continue
            break
        else: raise ValueError('CANDIDATE_ATTEMPTS_EXHAUSTED')
        out.mkdir()
        while True:
            mem={line.split(':')[0]:int(line.split()[1]) for line in
                 Path('/proc/meminfo').read_text().splitlines() if ':' in line}
            gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                '--format=csv,noheader,nounits','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
            if mem['MemAvailable']>5*1024**2 and gpu>3072: break
            write(out/'PROGRESS.json',dict(stage='WAIT_RESOURCES',time=now()))
            time.sleep(60)
        configure(seed)
        if not torch.cuda.is_available(): raise ValueError('CUDA_REQUIRED')
        parent=s['parent']; attach_engine(parent['engine_root'])
        recipe=read(parent['inputs']['recipe']); risk=read(args.root/'RISK.json')
        d=dict(s['design'],**{k:c[k] for k in ('gamma','learning_rate','entropy_fraction','batch_size','update_every')})
        model=make_model(recipe,c,seed,'cuda')
        write(out/'EFFECTIVE.json',assert_consumed(model,c,d))
        controls=read(args.root/'CONTROLS.json')
        qualification_dir=out/'qualification'; qualification_dir.mkdir()
        def qualify(step,m):
            r=assess_synthetic(m,recipe,risk,d,controls)
            r['stop']=r['passed'] and step>=d['qualification_minimum']
            return r
        factory=lambda episode:synthetic_env(recipe,risk,100000+seed*1000+episode,
            1 if episode%2==0 else -1,d['gamma'],flat=episode%5==4)[0]
        try:
            _,history=train_loop(model,factory,d,seed,qualification_dir,qualify,True)
        except ValueError as exc:
            if str(exc)!='NONFINITE_SAC_LOSS': raise
            finish(out,s,c,seed,eligible=False,status='NONFINITE_SAC_LOSS',stage='qualification'); return
        if not history[-1]['report']['passed']:
            finish(out,s,c,seed,eligible=False,status='SYNTHETIC_NOT_LEARNED'); return
        del model; torch.cuda.empty_cache(); configure(seed)
        model=make_model(recipe,c,seed,'cuda'); market=RecipeMarket(parent)
        assert_consumed(model,c,d)
        stopping=Patience(d['minimum_steps'],10,d['min_delta']); best={}
        def assess(step,m):
            r=monitor(m,market,risk,d)
            v=stopping.observe(r['score'],r['eligible'],step)
            if v['improved_any']: m.save(out/'best_unrestricted.pt')
            if v['improved']:
                m.save(out/'best.pt'); best.update(step=step,score=r['score'],report=r)
                write(out/'BEST.json',best)
            return dict(r,**v)
        factory=lambda episode:environment(market,risk,d['horizon'],
            recipe['split']['periods']['TRAIN'],d['gamma'])
        try:
            train_loop(model,factory,d,seed,out,assess)
        except ValueError as exc:
            if str(exc)!='NONFINITE_SAC_LOSS': raise
            finish(out,s,c,seed,eligible=False,status='NONFINITE_SAC_LOSS',stage='financial'); return
        if not best:
            finish(out,s,c,seed,eligible=False,status='NO_ACTIVE_CHECKPOINT'); return
        model.load(out/'best.pt')
        logits,_=emit(model,market.inputs)
        with torch.no_grad():
            sample=torch.tensor(market.inputs.windows[np.flatnonzero(market.inputs.valid)[:8]],device='cuda')
            before=model.surfaces(sample).clone(); model.load(out/'best.pt')
            if not torch.equal(before,model.surfaces(sample)): raise ValueError('RELOAD_CHANGED')
        metrics={p:evaluate(environment(market,risk,d['horizon'],recipe['split']['periods'][p],d['gamma']),
            logits,out/p,diagnose=True) for p in ('TRAIN','INNER_A')}
        finish(out,s,c,seed,eligible=True,status='COMPLETE',score=best['score'],
               selected_step=best['step'],metrics=metrics,reload_exact=True)


def individual_type():
    if not hasattr(creator,'SimpleSACFitness'): creator.create('SimpleSACFitness',base.Fitness,weights=(1.,))
    if not hasattr(creator,'SimpleSACGenome'): creator.create('SimpleSACGenome',list,fitness=creator.SimpleSACFitness)
    return creator.SimpleSACGenome


def next_population(pop, settings):
    cls=individual_type()
    offspring=[cls(x) for x in dt.selBest(pop,settings['elite'])]
    while len(offspring)<settings['population']:
        a,b=[cls(x) for x in dt.selTournament(pop,2,tournsize=settings['tournament'])]
        if random.random()<settings['crossover']: dt.cxUniform(a,b,indpb=.5)
        for child in (a,b):
            if random.random()<settings['mutation']:
                dt.mutGaussian(child,mu=0,sigma=.15,indpb=1/len(GENES))
            child[:]=[min(1.,max(0.,v)) for v in child]
            offspring.append(child)
            if len(offspring)==settings['population']: break
    return offspring


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value,list) else value


def dispatch(node, task, root, s):
    c=decode(task['genome']); seed=task['seed']; key=digest(dict(config=c,seed=seed))
    local=root/'collected'/node['role']/'runs'/key
    for cached in sorted((root/'collected').glob('*/runs/'+key+'/attempt*/RESULT.json')):
        r=verified(cached,s['identity'],c,seed)
        if r: return r
    taskfile=root/'tasks'/(key+'.json'); write(taskfile,task)
    remote_root=Path(node['root']); remote_task=remote_root/'tasks'/taskfile.name
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',node['host']] if node.get('host') else []
    def command(tokens,**kw):
        cmd=ssh+[shlex.join(tokens)] if ssh else tokens
        return subprocess.run(cmd,check=True,**kw)
    command(['mkdir','-p',str(remote_root/'tasks')],timeout=40)
    if ssh:
        subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=20',
            str(taskfile),node['host']+':'+str(remote_task)],check=True,timeout=60)
    elif taskfile!=remote_task:
        import shutil
        shutil.copy2(taskfile,remote_task)
    tokens=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=6G','-p','MemorySwapMax=1G',
            'env','PYTHONPATH='+node['code'],'CUDA_VISIBLE_DEVICES='+str(node['gpu']),
            'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',node['python'],
            '-m','tools.run_simple_sac_deap','worker','--root',str(remote_root),'--task',str(remote_task)]
    for attempt in (1,2):
        write(root/('WORKER_'+node['role']+'.json'),dict(task=key,seed=seed,config=c,state='RUNNING',time=now()))
        with (root/'tasks'/(key+'.log')).open('a') as log:
            try:
                command(tokens,stdout=log,stderr=subprocess.STDOUT,timeout=259200)
            except subprocess.CalledProcessError as exc:
                if exc.returncode==78: raise ValueError('WORKER_CONTRACT_FAILURE:'+key) from exc
                if attempt==2: raise
                time.sleep(60); continue
        local.mkdir(parents=True,exist_ok=True)
        source=(node['host']+':' if ssh else '')+str(remote_root/'runs'/key)+'/'
        subprocess.run(['rsync','-a','--exclude=worker.lock','-e','ssh -o BatchMode=yes -o ConnectTimeout=20',
            source,str(local)+'/'],check=True,timeout=300)
        for a in (1,2):
            r=verified(local/f'attempt{a}'/'RESULT.json',s['identity'],c,seed)
            if r:
                write(root/('WORKER_'+node['role']+'.json'),dict(task=key,state='COMPLETE',time=now()))
                return r
        raise ValueError('MISSING_CANDIDATE_RECEIPT')


def run(args):
    s=check_spec(args.root); settings=s['search']; fleet=read(args.fleet)['nodes']
    if len({n['role'] for n in fleet})!=len(fleet) or not fleet: raise ValueError('FLEET_ROLES')
    with (args.root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'CLOSE.json').exists():
            if read(args.root/'CLOSE.json')['identity']!=s['identity']:
                raise ValueError('CLOSE_IDENTITY')
            return
        cls=individual_type(); statefile=args.root/'EVOLUTION.json'
        if statefile.exists():
            state=read(statefile)
            if state['identity']!=s['identity']: raise ValueError('EVOLUTION_IDENTITY')
            random.setstate(tuples(state['random_state']))
        else:
            random.seed(settings['seed'])
            state=dict(identity=s['identity'],generation=0,bad=0,best=None,
                population=[baseline_genome()]+[[random.random() for _ in GENES] for _ in range(19)],
                random_state=random.getstate())
            write(statefile,state)
        while state['generation']<settings['max_generations']:
            pop=[cls(x) for x in state['population']]
            tasks={}
            for x in pop:
                for seed in s['design']['seeds']:
                    key=digest(dict(config=decode(x),seed=seed))
                    tasks[key]=dict(genome=list(x),seed=seed)
            write(args.root/'STATUS.json',dict(state='RUNNING',generation=state['generation'],
                unique_seed_tasks=len(tasks),best=state['best'],bad_generations=state['bad'],time=now()))
            # Each worker consumes its own sequential queue; no competing GPU jobs.
            task_items=list(tasks.items()); partitions=[task_items[i::len(fleet)] for i in range(len(fleet))]
            def consume(pair):
                node,items=pair
                return {key:dispatch(node,task,args.root,s) for key,task in items}
            with ThreadPoolExecutor(max_workers=len(fleet)) as executor:
                rows={k:v for part in executor.map(consume,zip(fleet,partitions)) for k,v in part.items()}
            evaluated=[]
            for x in pop:
                c=decode(x); results=[rows[digest(dict(config=c,seed=seed))] for seed in s['design']['seeds']]
                eligible=all(r['eligible'] for r in results)
                score=float(np.mean([r['score'] for r in results])) if eligible else None
                x.fitness.values=(score if eligible else -1e30,)
                evaluated.append(dict(config=c,genome=list(x),eligible=eligible,score=score,
                    seeds=[dict(seed=r['seed'],status=r['status'],metrics=r.get('metrics')) for r in results]))
            best=max((r for r in evaluated if r['eligible']),key=lambda r:r['score'],default=None)
            improved=best is not None and (state['best'] is None or best['score']>state['best']['score']+settings['min_delta'])
            if improved: state['best']=best; state['bad']=0
            else: state['bad']+=1
            write(args.root/'generations'/f"{state['generation']:04d}.json",dict(
                identity=s['identity'],generation=state['generation'],results=evaluated,time=now(),improved=improved))
            state['generation']+=1
            if state['bad']>=settings['patience'] or state['generation']>=settings['max_generations']:
                write(args.root/'CLOSE.json',dict(identity=s['identity'],generations=state['generation'],
                    best=state['best'],reason='combined_patience' if state['bad']>=settings['patience'] else 'generation_cap',
                    phase2_launched=False,time=now()))
                write(args.root/'STATUS.json',dict(state='COMPLETE',time=now(),phase2_launched=False))
                return
            state['population']=[list(x) for x in next_population(pop,settings)]
            state['random_state']=random.getstate(); write(statefile,state)


def smoke(args):
    from agent_plugins.discrete_curve_sac import tensor_observation
    from tools.run_discrete_curve_probe import transition
    from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
    s=check_spec(args.root); configure(7)
    market=RecipeMarket(s['parent']); risk=read(args.root/'RISK.json'); d=s['design']
    out=args.root/'SMOKE'; out.mkdir(exist_ok=False)
    rows=[]
    for label,g in [('baseline',baseline_genome()),('upper', [1.]*len(GENES))]:
        c=decode(g); design=dict(d,**{k:c[k] for k in ('gamma','learning_rate','entropy_fraction','batch_size','update_every')})
        model=make_model(market.recipe,c,7,'cpu'); proof=assert_consumed(model,c,design)
        env=environment(market,risk,d['horizon'],market.recipe['split']['periods']['TRAIN'],c['gamma'])
        replay=TensorDictReplayBuffer(storage=LazyTensorStorage(512),batch_size=c['batch_size'])
        obs=env.reset()
        for i in range(512):
            nxt,reward,done,duration=env.step(i%3)
            replay.extend(transition(obs,i%3,nxt,reward,done,duration))
            obs=env.reset() if done else nxt
        losses=[model.update(replay.sample()) for _ in range(2)]
        sample=tensor_observation(obs)['features']
        model.save(out/(label+'.pt'))
        before=model.surfaces(sample).detach().clone(); model.load(out/(label+'.pt'))
        if not torch.equal(before,model.surfaces(sample)): raise ValueError('SMOKE_RELOAD')
        before_rng=torch.get_rng_state().clone(); count=len(replay)
        report=monitor(model,market,risk,design)
        if not torch.equal(before_rng,torch.get_rng_state()) or len(replay)!=count:
            raise ValueError('MONITOR_MUTATED_TRAINING')
        rows.append(dict(label=label,proof=proof,losses=losses,monitor=report,transitions=count))
    write(out/'RESULT.json',dict(passed=True,scope='engineering only; not DEAP fitness',rows=rows))
    print(json.dumps(dict(passed=True,configurations=2,updates=4,full_accounts=12)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','worker','run','smoke'])
    p.add_argument('--root',type=Path,required=True)
    for name in ('parent','baseline','wavelet-audit','task','fleet'):
        p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    try: globals()[args.command](args)
    except (ValueError,AssertionError) as exc:
        print('CONTRACT_FAILURE:'+str(exc),flush=True); raise SystemExit(78) from exc


if __name__=='__main__': main()
