"""Sequential native SAC qualification, early-stopped financial fit and sealed reports."""
import argparse
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from agent_plugins.three_signal_sac import ThreeSignalSAC
from agent_plugins.discrete_curve_sac import tensor_observation
from app.three_signal_learning import Patience, synthetic_env, environment, emit, evaluate
from app.recipe_runtime import RecipeMarket, attach_engine
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from tools.run_discrete_curve_probe import authenticate as parent_authenticate, read, write, now, configure, transition

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/run_three_signal_baseline.py','app/three_signal_learning.py',
           'agent_plugins/three_signal_sac.py','env_plugins/three_signal_native.py']


def authenticate(root):
    spec = read(root/'SPEC.private.json')
    if spec['identity'] != digest({k:v for k,v in spec.items() if k!='identity'}):
        raise ValueError('BASELINE_SPEC_IDENTITY')
    parent = parent_authenticate(Path(spec['parent_root']))
    if parent['identity'] != spec['parent_identity']:
        raise ValueError('BASELINE_PARENT_IDENTITY')
    for name,sha in spec['sources'].items():
        if file_sha(ROOT/name) != sha: raise ValueError('BASELINE_SOURCE_CHANGED:'+name)
    if file_sha(root/'RISK.json') != spec['risk_sha256']: raise ValueError('BASELINE_RISK_CHANGED')
    if file_sha(root/'CONTROL_PREFLIGHT.json') != spec['control_sha256']: raise ValueError('BASELINE_CONTROL_CHANGED')
    return spec,parent


def prepare(args):
    parent = parent_authenticate(args.parent)
    design = read(args.config)
    if args.root.exists(): raise ValueError('NEW_BASELINE_ROOT_REQUIRED')
    if (design['schema']!='three_signal_sac.v1' or design['horizon']!=6 or design['seeds']!=[7,19,43]
            or design['minimum_steps']>=design['steps'] or design['patience']<1):
        raise ValueError('BASELINE_DESIGN')
    recipe = read(parent['inputs']['recipe'])
    validate_monitor_scope(design,recipe)
    if (recipe['list']['n']!=120 or len(recipe['representation']['resolved']['names'])!=31
            or recipe['tensor']['window']['observations']!=13
            or recipe['tensor']['window']['stride_rows']!=2
            or recipe['scaler']['fit_period']!=['2020-01-01','2023-07-01']):
        raise ValueError('PHASE1_RECIPE')
    risk = copy.deepcopy(read(parent['risk'])); risk['baseline'].update(design['risk_overrides'])
    args.root.mkdir(parents=True)
    write(args.root/'RISK.json',risk)
    spec = dict(schema='three_signal_campaign.v1',parent_root=str(args.parent.resolve()),
        parent_identity=parent['identity'],design=design,risk_sha256=file_sha(args.root/'RISK.json'),
        sources={n:file_sha(ROOT/n) for n in SOURCES},created_at=now())
    spec['identity'] = digest(spec)
    attach_engine(parent['engine_root'])
    controls = synthetic_controls(recipe,risk,design)
    write(args.root/'CONTROL_PREFLIGHT.json',controls)
    if not controls['passed']: raise ValueError('DETERMINISTIC_CONTROL_FAILED')
    spec['control_sha256']=file_sha(args.root/'CONTROL_PREFLIGHT.json')
    spec['identity']=digest({k:v for k,v in spec.items() if k!='identity'})
    write(args.root/'SPEC.private.json',spec)
    print(json.dumps(dict(identity=spec['identity'],controls=controls)),flush=True)


def known_logits(signal):
    x = np.zeros((len(signal),3),np.float32)
    x[np.arange(len(signal)),np.where(signal>0,1,np.where(signal<0,2,0))] = 10.
    return x


CONFIRMATION = [(91001,1),(91002,-1),(91003,1)]


def validate_monitor_scope(d,recipe):
    integer_keys=['steps','minimum_steps','warmup','update_every','batch_size','replay_size',
                  'evaluation_every','patience','qualification_steps','qualification_check_every','qualification_minimum']
    if any(type(d[k]) is not int or d[k]<=0 for k in integer_keys): raise ValueError('BASELINE_INTEGER_CONFIG')
    if (not all(np.isfinite(d[k]) and d[k]>0 for k in ('gamma','alpha','learning_rate','entropy_fraction','min_delta'))
            or not 0<d['gamma']<=1 or not 0<d['entropy_fraction']<1
            or d['minimum_steps']>=d['steps'] or d['warmup']>=d['minimum_steps']
            or d['steps']%d['evaluation_every'] or d['qualification_steps']%d['qualification_check_every']):
        raise ValueError('BASELINE_NUMERIC_CONFIG')
    for key,period in [('train_blocks','TRAIN'),('validation_blocks','INNER_A')]:
        lo,hi=[pd.Timestamp(t,tz='UTC') for t in recipe['split']['periods'][period]]
        previous=None
        if len(d[key])!=3: raise ValueError('MONITOR_BLOCK_COUNT')
        for a,b in d[key]:
            a,b=pd.Timestamp(a,tz='UTC'),pd.Timestamp(b,tz='UTC')
            if not (lo<=a<b<=hi) or b-a!=pd.Timedelta(weeks=8) or (previous is not None and a<previous):
                raise ValueError('MONITOR_SCOPE_OR_OVERLAP')
            previous=b


def synthetic_controls(recipe,risk,d):
    positive = []
    for seed,mirror in CONFIRMATION:
        env,s = synthetic_env(recipe,risk,seed,mirror,d['gamma'])
        positive.append(evaluate(env,known_logits(s)))
    flat,_ = synthetic_env(recipe,risk,91004,1,d['gamma'],flat=True)
    nop = evaluate(flat,known_logits(np.zeros(len(flat.engine.v))))
    forced = evaluate(flat,known_logits(np.ones(len(flat.engine.v))))
    passed = all(m['net_profit_usd']>0 and m['buy']['trades']>0 and m['sell']['trades']>0
                 and all(m['early_by_side'][str(s)]>0 for s in (-1,1)) for m in positive)
    return dict(passed=passed and nop['trades']==0 and forced['net_profit_usd']<0,
                known=positive,flat_nop=nop,flat_forced=forced)


def make_model(recipe,d,seed,device):
    return ThreeSignalSAC(recipe,device,seed,d['gamma'],d['alpha'],d['learning_rate'],d['entropy_fraction'])


def assess_synthetic(model,recipe,risk,d,controls):
    cases=[]
    for seed,mirror in CONFIRMATION:
        env,_ = synthetic_env(recipe,risk,seed,mirror,d['gamma'])
        logits,_ = emit(model,env.inputs)
        cases.append(evaluate(env,logits))
    flat,_ = synthetic_env(recipe,risk,91004,1,d['gamma'],flat=True)
    x,_ = emit(model,flat.inputs); fm=evaluate(flat,x)
    ratio = sum(c['net_profit_usd'] for c in cases)/sum(c['net_profit_usd'] for c in controls['known'])
    passed = (all(c['net_profit_usd']>0 and c['buy']['trades'] and c['sell']['trades']
                  and all(c['early_by_side'][str(s)] for s in (-1,1)) for c in cases)
              and ratio>=d['qualification_profit_fraction'] and fm['trades']==0)
    return dict(passed=bool(passed),cases=cases,flat=fm,profit_fraction=ratio,
                scope='sequential native economic learning on causal synthetic regime; development qualification')


def train_loop(model, env_factory, d, seed, output, assess, synthetic=False):
    rng=np.random.default_rng(seed)
    replay=TensorDictReplayBuffer(storage=LazyTensorStorage(d['replay_size'],device='cpu'),batch_size=d['batch_size'])
    env=env_factory(0); obs=env.reset()
    episodes=updates=native_bars=0; history=[]; last_loss=None
    maximum=d['qualification_steps'] if synthetic else d['steps']
    every=d['qualification_check_every'] if synthetic else d['evaluation_every']
    start=time.monotonic()
    rewards=[]
    for step in range(1,maximum+1):
        if step<=d['warmup']:
            action=int(rng.integers(0,3))
        else:
            with torch.no_grad(): action=int(model.distribution(tensor_observation(obs,model.device)).sample()[0])
        nxt,reward,done,duration=env.step(action)
        replay.extend(transition(obs,action,nxt,reward,done,duration))
        native_bars+=duration; rewards.append(reward)
        if done:
            episodes+=1; env=env_factory(episodes); obs=env.reset()
        else: obs=nxt
        if step>=d['warmup'] and step%d['update_every']==0:
            last_loss=model.update(replay.sample().to(model.device)); updates+=1
        progress=dict(stage='SYNTHETIC_SEQUENTIAL' if synthetic else 'FINANCIAL_TRAIN',
            step=step,maximum=maximum,updates=updates,native_bars=native_bars,completed_episodes=episodes,
            seconds=time.monotonic()-start,replay_transitions=len(replay),loss=last_loss)
        if step%256==0: write(output/'PROGRESS.json',progress)
        if step%every==0:
            # Assessment is deterministic, uses separate native accounts and no replay.
            torch_state=torch.get_rng_state(); cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            report=assess(step,model)
            if not torch.equal(torch_state,torch.get_rng_state()): raise ValueError('EVALUATION_MUTATED_TRAIN_RNG')
            if cuda_state and any(not torch.equal(a,b) for a,b in zip(cuda_state,torch.cuda.get_rng_state_all())):
                raise ValueError('EVALUATION_MUTATED_CUDA_RNG')
            progress.update(report=report,reward_mean=float(np.mean(rewards)),reward_std=float(np.std(rewards)))
            rewards=[]; history.append(progress); write(output/'HISTORY.json',history)
            if report.get('stop'):
                break
    model.save(output/'last.pt')
    result=dict(progress,stop_reason='qualification_passed' if synthetic and history and history[-1]['report'].get('stop')
                else history[-1]['report']['stop_reason'] if not synthetic and history and history[-1]['report'].get('stop') else 'step_budget',
                training_seconds=time.monotonic()-start,history_checks=len(history),mid_fit_resume_supported=False)
    write(output/'TRAINING.json',result)
    return result,history


def monitor(model,market,risk,d):
    logits,_=emit(model,market.inputs)
    halves=[]
    for name in ('train_blocks','validation_blocks'):
        rows=[evaluate(environment(market,risk,d['horizon'],interval,d['gamma']),logits) for interval in d[name]]
        halves.append(rows)
    score=.5*float(np.mean([r['return_per_week'] for r in halves[0]]))+.5*float(np.mean([r['return_per_week'] for r in halves[1]]))
    eligible=all(np.mean([r['orders_per_week'] for r in rows])>2 for rows in halves)
    return dict(score=score,eligible=bool(eligible),train=halves[0],internal_validation=halves[1])


def complete_result(output,spec,result):
    result.update(identity=spec['identity'],finished_at=now(),
                  artifacts={p.name:file_sha(p) for p in output.iterdir() if p.is_file() and p.name not in ('PROGRESS.json','RESULT.json')})
    result['children']={str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file() and p.parent!=output}
    write(output/'RESULT.json',result)


def verified(path,identity):
    if not path.exists(): return False
    d=read(path)
    if d['identity']!=identity or not {'TRAINING.json','HISTORY.json','last.pt','qualification.pt','QUALIFICATION.json'}<=set(d['artifacts']):
        raise ValueError('BASELINE_RECEIPT_CONTRACT')
    if d.get('financial_training'):
        selected='best.pt' if d['activity_admissible'] else 'best_unrestricted.pt'
        if not {selected,'EMISSIONS.npz','QUALIFICATION.json'}<=set(d['artifacts']):
            raise ValueError('BASELINE_FINANCIAL_ARTIFACTS')
        required={f'{period}/{name}' for period in ('TRAIN','INNER_A')
                  for name in ('METRICS.json','TRADES.json','ACTIONS.json','EQUITY.npz','CLOSE_COUNTERFACTUALS.json')}
        if not required<=set(d['children']): raise ValueError('BASELINE_ACCOUNT_ARTIFACTS')
    for name,sha in {**d['artifacts'],**d['children']}.items():
        p=(path.parent/name).resolve()
        if not p.is_relative_to(path.parent.resolve()) or file_sha(p)!=sha: raise ValueError('BASELINE_ARTIFACT_CHANGED')
    return True


def worker(args):
    spec,parent=authenticate(args.root); d=spec['design']; seed=args.seed
    if seed not in d['seeds']: raise ValueError('UNKNOWN_BASELINE_SEED')
    output=args.root/'runs'/f'seed_{seed}'/f'attempt{args.attempt}'
    output.mkdir(parents=True,exist_ok=False)
    configure(seed)
    if not torch.cuda.is_available(): raise ValueError('CUDA_REQUIRED')
    attach_engine(parent['engine_root'])
    recipe=read(parent['inputs']['recipe']); risk=read(args.root/'RISK.json')
    controls=read(args.root/'CONTROL_PREFLIGHT.json')
    if not controls['passed']: raise ValueError('UNQUALIFIED_EXECUTION')
    model=make_model(recipe,d,seed,'cuda')
    def qualify(step,model):
        q=assess_synthetic(model,recipe,risk,d,controls)
        q['stop']=q['passed'] and step>=d['qualification_minimum']
        return q
    factory=lambda episode:synthetic_env(recipe,risk,100000+seed*1000+episode,
        1 if episode%2==0 else -1,d['gamma'],flat=episode%5==4)[0]
    qt,qh=train_loop(model,factory,d,seed,output,qualify,True)
    qualification=qh[-1]['report']; write(output/'QUALIFICATION.json',qualification)
    model.save(output/'qualification.pt')
    if not qualification['passed']:
        complete_result(output,spec,dict(seed=seed,financial_training=False,status='SYNTHETIC_NOT_LEARNED',training=qt))
        return
    # Preserve the synthetic learning curve before the fresh financial run.
    (output/'HISTORY.json').rename(output/'QUALIFICATION_HISTORY.json')
    (output/'TRAINING.json').rename(output/'QUALIFICATION_TRAINING.json')
    del model; torch.cuda.empty_cache(); configure_fresh_seed(seed)
    model=make_model(recipe,d,seed,'cuda'); market=RecipeMarket(parent)
    stopping=Patience(d['minimum_steps'],d['patience'],d['min_delta'])
    best={}
    def assess(step,model):
        report=monitor(model,market,risk,d)
        verdict=stopping.observe(report['score'],report['eligible'],step)
        if verdict['improved_any']: model.save(output/'best_unrestricted.pt')
        if verdict['improved']:
            model.save(output/'best.pt'); best.update(step=step,score=report['score'])
            write(output/'BEST.json',best)
        return dict(report,**verdict)
    factory=lambda episode:environment(market,risk,6,market.recipe['split']['periods']['TRAIN'],d['gamma'])
    training,_=train_loop(model,factory,d,seed,output,assess)
    admitted=(output/'best.pt').exists()
    selected=output/('best.pt' if admitted else 'best_unrestricted.pt')
    model.load(selected)
    logits,q=emit(model,market.inputs,twins=True)
    valid=np.flatnonzero(market.inputs.valid)
    np.savez_compressed(output/'EMISSIONS.npz',indices=valid,logits=logits[valid],qvalues=q[:,valid],
                        timestamp=market.inputs.row_ts_ns[valid])
    with torch.no_grad():
        sample=torch.tensor(market.inputs.windows[valid[:8]],device='cuda')
        before=model.surfaces(sample).clone(); model.load(selected)
        if not torch.equal(before,model.surfaces(sample)): raise ValueError('SIGNAL_RELOAD_CHANGED')
    metrics={p:evaluate(environment(market,risk,6,market.recipe['split']['periods'][p],d['gamma']),
                        logits,output/p,diagnose=True) for p in ('TRAIN','INNER_A')}
    complete_result(output,spec,dict(seed=seed,financial_training=True,activity_admissible=admitted,
        selected_checkpoint='best_active' if admitted else 'best_unrestricted_no_active_checkpoint',
        qualification=qualification,training=training,best=best,metrics=metrics,reload_exact=True,
        scope='DEVELOPMENT_ONLY_NO_PHASE2_SELECTION_OR_STRATEGY_PROMOTION'))


def configure_fresh_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)


def run(args):
    spec,_=authenticate(args.root); seeds=read(args.assignment)['seeds']
    if not seeds or len(seeds)!=len(set(seeds)) or not set(seeds)<=set(spec['design']['seeds']):
        raise ValueError('BASELINE_ASSIGNMENT')
    with (args.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        outcomes={}
        for seed in seeds:
            for attempt in (1,2):
                folder=args.root/'runs'/f'seed_{seed}'/f'attempt{attempt}'
                if verified(folder/'RESULT.json',spec['identity']):
                    outcomes[str(seed)]=read(folder/'RESULT.json'); break
                if folder.exists(): continue
                while True:
                    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
                    gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits',
                                                     '-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
                    if memory['MemAvailable']>4*1024**2 and gpu>3072: break
                    write(args.root/'STATUS.json',dict(state='WAIT_RESOURCES',seed=seed,time=now())); time.sleep(30)
                write(args.root/'STATUS.json',dict(state='RUNNING',seed=seed,attempt=attempt,completed=len(outcomes),time=now()))
                with (args.root/f'seed_{seed}_attempt{attempt}.log').open('a') as log:
                    try:
                        code=subprocess.run([sys.executable,'-m','tools.run_three_signal_baseline','worker',
                            '--root',str(args.root),'--seed',str(seed),'--attempt',str(attempt)],stdout=log,stderr=subprocess.STDOUT,
                            timeout=172800).returncode
                    except subprocess.TimeoutExpired: code='TIMEOUT'
                write(args.root/f'seed_{seed}_attempt{attempt}_EXIT.json',dict(code=code,time=now()))
                if code==0 and verified(folder/'RESULT.json',spec['identity']):
                    outcomes[str(seed)]=read(folder/'RESULT.json'); break
                if code==78: break
            if str(seed) not in outcomes: outcomes[str(seed)]=dict(status='PROCESS_FAILED',financial_training=False)
            write(args.root/'SUMMARY.json',outcomes)
        write(args.root/'STATUS.json',dict(state='COMPLETE',completed=len(outcomes),time=now(),
            financial_completed=sum(r.get('financial_training',False) for r in outcomes.values()),
            failures=sum(r.get('status')=='PROCESS_FAILED' for r in outcomes.values())))


def collect(args):
    config=read(args.assignment); args.root.mkdir(parents=True,exist_ok=True)
    requested=[s for node in config['nodes'] for s in node['seeds']]
    if sorted(requested)!=[7,19,43]: raise ValueError('BASELINE_FLEET_PARTITION')
    while True:
        rows={}; issues=[]
        for node in config['nodes']:
            mirror=args.root/node['role']; mirror.mkdir(exist_ok=True)
            source=(node['host']+':' if node.get('host') else '')+node['root']+'/'
            try:
                subprocess.run(['rsync','-a','--exclude=*.tmp','--exclude=*.log','--exclude=queue.lock',
                    '-e','ssh -o BatchMode=yes -o ConnectTimeout=10',source,str(mirror)+'/'],
                    check=True,timeout=120,capture_output=True)
            except (subprocess.SubprocessError,OSError): issues.append(node['role']); continue
            spec=read(mirror/'SPEC.private.json')
            if (spec['identity']!=config['identity']
                    or spec['identity']!=digest({k:v for k,v in spec.items() if k!='identity'})):
                raise ValueError('BASELINE_COLLECT_IDENTITY')
            for seed in node['seeds']:
                for attempt in (1,2):
                    file=mirror/'runs'/f'seed_{seed}'/f'attempt{attempt}'/'RESULT.json'
                    if verified(file,spec['identity']): rows[str(seed)]=read(file); break
            if (mirror/'STATUS.json').exists() and read(mirror/'STATUS.json')['state']=='COMPLETE':
                for seed in node['seeds']:
                    if str(seed) not in rows: rows[str(seed)]=dict(status='PROCESS_FAILED',financial_training=False)
        write(args.root/'AGGREGATE.json',dict(results=rows,issues=issues,complete=len(rows),time=now()))
        lines=['# Three-Signal SAC Baseline','','Development: TRAIN2020-2023H1; repeatedly monitored INNER_A2023H2. No2024/TEST.',
               '', '| Seed | Period | Return % | Sharpe | DD % | Orders | Buy / Sell | Win % |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
        for seed,row in sorted(rows.items()):
            if not row.get('financial_training'):
                lines.append(f"| {seed} | {row.get('status')} | - | - | - | - | - | - |"); continue
            for period,m in row['metrics'].items():
                fmt=lambda x:'-' if x is None else f'{x:.3f}'
                lines.append(f"| {seed} | {period} | {m['return_pct']:.3f} | {fmt(m['sharpe'])} | {m['max_drawdown_pct']:.3f} | "
                    f"{m['trades']} | {m['buy']['trades']} / {m['sell']['trades']} | {fmt(None if m['win_rate'] is None else 100*m['win_rate'])} |")
        (args.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        if len(rows)==3:
            (args.root/'AUDIT_REQUEST.md').write_text('# Three-Signal Baseline Audit\n\nAudit sequential native qualification, source/data pins, three-signal decoder, market-only inputs, constant action alphabet, reward and duration units, stopping/checkpoint selection, fees, both directions, close attribution and exposed internal validation. No strategy promotion is claimed.\n')
            write(args.root/'CLOSE.json',dict(time=now(),results_sha256=file_sha(args.root/'AGGREGATE.json'),
                                           strategy_promoted=False,expected=3)); return
        time.sleep(120)


def smoke(args):
    spec,parent=authenticate(args.root); d=spec['design']
    configure(7)
    market=RecipeMarket(parent); risk=read(args.root/'RISK.json')
    model=make_model(market.recipe,d,7,args.device)
    env=environment(market,risk,6,market.recipe['split']['periods']['TRAIN'],d['gamma'])
    obs=env.reset(); rng=np.random.default_rng(7)
    replay=TensorDictReplayBuffer(storage=LazyTensorStorage(128),batch_size=32)
    for _ in range(128):
        action=int(rng.integers(0,3)); nxt,reward,done,duration=env.step(action)
        replay.extend(transition(obs,action,nxt,reward,done,duration))
        obs=env.reset() if done else nxt
    started=time.monotonic()
    losses=[model.update(replay.sample().to(args.device)) for _ in range(8)]
    seconds=(time.monotonic()-started)/8
    root=args.root/'SMOKE'; root.mkdir(exist_ok=False)
    model.save(root/'model.pt')
    sample=tensor_observation(obs,args.device)
    with torch.no_grad():
        before=model.distribution(sample).probs.clone(); model.load(root/'model.pt')
        if not torch.equal(before,model.distribution(sample).probs): raise ValueError('SMOKE_RELOAD')
    rng_before=torch.get_rng_state().clone()
    native_before=(env.bar,env.engine.balance,len(env.engine.trades),len(replay))
    evaluation=monitor(model,market,risk,d)
    if native_before!=(env.bar,env.engine.balance,len(env.engine.trades),len(replay)) or not torch.equal(rng_before,torch.get_rng_state()):
        raise ValueError('SMOKE_EVAL_MUTATED_TRAIN')
    result=dict(scope='ENGINEERING_ONLY_NOT_A_CANDIDATE',passed=True,transitions=128,updates=8,
                update_seconds=seconds,monitor_blocks=6,reload_exact=True,
                max_cash_error=max(m['cash_error'] for k in ('train','internal_validation') for m in evaluation[k]),
                losses=losses,device=args.device)
    write(root/'RESULT.json',result); print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','worker','run','collect','smoke'])
    p.add_argument('--root',required=True,type=Path); p.add_argument('--parent',type=Path)
    p.add_argument('--config',type=Path); p.add_argument('--assignment',type=Path)
    p.add_argument('--seed',type=int); p.add_argument('--attempt',type=int,default=1)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    try: globals()[args.command](args)
    except (ValueError,AssertionError) as e:
        print(f'CONTRACT_FAILURE:{type(e).__name__}:{e}',file=sys.stderr,flush=True)
        raise SystemExit(78) from e


if __name__=='__main__': main()
