"""Paired six-rule stopping contrast with qualified seeds and immutable receipts."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from app.stopping_contrast import make_rules, observe_rules
from app.three_signal_learning import environment, emit, evaluate
from app.recipe_runtime import RecipeMarket
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from tools.run_discrete_curve_probe import read, write, now, configure
from tools.run_three_signal_baseline import authenticate as baseline_authenticate
from tools.run_three_signal_baseline import verified as baseline_verified
from tools.run_three_signal_baseline import make_model, monitor, train_loop

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['app/stopping_contrast.py','tools/run_stopping_contrast.py']


def prepare(args):
    baseline,parent=baseline_authenticate(args.parent)
    if args.root.exists(): raise ValueError('NEW_CONTRAST_ROOT_REQUIRED')
    proofs={}; histories={}
    for role,seed in [('primary',7),('secondary',19),('primary',43)]:
        p=args.collection/role/'runs'/f'seed_{seed}'/'attempt1'/'RESULT.json'
        if not baseline_verified(p,baseline['identity']): raise ValueError('UNVERIFIED_BASELINE')
        proofs[str(seed)]=read(p)
        histories[str(seed)]=read(p.parent/'HISTORY.json')
    if not all(proofs[str(s)].get('qualification',{}).get('passed') for s in (7,19)):
        raise ValueError('QUALIFICATION_NOT_PASSED')
    if proofs['43'].get('status')!='SYNTHETIC_NOT_LEARNED': raise ValueError('UNEXPECTED_THIRD_SEED')
    args.root.mkdir(parents=True)
    write(args.root/'QUALIFICATION_PROOFS.json',proofs)
    write(args.root/'ORIGINAL_MONITORS.json',histories)
    spec=dict(schema='paired_stopping.v1',baseline_root=str(args.parent.resolve()),
        baseline_identity=baseline['identity'],design=baseline['design'],seeds=[7,19],
        excluded_seed43='synthetic qualification failed; not replaced',
        proofs_sha256=file_sha(args.root/'QUALIFICATION_PROOFS.json'),
        monitors_sha256=file_sha(args.root/'ORIGINAL_MONITORS.json'),
        sources={n:file_sha(ROOT/n) for n in SOURCES},created_at=now())
    spec['identity']=digest(spec); write(args.root/'SPEC.private.json',spec)
    print(json.dumps(dict(identity=spec['identity'],seeds=spec['seeds'])),flush=True)


def authenticate(root):
    spec=read(root/'SPEC.private.json')
    if spec['identity']!=digest({k:v for k,v in spec.items() if k!='identity'}): raise ValueError('CONTRAST_IDENTITY')
    baseline,parent=baseline_authenticate(Path(spec['baseline_root']))
    if baseline['identity']!=spec['baseline_identity'] or spec['design']!=baseline['design']:
        raise ValueError('CONTRAST_BASELINE_CHANGED')
    for name,sha in spec['sources'].items():
        if file_sha(ROOT/name)!=sha: raise ValueError('CONTRAST_SOURCE_CHANGED')
    if file_sha(root/'QUALIFICATION_PROOFS.json')!=spec['proofs_sha256']: raise ValueError('CONTRAST_PROOFS_CHANGED')
    if file_sha(root/'ORIGINAL_MONITORS.json')!=spec['monitors_sha256']: raise ValueError('CONTRAST_MONITORS_CHANGED')
    proofs=read(root/'QUALIFICATION_PROOFS.json')
    if spec['seeds']!=[7,19] or any(not proofs[str(s)].get('qualification',{}).get('passed') for s in spec['seeds']):
        raise ValueError('CONTRAST_QUALIFICATION')
    return spec,parent


def seal(output,spec,result):
    result.update(identity=spec['identity'],finished_at=now(),artifacts={
        str(p.relative_to(output)):file_sha(p) for p in output.rglob('*')
        if p.is_file() and p.name not in ('PROGRESS.json','RESULT.json')})
    write(output/'RESULT.json',result)


def verified(path,identity):
    if not path.exists(): return False
    r=read(path)
    required={'HISTORY.json','TRAINING.json','RULES.json','last.pt','INITIAL_PROBE.json'}
    names={x.name for x in make_rules(dict(minimum_steps=1,min_delta=1e-5))}
    if r['identity']!=identity or r['seed'] not in (7,19) or set(r['rules'])!=names or not required<=set(r['artifacts']):
        raise ValueError('CONTRAST_RECEIPT')
    for rule in r['rules'].values():
        step=rule['selected_step']
        if step is None: continue
        expected={f'checkpoints/{step}.pt',f'accounts/{step}/EMISSIONS.npz'}
        expected.update(f'accounts/{step}/{period}/{name}' for period in ('TRAIN','INNER_A')
                        for name in ('METRICS.json','TRADES.json','ACTIONS.json','EQUITY.npz','CLOSE_COUNTERFACTUALS.json'))
        if not expected<=set(r['artifacts']) or str(step) not in r['accounts']: raise ValueError('CONTRAST_ACCOUNTS')
    for name,sha in r['artifacts'].items():
        p=(path.parent/name).resolve()
        if not p.is_relative_to(path.parent.resolve()) or file_sha(p)!=sha: raise ValueError('CONTRAST_ARTIFACT_CHANGED')
    return True


def financial(output,spec,parent,seed,device,smoke=False,original_history=()):
    d=copy.deepcopy(spec['design'])
    if smoke:
        d.update(steps=128,minimum_steps=32,warmup=16,evaluation_every=64,batch_size=16,replay_size=128)
    configure(seed)
    recipe=read(parent['inputs']['recipe'])
    model=make_model(recipe,d,seed,device)
    market=RecipeMarket(parent)
    risk=read(Path(spec['baseline_root'])/'RISK.json')
    valid=np.flatnonzero(market.inputs.valid)
    with torch.no_grad():
        probe=model.surfaces(torch.tensor(market.inputs.windows[valid[:8]],device=device)).cpu().tolist()
    write(output/'INITIAL_PROBE.json',dict(seed=seed,values=probe,fresh_financial_weights=True))
    rules=make_rules(d)
    original={r['step']:r['report'] for r in original_history}
    prefix=[]
    def assess(step,model):
        report=monitor(model,market,risk,d)
        model.save(output/'checkpoints'/f'{step}.pt')
        result=observe_rules(rules,report,step)
        if step in original:
            prefix.append(dict(step=step,report_exact=digest(report)==digest({k:v for k,v in original[step].items()
                if k in report}),score_difference=report['score']-original[step]['score']))
            write(output/'PREFIX_COMPARISON.json',prefix)
        write(output/'RULES.json',{r.name:r.state() for r in rules})
        return dict(report,**result)
    (output/'checkpoints').mkdir()
    factory=lambda episode:environment(market,risk,6,recipe['split']['periods']['TRAIN'],d['gamma'])
    training,history=train_loop(model,factory,d,seed,output,assess)
    for rule in rules:
        if not rule.stopped:
            rule.stopped,rule.stop_step,rule.reason=True,training['step'],'step_budget'
    states={r.name:r.state() for r in rules}
    write(output/'RULES.json',states)
    accounts={}
    for step in sorted({r.selected_step for r in rules if r.selected_step is not None}):
        model.load(output/'checkpoints'/f'{step}.pt')
        x=torch.tensor(market.inputs.windows[valid[:8]],device=device)
        with torch.no_grad():
            before=model.surfaces(x).clone(); model.load(output/'checkpoints'/f'{step}.pt')
            if not torch.equal(before,model.surfaces(x)): raise ValueError('CONTRAST_RELOAD_CHANGED')
        logits,q=emit(model,market.inputs,twins=True)
        folder=output/'accounts'/str(step); folder.mkdir(parents=True)
        np.savez_compressed(folder/'EMISSIONS.npz',indices=valid,logits=logits[valid],qvalues=q[:,valid],timestamp=market.inputs.row_ts_ns[valid])
        accounts[str(step)]={p:evaluate(environment(market,risk,6,recipe['split']['periods'][p],d['gamma']),
            logits,folder/p,diagnose=True) for p in ('TRAIN','INNER_A')}
    result=dict(seed=seed,rules=states,accounts=accounts,training={k:v for k,v in training.items() if k!='report'},
        shared_trajectory=True,independent_replicates=1,smoke=smoke,scope='DEVELOPMENT_CONDITIONAL_ON_TWO_QUALIFIED_SEEDS')
    result['original_prefix_comparison']=prefix
    seal(output,spec,result)
    if not verified(output/'RESULT.json',spec['identity']): raise ValueError('CONTRAST_SELF_VERIFY')
    return result


def worker(args):
    spec,parent=authenticate(args.root)
    if args.seed not in spec['seeds']: raise ValueError('UNQUALIFIED_SEED')
    if not torch.cuda.is_available(): raise ValueError('CUDA_REQUIRED')
    output=args.root/'runs'/f'seed_{args.seed}'/f'attempt{args.attempt}'
    output.mkdir(parents=True,exist_ok=False)
    financial(output,spec,parent,args.seed,'cuda',original_history=read(args.root/'ORIGINAL_MONITORS.json')[str(args.seed)])


def smoke(args):
    spec,parent=authenticate(args.root)
    output=args.root/'SMOKE'; output.mkdir(exist_ok=False)
    result=financial(output,spec,parent,7,'cpu',True)
    print(json.dumps(dict(passed=True,updates=result['training']['updates'],accounts=len(result['accounts']))),flush=True)


def run(args):
    spec,_=authenticate(args.root)
    if args.seed not in spec['seeds']: raise ValueError('UNQUALIFIED_SEED')
    with (args.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for attempt in (1,2):
            folder=args.root/'runs'/f'seed_{args.seed}'/f'attempt{attempt}'
            if verified(folder/'RESULT.json',spec['identity']): break
            if folder.exists(): continue
            while True:
                mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if ':' in l}
                gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits',
                    '-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
                if mem['MemAvailable']>4*1024**2 and gpu>3072: break
                write(args.root/'STATUS.json',dict(state='WAIT_RESOURCES',seed=args.seed,time=now())); time.sleep(30)
            write(args.root/'STATUS.json',dict(state='RUNNING',seed=args.seed,attempt=attempt,time=now()))
            with (args.root/f'seed_{args.seed}_attempt{attempt}.log').open('a') as log:
                try:
                    code=subprocess.run([sys.executable,'-m','tools.run_stopping_contrast','worker','--root',str(args.root),
                        '--seed',str(args.seed),'--attempt',str(attempt)],stdout=log,stderr=subprocess.STDOUT,timeout=172800).returncode
                except subprocess.TimeoutExpired: code='TIMEOUT'
            write(args.root/f'EXIT_{args.seed}_{attempt}.json',dict(code=code,time=now()))
            if code==0 and verified(folder/'RESULT.json',spec['identity']): break
            if code==78: break
        success=any(verified(p,spec['identity']) for p in (args.root/'runs'/f'seed_{args.seed}').glob('attempt*/RESULT.json'))
        write(args.root/'STATUS.json',dict(state='COMPLETE' if success else 'FAILED',seed=args.seed,time=now()))


def collect(args):
    cfg=read(args.assignment); args.root.mkdir(parents=True,exist_ok=True)
    if sorted(n['seed'] for n in cfg['nodes'])!=[7,19]: raise ValueError('CONTRAST_FLEET')
    if (args.root/'CLOSE.json').exists():
        c=read(args.root/'CLOSE.json')
        if c['identity']!=cfg['identity'] or c['aggregate_sha256']!=file_sha(args.root/'AGGREGATE.json'):
            raise ValueError('CONTRAST_CLOSED_CHANGED')
        return
    while True:
        results={}; issues=[]
        for node in cfg['nodes']:
            dest=args.root/node['role']; dest.mkdir(exist_ok=True)
            try:
                subprocess.run(['rsync','-a','--exclude=*.tmp','--exclude=*.log','--exclude=queue.lock','-e',
                    'ssh -o BatchMode=yes -o ConnectTimeout=10',node['host']+':'+node['root']+'/',str(dest)+'/'],
                    check=True,timeout=120,capture_output=True)
            except (subprocess.SubprocessError,OSError): issues.append(node['role']); continue
            s=read(dest/'SPEC.private.json')
            if s['identity']!=cfg['identity'] or s['identity']!=digest({k:v for k,v in s.items() if k!='identity'}):
                raise ValueError('CONTRAST_FLEET_IDENTITY')
            for p in sorted((dest/'runs'/f"seed_{node['seed']}").glob('attempt*/RESULT.json')):
                if verified(p,cfg['identity']):
                    r=read(p)
                    if r['seed']!=node['seed']: raise ValueError('CONTRAST_SEED_MISMATCH')
                    results[str(node['seed'])]=r; break
            if str(node['seed']) not in results and (dest/'STATUS.json').exists() and read(dest/'STATUS.json')['state']=='FAILED':
                results[str(node['seed'])]=dict(status='PROCESS_FAILED')
        write(args.root/'AGGREGATE.json',dict(identity=cfg['identity'],results=results,issues=issues,time=now()))
        lines=['# Paired stopping contrast','', 'Development only; seed43 failed prior qualification; two qualified seeds only.',
               '', '| Seed | Rule | Stop | Chosen | TRAIN return % | INNER_A return % | INNER_A Sharpe | Orders |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
        for seed,r in results.items():
            if 'rules' not in r: lines.append(f"| {seed} | PROCESS_FAILED | - | - | - | - | - | - |"); continue
            for name,rule in r['rules'].items():
                step=rule['selected_step']
                if step is None: lines.append(f"| {seed} | {name}: no eligible checkpoint | {rule['stop_step']} | - | - | - | - | - |"); continue
                m=r['accounts'][str(step)]; val=m['INNER_A']; sh='-' if val['sharpe'] is None else f"{val['sharpe']:.3f}"
                lines.append(f"| {seed} | {name} | {rule['stop_step']} | {step} | {m['TRAIN']['return_pct']:.3f} | {val['return_pct']:.3f} | {sh} | {val['trades']} |")
        (args.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        if len(results)==2:
            (args.root/'AUDIT_REQUEST.md').write_text('# Paired stopping audit\n\nVerify frozen six-rule definitions, shared-trajectory independence, stop-time freezing, selected checkpoint hashes and reload, full accounts, prior qualification and explicit failed seed43, internal-period reuse, and no2024/TEST access. No phase2 launch or strategy promotion.\n')
            write(args.root/'CLOSE.json',dict(identity=cfg['identity'],time=now(),aggregate_sha256=file_sha(args.root/'AGGREGATE.json'))); return
        time.sleep(120)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','worker','run','collect','smoke'])
    p.add_argument('--root',required=True,type=Path); p.add_argument('--parent',type=Path)
    p.add_argument('--collection',type=Path); p.add_argument('--assignment',type=Path)
    p.add_argument('--seed',type=int); p.add_argument('--attempt',type=int,default=1)
    args=p.parse_args()
    try: globals()[args.command](args)
    except (ValueError,AssertionError) as e:
        print(f'CONTRACT_FAILURE:{e}',file=sys.stderr,flush=True); raise SystemExit(78) from e


if __name__=='__main__': main()
