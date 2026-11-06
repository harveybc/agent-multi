"""Bounded G2 GPU queue, automatic paired replication and G3 coverage audit."""
import argparse
import copy
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def matched_reference(history,design):
    from app.structural_refit import EpochPatience
    monitor=EpochPatience(design['patience'],design['min_delta'],design['monitor_mode'],
                         design['monitor_start_epoch'],design['select_during_warmup'])
    selected=None
    for expected,row in enumerate(history,1):
        if row['epoch']!=expected: raise ValueError('CONTROL_HISTORY_SEQUENCE')
        m=row['metrics']
        outcome=monitor.observe(m['TRAIN']['return_per_week'],m['VALIDATION']['return_per_week'])
        if outcome['improved']: selected=row
        if outcome['stop']: return dict(selected=selected,stop_epoch=row['epoch'],rule=vars(monitor))
    raise ValueError('CONTROL_HISTORY_DOES_NOT_REACH_MATCHED_STOP')


def pilot_gate(candidate,control,delta=1e-8):
    return (all(candidate[p]['orders_per_week']>2 for p in ('TRAIN','VALIDATION'))
            and candidate['VALIDATION']['return_per_week']>control['VALIDATION']['return_per_week']+delta)


def prepare(args):
    import numpy as np
    from app.structural_refit import build_model
    from app.rl_subset import digest,support_hash
    from tools.run_structural_sac_refit import authenticate,market_for,SOURCES
    source=authenticate(args.source_root,args.source_runtime)
    control=read(args.control_root/'SPEC.json')
    if control['identity']!=digest({k:v for k,v in control.items() if k!='identity'}):
        raise ValueError('CONTROL_IDENTITY')
    if read(args.control_root/'runs/seed_7/RESULT.json')['identity']!=control['identity']:
        raise ValueError('CONTROL_RESULT_IDENTITY')
    for name in ('RECIPE.json','RISK.json','SUPPORT_RECIPE.json'):
        if sha(args.control_root/name)!=source['files'][name] or control['files'][name]!=source['files'][name]:
            raise ValueError('CONTROL_DATA_CONTRACT:'+name)
    for key,value in control['inputs'].items():
        if source['files'][source['inputs'][key]]!=control['files'][value]:
            raise ValueError('CONTROL_INPUT:'+key)
    if args.root.exists(): raise ValueError('NEW_ROOT_REQUIRED')
    args.root.mkdir(parents=True)
    for name in source['files']:
        dest=args.root/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(args.source_root/name,dest)
    spec=copy.deepcopy(source);spec.pop('identity',None);spec.pop('warmup_migration',None)
    spec['created_at']=datetime.now(timezone.utc).isoformat()
    architecture=copy.deepcopy(control['effective']['architecture'])
    if args.arm=='G2':
        architecture['long_context']=dict(observations=13,stride_rows=6,branch_width=4,branch_blocks=2)
    effective=copy.deepcopy(control['effective']);effective['architecture']=architecture
    spec['effective']=build_model(read(args.root/'RECIPE.json'),effective).effective
    spec['design'].update(seeds=[7,19,31] if args.arm=='G2' else [19,31],
        max_epochs=2000,patience=20,min_delta=1e-8,monitor_start_epoch=20,
        select_during_warmup=True,monitor_mode='validation',diagnostics=True)
    for key in ('batch_size','replay_size','warmup','update_every','horizon'):
        if spec['design'][key]!=control['design'][key]: raise ValueError('CONTROL_TRAINING:'+key)
    reference=matched_reference(read(args.control_root/'runs/seed_7/HISTORY.json'),spec['design'])
    write(args.root/'REFERENCE.json',dict(control_identity=control['identity'],matched_monitor=reference,
        history_sha256=sha(args.control_root/'runs/seed_7/HISTORY.json'),
        original_result=read(args.control_root/'runs/seed_7/RESULT.json'),
        scope='reuse existing seed7 metrics, no new control training; warmup selection replayed'))
    spec['structural_round']=dict(variant=args.arm,parent_identity=source['identity'],
        control_identity=control['identity'],fresh_weights=True,fresh_replay=True,
        context_transport='two independent windows, not26 successive TCN steps' if args.arm=='G2' else 'original13x93',
        missing_context='NaN sentinel; disabled latent branch; baseline support retained')
    spec['sources']={name:sha(ROOT/name) for name in SOURCES+['tools/run_generalization_queue.py']}
    spec['files']={str(p.relative_to(args.root)):sha(p) for p in args.root.rglob('*') if p.is_file()}
    spec['identity']=digest(spec);write(args.root/'SPEC.json',spec)
    market=market_for(args.root,spec);pre=read(args.source_root/'PREFLIGHT.json')
    pre.pop('continuation_of_fresh_fit',None);pre.pop('warmup_migration',None)
    if support_hash(market.inputs)!=pre['support_sha256']: raise ValueError('BASE_SUPPORT_CHANGED')
    if not np.isfinite(market.inputs.windows[market.inputs.valid,:13]).all():
        raise ValueError('BASE_INPUT_NONFINITE')
    pre.update(identity=spec['identity'],tensor_shape=list(market.inputs.windows.shape),
        structural_round=spec['structural_round'],fresh_training=True,native_controls_reused_from=source['identity'])
    if args.arm=='G2':
        pre['context_unavailable']={p:int(np.sum(market.inputs.valid & (market.inputs.period==p)
            & ~np.isfinite(market.inputs.windows[:,13:]).all((1,2)))) for p in ('TRAIN','VALIDATION')}
    write(args.root/'PREFLIGHT.json',pre)
    print(json.dumps(dict(identity=spec['identity'],arm=args.arm,preflight=pre)),flush=True)


def collect(root,seed):
    from tools.run_structural_sac_refit import authenticate
    from app.structural_refit import load_bundle
    spec=authenticate(root);run=root/'runs'/f'seed_{seed}'
    result=read(run/'RESULT.json')
    if result['identity']!=spec['identity'] or result['seed']!=seed: raise ValueError('RESULT_IDENTITY')
    bundle=run/result['best_bundle']['relative_path']
    if not bundle.resolve().is_relative_to(run.resolve()): raise ValueError('BUNDLE_PATH')
    if sha(bundle/'MANIFEST.json')!=result['best_bundle']['manifest_sha256']: raise ValueError('BUNDLE_HASH')
    _,manifest=load_bundle(bundle)
    return dict(identity=spec['identity'],result=result,selected_epoch=manifest['checkpoint']['epoch'],
        result_sha256=sha(run/'RESULT.json'),history_sha256=sha(run/'HISTORY.json'),
        training=read(run/'TRAINING.json'))


def task(args):
    from tools.run_structural_sac_refit import authenticate,worker
    state=args.root/'runs'/f'seed_{args.seed}'/'TASK.json'
    spec=authenticate(args.root)
    if (state.parent/'RESULT.json').exists():
        receipt=collect(args.root,args.seed)
        write(state,dict(status='COMPLETE',receipt=receipt));return
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu.startswith('GPU-') or ',' in gpu: raise ValueError('EXPLICIT_GPU_REQUIRED')
    with (args.root.parent/'gpu.lock').open('a') as lock:
        write(state,dict(status='WAITING_FOR_GPU_LOCK',identity=spec['identity']))
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            write(state,dict(status='READINESS',identity=spec['identity']))
            subprocess.run([sys.executable,'tools/gpu_readiness_probe.py','--gate','--uuid',gpu,
                '--python',sys.executable,'--output-fs',str(args.root),
                '--expected-artifact-bytes',str(6*1024**3)],cwd=ROOT,check=True)
            write(state,dict(status='CUDA_ACCEPTANCE',identity=spec['identity']))
            subprocess.run([sys.executable,'-m','pytest','tests/unit/test_multiscale_context.py','-q'],cwd=ROOT,check=True)
            subprocess.run([sys.executable,'-m','tools.run_generalization_queue','canary',
                '--root',str(args.root),'--seed',str(args.seed)],cwd=ROOT,check=True)
            write(state,dict(status='TRAINING',identity=spec['identity']))
            worker(args)
            receipt=collect(args.root,args.seed)
            write(state,dict(status='COMPLETE',receipt=receipt))
        except Exception as exc:
            write(state,dict(status='FAILED',identity=spec['identity'],error=repr(exc)))
            raise


def canary(args):
    import resource
    import numpy as np
    import torch
    from tensordict import cat
    from tools.run_structural_sac_refit import authenticate,market_for
    from tools.run_discrete_curve_probe import configure,transition
    from app.structural_refit import build_model
    from app.three_signal_learning import environment
    spec=authenticate(args.root);configure(args.seed)
    if not torch.cuda.is_available(): raise ValueError('CUDA_CANARY_REQUIRED')
    market=market_for(args.root,spec)
    model=build_model(market.recipe,spec['effective'],args.seed,'cuda')
    env=environment(market,read(args.root/'RISK.json'),spec['design']['horizon'],
                    market.recipe['split']['periods']['TRAIN'],model.effective['gamma'])
    obs=env.reset();items=[]
    for i in range(spec['design']['batch_size']):
        nxt,reward,done,duration=env.step(i%3)
        items.append(transition(obs,i%3,nxt,reward,done,duration));obs=nxt
    batch=cat(items,dim=0).to('cuda')
    loss=model.update(batch)
    if not np.isfinite(list(loss.values())).all(): raise ValueError('CANARY_LOSS')
    torch.cuda.synchronize()
    write(args.root/'runs'/f'seed_{args.seed}'/'CANARY.json',dict(
        identity=spec['identity'],scope='throwaway integration update on real TRAIN transitions; not experiment metrics',
        rows=len(items),input_shape=list(obs['features'].shape),loss=loss,
        gpu_peak_bytes=torch.cuda.max_memory_allocated(),cpu_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        passed=True))


def coverage(recipe,output):
    """Admission audit, not a synthetic generator or automatic feature removal."""
    rec=read(recipe);rows=[]
    for name in rec['list']['columns']:
        if name.startswith(('cal.','ev.')) or name=='px.hours_since_session_open':
            group='calendar';requirement='recompute from causal timestamps and trading calendar'
        elif name.startswith(('cot.','dly.','fac.','wv.usd.','hb.eur.')):
            group='external';requirement='joint external series with publication availability and fitted transforms'
        elif name.startswith(('act.','rg.')):
            group='fitted';requirement='pinned TRAIN-only fitted producer and joint primitive inputs'
        else:
            group='price_derived';requirement='recompute pinned feature producer from coherent OHLCV'
        rows.append(dict(column=name,group=group,required=requirement,validated_joint_producer=False))
    result=dict(schema='synthetic_coverage.v1',recipe_identity=rec['identity'],columns=rows,
        status='NO_ADMISSIBLE_JOINT_PRODUCER',training_launched=False,
        reason='Single-asset OHLCV producers do not validate the full93-column joint contract.',
        next_engineering_action='implement joint primitive adapter and pinned feature regeneration; no user approval needed',
        generator_fitted=False,synthetic_transitions_inserted=0)
    write(output,result)
    return result


def remote(host,argv,timeout=90):
    import shlex
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,
        shlex.join(argv)],check=True,text=True,capture_output=True,timeout=timeout).stdout


def service_state(host,unit):
    data=remote(host,['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState',
                            '-p','Result','-p','ExecMainStatus','-p','NRestarts'])
    return dict(line.split('=',1) for line in data.splitlines() if '=' in line)


def resource_failure(state):
    return state.get('Result')=='oom-kill' or state.get('ExecMainStatus') in ('9','137')


def migrate_pilot(source,destination,out):
    if (not source['unit'].startswith('sac-generalization-') or source['key']!=destination['key']
            or source['identity']!=destination['identity']):
        raise ValueError('FALLBACK_IDENTITY')
    remote(source['host'],['systemctl','--user','disable','--now',source['unit']])
    if service_state(source['host'],source['unit']).get('ActiveState') in ('active','activating','deactivating'):
        raise ValueError('MIGRATION_SOURCE_NOT_QUIESCENT')
    staging=out/'migration'/source['key'];staging.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--exclude=worker.lock',source['host']+':'+source['root']+'/',str(staging)+'/'],
                   check=True,timeout=1200)
    subprocess.run(['rsync','-a','--exclude=worker.lock',str(staging)+'/',
                    destination['host']+':'+destination['root']+'/'],check=True,timeout=1200)
    remote(destination['host'],['systemctl','--user','reset-failed',destination['unit']])


def launch(job):
    state=service_state(job['host'],job['unit'])
    if state.get('LoadState')!='loaded': raise ValueError('WORKER_UNIT_NOT_INSTALLED')
    if state.get('ActiveState') in ('active','activating','reloading','deactivating'): return
    if state.get('ActiveState')=='failed': raise ValueError('WORKER_RETRIES_EXHAUSTED:'+job['unit'])
    remote(job['host'],['systemctl','--user','enable','--now',job['unit']])


def progress(job):
    code="import json,pathlib; p=pathlib.Path(%r); print(json.dumps({n:json.loads((p/n).read_text()) if (p/n).exists() else None for n in ['TASK.json','PROGRESS.json']}))" % (job['root']+'/runs/seed_'+str(job['seed']))
    return json.loads(remote(job['host'],[job['python'],'-c',code]))


def fetch_receipt(job,out):
    payload=remote(job['host'],['env','CUDA_VISIBLE_DEVICES=','PYTHONPATH='+job['runtime'],
        job['python'],job['runtime']+'/tools/run_generalization_queue.py','collect',
        '--root',job['root'],'--seed',str(job['seed'])],timeout=180)
    receipt=json.loads(payload.strip().splitlines()[-1])
    if receipt['identity']!=job['identity']: raise ValueError('QUEUE_RUN_IDENTITY')
    write(out/'receipts'/f"{job['key']}.json",receipt)
    dest=out/'collected'/job['key'];dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--exclude=recovery/','--exclude=worker.lock',
        job['host']+':'+job['root']+'/runs/seed_'+str(job['seed'])+'/',str(dest)+'/'],check=True,timeout=600)
    if sha(dest/'RESULT.json')!=receipt['result_sha256'] or sha(dest/'HISTORY.json')!=receipt['history_sha256']:
        raise ValueError('COLLECTION_HASH')
    return receipt


def render_report(out,state,reference):
    lines=['# Automated generalization queue','', 'Status: '+state['status'], '',
        'No TEST reads. Pilot comparisons are development results, not replicated proof.', '',
        '| Arm | Seed | Epoch | TRAIN % | VAL % | VAL Sharpe | VAL orders | VAL win % | VAL DD % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def row(arm,seed,epoch,m):
        v=m['VALIDATION'];s=v['sharpe'];w=v['win_rate']
        lines.append(f"| {arm} | {seed} | {epoch} | {m['TRAIN']['return_pct']:.4f} | {v['return_pct']:.4f} | "
            +(f'{s:.4f}' if s is not None else 'undefined')+f" | {v['trades']} | "
            +(f'{100*w:.2f}' if w is not None else 'undefined')+f" | {v['max_drawdown_pct']:.4f} |")
    r=reference['matched_monitor']['selected'];row('R0 reused; matched monitor',7,r['epoch'],r['metrics'])
    for path in sorted((out/'receipts').glob('*.json')):
        r=read(path);row(path.stem,r['result']['seed'],r['selected_epoch'],r['result']['metrics'])
    lines.extend(['','G3: producer coverage audit is separate; no synthetic training is inferred from G2 completion.'])
    temp=out/'REPORT.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(out/'REPORT.md')


def coordinator(args):
    cfg=read(args.config);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=out/'QUEUE.json';cfg_hash=sha(args.config)
        state=read(state_path) if state_path.exists() else dict(config_sha256=cfg_hash,status='PILOT',completed=[])
        if state['config_sha256']!=cfg_hash: raise ValueError('QUEUE_CONFIG_CHANGED')
        reference=read(cfg['reference'])
        coverage(Path(cfg['recipe']),out/'SYNTHETIC_COVERAGE.json')
        while state['status'] not in ('COMPLETE','COMPLETE_PILOT_ONLY'):
            try:
                if state.get('migration'):
                    migrate_pilot(cfg['pilot'],cfg['pilot_fallback'],out)
                    state['pilot_override']=cfg['pilot_fallback'];state['fallback_used']=True
                    state.pop('migration');write(state_path,state)
                jobs=[state.get('pilot_override',cfg['pilot'])] if not state.get('replicate') else copy.deepcopy(cfg['replications'])
                if state.get('fallback_used') and state.get('replicate'):
                    for job in jobs: job['host']=cfg['pilot_fallback']['host']
                active=[]
                for host in dict.fromkeys(j['host'] for j in jobs):
                    remaining=[j for j in jobs if j['host']==host and j['key'] not in state['completed']]
                    if not remaining: continue
                    job=remaining[0];p=progress(job)
                    if p['TASK.json'] and p['TASK.json']['status']=='COMPLETE':
                        fetch_receipt(job,out);state['completed'].append(job['key']);continue
                    s=service_state(host,job['unit'])
                    if s.get('ActiveState')=='failed':
                        if job['key']==cfg['pilot']['key'] and not state.get('fallback_used') and resource_failure(s):
                            state['migration']=True;write(state_path,state);break
                        raise ValueError('WORKER_RETRIES_EXHAUSTED:'+job['key'])
                    launch(job);active.append(dict(key=job['key'],service=s,progress=p))
                state['active']=active
                if cfg['pilot']['key'] in state['completed'] and not state.get('pilot_decided'):
                    receipt=read(out/'receipts'/f"{cfg['pilot']['key']}.json")
                    state['replicate']=pilot_gate(receipt['result']['metrics'],reference['matched_monitor']['selected']['metrics'])
                    state['pilot_decided']=True
                    state['status']='REPLICATION' if state['replicate'] else 'COMPLETE_PILOT_ONLY'
                if state.get('replicate') and all(j['key'] in state['completed'] for j in cfg['replications']):
                    state['status']='COMPLETE'
                state.pop('transient_error',None)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as exc:
                # Network outages do not discard a durable completed epoch or start a duplicate job.
                state['transient_error']=repr(exc)
            except Exception as exc:
                state['status']='FAILED';state['error']=repr(exc)
                write(state_path,state);render_report(out,state,reference);raise
            state['updated_unix']=time.time()
            write(state_path,state);render_report(out,state,reference)
            if state['status'].startswith('COMPLETE'): break
            time.sleep(cfg.get('poll_seconds',60))
        render_report(out,state,reference)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare')
    for k in ('source-root','source-runtime','control-root','root'): a.add_argument('--'+k,type=Path,required=True)
    a.add_argument('--arm',choices=['G2','R0'],required=True)
    for cmd in ('task','collect','canary'):
        a=sub.add_parser(cmd);a.add_argument('--root',type=Path,required=True);a.add_argument('--seed',type=int,required=True)
    a=sub.add_parser('coordinator');a.add_argument('--config',type=Path,required=True)
    a=sub.add_parser('coverage');a.add_argument('--recipe',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.cmd=='collect': print(json.dumps(collect(args.root,args.seed)),flush=True)
    elif args.cmd=='coverage': coverage(args.recipe,args.output)
    else: globals()[args.cmd](args)


if __name__=='__main__': main()
