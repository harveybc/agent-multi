"""CPU-only finite observer and authenticated input inventory; never starts training."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from app.prioritized_training_diagnosis import summarize_history, data_profile

FILES=('SPEC.json','HISTORY.json','PROGRESS.json','EFFECTIVE.json','RESULT.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def fetch(run):
    script="""import sys,json,base64,pathlib
r=pathlib.Path(sys.argv[1]); names=json.loads(sys.argv[2])
print(json.dumps({n:base64.b64encode((r/n).read_bytes()).decode() for n in names if (r/n).is_file()}))
"""
    argv=['python3','-c',script,run['root'],json.dumps(FILES)]
    if run.get('host'):
        import shlex
        argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',run['host'],shlex.join(argv)]
    raw=subprocess.run(argv,check=True,capture_output=True,text=True,timeout=40)
    decoded={k:base64.b64decode(v) for k,v in json.loads(raw.stdout).items()}
    docs={k:json.loads(v) for k,v in decoded.items()}
    if any(k not in docs for k in FILES[:3]):
        raise ValueError('MISSING_REQUIRED_SNAPSHOT')
    if docs['SPEC.json']['identity']!=docs['PROGRESS.json']['identity']:
        raise ValueError('SNAPSHOT_IDENTITY')
    if docs['SPEC.json']['identity']!=run['identity']:
        raise ValueError('UNEXPECTED_RUN_IDENTITY')
    if 'RESULT.json' in docs and docs['RESULT.json'].get('identity') != run['identity']:
        raise ValueError('RESULT_IDENTITY')
    return decoded,docs


def inventory(config,out):
    from app.frozen_recipe_inputs import file_sha
    from app.structural_refit import verify_files
    from app.selected_input_market import market_for
    from app.rl_subset import support_hash
    parent=Path(config['parent']);bundle=Path(config['bundle']);spec=read(parent/'SPEC.json')
    manifest=read(bundle/'MANIFEST.json');verify_files(bundle,manifest['files'])
    verify_files(parent,{k:v for k,v in spec['files'].items()
        if k.startswith(('data/','engine/')) or k in ('SUPPORT_RECIPE.json','RISK.json')})
    if file_sha(bundle/'model.pt')!=config['source']:
        raise ValueError('SOURCE_IDENTITY')
    identity=sha(json.dumps(dict(source=config['source'],support=config['support'],
        parent=file_sha(parent/'SPEC.json'),manifest=file_sha(bundle/'MANIFEST.json'),
        code=file_sha(Path(__file__).parents[1]/'app/prioritized_training_diagnosis.py')),
        sort_keys=True).encode())
    target=out/'DATA.json'
    if target.exists():
        previous=read(target)
        if previous['identity']!=identity:raise ValueError('DATA_CACHE_IDENTITY')
        if previous['profile_sha256']!=sha(json.dumps(previous['profile'],sort_keys=True,allow_nan=False).encode()):
            raise ValueError('DATA_CACHE_CONTENT')
        return previous
    recipe=read(bundle/'RECIPE.json');market=market_for(parent,spec,recipe);inputs=market.inputs
    if support_hash(inputs)!=config['support']:raise ValueError('SUPPORT_IDENTITY')
    result=data_profile(inputs.windows[:,-1,:],inputs.row_ts_ns,inputs.valid,inputs.period,
                        recipe['list']['columns'])
    payload=dict(identity=identity,source=config['source'],support=config['support'],
        source_files_verified=True,profile=result,
        profile_sha256=sha(json.dumps(result,sort_keys=True,allow_nan=False).encode()),
        training_started=False,test_read=False)
    write(target,payload)
    return payload


def plot(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,6))
    for key in ('train_return_pct','validation_return_pct'):
        axes[0].plot([r['epoch'] for r in rows],[r[key] for r in rows],label=key)
    axes[0].legend();axes[0].set_ylabel('Return %')
    axes[1].plot([r['epoch'] for r in rows],[r['validation_trades'] for r in rows])
    axes[1].set(xlabel='Completed epoch',ylabel='Validation trades')
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def run(config,out,once=False):
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=out/'CONFIG.json'
    if cfg.exists() and read(cfg)!=config:raise ValueError('OBSERVER_CONFIG_CHANGED')
    write(cfg,config)
    started=out/'START.json'
    if not started.exists():write(started,dict(at=time.time()))
    start=read(started)['at'];deadline=start+config['max_hours']*3600
    write(out/'PROGRESS.json',dict(stage='DATA_AUTHENTICATION',at=time.time()))
    data=inventory(config,out)
    while True:
        summaries={};errors={};terminal=[]
        for name,spec in config['runs'].items():
            try:
                raw,docs=fetch(spec)
                summary=summarize_history(docs['HISTORY.json'],docs['PROGRESS.json'])
                signature=sha(b''.join(k.encode()+raw[k] for k in sorted(raw)))
                snap=out/'snapshots'/name/signature;snap.mkdir(parents=True,exist_ok=True)
                for key,value in raw.items():
                    target=snap/key
                    if target.exists() and target.read_bytes()!=value:raise ValueError('SNAPSHOT_MUTATION')
                    if not target.exists():target.write_bytes(value)
                receipt=snap/'RECEIPT.json'
                if not receipt.exists():
                    write(receipt,dict(identity=spec['identity'],at=time.time(),files={k:sha(v) for k,v in raw.items()}))
                summary['snapshot']=str(snap.relative_to(out));summaries[name]=summary
                if 'RESULT.json' in docs:terminal.append(name)
                history_hash=sha(raw['HISTORY.json']); chart=out/'plots'/f'{name}-{history_hash}.png'
                chart.parent.mkdir(exist_ok=True)
                if not chart.exists():plot(summary['rows'],chart)
            except Exception as exc:
                errors[name]=dict(type=type(exc).__name__,message=str(exc))
        repair=read(Path(config['repair'])/'DECISION.json')
        canonical=json.dumps(repair['value'],sort_keys=True,separators=(',',':')).encode()
        if sha(canonical)!=repair['sha256'] or repair['identity']!=config['repair_identity']:
            raise ValueError('REPAIR_RECEIPT_IDENTITY')
        state=dict(at=time.time(),stage='OBSERVING',runs=summaries,errors=errors,
            terminal=terminal,data_identity=data['identity'],repair=repair['value'],
            new_gpu_work=False,workers_interrupted=False,
            next_action='Implement and test nested-history producer/scaler fit boundaries before GPU learning curve; no repeat369')
        write(out/'RESULTS.json',state);write(out/'PROGRESS.json',state)
        if not errors and len(terminal)==len(config['runs']):
            write(out/'FINAL.json',dict(**state,reason='ALL_RUNS_TERMINAL'));return
        if time.time()>=deadline:
            write(out/'FINAL.json',dict(**state,reason='OBSERVATION_BUDGET_ENDED_NOT_TRAINING_COMPLETION'));return
        if once:return
        time.sleep(min(config['poll_seconds'],max(0,deadline-time.time())))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--once',action='store_true')
    a=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('FORCE_CPU')
    try:run(read(a.config),a.output,a.once)
    except Exception as exc:
        write(a.output/'FAILURE.json',dict(at=time.time(),type=type(exc).__name__,message=str(exc)))
        raise


if __name__=='__main__':main()
