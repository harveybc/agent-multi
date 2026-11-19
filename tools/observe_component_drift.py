"""Bounded CPU-only, hash-checked component/policy observation of existing runs."""
import argparse
import base64
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
import torch

from app.component_observer import selected_epochs, component_stats, policy_stats
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from app.structural_refit import build_model, verify_files
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_nested_history_training import market_for, input_parity
from tools.run_sac_training_reliability import code_identity


def fetch(run,names):
    script="""import sys,json,base64,pathlib
r=pathlib.Path(sys.argv[1]).resolve();out={}
for n in json.loads(sys.argv[2]):
 p=(r/n).resolve()
 if not p.is_relative_to(r):raise ValueError('OBSERVATION_PATH')
 if p.is_file():out[n]=base64.b64encode(p.read_bytes()).decode()
print(json.dumps(out))
"""
    argv=['python3','-c',script,run['root'],json.dumps(names)]
    if run.get('host'):argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',run['host'],shlex.join(argv)]
    response=subprocess.run(argv,check=True,capture_output=True,text=True,timeout=60)
    return {k:base64.b64decode(v) for k,v in json.loads(response.stdout).items()}


def chart(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    epochs=[r['epoch'] for r in rows]
    for period in ('TRAIN','VALIDATION'):
        axes[0].plot(epochs,[r['metrics'][period]['return_pct'] for r in rows],label=period)
    for field in ('action_flip_fraction','critic_actor_disagreement','near_tie_fraction'):
        axes[1].plot(epochs,[r['policy'][field] for r in rows],label=field)
    for group in ('actor.attention','actor.core','actor.head','critic.attention','critic.core'):
        if group in rows[0]['components']:
            axes[2].plot(epochs,[r['components'][group]['relative_drift'] for r in rows],label=group)
    for axis in axes:axis.legend(fontsize=8)
    axes[0].set_ylabel('Account return %');axes[1].set_ylabel('Fixed TRAIN panel fraction')
    axes[2].set(ylabel='Relative tensor drift',xlabel='Completed epoch')
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def inspect_run(c,name,run,out):
    raw=fetch(run,['SPEC.json','FORK.json','HISTORY.json','PROGRESS.json','RESULT.json'])
    docs={k:json.loads(v) for k,v in raw.items()}
    if 'HISTORY.json' not in docs:
        return dict(stage='WAITING_FOR_FIRST_COMPLETED_EPOCH',rows=[])
    if docs['PROGRESS.json']['identity']!=run['identity']:raise ValueError('OBSERVATION_RUN_IDENTITY')
    if 'spec_override' in run:
        if (file_sha(run['spec_override'])!=run['spec_sha256']
                or docs['FORK.json']['identity']!=run['identity']):raise ValueError('OBSERVATION_FORK')
        spec=read(Path(run['spec_override']))
        if docs['FORK.json']['source_identity']!=spec['identity']:raise ValueError('OBSERVATION_FORK_SOURCE')
    else:
        spec=docs['SPEC.json']
        if spec['identity']!=run['identity']:raise ValueError('OBSERVATION_SPEC_IDENTITY')
    if digest({k:v for k,v in spec.items() if k!='identity'})!=spec['identity']:
        raise ValueError('OBSERVATION_SPEC_CONTENT')
    if 'RESULT.json' in docs and docs['RESULT.json']['identity']!=run['identity']:
        raise ValueError('OBSERVATION_RESULT_IDENTITY')
    history=docs['HISTORY.json'];epochs=selected_epochs(history)
    if not epochs:return dict(stage='WAITING_FOR_FIRST_COMPLETED_EPOCH',rows=[])
    base=out/name;base.mkdir(parents=True,exist_ok=True)
    history_key=digest(dict(raw={k:digest(json.loads(v)) for k,v in raw.items()},code=c['code_identity']))
    dest=base/history_key
    if (dest/'RESULT.json').exists():return read(dest/'RESULT.json')
    dest.mkdir(exist_ok=True)
    for k,v in raw.items():(dest/k).write_bytes(v)
    store=Path(c['stores'])/f'months_{spec["months"]}'
    verify_files(store,read(store/'DONE.json')['files'])
    if file_sha(store/'RECIPE.json')!=spec['recipe']:raise ValueError('OBSERVATION_RECIPE')
    market=market_for(c,spec['months']);support=input_parity(c,spec['months'],market)
    ids=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    ids=ids[np.linspace(0,len(ids)-1,min(256,len(ids)),dtype=int)]
    probe=market.inputs.windows[ids].copy();x=torch.tensor(probe,device='cpu')
    model=build_model(market.recipe,spec['effective'],spec['seed'],'cpu')
    selected={r['epoch']:r for r in history if r['epoch'] in epochs}
    tensors={}
    for e,row in selected.items():
        snap=row['weight_snapshot'];cache=base/(snap['sha256']+'.pt')
        if not cache.exists():
            received=fetch(run,[snap['path']])
            if snap['path'] not in received:raise ValueError('OBSERVATION_MISSING_TENSOR')
            tmp=cache.with_suffix('.tmp');tmp.write_bytes(received[snap['path']])
            if file_sha(tmp)!=snap['sha256']:raise ValueError('OBSERVATION_TENSOR_HASH')
            tmp.replace(cache)
        if file_sha(cache)!=snap['sha256']:raise ValueError('OBSERVATION_CACHE_HASH')
        tensors[e]=cache
    first=torch.load(tensors[epochs[0]],map_location='cpu',weights_only=True)
    model.loss.load_state_dict(first,strict=True)
    anchor={n:p.detach().clone() for n,p in model.loss.named_parameters()}
    with torch.no_grad():reference=model.surfaces(x).argmax(-1)
    rows=[]
    for e in epochs:
        key=digest(dict(code=c['code_identity'],snapshot=selected[e]['weight_snapshot']['sha256'],
            anchor=selected[epochs[0]]['weight_snapshot']['sha256'],support=support['support_sha256']))
        cached=base/(key+'.json')
        if cached.exists():row=read(cached)
        else:
            model.loss.load_state_dict(torch.load(tensors[e],map_location='cpu',weights_only=True),strict=True)
            with torch.no_grad():stats=policy_stats(model.surfaces(x),model.q_surfaces(x),reference)
            row=dict(epoch=e,updates=selected[e]['updates'],metrics=selected[e]['metrics'],policy=stats,
                components=component_stats(model,anchor),snapshot_sha256=selected[e]['weight_snapshot']['sha256'])
            write(cached,row)
        rows.append(row)
    result=dict(stage='COMPLETE' if 'RESULT.json'in docs else 'OBSERVING',identity=run['identity'],
        anchor_epoch=epochs[0],panel_rows=len(ids),panel_period='TRAIN',support=support,rows=rows,
        source_metrics_recomputed=False,training_started=False,promoted=False,
        interpretation='Descriptive same-lineage drift; rank is not model capacity and correlations are not causal')
    chart(rows,dest/'CURVES.png');write(dest/'RESULT.json',result)
    write(base/'LATEST.json',dict(path=str(dest.relative_to(out)),epoch=epochs[-1]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--once',action='store_true')
    a=parser.parse_args();c=read(a.config)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('OBSERVATION_FORCE_CPU')
    if c['code_identity']!=code_identity():raise ValueError('OBSERVATION_CODE_CHANGED')
    if not 1<=c['max_hours']<=96 or c['poll_seconds']<1800:raise ValueError('OBSERVATION_BUDGET')
    configure(19);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    parent=Path(c['parent'])
    if file_sha(parent/'SPEC.json')!=c['parent_spec_sha256']:raise ValueError('OBSERVATION_PARENT')
    ps=read(parent/'SPEC.json')
    verify_files(parent,{k:v for k,v in ps['files'].items()
        if k.startswith(('data/','engine/')) or k in ('SUPPORT_RECIPE.json','RISK.json')})
    lock=(out/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'CONFIG.json').exists() and read(out/'CONFIG.json')!=c:raise ValueError('OBSERVATION_CONFIG_CHANGED')
    write(out/'CONFIG.json',c)
    if not(out/'START.json').exists():write(out/'START.json',dict(time=time.time()))
    deadline=read(out/'START.json')['time']+c['max_hours']*3600
    while True:
        results={};errors={}
        for name,run in c['runs'].items():
            write(out/'PROGRESS.json',dict(stage='ANALYZING',run=name,at=now()))
            try:results[name]=inspect_run(c,name,run,out)
            except Exception as exc:errors[name]=dict(type=type(exc).__name__,message=str(exc))
        write(out/'RESULTS.json',dict(at=now(),runs=results,errors=errors))
        complete=not errors and all(r['stage']=='COMPLETE' for r in results.values())
        terminal=complete or time.time()>=deadline
        write(out/'PROGRESS.json',dict(stage='COMPLETE' if complete else 'BUDGET_ENDED' if terminal else 'WAITING',
            errors=errors,at=now(),next_poll_seconds=c['poll_seconds']))
        if terminal or a.once:return
        time.sleep(min(c['poll_seconds'],max(0,deadline-time.time())))


if __name__=='__main__':main()
