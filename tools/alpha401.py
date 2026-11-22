"""Bounded paired history399 temperature intervention, one GPU, no promotion."""
import argparse
import csv
import fcntl
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from tools import series399 as s
from tools import run_structural_sac_refit as trainer


def copy_state(src,dst):
    dst.mkdir(parents=True,exist_ok=True)
    pointer=s.p.read(src/'RESUME.json')
    assert pointer['directory']=='epoch_0003'
    rec=src/'recovery'/pointer['directory']
    assert s.p.file_sha(rec/'MANIFEST.json')==pointer['manifest_sha256']
    s.p.verify_files(rec,s.p.read(rec/'MANIFEST.json')['files'])
    for name in ['bundles','recovery/'+pointer['directory']]:
        shutil.copytree(src/name,dst/name,dirs_exist_ok=True)
    for name in ['EPOCH_ZERO.json','RESUME.json']:
        shutil.copy2(src/name,dst/name)


def branch(root,base,arm):
    adaptive=arm=='adaptive'
    out=base/arm
    out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists(): return
    if adaptive and not (out/'RESUME.json').exists(): copy_state(base/'fork',out)
    original=s.p.fit_epochs
    def fit(model,env_factory,assess,d,seed,path,identity,probe,epoch_hook):
        design=dict(d,max_epochs=8,matched_budget=True)
        experiment=dict(seed=seed,epochs=8,fork_after=3,arm=arm,baseline=identity,
                        code=s.p.file_sha(__file__),gpu=torch.cuda.get_device_name(),change='only_log_alpha_requires_grad')
        if (path/'EXPERIMENT.json').exists() and s.p.read(path/'EXPERIMENT.json')!=experiment:
            raise ValueError('changed experiment')
        s.p.write(path/'EXPERIMENT.json',experiment)
        load,save=trainer.load_recovery,trainer.save_recovery
        def restore(*args):
            state=load(*args)
            if adaptive and state['epoch']<3: raise ValueError('early adaptive fork')
            param=model.loss.log_alpha
            if not any(param is v for g in model.optimizer.param_groups for v in g['params']):
                raise ValueError('temperature absent from optimizer')
            if state['epoch']==3 and param.requires_grad: raise ValueError('contaminated fork')
            param.requires_grad_(adaptive)
            return state
        def archive(output,model_,replay,rng,state):
            save(output,model_,replay,rng,state)
            epoch=state['epoch']
            rec=output/'recovery'/f'epoch_{epoch:04d}'
            dst=output/'snapshots'/rec.name
            shutil.copytree(rec,dst,dirs_exist_ok=True)
            if epoch==3 and not adaptive: copy_state(output,base/'fork')
        def check(m,row):
            epoch_hook(m,row)
            if bool(m.loss.log_alpha.requires_grad)!=adaptive: raise ValueError('unexpected trainability')
            if not adaptive and abs(row['last_update']['alpha']-0.005312912631779909)>1e-9:
                raise ValueError('fixed temperature changed')
        loss=model.loss
        qfun,target,actor,update=loss._vmap_qnetworkN0,loss._compute_target,loss.actor_loss,model.update
        tap={}
        def qs(td,params):
            y=qfun(td,params)
            if params is loss._cached_detached_qvalue_params:
                tap['q']=y['action_value'].detach().amin(0)
            return y
        def targets(*args,**kw):
            y=target(*args,**kw);tap['target']=y.detach();return y
        def actorloss(*args,**kw):
            y=actor(*args,**kw);tap['entropy']=-y[1]['log_prob'].detach().mean();return y
        cols=['update','alpha','entropy','gradient_preclip']+[f'{k}_{a}' for k in
              ['count','q_sum','target_sum','reward_sum'] for a in ['nop','buy','sell']]
        pointer=path/'RESUME.json'
        step=(s.p.read(path/'recovery'/s.p.read(pointer)['directory']/'STATE.json')['updates']
              if pointer.exists() else 0)
        # Each restart gets a new stream keyed by its committed starting update.
        with (path/f'updates_from_{step:07d}.csv').open('w',buffering=65536) as f, ExitStack() as stack:
            writer=csv.writer(f); writer.writerow(cols)
            def observed(batch):
                nonlocal step
                tap.clear()
                vals=update(batch)
                act=batch['action'].reshape(-1)
                ind=(act[:,None]==torch.arange(3,device=act.device)).float()
                q=tap['q'].gather(1,act[:,None])
                vec=torch.cat([tap['entropy'].reshape(1),ind.sum(0),(ind*q).sum(0),
                    (ind*tap['target'].reshape(-1,1)).sum(0),(ind*batch['next','reward'].reshape(-1,1)).sum(0)]).tolist()
                step+=1
                writer.writerow([step,vals['alpha'],vec[0],vals['gradient_norm'],*vec[1:]])
                if step%256==0: f.flush()
                return vals
            for obj,k,fn in [(trainer,'load_recovery',restore),(trainer,'save_recovery',archive),
                    (loss,'_vmap_qnetworkN0',qs),(loss,'_compute_target',targets),
                    (loss,'actor_loss',actorloss),(model,'update',observed)]:
                stack.enter_context(patch.object(obj,k,fn))
            return original(model,env_factory,assess,design,seed,path,identity,probe,epoch_hook=check)
    with patch.object(s.p,'fit_epochs',fit): s.run(root,8,19,out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('/var/tmp/sac399-series'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--worker',choices=['fixed','adaptive'])
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.worker:
        branch(a.root,a.out,a.worker)
        sys.exit(0)
    with (a.out/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('fixed','adaptive'):
            subprocess.run([sys.executable,'-m','tools.alpha401','--root',str(a.root),'--out',str(a.out),'--worker',name],check=True)
        def metrics(name):
            return [dict(epoch=r['epoch'],coverage=r['metrics']['TRAIN']['signals'],
                    entropy=r['metrics']['TRAIN']['mean_signal_entropy'],alpha=r['last_update']['alpha'],
                    validation_trades=r['metrics']['VALIDATION']['trades'],
                    validation_net_pct=r['metrics']['VALIDATION']['return_pct']) for r in s.p.read(a.out/name/'HISTORY.json')]
        s.p.write(a.out/'COMPARISON.json',dict(fixed=metrics('fixed'),adaptive=metrics('adaptive'),promoted=False))
