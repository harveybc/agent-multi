"""Supplement322 with FIT-balanced representable three-action teacher controls."""
import argparse
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write,now

ROOT=Path(__file__).resolve().parents[1]


def run(a):
    begun=time.monotonic();previous=read(a.previous/'RESULT.json');verify_done(a.previous,previous['identity'])
    if not previous['reproducibility']['exact']:raise ValueError('BALANCED_REPRODUCIBILITY')
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('BALANCED_DEVICE')
    cfg=dict(schema='balanced_teacher_forensics.v1',previous=previous['identity'],
        cache_sha256=file_sha(a.previous/'cache/DATA.pt'),lr=read(a.previous/'units/A_19_base/RESULT.json')['task']['lr'],
        code={k:file_sha(ROOT/k) for k in ('tools/run_balanced_teacher_forensics.py','tools/run_training_forensics.py','app/training_forensics.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('BALANCED_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    cache=a.output/'cache';cache.mkdir(parents=True,exist_ok=True)
    if not (cache/'DONE.json').exists():
        for name in ('DATA.pt','FOLD_A.json','FOLD_B.json'):shutil.copyfile(a.previous/'cache'/name,cache/name)
        seal(cache,cfg['identity'])
    else:verify_done(cache,cfg['identity'])
    tasks=[dict(name=f'{fold}_{seed}_balanced',fold=fold,seed=seed,lr=cfg['lr'],updates=8192,target='balanced_teacher')
        for fold in ('A','B') for seed in (19,43)]
    tasks.append(dict(name='balanced_sinusoid',fold='A',seed=19,lr=cfg['lr'],updates=8192,target='balanced_sinusoid'))
    results=[]
    for task in tasks:
        path=a.output/'tasks'/f"{task['name']}.json";write(path,task)
        write(a.output/'PROGRESS.json',dict(stage='RUNNING',unit=task['name'],completed=len(results),at=now()))
        with (a.output/'WORKERS.log').open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'tools/run_training_forensics.py'),'--worker',
                '--output',str(a.output),'--task',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
        d=a.output/'units'/task['name'];r=read(d/'RESULT.json');verify_done(d,r['identity']);results.append(r)
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],units=results,original_322_retained=True,
        source_changed=False,model_promoted=False,validation_read=False,test_read=False))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',completed=5,seconds=time.monotonic()-begun,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('previous','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
