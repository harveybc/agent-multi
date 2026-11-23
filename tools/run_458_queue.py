"""Launch three persistent seed queues; sync small live receipts and finished champions."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess as sp
import time

from role458_full_train import OUT,emit,file_sha

REPO=Path(__file__).resolve().parents[1]
PY='/var/tmp/discrete-sac-probe-venv/bin/python'
def main():
    host=os.environ['ROLE431_SSH_TARGET']
    lease=(OUT/'coordinator.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    worker=REPO/'tools/role458_full_train.py'
    sp.run(['ssh',host,f'mkdir -p {OUT}/seed-7 {OUT}/seed-19'],check=True)
    sp.run(['rsync','-a',str(worker),str(OUT/'encoder.pt'),str(OUT/'TRANSFER.json'),f'{host}:{OUT}/'],check=True)
    jobs=[(7,1,True),(19,0,True),(43,0,False)]
    def launch(seed,dev,remote):
        unit=f'role458-seed-{seed}'
        cmd=lambda a:['ssh',host,shlex.join(a)] if remote else a
        work='/var/tmp/role448-temporal-4plus1/runtime' if remote else str(REPO)
        if sp.run(cmd(['systemctl','--user','is-active','--quiet',unit])).returncode!=0:
            sp.run(cmd(['systemd-run','--user','--collect','--unit='+unit,'--working-directory='+work,
                  '--setenv=PYTHONPATH=.:tools','--setenv=CUDA_VISIBLE_DEVICES='+str(dev),
                  '--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
                  '--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8',PY,'-Wignore::UserWarning',
                  str(OUT/worker.name if remote else worker),str(seed)]),check=True)
    for job in jobs:launch(*job)
    retries={seed:0 for seed,_,_ in jobs}
    while True:
        for seed,dev,remote in jobs:
            if remote:
                res=sp.run(['rsync','-a','--exclude=*.writing','--exclude=resume.pt','--exclude=champ.pt',
                            f'{host}:{OUT}/seed-{seed}',str(OUT)+'/'])
                if res.returncode:continue
            if (OUT/f'seed-{seed}/DONE.json').exists():continue
            query=['systemctl','--user','is-active','--quiet',f'role458-seed-{seed}']
            status=sp.run(['ssh',host,shlex.join(query)] if remote else query).returncode
            if status not in (0,255):
                if retries[seed]>=2:
                    emit(OUT/'FAILED.json',dict(seed=seed,retries=retries,reason='worker stopped without DONE'))
                    raise RuntimeError(f'seed {seed} repeatedly stopped')
                retries[seed]+=1;launch(seed,dev,remote)
        if all((OUT/f'seed-{s}/DONE.json').exists() for s,_,_ in jobs):break
        time.sleep(30)
    for seed,_,remote in jobs:
        if remote:sp.run(['rsync','-a','--include=*/','--include=champ.pt','--exclude=*',
                          f'{host}:{OUT}/seed-{seed}',str(OUT)+'/'],check=True)
    results={}
    for s,_,_ in jobs:
        results[s]={p.parent.name:json.loads(p.read_text()) for p in (OUT/f'seed-{s}').glob('*/DONE.json')}
        assert len(results[s])==4
    emit(OUT/'COMPARISON.json',dict(results=results,promoted=False,worker_sha=file_sha(worker)))

if __name__=='__main__':main()
