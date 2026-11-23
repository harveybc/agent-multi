"""Stage workers on the two authorized hosts, collect artifacts and draw stage curves."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess as sp
import time
from concurrent.futures import ThreadPoolExecutor

from role448_prepare import OUT,base,atomic

ROOT=Path(__file__).resolve().parents[1]
PY='/var/tmp/discrete-sac-probe-venv/bin/python'
HOST=os.environ['ROLE431_SSH_TARGET']
REMOTE=OUT/'runtime'


def run(args):
    sp.run(args,check=True)


def launch(seed,device,remote,policy_output=None):
    work=REMOTE if remote else ROOT
    unit=f'role{450 if policy_output else 448}-s{seed}'
    worker=str(policy_output/'role448_worker.py') if policy_output else 'tools/role448_worker.py'
    args=['systemd-run','--user',f'--unit={unit}','--collect',f'--working-directory={work}',
          '--setenv=PYTHONPATH=.:tools',f'--setenv=CUDA_VISIBLE_DEVICES={device}',
          '--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1',
          PY,'-W','ignore::UserWarning',worker,str(OUT),str(seed)]
    if policy_output:args.append(str(policy_output))
    def cmd(a):
        return ['ssh','-o','ConnectTimeout=8',HOST,shlex.join(a)] if remote else a
    query=cmd(['systemctl','--user','is-active',unit])
    dest=(policy_output or OUT)/f'run-{seed}'
    dest.mkdir(exist_ok=True)
    if not (dest/'DONE.json').exists() and sp.call(query,stdout=sp.DEVNULL)!=0:
        run(cmd(args))
    while not (dest/'DONE.json').exists():
        time.sleep(15)
        if remote:
            run(['rsync','-a','--exclude=*.resume','--exclude=*.tmp','--exclude=*.writing',
                 f'{HOST}:{dest}/',str(dest)+'/'])
        if not (dest/'DONE.json').exists() and sp.call(query,stdout=sp.DEVNULL)!=0:
            raise RuntimeError(f'worker stopped without DONE: {unit}')
    return seed


def main():
    lock=(OUT/'lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for seed in (7,19,43):
        assert (OUT/f'{seed}.json').is_file()
    run(['ssh',HOST,f'mkdir -p {REMOTE}'])
    for sub in ('app','agent_plugins','env_plugins','tools'):
        run(['rsync','-a','--include=*/','--include=*.py','--exclude=*',str(ROOT/sub),f'{HOST}:{REMOTE}/'])
    run(['rsync','-a',str(base.PARENT/'engine'),str(OUT)+'/'])
    run(['rsync','-a',str(OUT/'engine'),str(OUT/'recipe.json'),str(OUT/'arch.json'),
         *[str(OUT/f'{seed}.{ext}') for seed in (7,19) for ext in ('npz','sim.gz','json')],f'{HOST}:{OUT}/'])
    with ThreadPoolExecutor(3) as pool:
        fs=[pool.submit(launch,*p) for p in [(7,1,True),(19,0,True),(43,0,False)]]
        done=[f.result() for f in fs]
    summarize(OUT,done)


def summarize(output,done):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    summary=[]
    for seed in done:
        root=output/f'run-{seed}'
        for path in sorted(root.glob('s*.pt')):
            d=torch.load(path,map_location='cpu',weights_only=False)
            hist=d['history'];fig,ax=plt.subplots()
            ax.plot([r['epoch'] for r in hist],[r['val']['ret'] for r in hist],label='Validation 2024')
            for j in range(len(hist[0]['prior'])):
                ax.plot([r['epoch'] for r in hist],[r['prior'][j]['ret'] for r in hist],alpha=.6,label=f'Train {j+1}')
            ax.set(xlabel='Epoch',ylabel='Net return (%)');ax.legend();fig.tight_layout()
            fig.savefig(path.with_suffix('.png'));plt.close(fig)
        summary.append({arm:json.loads((root/f's7-{arm}.json').read_text()) for arm in ('validation','retention')}|dict(seed=seed))
    atomic(output/'DONE.json',dict(cells=summary,promoted=False))


if __name__=='__main__':main()
