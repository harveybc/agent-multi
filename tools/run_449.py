"""Run each checkpoint replay on its original device, no training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import shlex
import subprocess as sp
from role448_worker import emit

ROOT=Path(__file__).resolve().parents[1]
SRC=Path('/var/tmp/role448-temporal-4plus1')
OUT=Path('/var/tmp/role449-cross')
PY='/var/tmp/discrete-sac-probe-venv/bin/python'


def lane(seed,device,remote,host):
    root=SRC/'runtime' if remote else ROOT
    dst=OUT/f'{seed}.json'
    cmd=['env','PYTHONPATH=.:tools','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
         'CUBLAS_WORKSPACE_CONFIG=:4096:8',f'CUDA_VISIBLE_DEVICES={device}',
         PY,'-W','ignore::UserWarning','tools/role449_policy_cross.py',str(SRC),str(seed),str(dst)]
    if remote:
        sp.run(['ssh',host,f'cd {root} && '+shlex.join(cmd)],check=True)
        sp.run(['rsync','-a',f'{host}:{dst}',str(dst)],check=True)
        sp.run(['rsync','-a',f'{host}:{OUT}/policies-{seed}',str(OUT)+'/'],check=True)
    else:sp.run(cmd,cwd=root,check=True)
    return json.loads(dst.read_text())


if __name__=='__main__':
    host=os.environ['ROLE431_SSH_TARGET']
    assert json.loads((SRC/'DONE.json').read_text())['promoted'] is False
    OUT.mkdir(exist_ok=True)
    sp.run(['ssh',host,f'mkdir -p {OUT}'],check=True)
    sp.run(['rsync','-a',str(ROOT/'tools/role449_policy_cross.py'),f'{host}:{SRC}/runtime/tools/'],check=True)
    with ThreadPoolExecutor(3) as pool:
        fs=[pool.submit(lane,*args,host) for args in [(7,1,True),(19,0,True),(43,0,False)]]
        rows=[r for f in fs for r in f.result()]
    assert len(rows)==42
    emit(OUT/'DONE.json',rows)
