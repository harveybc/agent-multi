"""Run one bounded price audit per explicitly assigned GPU and verify collection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

def run(argv):return subprocess.check_output(argv,text=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def worker(job):
    seed,host,device=job.split(':');base=a.out;data=base/f'{seed}.npz'
    script=Path(__file__).with_name('segment_business388.py').resolve()
    result=base/f'{seed}_metrics.npz'
    if host=='local':
        receipt=json.loads(run(['env','CUDA_VISIBLE_DEVICES='+device,'OMP_NUM_THREADS=1',
            sys.executable,str(script),'--data',str(data),'--out',str(result)]))
    else:
        dest='/var/tmp/segment-business388-'+seed
        ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host]
        run(ssh+['mkdir','-p',dest])
        run(['rsync','-a','--checksum',str(script),str(data),host+':'+dest+'/'])
        line='cd '+shlex.quote(dest)+' && sha256sum '+script.name+' '+data.name
        hashes=run(ssh+[line]).splitlines()
        if [x.split()[0] for x in hashes]!=[digest(script),digest(data)]:raise ValueError('input identity')
        cmd=['env','CUDA_VISIBLE_DEVICES='+device,'OMP_NUM_THREADS=1',a.python,script.name,
             '--data',data.name,'--out',result.name]
        receipt=json.loads(run(ssh+['cd '+shlex.quote(dest)+' && '+shlex.join(cmd)]))
        run(['rsync','-a','--checksum',host+':'+dest+'/'+result.name,str(result)])
    if receipt['sha256']!=digest(result):raise ValueError('result identity')
    receipt.update(seed=seed,input_sha256=digest(data),worker_sha256=digest(script))
    (base/f'{seed}_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(seed,receipt['device'],receipt['stress'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--python',required=True);p.add_argument('jobs',nargs=4)
    a=p.parse_args()
    if {x.split(':')[0] for x in a.jobs}!={'19','7','43','71'}:raise ValueError('seed allocation')
    with ThreadPoolExecutor(4) as pool:list(pool.map(worker,a.jobs))
