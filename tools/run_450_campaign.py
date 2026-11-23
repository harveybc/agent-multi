"""Paired full-policy inheritance, identical 448 optimizer and data; no automatic promotion."""
import fcntl
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from run_448_campaign import ROOT,OUT as SRC,HOST,atomic,base,launch,run,summarize

OUT=Path('/var/tmp/role450-full-policy')


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert len(base.read(SRC/'DONE.json')['cells'])==3
    assert len(base.read(Path('/var/tmp/role449-cross/DONE.json')))==42
    worker=ROOT/'tools/role448_worker.py'
    spec=dict(worker=base.file_sha(worker),seeds=[7,19,43],control='448 archived',rule='parent full policy vs epoch-zero recalibration; ties keep incumbent; trained candidates identical to 448',source={str(s):base.file_sha(SRC/f'{s}.json') for s in (7,19,43)})
    if (OUT/'PROTOCOL.json').exists():assert base.read(OUT/'PROTOCOL.json')==spec
    atomic(OUT/'PROTOCOL.json',spec);shutil.copyfile(worker,OUT/worker.name)
    run(['ssh',HOST,f'mkdir -p {OUT}'])
    run(['rsync','-a',str(OUT/worker.name),str(OUT/'PROTOCOL.json'),f'{HOST}:{OUT}/'])
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(launch,*args,OUT) for args in [(7,1,True),(19,0,True),(43,0,False)]]
        done=[j.result() for j in jobs]
    summarize(OUT,done)
    comparisons=[]
    for seed in done:
        for arm in ('validation','retention'):
            original,new=[base.read(d/f'run-{seed}'/f's7-{arm}.json')['val'] for d in (SRC,OUT)]
            comparisons.append(dict(seed=seed,arm=arm,original=original,new=new,
                            improved=new['ret']>original['ret'],positive_active=new['ret']>0 and new['orders_week']>2))
    atomic(OUT/'COMPARISON.json',dict(rows=comparisons,promoted=False))


if __name__=='__main__':main()
