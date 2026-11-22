"""Conditional CPU fan-out: ablate only the positive EV gate of failed 435 probes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from role429_campaign import base,atomic,common,panel,score

SRC=Path('/var/tmp/role435-within-week')
OUT=Path('/var/tmp/role436-gate-replay')


def cell(tag,bank):
    year,seed=map(int,tag.split('-'));ids,end,_,y,z=common(year,seed,bank[1]);dates=bank[1][ids];ts=z['origin_ns']
    ix=np.searchsorted(ts,dates);jul=base.ns(f'{year}-07-01');cal=(dates<jul)&(end<jul)
    pn=panel(year,seed,*bank);assert np.array_equal(pn.p,z['p']) and np.array_equal(ts[ix],dates)
    ref=base.read(SRC/f'{tag}.json');rows={}
    with np.load(SRC/f'{tag}.scores.npz') as d:
        for k in ('linear','nonlinear'):
            p=d[k];p=p+(y[cal]-p[ix[cal]]).mean(0)
            closed=score(p,pn,dates,ix,y,jul,ts,True)
            assert closed['account']==ref[k]['account']
            relaxed=score(p,pn,dates,ix,y,jul,ts,False)
            rows[k]=dict(positive_gate=closed['account'],rank_only=relaxed['account'])
    atomic(OUT/f'{tag}.json',rows);return rows


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    done=base.read(SRC/'DONE.json');assert done['cells']==6
    if done['replication']:
        atomic(OUT/'DONE.json',dict(skipped='probe qualified for independent replication'));return
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC/'PROTOCOL.json')['market']
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    with ThreadPoolExecutor(3) as pool:
        rows=list(pool.map(lambda tag:cell(tag,bank),[f'{y}-{s}' for y in (2022,2023) for s in (7,19,43)]))
    atomic(OUT/'DONE.json',dict(cells=len(rows),improved={k:sum(r[k]['rank_only']['ret']>r[k]['positive_gate']['ret'] for r in rows)
                                for k in ('linear','nonlinear')},promoted=False))


if __name__=='__main__':main()
