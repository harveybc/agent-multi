"""Test the single weekly update on the assimilated and immediately following blocks."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from role443_weekly_refit import atomic,base,cell,week,SRC
from role424_sell_bias import selected

OUT = Path('/var/tmp/role445-weekly-transfer')


def run(key):
    year,seed = key
    rows = []
    def observe(step,label,ts,y,pre,post):
        group = np.unique(week(ts))
        assert len(group)==1 and (group[0]==step if label=='next' else group[0]<step)
        a,b = selected(pre,ts),selected(post,ts)
        for j,h in enumerate(base.HS):
            n = int(a[:,j].sum())
            assert b[:,j].sum()==n
            gain = y[b[:,j],j].sum()-y[a[:,j],j].sum()
            rows.append(dict(year=year,seed=seed,update=step,block=int(group[0]),scope=label,
                             h=int(h),n=n,gain=float(gain),swaps=int((b[:,j]&~a[:,j]).sum())))
    cell(key,observe_update=observe)
    scopes = {s:{r['update'] for r in rows if r['scope']==s} for s in ('new_fit','next')}
    assert scopes['next'] and scopes['next']==scopes['new_fit']
    atomic(OUT/f'{year}-{seed}.json',rows)
    return rows


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC/'PROTOCOL.json')['market']
    with ProcessPoolExecutor(3) as pool:
        rows = [r for block in pool.map(run,[(y,s) for y in (2022,2023) for s in (7,19,43)]) for r in block]
    atomic(OUT/'DONE.json',rows)
