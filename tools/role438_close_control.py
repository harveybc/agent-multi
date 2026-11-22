"""Research-only account counterfactual; retain all forced exits and opening bids."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from role437_sell_attribution import SRC, atomic, base, common, panel
from role421_sell_value import score

OUT = Path('/var/tmp/role438-close-control')


def cell(key):
    year, seed = key
    tag = f'{year}-{seed}'
    base.torch.set_num_threads(1)
    with np.load(base.DEST/'market.npz') as bank:
        t = bank['t']
        ids,end,_,y,z = common(year,seed,t)
        pn = panel(year,seed,bank['x'],t,bank['valid'])
    dates, ts = t[ids], z['origin_ns']
    ix = np.searchsorted(ts,dates)
    assert np.array_equal(ts[ix],dates) and np.array_equal(pn.p,z['p'])
    jul = base.ns(f'{year}-07-01')
    fit = (dates<jul)&(end<jul)
    with np.load(SRC/f'{tag}.scores.npz') as d:
        assert str(d['sha']) == base.file_sha(SRC/f'{tag}.npz')
        p = d['nonlinear']
    p = p+(y[fit]-p[ix[fit]]).mean(0)

    def replay():
        return score(p,pn,dates,ix,y,jul,ts,True)['account']

    active = replay()
    assert active == base.read(SRC/f'{tag}.json')['nonlinear']['account']
    with patch.object(pn,'close',lambda eng,k,j: 0):
        passive = replay()
    assert replay()==active
    row = dict(year=year,seed=seed,managed=active,barriers=passive)
    atomic(OUT/f'{tag}.json',row)
    return row


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC/'PROTOCOL.json')['market']
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(cell,[(y,s) for y in (2022,2023) for s in (7,19,43)]))
    atomic(OUT/'DONE.json',rows)
