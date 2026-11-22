"""Expanding weekly linear readout versus frozen 435 and time-matched shifted updates."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from role438_close_control import SRC, base, common, panel, atomic, score

OUT = Path('/var/tmp/role443-weekly-refit')
DAY = 86400000000000


def week(t):
    return (t//DAY+3)//7


def cell(key, audit=None, observe_update=None, account_check=None):
    year,seed = key
    tag = f'{year}-{seed}'
    base.torch.set_num_threads(1)
    with np.load(base.DEST/'market.npz') as bank:
        t = bank['t']
        ids,end,_,y,z = common(year,seed,t)
        pn = panel(year,seed,bank['x'],t,bank['valid'])
    ts = z['origin_ns']
    dates = t[ids]
    ix = np.searchsorted(ts,dates)
    assert np.array_equal(ts[ix],dates) and np.array_equal(pn.p,z['p'])
    assert np.array_equal(pn.e.v[pn.bars]+base.HOUR,ts)
    with np.load(SRC/f'{tag}.scores.npz') as d:
        assert str(d['sha'])==base.file_sha(SRC/f'{tag}.npz')
        frozen = d['linear']
    with np.load(SRC/f'{tag}.npz') as d:
        x,u,target,wk = d['x'].astype(float),d['u'].astype(float),d['y'].astype(float),d['week']
    mu = x.mean(0)
    sd = np.maximum(x.std(0),1e-6)
    x,u = (x-mu)/sd,(u-mu)/sd
    sums,n = x.sum(0),len(x)
    for w in np.unique(wk):
        i = wk==w
        x[i] -= x[i].mean(0)
        target[i] -= target[i].mean(0)
    gram = x.T@x
    rhs = x.T@target
    identity = np.eye(len(gram))
    def predict(rows,right):
        coef = np.linalg.solve(gram+.01*n*identity,right)
        return (u[rows]-sums/n)@coef
    assert np.allclose(predict(slice(None),rhs),frozen,atol=1e-7,rtol=1e-7)
    jul = base.ns(f'{year}-07-01')
    cal = (dates<jul)&(end<jul)
    intercept = (y[cal]-frozen[ix[cal]]).mean(0)
    dynamic,placebo = frozen.copy(),frozen.copy()
    if audit is not None:
        initial_rhs = rhs.copy()
        initial_system = gram+.01*n*identity
        initial_center = sums/n
        geometry,labels_only = frozen.copy(),frozen.copy()
    shifted = rhs.copy()
    seen = set()
    exposures = []
    clock = week(ts)
    source_week = week(dates)
    maturity = np.maximum(end,pn.e.ff[pn.e.seg_of[pn.bars[ix]]])
    rng = np.random.default_rng(443)
    for w in np.unique(clock[ts>=jul]):
        boundary = (int(w)*7-3)*DAY
        if observe_update is not None:
            previous_coef = np.linalg.solve(gram+.01*n*identity,rhs)
            previous_center = sums/n
            incorporated = []
        for old in sorted(set(source_week[(dates>=jul)&(source_week<w)])-seen):
            ii = np.flatnonzero(source_week==old)
            if dates[ii].min()<jul or maturity[ii].max()>=boundary:
                continue
            seen.add(old)
            if len(ii)<2:
                continue
            f = u[ix[ii]]
            dx = f-f.mean(0)
            dy = y[ii]-y[ii].mean(0)
            gram += dx.T@dx
            rhs += dx.T@dy
            shifted += dx.T@np.roll(dy,rng.integers(1,len(ii)),axis=0)
            n += len(ii)
            sums += f.sum(0)
            if observe_update is not None:
                incorporated.append(ii)
        rows = (clock==w)&(ts>=jul)
        dynamic[rows] = predict(rows,rhs)
        if observe_update is not None and incorporated:
            held_week = np.flatnonzero((source_week==w)&(dates>=jul))
            for label,block in [('new_fit',ii) for ii in incorporated]+[('next',held_week)]:
                if len(block):
                    assert label=='next' or maturity[block].max()<boundary
                    old_p = (u[ix[block]]-previous_center)@previous_coef
                    new_p = predict(ix[block],rhs)
                    observe_update(int(w),label,dates[block],y[block],old_p,new_p)
        placebo[rows] = predict(rows,shifted)
        if audit is not None:
            geometry[rows] = predict(rows,initial_rhs)
            labels_only[rows] = (u[rows]-initial_center)@np.linalg.solve(initial_system,rhs)
        exposures.append(dict(week=int(w),added=n-len(x)))
    assert exposures[0]['added']==0 and exposures[-1]['added']>0
    arms = {k:score(p+intercept,pn,dates,ix,y,jul,ts,True) for k,p in
            [('frozen',frozen),('weekly',dynamic),('shifted',placebo)]}
    ref = base.read(SRC/f'{tag}.json')['linear']
    assert arms['frozen']['account']==ref['account'] and abs(arms['frozen']['bp']-ref['bp'])<1e-7
    result = dict(year=year,seed=seed,arms=arms,updates=exposures)
    if account_check is not None:
        assert result==base.read(OUT/f'{tag}.json')
        return account_check(key,pn,dates,ix,y,jul,ts,maturity,frozen+intercept,dynamic+intercept,result)
    if audit is not None:
        assert result==base.read(OUT/f'{tag}.json')
        held = dates>=jul
        return audit(key,pn,ix[held],dates[held],y[held],result,
                     {k:(v+intercept)[ix[held]] for k,v in
                      [('frozen',frozen),('weekly',dynamic),('geometry',geometry),
                       ('labels',labels_only),('shifted',placebo)]})
    if observe_update is not None:
        assert result==base.read(OUT/f'{tag}.json')
        return result
    atomic(OUT/f'{tag}.json',result)
    return result


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC/'PROTOCOL.json')['market']
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(cell,[(y,s) for y in (2022,2023) for s in (7,19,43)]))
    atomic(OUT/'DONE.json',rows)
