"""Paired SELL offer ranking versus expected equal-count random selection."""
from pathlib import Path

import numpy as np

from role425_sell_probability import calibrated, OUT as CAL
from role424_sell_bias import (
    SOURCE, NAMES, Split, atomic, base, common, nn, readout, selected, torch,
)

OUT = Path('/var/tmp/role426-sell-rank-control')


def contrasts(pred, payoff, dates):
    # For k draws without replacement from n, E[sum] = k * population mean.
    mask = selected(pred, dates)
    wk = (dates // 86400000000000 + 3) // 7
    rows = []
    for week in np.unique(wk):
        at = wk == week
        for j, horizon in enumerate(base.HS):
            y = payoff[at, j]
            picked = y[mask[at, j]]
            k = max(1, len(y) // 10)
            assert len(picked) == k and np.isfinite(y).all()
            rows.append(dict(week=int(week), horizon=int(horizon), n=len(y), k=k,
                             model=float(picked.mean()), control=float(y.mean())))
    return rows


def weighted(rows, key):
    return float(np.average([r[key] for r in rows], weights=[r['k'] for r in rows]))


def summarize(rows):
    a, b = [weighted(rows, key) for key in ('model', 'control')]
    weeks = sorted({r['week'] for r in rows})
    blocks = [[r for r in rows if r['week'] == week] for week in weeks]
    sums = np.array([[sum(r['k']*(r['model']-r['control']) for r in block),
                      sum(r['k'] for r in block)] for block in blocks])
    # Resample weeks jointly for all heads, seeds and horizons, within semester.
    rng = np.random.default_rng(426)
    starts = rng.integers(len(blocks), size=(10000,(len(blocks)+3)//4))
    idx = ((starts[:,:,None]+np.arange(4)) % len(blocks)).reshape(10000,-1)[:,:len(blocks)]
    totals = sums[idx].sum(axis=1)
    return dict(weeks=len(weeks), model=a, control=b, delta=a-b,
                ci95=np.quantile(totals[:,0]/totals[:,1], [.025,.975]).tolist())


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz') == base.read(base.DEST/'CACHE.json')['sha']
    refs = {(r['year'],r['seed'],r['arm'],r['method']):r for r in base.read(CAL/'DETAIL.json')}
    with np.load(base.DEST/'market.npz') as z:
        x,t,valid = z['x'],z['t'],z['valid']
    rows = []
    for year in (2022,2023):
        jul = base.ns(f'{year}-07-01')
        for seed in (7,19,43):
            ids,end,_,y,z = common(year,seed,t)
            dt = t[ids]
            ts = z['origin_ns']
            ii,ix = np.searchsorted(t,ts),np.searchsorted(ts,dt)
            fit,ev = (dt<jul)&(end<jul),dt>=jul
            assert np.array_equal(t[ii],ts) and np.array_equal(ts[ix],dt) and valid[ii].all()
            assert end[fit].max()<dt[ev].min() and dt[ev].max()<base.ns(f'{year+1}-01-01')
            for arm in NAMES:
                width = 8 if arm=='shared8' else 32
                net = Split() if arm=='split8' else nn.Sequential(nn.Linear(208,width),nn.SiLU(),nn.Linear(width,12))
                ck = torch.load(SOURCE/f'{year}-{seed}-{arm}.pt',map_location='cpu',weights_only=True)
                net.load_state_dict(ck['weights'])
                net.eval()
                raw,p,w,l = readout(net,x[ii],ck['scale'])
                for method in ('original','conditional'):
                    pred = raw
                    if method=='conditional':
                        pp = np.column_stack([calibrated(p[:,h],w[:,h],l[:,h],ix[fit],y[fit,h]>0,True)[0]
                                              for h in range(4)])
                        pred = pp*w-(1-pp)*l
                    r = contrasts(pred[ix[ev]],y[ev],dt[ev])
                    ref = refs[year,seed,arm,method]
                    assert abs(weighted(r,'model')-ref['matched_bp'])<.001
                    rows.extend(dict(year=year,seed=seed,arm=arm,method=method,**a) for a in r)
    result = {}
    for method in ('original','conditional'):
        for year in (2022,2023):
            r = [v for v in rows if v['year']==year and v['method']==method]
            cells = [[v for v in r if v['arm']==a and v['seed']==s] for a in NAMES for s in (7,19,43)]
            result[f'{method}-{year}'] = dict(**summarize(r),
                cells_beating_control=sum(weighted(c,'model')>weighted(c,'control') for c in cells),
                by_horizon={str(h):summarize([v for v in r if v['horizon']==h]) for h in base.HS})
    atomic(OUT/'WEEKS.json',rows)
    atomic(OUT/'RESULTS.json',result)
    for key,r in result.items():
        print(key,{k:v for k,v in r.items() if k!='by_horizon'},flush=True)


if __name__=='__main__':
    main()
