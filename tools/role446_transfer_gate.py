"""Shadow weekly and frozen rankings; use only fully matured previous H2 weeks to switch."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from role424_sell_bias import selected
from role443_weekly_refit import atomic,base,cell,week,score,SRC,DAY

OUT = Path('/var/tmp/role446-transfer-gate')


def replay(key,pn,dates,ix,y,jul,ts,maturity,frozen,weekly,ref):
    year,seed = key
    stream = frozen.copy()
    hist = []
    for w in np.unique(week(ts[ts>=jul])):
        cut = (int(w)*7-3)*DAY
        past = [r for r in hist if r['mature']<cut]
        credit = sum(r['net'] for r in past)
        use_weekly = credit>0
        current = (ts>=jul)&(week(ts)==w)
        if use_weekly:
            stream[current] = weekly[current]
        sel = (dates>=jul)&(week(dates)==w)
        if sel.any():
            d = dates[sel]
            assert np.array_equal(ts[ix[sel]],d)
            a,b = [selected(p[ix[sel]],d) for p in (frozen,weekly)]
            assert a.sum()==b.sum()
            payoff = y[sel]
            hist.append(dict(w=int(w),net=float(payoff[b].sum()-payoff[a].sum()),
                             mature=int(maturity[sel].max()),credit=float(credit),weekly=use_weekly))
    assert not hist[0]['weekly'] and np.array_equal(stream[ts<jul],frozen[ts<jul])
    out = score(stream,pn,dates,ix,y,jul,ts,True)
    row = dict(year=year,seed=seed,frozen=ref['arms']['frozen'],weekly=ref['arms']['weekly'],
               switched=out,weeks=hist)
    atomic(OUT/f'{year}-{seed}.json',row)
    return row


def run(key):
    return cell(key,account_check=replay)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC/'PROTOCOL.json')['market']
    with ProcessPoolExecutor(3) as pool:
        rows=list(pool.map(run,[(y,s) for y in (2022,2023) for s in (7,19,43)]))
    atomic(OUT/'DONE.json',rows)
