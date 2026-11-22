"""Offer-level perfect-foresight ceiling for the frozen role427 rank comparisons."""
from pathlib import Path

import numpy as np

from role427_sell_rank_stages import OUT as PRIOR
from role426_sell_rank_control import contrasts, weighted, common, atomic, base

OUT = Path('/var/tmp/role428-sell-oracle')


def metrics(rows):
    rnd, model, oracle = (weighted(rows,key) for key in ('control','model','oracle'))
    return dict(random_bp=rnd, model_bp=model, oracle_bp=oracle, gap_bp=oracle-model,
                capture_pct=100*(model-rnd)/(oracle-rnd) if oracle>rnd else None)


def main():
    refs = base.read(PRIOR/'WEEKS.json')
    assert base.file_sha(base.DEST/'market.npz') == base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:
        t = z['t']
    ceilings = {}
    for year in (2022,2023):
        cut,jan,jul = map(base.ns,(f'{year-1}-07-01',f'{year}-01-01',f'{year}-07-01'))
        for seed in (7,19,43):
            yrs = [common(yr,seed,t) for yr in range(2020,year+1)]
            ids,end,y = [np.concatenate([r[k] for r in yrs[:-1]]) for k in (0,1,3)]
            tr,mon = (t[ids]<cut)&(end<cut),(t[ids]>=cut)&(end<jan)
            cur,ends,_,cy,_ = yrs[-1]
            h1,h2 = (t[cur]<jul)&(ends<jul),t[cur]>=jul
            for stage,i,v in [('fit',ids[tr],y[tr]),('monitor',ids[mon],y[mon]),
                              ('calibration',cur[h1],cy[h1]),('held',cur[h2],cy[h2])]:
                for r in contrasts(v,v,t[i]):
                    key = year,seed,stage,r['week'],r['horizon']
                    assert key not in ceilings
                    ceilings[key] = r
    used = set()
    for r in refs:
        key = tuple(r[k] for k in ('year','seed','stage','week','horizon'))
        a = ceilings[key]
        assert (a['n'],a['k']) == (r['n'],r['k']) and abs(a['control']-r['control'])<1e-7
        r['oracle'] = a['model']
        assert r['oracle']+1e-5>=max(r['model'],r['control'])
        used.add(key)
    assert used==ceilings.keys()
    out = {}
    for year in (2022,2023):
        for stage in ('fit','monitor','calibration','held'):
            rows = [r for r in refs if r['year']==year and r['stage']==stage]
            out[f'{year}-{stage}'] = dict(**metrics(rows),
                by_horizon={str(h):metrics([r for r in rows if r['horizon']==h]) for h in base.HS})
            print(year,stage,metrics(rows),flush=True)
    OUT.mkdir(exist_ok=True)
    atomic(OUT/'WEEKS.json',refs)
    atomic(OUT/'RESULTS.json',out)


if __name__=='__main__':
    main()
