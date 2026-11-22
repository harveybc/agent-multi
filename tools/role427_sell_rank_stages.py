"""Where does the original SELL ranker's equal-count advantage disappear?"""
from pathlib import Path

import numpy as np

from role424_sell_bias import OUT as AUDIT, SOURCE, NAMES, Split, atomic, base, common, nn, readout, torch
from role426_sell_rank_control import OUT as HELD, contrasts, summarize, weighted

OUT = Path('/var/tmp/role427-sell-rank-stages')


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    refs = {(r['year'],r['seed'],r['arm']):r['slices'] for r in base.read(AUDIT/'DETAIL.json')}
    held = base.read(HELD/'RESULTS.json')
    assert base.file_sha(base.DEST/'market.npz') == base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:
        x,t,valid = z['x'],z['t'],z['valid']
    rows = []
    for year in (2022,2023):
        cut,jan,jul = map(base.ns,(f'{year-1}-07-01',f'{year}-01-01',f'{year}-07-01'))
        for seed in (7,19,43):
            past = [common(yr,seed,t) for yr in range(2020,year)]
            ids,end,y = [np.concatenate([z[k] for z in past]) for k in (0,1,3)]
            tr = (t[ids]<cut)&(end<cut)
            mon = (t[ids]>=cut)&(end<jan)
            cur,ends,_,net_y,_ = common(year,seed,t)
            h1,h2 = (t[cur]<jul)&(ends<jul),t[cur]>=jul
            assert end[tr].max()<t[ids[mon]].min() and end[mon].max()<t[cur[h1]].min()
            assert ends[h1].max()<t[cur[h2]].min() and t[cur[h2]].max()<base.ns(f'{year+1}-01-01')
            for arm in NAMES:
                width = 8 if arm=='shared8' else 32
                net = Split() if arm=='split8' else nn.Sequential(nn.Linear(208,width),nn.SiLU(),nn.Linear(width,12))
                ck = torch.load(SOURCE/f'{year}-{seed}-{arm}.pt',map_location='cpu',weights_only=True)
                assert np.isclose(ck['scale'],y[tr].std())
                net.load_state_dict(ck['weights'])
                net.eval()
                for stage,i,v in [('fit',ids[tr],y[tr]),('monitor',ids[mon],y[mon]),
                                  ('calibration',cur[h1],net_y[h1]),('held',cur[h2],net_y[h2])]:
                    assert valid[i].all() and np.all(np.diff(t[i])>0)
                    pred = readout(net,x[i],ck['scale'])[0]
                    r = contrasts(pred,v,t[i])
                    ref = refs[year,seed,arm][stage]['top']
                    assert abs(weighted(r,'model')-ref['actual'])<.001 and sum(a['k'] for a in r)==ref['n']
                    rows.extend(dict(year=year,seed=seed,arm=arm,stage=stage,**a) for a in r)
    result = {}
    for year in (2022,2023):
        for stage in ('fit','monitor','calibration','held'):
            r = [v for v in rows if v['year']==year and v['stage']==stage]
            cells = [[v for v in r if v['arm']==a and v['seed']==s] for a in NAMES for s in (7,19,43)]
            stats = dict(**summarize(r),better=sum(weighted(c,'model')>weighted(c,'control') for c in cells),
                         by_horizon={str(h):summarize([v for v in r if v['horizon']==h]) for h in base.HS})
            if stage=='held':
                assert abs(stats['delta']-held[f'original-{year}']['delta'])<.001
            result[f'{year}-{stage}']=stats
            print(year,stage,{k:v for k,v in stats.items() if k!='by_horizon'},flush=True)
    atomic(OUT/'WEEKS.json',rows)
    atomic(OUT/'RESULTS.json',result)


if __name__=='__main__':
    main()
