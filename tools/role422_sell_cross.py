"""Replay saved SELL heads across closes; compare at weekly matched exposure."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from role421_sell_value import OUT as SOURCE, ARMS, base, atomic, panel, common, score, output

OUT=Path('/var/tmp/role422-sell-cross')


def main():
    assert (SOURCE/'DONE.json').is_file()
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    x,t,valid=bank;rows=[]
    for year in (2022,2023):
        jul=base.ns(f'{year}-07-01')
        for close in (7,19,43):
            pn=panel(year,close,*bank)
            ids,_,_,y,z=common(year,close,t);dates=t[ids];ts=z['origin_ns'];ii=np.searchsorted(t,ts);ix=np.searchsorted(ts,dates)
            assert np.array_equal(t[ii],ts) and np.array_equal(ts[ix],dates) and np.array_equal(pn.p,z['p'])
            weeks=(dates//86400000000000+3)//7;ev=dates>=jul
            groups=[np.flatnonzero(ev&(weeks==w)) for w in np.unique(weeks[ev])]
            counts=np.array([max(1,len(g)//10) for g in groups])
            control=float(np.average([y[g].mean() for g in groups],weights=counts))
            for op in (7,19,43):
                own=base.read(SOURCE/f'{year}-{op}.json')
                for a in ARMS:
                    net=base.Expert(208,False,'cls').net
                    if a=='severity':net[-1]=nn.Linear(8,12)
                    ck=torch.load(SOURCE/f'{year}-{op}-{a}.pt',weights_only=True,map_location='cpu')
                    net.load_state_dict(ck['weights']);net.eval()
                    with torch.no_grad():pred=(output(a,net(torch.from_numpy(x[ii])),ck['means'])*ck['scale']).numpy()
                    for gate in ('rank','positive'):
                        s=score(pred,pn,dates,ix,y,jul,ts,gate=='positive');ac=s['account']
                        if op==close:
                            ref=own['arms'][a][gate]
                            assert ref['account']==ac and abs(ref['bp']-s['bp'])<1e-9
                        rows.append(dict(year=year,open_seed=op,close_seed=close,arm=a,gate=gate,ret=ac['ret'],
                            trades=ac['trades'],weeks=own['weeks'],matched_bp=s['bp'],control_bp=control,lift_bp=s['bp']-control))
            atomic(OUT/f'progress-{year}-{close}.json',dict(year=year,closer=close,completed=len(rows)))
    atomic(OUT/'REPLAYS.json',rows)
    metrics={}
    for a in ARMS:
        for gate in ('rank','positive'):
            group=[r for r in rows if (r['arm'],r['gate'])==(a,gate)]
            metrics[a+'_'+gate]=dict(cells=len(group),positive_accounts=sum(r['ret']>0 for r in group),
                active_positive=sum(r['ret']>0 and r['trades']>2*r['weeks'] for r in group),
                positive_lift=sum(r['lift_bp']>0 for r in group),positive_matched=sum(r['matched_bp']>0 for r in group),
                median_net_pct=float(np.median([r['ret'] for r in group])),median_lift=float(np.median([r['lift_bp'] for r in group])))
    atomic(OUT/'DONE.json',dict(metrics=metrics,historical_semesters=2,promoted=False))


if __name__=='__main__':main()
