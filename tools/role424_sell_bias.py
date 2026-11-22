"""Read-only audit of 423 champions, chronological slices and frozen bias correction."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from role423_sell_horizons import OUT as SOURCE, Split, NAMES, common, atomic, base, panel, score

OUT=Path('/var/tmp/role424-sell-bias')


def selected(pred,ts):
    week=(ts//86400000000000+3)//7
    mask=np.zeros(pred.shape,bool)
    for w in np.unique(week):
        i=np.flatnonzero(week==w)
        k=max(1,len(i)//10)
        loc=np.argsort(-pred[i],axis=0,kind='stable')[:k]
        mask[i[loc],np.arange(4)]=True
    return mask


def readout(model,feats,scale):
    with torch.inference_mode():
        z=model(torch.from_numpy(feats)).reshape(-1,4,3)
        prob=z[:,:,0].sigmoid()
        win=F.softplus(z[:,:,1])
        lose=F.softplus(z[:,:,2])
        pred=((prob*win-(1-prob)*lose)*scale).numpy()
    return (pred,prob.numpy(),(win*scale).numpy(),(lose*scale).numpy())


def summary(outputs,y,ts):
    pred,p,w,l=[v.astype(float) for v in outputs]
    b=y>0
    terms=[(p-b)*(w+l),b*(w-y),(~b)*(-l-y)]
    # Float32 output heads are compared to a double-precision attribution.
    assert np.allclose(sum(terms),pred-y,rtol=1e-5,atol=4e-5)
    top=selected(pred,ts)
    res={}
    for label,m in [('all',np.ones_like(top)),('top',top)]:
        res[label]=dict(n=int(m.sum()),prediction=float(pred[m].mean()),actual=float(y[m].mean()),
            bias=float((pred-y)[m].mean()),prob_error=float(terms[0][m].mean()),win_size_error=float(terms[1][m].mean()),
            loss_size_error=float(terms[2][m].mean()),win_fraction=float(b[m].mean()),mean_probability=float(p[m].mean()))
    return res


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    assert not base.read(SOURCE/'DONE.json')['promoted']
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    x,t,valid=bank;rows=[]
    for year in (2022,2023):
        jul=base.ns(f'{year}-07-01');cut=base.ns(f'{year-1}-07-01')
        for seed in (7,19,43):
            tag=f'{year}-{seed}'
            ref=base.read(SOURCE/f'{tag}.json')
            past=[common(y,seed,t) for y in range(2020,year)]
            i,end,v=[np.concatenate([z[k] for z in past]) for k in (0,1,3)]
            train=(t[i]<cut)&(end<cut);mon=(t[i]>=cut)&(end<base.ns(f'{year}-01-01'))
            idx,ends,_,economic,z=common(year,seed,t);dates=t[idx];ts=z['origin_ns']
            h1=(dates<jul)&(ends<jul);h2=dates>=jul
            assert end[train].max()<t[i[mon]].min() and ends[h1].max()<dates[h2].min()
            pn=panel(year,seed,*bank)
            assert np.array_equal(pn.p,z['p'])
            ii=np.searchsorted(t,ts);ix=np.searchsorted(ts,dates)
            assert np.array_equal(t[ii],ts) and np.array_equal(ts[ix],dates) and valid[ii].all() and valid[i].all()
            for arm in NAMES:
                ck=torch.load(SOURCE/f'{tag}-{arm}.pt',weights_only=True,map_location='cpu')
                assert np.isclose(ck['scale'],v[train].std())
                width=8 if arm=='shared8' else 32
                model=Split() if arm=='split8' else nn.Sequential(nn.Linear(208,width),nn.SiLU(),nn.Linear(width,12))
                model.load_state_dict(ck['weights']);model.eval()
                slices={}
                for name,k,y in [('fit',i[train],v[train]),('monitor',i[mon],v[mon]),
                                  ('calibration',idx[h1],economic[h1]),('held',idx[h2],economic[h2])]:
                    slices[name]=summary(readout(model,x[k],ck['scale']),y,t[k])
                p=readout(model,x[ii],ck['scale'])[0]
                # Match original GEMM batching, which may differ by float roundoff.
                for key,a in [('pred','prediction'),('bp','actual')]:
                    assert abs(slices['held']['top'][a]-ref['arms'][arm]['rank'][key])<.001
                # Fit a constant only on the high-score H1 tail, one per horizon.
                # Correcting by a constant leaves ranks unchanged; only the >0
                # gate can change, not the fixed-volume selection.
                cp=p[ix[h1]];tail=selected(cp,dates[h1])
                offsets=np.array([(cp[:,j]-economic[h1,j])[tail[:,j]].mean() for j in range(4)])
                before=score(p,pn,dates,ix,economic,jul,ts,True)
                assert before['account']==ref['arms'][arm]['positive']['account']
                after=score(p.astype(float)-offsets,pn,dates,ix,economic,jul,ts,True)
                rows.append(dict(year=year,seed=seed,arm=arm,slices=slices,h1_offsets=offsets.tolist(),
                    original=before['account'],corrected=after['account'],adjusted_top_bias=after['pred']-after['bp'],
                    weeks=ref['weeks']))
    atomic(OUT/'DETAIL.json',rows)
    lines=['# 424: chronological SELL calibration audit','','Median bp per cell (18 overlapping seed/year/architecture cells).',
           'Top = same-volume weekly top tenth per horizon; diagnostic only. Bias = predicted minus realized.',
           '', '|Time slice|All bias|Top predicted|Top realized|Top bias|Cells top bias > 0|','|---|---:|---:|---:|---:|---:|']
    for s in ('fit','monitor','calibration','held'):
        a=[r['slices'][s] for r in rows]
        vals=[np.median([z[g][k] for z in a]) for g,k in [('all','bias'),('top','prediction'),('top','actual'),('top','bias')]]
        lines.append('|'+s+'|'+ '|'.join(f'{v:.2f}' for v in vals)+f"|{sum(z['top']['bias']>0 for z in a)} / 18|")
    lines+=['','|Error contributor (top cells)|Fit|Monitor|Calibration|Held|','|---|---:|---:|---:|---:|']
    for k in ('prob_error','win_size_error','loss_size_error'):
        lines+=['|'+k+'|'+'|'.join(f"{np.median([r['slices'][s]['top'][k] for r in rows]):.2f}" for s in ('fit','monitor','calibration','held'))+'|']
    good=sum(r['corrected']['ret']>0 and r['corrected']['trades']>2*r['weeks'] for r in rows)
    lines+=['',f'Positive active corrected accounts: {good}/18. No promotion.',
            'The H1 correction is a fixed offset per horizon; it cannot improve within-horizon ranking.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic(OUT/'DONE.json',dict(cells=len(rows),corrected_positive_active=good,promoted=False))


if __name__=='__main__':main()
