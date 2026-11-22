"""Calibrate only win probability, never the two conditional amounts, prior to replay."""
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from role424_sell_bias import torch, nn, SOURCE, NAMES, common, base, Split, panel, score, readout, selected, atomic

OUT=Path('/var/tmp/role425-sell-probability')


def calibrated(p,w,l,idx,win,conditional):
    assert len(idx)==len(win) and np.isfinite(np.column_stack((p,w,l))).all()
    initial=logit(np.clip(p,1e-7,1-1e-7))
    x=initial[:,None] if not conditional else np.column_stack((initial,np.log1p(w),np.log1p(l)))
    sd=x[idx].std(0);sd[sd<1e-8]=1
    x=np.column_stack((np.ones(len(x)),(x-x[idx].mean(0))/sd))
    a=x[idx];b=initial[idx]
    def obj(coef):
        z=b+a@coef
        return (np.mean(np.logaddexp(0,z)-win*z)+.005*np.sum(coef[1:]**2),
                a.T@(expit(z)-win)/len(z)+.01*np.r_[0.,coef[1:]])
    res=minimize(obj,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B')
    assert res.success
    return expit(initial+x@res.x),res.x.tolist()


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    x,t,valid=bank;out=[]
    for year in (2022,2023):
        jul=base.ns(f'{year}-07-01')
        for seed in (7,19,43):
            ids,end,_,y,z=common(year,seed,t);dt=t[ids];ts=z['origin_ns']
            ii=np.searchsorted(t,ts);ix=np.searchsorted(ts,dt)
            assert np.array_equal(t[ii],ts) and np.array_equal(ts[ix],dt)
            fit=(dt<jul)&(end<jul);ev=dt>=jul
            assert end[fit].max()<dt[ev].min()
            pn=panel(year,seed,*bank)
            assert np.array_equal(pn.p,z['p'])
            ref=base.read(SOURCE/f'{year}-{seed}.json')
            for arm in NAMES:
                width=8 if arm=='shared8' else 32
                model=Split() if arm=='split8' else nn.Sequential(nn.Linear(208,width),nn.SiLU(),nn.Linear(width,12))
                ck=torch.load(SOURCE/f'{year}-{seed}-{arm}.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(ck['weights']);model.eval()
                raw,p,w,l=readout(model,x[ii],ck['scale']);old=selected(raw[ix[ev]],dt[ev])
                for method in ('original','platt','conditional'):
                    coefs=[]
                    if method=='original':pp=p;pred=raw
                    else:
                        hs=[calibrated(p[:,h],w[:,h],l[:,h],ix[fit],y[fit,h]>0,method=='conditional') for h in range(4)]
                        pp=np.column_stack([a for a,b in hs]);coefs=[b for a,b in hs]
                        pred=pp*w-(1-pp)*l
                    pe=pp[ix[ev]];truth=y[ev]>0
                    s=score(pred,pn,dt,ix,y,jul,ts,True)
                    if method=='original':assert s['account']==ref['arms'][arm]['positive']['account']
                    out.append(dict(year=year,seed=seed,arm=arm,method=method,coefs=coefs,
                        fixed_top_p=float(pe[old].mean()),fixed_top_win=float(truth[old].mean()),
                        fixed_top_brier=float(((pe[old]-truth[old])**2).mean()),weeks=ref['weeks'],
                        matched_bp=s['bp'],account=s['account']))
    atomic(OUT/'DETAIL.json',out)
    metrics={}
    for k in ('original','platt','conditional'):
        a=[r for r in out if r['method']==k]
        metrics[k]=dict(n=len(a),positive_active=sum(r['account']['ret']>0 and r['account']['trades']>2*r['weeks'] for r in a),
            matched_positive=sum(r['matched_bp']>0 for r in a),median_net_pct=float(np.median([r['account']['ret'] for r in a])),
            median_trades=float(np.median([r['account']['trades'] for r in a])),
            fixed_top_probability=float(np.median([r['fixed_top_p'] for r in a])),
            fixed_top_wins=float(np.median([r['fixed_top_win'] for r in a])))
    atomic(OUT/'DONE.json',dict(metrics=metrics,promoted=False))


if __name__=='__main__':main()
