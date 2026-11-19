"""Independent scalar parity, distributional checks and matched-prefix stress pairs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tools.segment_business388 import NAMES


def scalar(v,d,sl,tp):
    opts=[];mae=0.;entry=v[0,0]
    for j,(o,h,l,c) in enumerate((v-entry)*d*10000):
        hi,lo=max(h,l),min(h,l)
        if o<=-sl or o>=tp:
            kind='sl' if o<=-sl else 'tp';p=o;break
        if j:opts.append(o)
        if hi>=tp and lo<=-sl:kind='both';p=np.nan;break
        if hi>=tp:kind='tp';p=tp;break
        if lo<=-sl:kind='sl';p=-sl;break
        mae=max(mae,-lo)
    else:kind='time';p=c
    diff=np.asarray(opts)-p
    return [kind=='sl',kind=='tp',kind=='both',kind=='time',j+1,mae,
            diff.mean() if opts else np.nan,np.mean(diff>0) if opts and np.isfinite(p) else np.nan,
            j+1 if kind=='sl' else np.nan,j+1 if kind=='tp' else np.nan]


def forks(v,w,ix,H,d,sl,tp):
    # Same real position and market through the current open. Only the future is replaced.
    prev=v[ix-1];cur=v[ix,0]
    init=(cur-prev[:,0])*10000*d
    signed=(prev-prev[:,:1])*10000*d
    ok=(signed.max(1)<tp)&(signed.min(1)>-sl)&(init<0)&(init>-sl)
    ix=ix[ok];init=init[ok];rows=np.arange(len(ix))
    def terminal(series):
        p=(series[ix[:,None]+np.arange(H)]-series[ix,0][:,None,None])*10000*d+init[:,None,None]
        o,hi,lo=p[:,:,0],p[:,:,1:3].max(2),p[:,:,1:3].min(2)
        s=(lo<=-sl)&(o<tp);t=(hi>=tp)&(o>-sl)
        k=(s|t).argmax(1);slfirst=s[rows,k]&~t[rows,k];tpfirst=t[rows,k]&~s[rows,k]
        return slfirst,tpfirst
    rs,rt=terminal(v);ss,st=terminal(w)
    flip=(rs&st)|(rt&ss)
    return pd.DataFrame(dict(origin=ix[flip],side=d,horizon=H,
        real_hits_sl=rs[flip],synthetic_hits_sl=ss[flip],unrealized_pips=init[flip]))


def audit(root):
    rng=np.random.default_rng(388);out=[];stress=[];checked=0;total={}
    for seed in (19,7,43,71):
        with np.load(root/f'{seed}.npz') as f: data=dict(f)
        with np.load(root/f'{seed}_metrics.npz') as m:
            total[seed]=0
            for h,d,sl,tp in data['risk']:
                H=int(h);d=int(d);key=f'{H}_{d}';i=m[f'real_{key}_idx']
                x=m['real_'+key];y=m['syn_'+key];q=data['quarter'][i];g=data['group'][i]
                total[seed]+=len(i)
                for tag,observed in [('real',x),('syn',y)]:
                    for j in rng.choice(len(i),64,replace=False):
                        ref=scalar(data[tag][i[j]:i[j]+H],d,sl,tp)
                        np.testing.assert_allclose(observed[j],ref,rtol=1e-9,atol=1e-8,equal_nan=True);checked+=1
                ids=i[(q<12)&(i>0)&(data['ts'][i]-data['ts'][i-1]==3600*10**9)]
                pairs=forks(data['real'],data['syn'],ids,H,d,sl,tp)
                pairs['seed']=seed;pairs['timestamp_ns']=data['ts'][pairs.origin.values];stress.append(pairs)
                for group in range(4):
                    # CDF on fixed 2020-21 quantile grid. Wrong-trend comparator keeps volatility.
                    for j in [4,5,6,8,9]:
                        def samples(arr,m):
                            z=arr[m,j];return z[np.isfinite(z)]
                        train=samples(x,(g==group)&(q<8))
                        if len(train)<20:continue
                        grid=np.unique(np.quantile(train,[.1,.25,.5,.75,.9]))
                        def cdf(z):return (z[:,None]<=grid).mean(0) if len(z)>=20 else None
                        qs=[cdf(samples(x,(g==group)&(q==k))) for k in range(12)]
                        qs=[a for a in qs if a is not None]
                        target=cdf(samples(x,(g==group)&(q>=12)))
                        if len(qs)<4 or target is None:continue
                        tol=np.quantile([abs(a-b).max() for k,a in enumerate(qs) for b in qs[:k]],.9)
                        for name,arr,which in [('real',x,group),('synthetic',y,group),('wrong_trend',y,group^2)]:
                            c=cdf(samples(arr,(g==which)&(q<12)))
                            if c is not None:out.append((seed,H,d,group,NAMES[j],name,tol,abs(c-target).max()))
    pairs=pd.concat(stress);pairs.to_parquet(root/'STRESS_PAIRS.parquet',index=False)
    df=pd.DataFrame(out,columns='seed horizon side group metric comparator tolerance distance'.split())
    df['breach']=df.distance>df.tolerance+1e-12;df.to_csv(root/'CDF.csv',index=False)
    result=dict(scalar_gpu_checks=checked,synthetic_entries=total,paired_stress=len(pairs),
                stress_unique_origins=int(pairs.timestamp_ns.nunique()),cdf=df.groupby('comparator').breach.agg(['count','sum']).to_dict('index'))
    (root/'VERIFICATION.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
