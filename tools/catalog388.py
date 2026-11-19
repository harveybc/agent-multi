"""Prepare four authenticated, TRAIN-only work units; aggregate quarterly gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import pandas as pd
from tools.segment_business388 import NAMES

SEEDS=(19,7,43,71)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def catalog(v,segs,kinds):
    rows=[]
    for (a,b),k in zip(segs,kinds,strict=True):
        x=np.log(v[a:b+1,3]);r=np.diff(x);n=b-a
        d=np.log(v[b,1 if k>0 else 2]/v[a,2 if k>0 else 1])
        t=np.arange(n+1);trend=np.polyfit(t,x,1)
        err=x-np.polyval(trend,t)
        path=(x-x[0])*k
        dd=np.maximum.accumulate(path)-path;j=int(dd.argmax())
        thirds=np.diff(np.interp([0,n/3,2*n/3,n],t,path))
        half=max(n//2,1)
        rows.append([a,b,k,n,d,d/n,r.std(),np.std(np.polyval(trend,t))/max(err.std(),1e-8),
                     np.polyfit(t[:-1],r,1)[0] if n>1 else 0,*thirds,dd[j],j/n,
                     r[:half].std(),r[half:].std() if n>1 else 0])
    return pd.DataFrame(rows,columns='start end side n displacement slope vol snr curvature first middle last retrace retrace_at vol_early vol_late'.split())

def prepare(a):
    from app.zigzag_opportunities import engine
    from app.pivot_segment_generator import pivots_by_chunk
    engine(a.engine)
    src=subprocess.check_output(['git','-C',str(a.engine),'cat-file','blob','1aaea5cb2262130cf4ce8ff7c87a745ccbe82290'])
    if hashlib.sha256(src).hexdigest()!='c06592f6478f3887213e985a80de5823dd953d3887ce25bec0272197860affda':raise ValueError('zigzag')
    z=types.ModuleType('app.short_horizon.zz388');z.__package__='app.short_horizon';sys.modules[z.__name__]=z
    exec(compile(src,'fixed_pivots','exec'),z.__dict__)
    real=pd.read_parquet(a.bars,filters=[('timestamp','>=',pd.Timestamp('2020-01-01',tz='UTC')),
        ('timestamp','<',pd.Timestamp('2024-01-01',tz='UTC'))]).set_index('timestamp')
    real=real[['BidOpen','BidHigh','BidLow','BidClose']].set_axis(['OPEN','HIGH','LOW','CLOSE'],axis=1)
    piv=pivots_by_chunk(real,z,z.Params(depth=4,backstep=2,deviation_pips=5))
    pairs={(u,v):kind for chunk in piv for (u,_),(v,kind) in zip(chunk,chunk[1:])}
    t=real.index.as_unit('ns').asi8
    q=(real.index.year.to_numpy()-2020)*4+real.index.quarter.to_numpy()-1
    risk=[[r['horizon'],1 if r['direction']=='buy' else -1,r['reference']['median']['mae_pips'],
           r['reference']['median']['tp_pips']] for r in json.loads(a.risk.read_text())['rows']]
    assert {tuple(r[:2]) for r in risk}=={(h,d) for h in (3,6,12,24) for d in (1,-1)}
    manifest=json.loads((a.root/'COHORTS.json').read_text())
    a.out.mkdir(parents=True,exist_ok=False)
    for seed in SEEDS:
        path=a.root/str(seed);receipt=path/'RESULT.json'
        assert manifest['release_for_training'] and sha(receipt)==manifest['receipts'][str(seed)]
        for f in ('OHLC.parquet','SEGMENTS.npy'):
            assert sha(path/f)==json.loads(receipt.read_text())['sha256'][f]
        syn=pd.read_parquet(path/'OHLC.parquet')
        assert syn.index.equals(real.index) and list(syn)==list(real)
        seg=np.load(path/'SEGMENTS.npy')[:,:2].astype(int)
        seg=seg[q[seg[:,0]]==q[seg[:,1]]]
        kinds=[pairs[tuple(p)] for p in seg]
        cat=catalog(real.values,seg,kinds);sc=catalog(syn.values,seg,kinds)
        med=cat.vol[q[seg[:,1]]<8].median()
        g=np.full(len(t),-1)
        for (l,r),k,vol in zip(seg,kinds,cat.vol):
            g[l+1:r+1]=int(k>0)*2+int(vol>med)
        for name,c in [('real',cat),('syn',sc)]:
            c['start_ns']=t[seg[:,0]];c['end_ns']=t[seg[:,1]]
            c['utc_hour']=real.index.hour[seg[:,0]];c['weekday']=real.index.dayofweek[seg[:,0]]
            if name!='real' or seed==19:c.to_parquet(a.out/(f'{seed}_catalog.parquet' if name=='syn' else 'real_catalog.parquet'))
        np.savez_compressed(a.out/f'{seed}.npz',real=real.values,syn=syn.values,ts=t,group=g,quarter=q,risk=risk)

def summarize(a):
    rows=[]
    for seed in SEEDS:
        with np.load(a.out/f'{seed}_metrics.npz') as f:
            for h in (3,6,12,24):
                for d in (-1,1):
                    key=f'{h}_{d}';ix=f['real_'+key+'_idx']
                    np.testing.assert_array_equal(ix,f['syn_'+key+'_idx'])
                    g=f['group'][ix];q=f['quarter'][ix]
                    real=f['real_'+key];syn=f['syn_'+key]
                    for group in range(4):
                        for j,name in enumerate(NAMES):
                            x=real[:,j];y=syn[:,j]
                            def mean(v,m):
                                z=v[m];z=z[np.isfinite(z)]
                                return z.mean() if len(z)>=20 else np.nan
                            quarters=np.array([mean(x,(g==group)&(q==k)) for k in range(12)])
                            quarters=quarters[np.isfinite(quarters)]
                            m=(g==group)&(q<12);held=mean(x,(g==group)&(q>=12))
                            base=mean(x,m);generated=mean(y,m)
                            tol=np.nan
                            if len(quarters)>=4:
                                u,v=np.triu_indices(len(quarters),1)
                                tol=float(np.quantile(abs(quarters[u]-quarters[v]),.9))
                            ok=np.isfinite([tol,held,base,generated]).all()
                            rows.append((seed,h,d,group,name,base,held,generated,tol,ok,
                                         ok and abs(held-generated)>tol+1e-12,ok and abs(held-base)>tol+1e-12))
    df=pd.DataFrame(rows,columns='seed horizon side group metric ref held generated tolerance supported breach real_breach'.split())
    df.to_csv(a.out/'GATES.csv',index=False)
    print(df.groupby('seed')[['supported','breach','real_breach']].sum().to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','summarize'])
    for k in ('out','root','risk','engine','bars'):p.add_argument('--'+k,type=Path)
    a=p.parse_args();(prepare if a.mode=='prepare' else summarize)(a)
