"""Bounded diagnostic, correction, account replay and conditional fallback."""
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from role411_payoff_parts import base, load

OUT = Path('/var/tmp/role412-router-campaign')
LEDGER = Path('/var/tmp/role411-train-attribution')
SEEDS = (7, 19, 43)
JUL = base.ns('2023-07-01')


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, allow_nan=False, indent=2)+'\n')
    tmp.replace(path)


def ridge(x, y, u):
    mu, sd = x.mean(0), x.std(0)
    sd[sd<1e-8] = 1
    z = (x-mu)/sd
    w = np.linalg.solve(z.T@z+len(x)*.1*np.eye(x.shape[1]), z.T@(y-y.mean(0)))
    return (u-mu)/sd@w+y.mean(0)


def pair(p, y, opened):
    order = np.argsort(-p, axis=1, kind='stable')
    i = np.flatnonzero(opened)
    a,b = order[i,0], order[i,1]
    v,w = y[i,a], y[i,b]
    margin = p[i,a]-p[i,b]
    def stats(ix):
        miss=(v[ix]<=0)&(w[ix]<=0)
        other_positive=y[i[ix]].max(1)>0
        return dict(n=len(ix), winner=float(v[ix].mean()), runner=float(w[ix].mean()),
                    fixed_buy6=float(y[i[ix],1].mean()), predicted_margin=float(margin[ix].mean()),
                    realized_margin=float((v[ix]-w[ix]).mean()),
                    neither=int(miss.sum()),no_positive_in_eight=int((~other_positive).sum()),
                    only_lower_six_positive=int((miss & other_positive).sum()),runner_only=int(((v[ix]<=0)&(w[ix]>0)).sum()),
                    winner_only=int(((v[ix]>0)&(w[ix]<=0)).sum()),both=int(((v[ix]>0)&(w[ix]>0)).sum()))
    return dict(all=stats(np.arange(len(i))),quartiles=[stats(idx) for idx in np.array_split(np.argsort(margin,kind='stable'),4)])


def winner(scores):
    side,h = base.decisions(scores.reshape(-1,2,4))
    return np.where(side>0,(side-1)*4+h,-1)


def policies(p,y,train):
    best=p.argmax(1)
    i=np.arange(len(p))
    raw=winner(p)
    features=np.column_stack((p,np.eye(8)[best]))
    gate=ridge(features[train],y[i[train],best[train]],features)
    values=ridge(p[train],y[train],p)
    # Reranking-only must preserve exactly the original opportunities to open.
    return dict(raw=raw, gate=np.where((raw>=0)&(gate>0),raw,-1),
                rerank=np.where(raw>=0,values.argmax(1),-1),joint=winner(values),
                fixed=np.where(raw>=0,1,-1),nop=np.full_like(raw,-1))


def analog(x,y,u):
    mu,sd=x.mean(0),x.std(0)
    sd[sd<1e-8]=1
    z=(x-mu)/sd
    k=int(np.ceil(np.sqrt(len(z))))
    chunks=[]
    for block in np.array_split(u,max(1,int(np.ceil(len(u)/128)))):
        d=cdist((block-mu)/sd,z,'sqeuclidean')
        near=np.argpartition(d,k-1,axis=1)[:,:k]
        chunks.append(y[near].mean(1))
    return np.vstack(chunks)


def checks():
    p=np.tile([4.,3.,0.,0.,0.,0.,0.,0.],(12,1)); y=p.copy()
    assert np.all(winner(p)==0) and np.all(winner(-np.ones_like(p))==-1)
    tie=p.copy();tie[:,4]=4
    assert np.all(winner(tie)==-1)
    a=pair(p,y,np.ones(len(p),bool))['all']
    assert a['both']==12 and a['realized_margin']==1
    alternative=-np.ones_like(y); alternative[:6,7]=1
    a=pair(p,alternative,np.ones(len(p),bool))['all']
    assert a['neither']==12 and a['only_lower_six_positive']==6 and a['no_positive_in_eight']==6
    train=np.arange(len(y))<6
    changed=y.copy();changed[~train]=-1e6
    assert all(np.array_equal(v,policies(p,changed,train)[k]) for k,v in policies(p,y,train).items())


def run(seed, stage):
    data,rc=load(2023,seed)
    dates=data['origin_ns']; y=data['y'].reshape(-1,8)
    assert np.isfinite(y).all()
    f=pd.read_csv(LEDGER/f'offers-{seed}.csv.gz')
    f['j']=f.side.map({'OPEN_BUY':0,'OPEN_SELL':4}) + f.h.map(dict(zip(base.HS,range(4))))
    def matrix(col):
        m=f.pivot(index='origin_ns',columns='j',values=col)
        assert np.array_equal(m.index,dates) and np.array_equal(m.columns,np.arange(8))
        return m.to_numpy()
    p=matrix('pred')
    np.testing.assert_allclose(matrix('target'),y,atol=1e-10,rtol=0)
    raw=winner(p)
    assert np.array_equal(matrix('chosen'),(raw[:,None]==np.arange(8)))
    market,_=base.assemble(base.PARENT,base.ROOT/'model/61/bundle',base.ROOT/'history397-61-r2')
    with np.load(base.DEST/'market.npz') as z:
        t,x,valid=z['t'],z['x'],z['valid']
    def at(ts):
        i=np.searchsorted(t,ts)
        assert np.array_equal(t[i],ts) and valid[i].all()
        return x[i].astype(float)
    ix=np.flatnonzero(valid&(t>=base.ns('2023-01-01'))&(t<base.ns('2024-01-01')))
    pn=base.Panel(market,2023,seed,base.torch.from_numpy(x[ix]),ix,base.class_means(2023),base.read(base.PARENT/'RISK.json'))
    assert pn.hashes==rc['spec']['heads'] and np.array_equal(data['bars'],pn.bars) and np.array_equal(pn.p,data['p'])
    end=pn.e.ff[pn.e.seg_of[pn.bars]]
    train=(dates<JUL)&(end<JUL)
    ev=dates>=JUL
    assert train.any() and end[train].max()<dates[ev].min()
    sessions=len(np.unique(pn.e.seg_of[pn.bars[ev]]))
    if stage=='main':
        arms=policies(p,y,train)
        arms['oracle']=winner(y)
        diag=pair(p,y,raw>=0)
    else:
        old,_=load(2022,seed)
        yo=old['y'].reshape(-1,8)
        ok=np.isfinite(yo).all(1)
        arms={'analog':np.full_like(raw,-1)}
        arms['analog'][ev]=winner(analog(at(old['origin_ns'][ok]),yo[ok],at(dates[ev])))
        diag=None
    result={}
    for name,actions in arms.items():
        enabled=(actions>=0)&ev
        scores=np.zeros_like(p)
        i=np.flatnonzero(enabled)
        scores[i,actions[i]]=1
        assert np.array_equal(winner(scores),np.where(ev,actions,-1))
        account=pn.account(scores.reshape(-1,2,4))
        values=y[i,actions[i]]
        result[name]=dict(trades=account['trades'],buy=account['buy'],sell=account['sell'],
                          ret=account['ret'],dd=account['dd'],wins=account['wins'],
                          offers=len(i),bp=float(values.mean()) if len(i) else None,
                          trades_per_session=account['trades']/sessions)
    if stage=='main':
        assert result['nop']['trades']==0 and result['nop']['ret']==0 and result['oracle']['ret']>0
    return dict(year=2023,seed=seed,stage=stage,train=int(train.sum()),evaluated=int(ev.sum()),
                sessions=sessions,diagnosis=diag,accounts=result)


def campaign():
    checks()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    base.torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    def stage(name):
        collected=[]
        for seed in SEEDS:
            deps=[Path(__file__),Path(__file__).with_name('role411_payoff_parts.py'),
                  Path(base.__file__),LEDGER/f'offers-{seed}.csv.gz',base.DEST/'market.npz',
                  base.PARENT/'RISK.json',base.ROOT/'model/61/bundle/model.pt']
            for year in (2022,2023):
                root=base.OUT/f'score-{seed}-{year}'
                deps.extend([root/'RECEIPT.json',root/'labels.npz'])
            sig={str(p):base.file_sha(p) for p in deps}
            assert sig[str(base.DEST/'market.npz')]==base.read(base.DEST/'CACHE.json')['sha']
            path=OUT/f'{name}-{seed}.json'
            if path.exists() and (old:=base.read(path))['inputs']==sig:
                row=old['result']
            else:
                row=run(seed,name)
                atomic(path,dict(inputs=sig,result=row))
            print(name,seed,row['accounts'],flush=True)
            collected.append(row)
        return collected
    def qualifies(candidate,baseline):
        return candidate['ret']>max(0,baseline['ret']) and candidate['trades_per_session']>2 and (candidate['bp'] or 0)>0
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main=stage('main')
        passed=[k for k in ('gate','rerank','joint') if all(qualifies(a['accounts'][k],a['accounts']['raw']) for a in main)]
        if not passed:
            alt=stage('analog')
            if all(qualifies(b['accounts']['analog'],a['accounts']['raw']) for a,b in zip(main,alt)):
                passed=['analog']
        atomic(OUT/'DONE.json',dict(passed=passed,status='replicate_only' if passed else 'reject_tested_routers',
                                    next='independent replication' if passed else 'do not advance these routers into feature selection'))


if __name__=='__main__':
    campaign()
