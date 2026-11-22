"""CPU-only sequential campaign with independent economic and ordinal endpoints."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import fcntl
import itertools
import os
import time

import numpy as np
from role412_campaign import atomic, ridge, winner, base
from role409_signed_advantage import fit
from tools import role408_folds as folds

OUT=Path('/var/tmp/role413-pair-rank-v1')
OLD=base.OUT
PAIRS=list(itertools.combinations(range(8),2))


def ranks(x,y,u):
    target=np.stack([np.sign(y[:,a]-y[:,b]) for a,b in PAIRS],axis=1)
    margin=np.clip(ridge(x,target,u),-1,1)
    votes=np.zeros((len(u),8))
    for k,(a,b) in enumerate(PAIRS):
        votes[:,a]+=margin[:,k]
        votes[:,b]-=margin[:,k]
    assert np.allclose(votes.sum(1),0)
    return votes/7


def head(task):
    folds.OUT=OUT
    return folds.one(task)


def labels(year,seed):
    root=OLD if year>=2022 else OUT
    path=root/f'score-{seed}-{year}'
    rc=base.read(path/'RECEIPT.json')
    assert rc['spec']['year']==year and rc['spec']['seed']==seed and rc['end']<base.ns(f'{year+1}-01-01')
    assert rc['sha']==base.file_sha(path/'labels.npz')
    with np.load(path/'labels.npz') as z:
        return dict(z),rc


def panel(year,seed,x,t,valid):
    base.OUT=OLD if year>=2022 else OUT
    market,_=base.assemble(base.PARENT,base.ROOT/'model/61/bundle',base.ROOT/'history397-61-r2')
    assert np.array_equal(market.inputs.row_ts_ns,t) and np.array_equal(market.inputs.valid,valid)
    ix=np.flatnonzero(valid&(t>=base.ns(f'{year}-01-01'))&(t<base.ns(f'{year+1}-01-01')))
    return base.Panel(market,year,seed,base.torch.from_numpy(x[ix]),ix,base.class_means(year),base.read(base.PARENT/'RISK.json'))


def build(year,seed,bank):
    dest=OUT/f'score-{seed}-{year}'
    dest.mkdir(exist_ok=True)
    pn=panel(year,seed,*bank)
    if (dest/'RECEIPT.json').exists():
        z,rc=labels(year,seed)
        assert rc['spec']['heads']==pn.hashes and np.array_equal(z['p'],pn.p)
        return
    b=pn.targets()
    f=dest/'labels.npz'
    tmp=dest/'building.npz'
    np.savez_compressed(tmp,y=b['y'],p=b['p'],bars=pn.bars,origin_ns=pn.e.v[pn.bars]+base.HOUR)
    tmp.replace(f)
    atomic(dest/'RECEIPT.json',dict(spec=dict(heads=pn.hashes,seed=seed,year=year),end=b['end'],sha=base.file_sha(f)))


def evaluate(years,year,seed,bank):
    x,t,valid=bank
    def fetch(y):
        z,rc=labels(y,seed)
        idx=np.searchsorted(t,z['origin_ns'])
        assert np.array_equal(t[idx],z['origin_ns']) and valid[idx].all()
        return x[idx].astype(float),z['y'].reshape(-1,8),z,rc
    xs=[];ys=[]
    for yy in years:
        assert yy<year
        xx,out,_,_=fetch(yy)
        ok=np.isfinite(out).all(1)
        xs.append(xx[ok]);ys.append(out[ok])
    u,y,z,rc=fetch(year)
    pn=panel(year,seed,*bank)
    assert pn.hashes==rc['spec']['heads'] and np.array_equal(pn.p,z['p']) and np.array_equal(pn.bars,z['bars'])
    dt=z['origin_ns'];jul=base.ns(f'{year}-07-01')
    cal=(dt<jul)&(pn.e.ff[pn.e.seg_of[pn.bars]]<jul)&np.isfinite(y).all(1)
    ev=dt>=jul
    assert ev.any() and cal.any() and np.isfinite(y[ev]).all()
    votes=ranks(np.concatenate(xs),np.concatenate(ys),u)
    action=votes.argmax(1);tie=(votes==votes.max(1)[:,None]).sum(1)>1
    feat=np.column_stack((votes,np.eye(8)[action]))
    i=np.arange(len(u))
    score=ridge(feat[cal],y[i[cal],action[cal]],feat)
    # Match the original economic baseline, always trained on the single preceding year.
    xx,yy,_,_=fetch(year-1)
    mu,sd,w,b,_=fit([xx],[yy.reshape(-1,2,4)])
    raw=winner(np.tensordot((u-mu)/sd,w,axes=1)+b)
    arms=dict(raw=raw,paired_timing=np.where((raw>=0)&~tie,action,-1),
              paired_gate=np.where((score>0)&~tie,action,-1),nop=np.full_like(raw,-1))
    res={}
    weeks=len(np.unique(pn.e.seg_of[pn.bars[ev]]))
    for name,a in arms.items():
        k=np.flatnonzero(ev&(a>=0))
        v=np.zeros_like(y);v[k,a[k]]=1
        ac=pn.account(v.reshape(-1,2,4))
        res[name]=dict(net_pct=ac['ret'],trades=ac['trades'],buy=ac['buy'],sell=ac['sell'],wins=ac['wins'],
                       max_dd=ac['dd'],per_week=ac['trades']/weeks,offers=len(k),
                       offer_bp=float(y[k,a[k]].mean()) if len(k) else None)
    assert res['nop']['net_pct']==0 and res['nop']['trades']==0
    if year==2023:
        prev=base.read(Path('/var/tmp/role412-router-campaign')/f'main-{seed}.json')['result']['accounts']['raw']
        assert abs(prev['ret']-res['raw']['net_pct'])<1e-6 and prev['trades']==res['raw']['trades']
    return dict(train=years,score_year=year,seed=seed,fit_examples=sum(map(len,xs)),
                calibration=int(cal.sum()),eval_origins=int(ev.sum()),native_weeks=weeks,arms=res)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    base.torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    with (OUT/'run.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        X=np.zeros((16,3));Y=np.tile(np.arange(8),(16,1))
        assert np.all(ranks(X,Y,X).argmax(1)==7) and np.all(ranks(X,-Y,X).argmax(1)==0)
        assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
        assert base.read(Path('/var/tmp/role412-router-campaign/DONE.json'))['status']=='reject_tested_routers'
        with np.load(base.DEST/'market.npz') as z:bank=(z['x'],z['t'],z['valid'])
        def state(stage,**kw):
            atomic(OUT/'STATUS.json',dict(stage=stage,utc_epoch=time.time(),**kw))
        def compare(tag,years,year):
            rows=[]
            for seed in folds.SEEDS:
                state(tag,seed=seed)
                row=evaluate(years,year,seed,bank)
                atomic(OUT/f'{tag}-{seed}.json',row);rows.append(row)
            return rows
        compare('short', [2022], 2023)
        tasks=list(itertools.product((2020,2021),folds.SEEDS,base.ROLES))
        start=time.time()
        state('historical_heads',completed=0,total=len(tasks),started=start)
        with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as ex:
            futures=[ex.submit(head,task) for task in tasks]
            for n,f in enumerate(as_completed(futures),1):
                item=f.result()
                state('historical_heads',completed=n,total=len(tasks),started=start,last=item)
        for year,seed in itertools.product((2020,2021),folds.SEEDS):
            state('managed_labels',year=year,seed=seed)
            build(year,seed,bank)
        results=compare('historical',[2020,2021],2022)+compare('extended',[2020,2021,2022],2023)
        passed=[]
        for name in ('paired_timing','paired_gate'):
            def accepts(row):
                a=row['arms'][name]
                return a['net_pct']>max(0,row['arms']['raw']['net_pct']) and a['per_week']>2 and (a['offer_bp'] or 0)>0
            if all(map(accepts,results)):passed.append(name)
        atomic(OUT/'DONE.json',dict(passed=passed,decision='replicate_only' if passed else 'reject_pair_routers'))
        state('complete',passed=passed)


if __name__=='__main__':
    main()
