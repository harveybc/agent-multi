"""Paired SELL-only refit: passive vs managed sign targets, identical opportunity rows."""
import copy
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import time
import fcntl
import numpy as np
import torch
from torch.nn import functional as F
from tools import role405_plateau as plateau
from role413_sequence import atomic, base, labels, panel

OUT=Path('/var/tmp/role420-sell-paired')
ARMS=('old_label','managed_label')


def common(yr,seed,t):
    z,_=labels(yr,seed)
    ts=z['origin_ns']
    idx=np.searchsorted(t,ts)
    assert np.array_equal(t[idx],ts) and np.all(np.diff(ts)>0)
    old=np.full((len(ts),4),np.nan)
    ends=np.zeros_like(old,dtype=np.int64)
    for j,h in enumerate(base.HS):
        d=base.DEST/f'labels-{yr}-{h}';f=d/'rows.npz'
        assert base.file_sha(f)==base.read(d/'DONE.json')['sha']
        with np.load(f) as a:
            ids=a['OPEN_SELL_id'];v=a['OPEN_SELL'][:,3]
        assert np.unique(ids[:,1]).size==len(ids)
        _,i,k=np.intersect1d(ts,ids[:,1],return_indices=True)
        assert np.array_equal(idx[i],ids[k,0])
        old[i,j]=v[k];ends[i,j]=ids[k,2]
    new=z['y'][:,1,:]
    good=np.isfinite(old).all(1)&np.isfinite(new).all(1)
    return idx[good],ends[good].max(1),old[good],new[good],z


def worker(task):
    year,seed=task
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    tag=f'{year}-{seed}'
    if (OUT/f'{tag}.json').exists():return tag
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    x,t,valid=bank
    past=[common(v,seed,t) for v in range(2020,year)]
    idx,end,old,new=[np.concatenate([v[k] for v in past]) for k in range(4)]
    cut=base.ns(f'{year-1}-07-01')
    train=(end<cut)&(t[idx]<cut);mon=(t[idx]>=cut)&(end<base.ns(f'{year}-01-01'))
    assert valid[idx].all() and train.any() and mon.any() and not (train&mon).any()
    X=torch.from_numpy(x[idx[train]])
    V=torch.from_numpy(x[idx[mon]])
    truth=torch.from_numpy((new[mon]>0).astype('f'))
    targets=[torch.from_numpy((y[train]>0).astype('f')) for y in (old,new)]
    torch.manual_seed(seed)
    nets=[base.Expert(208,False,'cls')];nets.append(copy.deepcopy(nets[0]))
    assert all(torch.equal(a,b) for a,b in zip(*[m.parameters() for m in nets]))
    opts=[torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001) for m in nets]
    best=[np.inf,np.inf];champ=[None,None];mom=[None,None];ep=[0,0];bad=[0,0];miss=[0,0]
    history=[];gen=torch.Generator().manual_seed(seed);t0=time.time()
    for epoch in range(1,2001):
        for ix in torch.randperm(len(X),generator=gen).split(512):
            for j,(net,opt) in enumerate(zip(nets,opts)):
                opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(net.net(X[ix]),targets[j][ix])
                loss.backward();opt.step()
        bces=[]
        for j,(net,opt) in enumerate(zip(nets,opts)):
            with torch.no_grad():bce=F.binary_cross_entropy_with_logits(net.net(V),truth).item()
            assert np.isfinite(bce)
            bces.append(bce);improved=bce<best[j]-1e-8
            if improved:
                best[j]=bce;ep[j]=epoch;champ[j]=copy.deepcopy(net.state_dict())
                mom[j]=plateau.snapshot(opt);bad[j]=0
            elif epoch>=100:bad[j]+=1
            miss[j],rate=plateau.observe(epoch,improved,miss[j],opt.param_groups[0]['lr'],False)
            if rate is not None:plateau.restore(net,opt,champ[j],mom[j],rate)
        history.append(dict(epoch=epoch,bce=bces,lr=[o.param_groups[0]['lr'] for o in opts],patience=bad.copy()))
        if epoch%10==0:atomic(OUT/f'progress-{tag}.json',dict(epoch=epoch,best=ep,bad=bad,seconds=time.time()-t0))
        if min(bad)>=80:break
    ix,_,old_test,new_test,z=common(year,seed,t);dates=t[ix];jul=base.ns(f'{year}-07-01')
    mask=dates>=jul
    # Coverage for the chronological replay uses every eligible, causal bar;
    # descriptive matched-label metrics use the same finite paired support.
    pn=panel(year,seed,*bank)
    all_ts=z['origin_ns'];ii=np.searchsorted(t,all_ts)
    assert np.array_equal(t[ii],all_ts) and valid[ii].all() and np.array_equal(pn.p,z['p'])
    weeks=(dates//86400000000000+3)//7
    res={}
    for j,arm in enumerate(ARMS):
        net=nets[j];net.load_state_dict(champ[j]);net.eval()
        torch.save(net.state_dict(),OUT/f'{tag}-{arm}.pt')
        with torch.no_grad():
            p=net.net(torch.from_numpy(x[ii])).sigmoid().numpy()
        rank=np.column_stack([np.searchsorted(np.sort(q[all_ts<jul]),q,side='right')/(all_ts<jul).sum() for q in p.T])
        # Highest probability percentile above H1 90th percentile, deterministic
        # horizon ties; no BUY proposals. Tied cutoff values do not generate opens.
        bids=np.zeros((len(ii),2,4))
        take=(all_ts>=jul)[:,None]&(p>np.quantile(p[all_ts<jul],.9,axis=0))
        bids[:,1,:]=np.where(take,rank,0.)
        ac=pn.account(bids);assert ac['buy']==0
        # Offline same-volume comparison; not a live quantile policy.
        pc=p[np.searchsorted(all_ts,dates)]
        chosen=[]
        for h in range(4):
            for w in np.unique(weeks[mask]):
                k=np.flatnonzero(mask&(weeks==w));n=max(1,len(k)//10)
                pick=k[np.argsort(-pc[k,h],kind='stable')[:n]]
                chosen.extend(new_test[pick,h].tolist())
        res[arm]=dict(account=ac,matched_n=len(chosen),matched_bp=float(np.mean(chosen)),
                      matched_win=float(np.mean(np.array(chosen)>0)),best_epoch=ep[j],best_bce=best[j])
    assert res[ARMS[0]]['matched_n']==res[ARMS[1]]['matched_n']
    atomic(OUT/f'{tag}.json',dict(seed=seed,year=year,train=int(train.sum()),monitor=int(mon.sum()),
        sign_flipped=float(np.mean(targets[0].numpy()!=targets[1].numpy())),eval_origins=int(mask.sum()),
        native_weeks=int(np.unique(pn.e.seg_of[pn.bars[all_ts>=jul]]).size),epochs=epoch,history=history,arms=res))
    return tag


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    OUT.mkdir(exist_ok=True)
    with (OUT/'lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sig=base.file_sha(__file__)
        if (OUT/'CONFIG.json').exists():assert base.read(OUT/'CONFIG.json')['sha']==sig
        assert base.read(base.DEST/'CACHE.json')['sha']==base.file_sha(base.DEST/'market.npz')
        atomic(OUT/'CONFIG.json',dict(sha=sig,batch_origins=512,policy=plateau.POLICY,
            monitor='managed BCE identical in both arms',stop='both patience >=80 or 2000',test_years=[2022,2023]))
        tasks=[(yr,s) for yr in (2022,2023) for s in (7,19,43)]
        with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as ex:
            for job in as_completed([ex.submit(worker,k) for k in tasks]):print(job.result(),flush=True)
        passed=True;lines=['# 420: paired SELL-only refit','','|Year|Seed|Epochs|Old net % / trades|Aligned net % / trades|Old / aligned matched bp|',
            '|---|---|---|---|---|---|']
        for y,s in tasks:
            r=base.read(OUT/f'{y}-{s}.json');a,b=[r['arms'][k] for k in ARMS]
            ra,rb=[m['account'] for m in (a,b)]
            passed &= rb['ret']>max(0.,ra['ret']) and rb['trades']>2*r['native_weeks'] and b['matched_bp']>max(0.,a['matched_bp'])
            lines.append(f"|{y}|{s}|{r['epochs']}|{ra['ret']:.2f} / {ra['trades']}|{rb['ret']:.2f} / {rb['trades']}|{a['matched_bp']:.2f} / {b['matched_bp']:.2f}|")
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        atomic(OUT/'DONE.json',dict(replication_candidate=bool(passed),promoted=False))


if __name__=='__main__':main()
