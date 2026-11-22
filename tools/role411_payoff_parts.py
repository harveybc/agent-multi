"""Paired payoff attribution on 2023 TRAIN, scored with the sealed 409 design."""
from functools import lru_cache
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from role409_signed_advantage import base, fit

OUT = Path('/var/tmp/role411-train-attribution')
COST = ('spread', 'slippage', 'commission', 'financing')


def load(year, seed):
    path = base.OUT/f'score-{seed}-{year}'
    rc = base.read(path/'RECEIPT.json')
    assert rc['sha']==base.file_sha(path/'labels.npz') and rc['end']<base.ns(f'{year+1}-01-01')
    assert rc['spec']['year']==year and rc['spec']['seed']==seed
    with np.load(path/'labels.npz') as z:
        return {k:z[k] for k in z}, rc


def run(seed):
    OUT.mkdir(exist_ok=True)
    base.torch.set_num_threads(1)
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:
        x, t, valid = z['x'], z['t'], z['valid']
    def at(ts):
        i=np.searchsorted(t,ts)
        assert np.array_equal(t[i],ts) and valid[i].all()
        return x[i]
    train,_=load(2022,seed)
    test,rc=load(2023,seed)
    mu,sd,w,bias,_=fit([at(train['origin_ns'])],[train['y']])
    score=np.tensordot((at(test['origin_ns']).astype(float)-mu)/sd,w,axes=1)+bias
    ds,hs=base.decisions(score)
    y=test['y']
    market,_=base.assemble(base.PARENT,base.ROOT/'model/61/bundle',base.ROOT/'history397-61-r2')
    assert np.array_equal(t,market.inputs.row_ts_ns) and np.array_equal(valid,market.inputs.valid)
    ix=np.flatnonzero(valid & (t>=base.ns('2023-01-01')) & (t<base.ns('2024-01-01')))
    pn=base.Panel(market,2023,seed,base.torch.from_numpy(x[ix]),ix,base.class_means(2023),base.read(base.PARENT/'RISK.json'))
    assert pn.hashes==rc['spec']['heads'] and np.array_equal(test['p'],pn.p)
    assert np.array_equal(pn.bars,test['bars']) and np.array_equal(pn.e.v[pn.bars]+base.HOUR,test['origin_ns'])
    e=pn.e
    e.reset()
    cap=e.cfg.capital
    scale=1e4/cap
    anchors={}
    for b in pn.bars:
        g=e.seg_of[b]
        if b>=e.first+3 and e.v[b]+base.HOUR>=e.op[g]+36*base.HOUR:
            anchors.setdefault(g,b)

    @lru_cache(4096)
    def extra(a,cash):
        direction=np.sign(e.c[a]-e.c[a-3])
        return pn.isolated(a,1 if direction>0 else 2,base.HS.index(6),cash)[1]-cash if direction else 0.

    def trade(b,r,j,managed):
        eng=base.clone(e)
        env=pn.envs[j]
        env.bar=b
        eng.next_decision=b
        filled=eng.step(b,r+1,entry_plan=env.entry_plan(1 if not r else -1))['entry']
        assert filled
        k=b+1
        while eng.pos is not None:
            assert k<eng.last
            eng.step(k,pn.close(eng,k,j) if managed else 0)
            k+=1
        tr,=eng.trades
        cost=sum(tr[c+'_usd'] for c in COST)
        assert abs(tr['gross_usd']-cost-tr['net_usd'])<1e-6 and abs(cap+tr['net_usd']-eng.balance)<1e-6
        return tr,k,eng.balance

    rows=[]
    for r,j in np.ndindex(2,4):
        ii=np.flatnonzero(np.isfinite(y[:,r,j]))
        dec=np.minimum(9,((rankdata(score[ii,r,j])-.5)*10/len(ii)).astype(int))+1
        for i,d in zip(ii,dec):
            b=pn.bars[i]
            a=anchors.get(e.seg_of[b],-1)
            wait=extra(a,cap) if a>=b+1 else 0.
            m,k,cash=trade(b,r,j,True)
            p,kp,cashp=trade(b,r,j,False)
            assert all(m[f]==p[f] for f in ('units','direction','entry_price','entry_bar_abs','stop_loss','take_profit'))
            cont=(extra(a,cash) if a>=k else 0.)-wait
            passive_cont=(extra(a,cashp) if a>=kp else 0.)-wait
            vals=dict(barrier_gross=p['gross_usd'],close_gross=m['gross_usd']-p['gross_usd'],
                      **{c:-m[c+'_usd'] for c in COST},continuation=cont)
            delta=sum(vals.values())*scale-y[i,r,j]
            assert abs(delta)<1e-7, (seed,i,r,j,delta)
            rows.append(dict(seed=seed,origin_ns=test['origin_ns'][i],side=base.ROLES[r],h=base.HS[j],decile=d,
                chosen=int(ds[i]==r+1 and hs[i]==j),pred=score[i,r,j],target=y[i,r,j],
                **{c:v*scale for c,v in vals.items()},error=delta,barrier_target=(p['net_usd']+passive_cont)*scale,
                exit=m['close_reason'],barrier_exit=p['close_reason'],exit_changed=k!=kp or m['close_reason']!=p['close_reason']))
    frame=pd.DataFrame(rows)
    chosen=frame[frame.chosen==1]
    n,ref={7:(2555,-4.2825793522500915),19:(2795,-3.6644036762070438),43:(2681,-3.887856430435991)}[seed]
    assert len(chosen)==n and abs(chosen.target.mean()-ref)<1e-9
    frame.to_csv(OUT/f'offers-{seed}.csv.gz',index=False)
    print(seed,len(frame),abs(frame.error).max(),flush=True)


if __name__=='__main__':
    for seed in base.SEEDS:
        run(seed)
