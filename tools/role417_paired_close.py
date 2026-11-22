"""Same-tick counterfactual pairs; terminal state adapter, no feedback to encoder."""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from role413_sequence import base, atomic, panel
from role412_campaign import LEDGER, SEEDS, winner

OUT=Path('/var/tmp/role417-paired-close')
ARMS=('curve','direct','adapter')


def design(arm,logit,g):
    if arm=='direct':return np.column_stack((logit,g))
    return np.column_stack((logit,g,(logit[:,:,None]*g[:,None,:]).reshape(len(g),24)))


def curve(logit,g,means):
    b=np.ones((len(g),1))
    for col in g.T:b=np.column_stack((b*(1-col[:,None]),b*col[:,None]))
    p=1/(1+np.exp(-np.clip((b*logit).sum(1),-50,50)))
    return means[0]+p*(means[1]-means[0])


def moments(a,group,y):
    # Equal tick mass for absolute payoff. Within each conflicted tick, give
    # equal mass to every Cartesian positive-negative pair without expanding it.
    _,i,n=np.unique(group,return_inverse=True,return_counts=True)
    w=1/n[i]
    A=a.T@(w[:,None]*a);B=a.T@(w*y)
    pos=y>0;neg=y<0
    npos=np.bincount(i,weights=pos,minlength=len(n))
    nneg=np.bincount(i,weights=neg,minlength=len(n))
    mixed=(npos>0)&(nneg>0)
    summaries=[]
    for keep,count in ((pos,npos),(neg,nneg)):
        w=np.where(keep & mixed[i],1/np.maximum(1,count[i]),0)
        A+=a.T@(a*w[:,None]);B+=a.T@(y*w)
        m=np.stack([np.bincount(i,weights=c*w,minlength=len(n))[mixed] for c in a.T],axis=1)
        summaries.append((m,np.bincount(i,weights=y*w,minlength=len(n))[mixed]))
    (p,yp),(q,yq)=summaries
    A-=p.T@q+q.T@p;B-=p.T@yq+q.T@yp
    return A,B,len(n),int(mixed.sum()),int((npos*nneg).sum())


def checks():
    a=np.array([[1.,2],[1,4],[1,1],[1,5],[1,2]])
    y=np.array([1.,2,-3,0,-1]);i=np.array([0,0,0,1,1])
    A,B,_,m,k=moments(a,i,y)
    ax=np.vstack((a,(a[[0,1]]-a[2])));v=np.r_[y,y[:2]-y[2]]
    w=np.array([1/3]*3+[.5]*4)
    assert m==1 and k==2 and np.allclose(A,ax.T@(ax*w[:,None])) and np.allclose(B,ax.T@(w*v))
    l=np.arange(8.)[None,:];g=np.array([[.2,.4,.7]])
    assert np.isclose(curve(l,g,[-1,1])[0],2/(1+np.exp(-base.lookup(l[0],g[0])))-1)


def getrows(yr,h,role,t):
    d=base.DEST/f'labels-{yr}-{h}'
    assert base.file_sha(d/'rows.npz')==base.read(d/'DONE.json')['sha']
    with np.load(d/'rows.npz') as z:v=z[role];ids=z[role+'_id']
    i=ids[:,0]; end=base.ns(f'{yr+1}-01-01')
    good=(ids[:,2]<end)&(t[i]>=base.ns(f'{yr}-01-01' if yr<2023 else '2023-07-01'))&(t[i]<end)
    assert (ids[good,1]<=t[i[good]]).all()
    i=i[good];v=v[good].astype(float);g=v[:,:3]
    return i,g/(g+[1.,1.,24.]),v[:,3]


def run(seed,bank,*,arms=ARMS,encode=design,out=OUT,route=None):
    x,t,valid=bank
    pn=panel(2023,seed,*bank)
    models={};result={}
    for r,role in enumerate(base.ROLES[2:]):
        if route is not None:assert int(np.argmax(route(role)))==r
        head=base.Expert(208,True,'cls').eval()
        path=base.OUT/'2023/champion'/f'cls-{role}-4-{seed}'/'expert.pt'
        assert base.file_sha(path)==pn.hashes[role]
        head.load_state_dict(base.torch.load(path,weights_only=True,map_location='cpu'))
        with base.torch.inference_mode():
            logits=head.net(base.torch.from_numpy(x)).reshape(-1,4,8).numpy().astype(float)
        for j,h in enumerate(base.HS):
            rows=[getrows(yr,h,role,t) for yr in (2019,2020,2021,2022)]
            i,g,y=map(np.concatenate,zip(*rows))
            assert valid[i].all()
            test,geo,val=getrows(2023,h,role,t)
            assert valid[test].all() and t[i].max()<t[test].min()
            order=np.argsort(test,kind='stable');sw=geo.copy()
            for ix in np.split(order,np.flatnonzero(np.diff(test[order]))+1):
                sw[ix]=geo[np.roll(ix,1)]
            _,inv,n=np.unique(test,return_counts=True,return_inverse=True)
            conflict=(np.bincount(inv,weights=val>0)>0)&(np.bincount(inv,weights=val<0)>0)
            means=pn.means[r+2,j]
            metrics={}
            for arm in arms:
                if arm!='curve':
                    a=encode(arm,logits[i,j],g)
                    mu=a.mean(0);sd=a.std(0);sd[sd<1e-8]=1
                    a=np.column_stack((np.ones(len(i)),(a-mu)/sd))
                    A,B,nt,nm,npairs=moments(a,i,y)
                    reg=np.eye(len(A))*.1*nt;reg[0,0]=0
                    coef=np.linalg.solve(A+reg,B)
                    models[arm,r,j]=(mu,sd,coef)
                for swapped,state in ((False,geo),(True,sw)):
                    l=logits[test,j]
                    s=(curve(l,state,means) if arm=='curve' else
                       coef[0]+((encode(arm,l,state)-mu)/sd)@coef[1:])
                    v=np.maximum(val,0)-np.where(s>0,val,0)
                    loss=np.bincount(inv,weights=v)/n
                    metrics[arm+('_swap' if swapped else '')]=float(loss[conflict].mean())
            result[f'{role}:{h}']=dict(conflict_ticks=int(conflict.sum()),n=len(test),fit_pairs=npairs,
                                        fit_mixed_ticks=nm,regret=metrics)
    # Replay identical eligible signals. Different holding times can block
    # different signals. Also replay each opportunity in isolation at fixed cash.
    df=pd.read_csv(LEDGER/f'offers-{seed}.csv.gz')
    df['j']=df.side.map({'OPEN_BUY':0,'OPEN_SELL':4})+df.h.map(dict(zip(base.HS,range(4))))
    p=df.pivot(index='origin_ns',columns='j',values='pred')
    assert np.array_equal(p.index,pn.e.v[pn.bars]+base.HOUR)
    v=p.to_numpy().copy();v[p.index<base.ns('2023-07-01')]=0
    raw=winner(v)
    k=np.flatnonzero(raw>=0);accounts={};fixed={};native=pn.close
    for arm in arms:
        def close(e,b,j):
            if not pn.envs[0].valid[b]:return 0
            po=e.pos;r=int(po.direction<0);ii=pn.index[pn.envs[0].rows[b]]
            if route is not None:r=int(np.argmax(route(po.direction)))
            px=e.quote_price(b,e.c[b],-po.direction,int(e.v[b])+base.HOUR)
            sc=abs(po.entry_price-po.sl)
            g=np.maximum(0.,[po.direction*(px-po.sl)/sc,po.direction*(po.tp-px)/sc,(e.ff[e.seg_of[b]]-int(e.v[b])-base.HOUR)/base.HOUR])
            mu,sd,c=models[arm,r,j]
            a=encode(arm,pn.cl[r][ii:ii+1,j].astype(float),(g/(g+[1,1,24]))[None,:])
            return 3 if (c[0]+((a-mu)/sd)@c[1:])[0]>0 else 0
        pn.close=native if arm=='curve' else close
        accounts[arm]=pn.account(v.reshape(-1,2,4))
        e=pn.e;e.reset();cap=e.cfg.capital
        profits=[]
        for ix in k:
            ok,cash,_=pn.isolated(pn.bars[ix],int(raw[ix]//4+1),int(raw[ix]%4),cap)
            assert ok
            profits.append(1e4*(cash-cap)/cap)
        fixed[arm]=float(np.mean(profits))
    ref=base.read(Path('/var/tmp/role412-router-campaign')/f'main-{seed}.json')['result']['accounts']['raw']
    assert abs(accounts['curve']['ret']-ref['ret'])<1e-6 and accounts['curve']['trades']==ref['trades']
    np.savez_compressed(out/f'weights-{seed}.npz',**{f'{a}-{r}-{j}-{k}':v for (a,r,j),m in models.items() for k,v in enumerate(m)})
    return dict(seed=seed,heads=pn.hashes,cells=result,account=accounts,same_entry_bp=fixed,entries=len(k))


def main():
    checks();base.torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    results=[]
    for seed in SEEDS:
        row=run(seed,bank);atomic(OUT/f'seed-{seed}.json',row);results.append(row)
        print(seed,row['same_entry_bp'],flush=True)
    passed=[]
    for arm in ARMS[1:]:
        def good(row):
            c=list(row['cells'].values())
            def loss(a):return np.average([z['regret'][a] for z in c],weights=[z['conflict_ticks'] for z in c])
            return (loss(arm)<min(loss('curve'),loss(arm+'_swap')) and
                    row['same_entry_bp'][arm]>row['same_entry_bp']['curve'] and
                    row['account'][arm]['ret']>row['account']['curve']['ret'])
        if all(good(row) for row in results):passed.append(arm)
    atomic(OUT/'DONE.json',dict(replication_candidates=passed,promoted=False))


if __name__=='__main__':main()
