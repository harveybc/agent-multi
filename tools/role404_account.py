"""Automatic CPU integration of a restored quartet. Fixed 2024 comparison."""
import json
import os
from pathlib import Path
import time
from collections import Counter
import numpy as np
import pandas as pd
import torch
from tools.role404_fit import Expert,arbitration
from tools.role404_data import ROOT,PARENT,DEST,HS,ROLES,save,file_sha,assemble,HOUR,environment
from app.structural_refit import load_bundle

def routing_sanity():
    assert arbitration(2,1)==1 and arbitration(1,2)==2 and arbitration(0,0)==0
    assert arbitration(1,1)==0 and arbitration(np.nan,1)==0
    a=np.arange(8,dtype=float)
    for v,target in [(np.zeros(3),0.),(np.ones(3),7.)]:
        assert np.isclose(lookup(a,v),target)

def lookup(a,v):
    basis=np.array([1.])
    for t in v:basis=np.r_[basis*(1-t),basis*t]
    return float(basis@a)

def run(only_root=None, cases=None):
    routing_sanity()
    torch.set_num_threads(1)
    bundle=ROOT/'model/61/bundle'
    _,market=assemble(PARENT,bundle,ROOT/'history397-61-r2')
    risk=json.loads((PARENT/'RISK.json').read_text())
    inputs=market.inputs;ts=inputs.row_ts_ns; lower=pd.Timestamp('2024',tz='UTC').value
    ix=np.flatnonzero(inputs.valid&(ts>=lower)&(ts<pd.Timestamp('2025',tz='UTC').value))
    index=np.full(len(ts),-1,dtype=int);index[ix]=np.arange(len(ix))
    src,_=load_bundle(bundle,'cpu');parts=[]
    with src.loss.actor_network_params.to_module(src.actor),torch.inference_mode():
        src.actor_net.encoder.eval()
        for b in range(0,len(ix),256):
            x=torch.from_numpy(inputs.windows[ix[b:b+256]])
            parts.append(src.actor_net.encoder({'features':x}))
    bank=torch.cat(parts)
    root=Path('/var/tmp/role405') if os.environ.get('ROLE_RUN')=='405' else DEST
    if only_root is not None:root=only_root
    accounts=root/'accounts';accounts.mkdir(exist_ok=True)
    def assess(method,years,seed):
        nets={};ids={};means={}
        for role in ROLES:
            d=root/f'{method}-{role}-{years}-{seed}'
            cfg=json.loads((d/'CONTRACT.json').read_text())
            res=json.loads((d/'RESULT.json').read_text())
            assert cfg['role']==role and cfg['method']==method and cfg['encoder']==file_sha(bundle/'model.pt')
            assert file_sha(d/'expert.pt')==res['weights']
            m=Expert(bank.shape[1],role.startswith('CLOSE'),method);m.load_state_dict(torch.load(d/'expert.pt',weights_only=True))
            with torch.inference_mode():
                v=m.net(bank).reshape(len(ix),4,m.n,m.m)[:,:,:,0].numpy()
            ids[role]=res['weights']
            nets[role]=v*cfg['scale'] if method=='adv' else v
            if method=='cls':
                for hi,h in enumerate(HS):
                    subset=[]
                    for y in range(2024-years,2024):
                        lab=DEST/f'labels-{y}-{h}'
                        with np.load(lab/'rows.npz') as f:
                            subset.append(f[role][f[role+'_id'][:,2]<pd.Timestamp('2023-07-01',tz='UTC').value,3])
                    a=np.concatenate(subset);means[role,hi]=(a[a<=0].mean(),a[a>0].mean())
        def score(role,i,j,g=None):
            raw=nets[role][i,j]
            a=float(raw[0]) if g is None else lookup(raw,g)
            if method=='adv':return a
            lo,up=means[role,j];p=1/(1+np.exp(-np.clip(a,-50,50)))
            return lo+p*(up-lo)
        envs=[environment(market,risk,h,['2024-01-01','2025-01-01'],.99) for h in HS]
        e=envs[0].engine;e.reset()
        for env in envs:env.engine=e
        active=None;stats=Counter();equity=[]
        for b in range(e.first,e.last):
            cmd=0;plan=None
            if envs[0].valid[b]:
                i=index[envs[0].rows[b]];assert i>=0
                if e.pos is None and e.entry_gate(b)[0]:
                    vals=[[score(r,i,j) for j in range(4)] for r in ROLES[:2]]
                    cmd=arbitration(*np.max(vals,axis=1))
                    if cmd:
                        active=int(np.argmax(vals[cmd-1]));env=envs[active];env.bar=b
                        plan=env.entry_plan(1 if cmd==1 else -1)
                elif e.pos is not None:
                    p=e.pos;px=e.quote_price(b,e.c[b],-p.direction,int(e.v[b])+HOUR)
                    scale=abs(p.entry_price-p.sl)
                    g=np.maximum(0.,[p.direction*(px-p.sl)/scale,p.direction*(p.tp-px)/scale,
                                     (e.ff[e.seg_of[b]]-int(e.v[b])-HOUR)/HOUR])
                    if score(ROLES[2 if p.direction==1 else 3],i,active,g/(g+[1.,1.,24.]))>0:cmd=3
            info=e.step(b,cmd,entry_plan=plan)
            if info['entry']:stats['entries_h'+str(HS[active])]+=1
            if info['close']:stats[info['close']]+=1
            equity.append((int(e.v[b+1])+HOUR,e.equity_at(b+1)))
        assert e.pos is None
        net=sum(t['net_usd'] for t in e.trades)
        assert abs(net-e.balance+e.cfg.capital)<1e-6
        eq=np.r_[e.cfg.capital,[v for _,v in equity]]
        daily=pd.Series([v for _,v in equity],index=pd.to_datetime([k for k,_ in equity],utc=True)).resample('D').last().dropna().pct_change().dropna()
        sides={name:dict(trades=len(a),wins=sum(bool(x['net_usd']>0) for x in a)) for side,name in [(1,'buy'),(-1,'sell')]
               for a in [[t for t in e.trades if t['direction']==side]]}
        return dict(method=method,years=years,seed=seed,weights=ids,
            validation='real 2024, historical conditional comparison',net_usd=net,
            return_pct=100*net/e.cfg.capital,max_dd_pct=100*float(np.max(1-eq/np.maximum.accumulate(eq))),
            sharpe_daily_252=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else None,
            counts=stats,sides=sides,promoted=False)
    if cases is not None:
        for m,y,s in cases:
            save(accounts/f'{m}-{y}-{s}.json',assess(m,y,s))
        return
    while True:
        for m in ('adv','cls'):
            for y in (4,8):
                for s in (7,19,43):
                    out=accounts/f'{m}-{y}-{s}.json'
                    if out.exists():continue
                    if all((root/f'{m}-{r}-{y}-{s}'/'RESULT.json').is_file() and
                           (root/f'{m}-{r}-{y}-{s}'/'expert.pt').is_file() for r in ROLES):
                        save(out,assess(m,y,s))
        if len(list(accounts.glob('*.json')))==12:break
        time.sleep(20)

if __name__=='__main__':run()
