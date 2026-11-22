"""Read-only paired routing audit; historical 2024 interventions, no tuning."""
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tools.role404_fit import Expert,arbitration
from tools.role404_account import lookup
from tools.role404_data import DEST,ROOT,PARENT,HS,ROLES,HOUR,assemble,environment,file_sha
from app.structural_refit import load_bundle
from app.sequential_account_pareto import clone

BASE=Path('/var/tmp/role406-cls4-paired')
read=lambda p:json.loads(p.read_text())
tags=('NOP','BUY','SELL')


def decisions(scores):
    if not np.isfinite(scores).all():raise ValueError('score')
    sides=scores.max(axis=2)
    cmd=np.array([arbitration(*row) for row in sides])
    hops=scores.argmax(axis=2)[np.arange(len(cmd)),np.maximum(cmd-1,0)]
    return cmd,np.where(cmd>0,hops,-1)


def classes(f,j,e,k):
    if f!=e:return tags[f]+'->'+tags[e]
    if f and j!=k:return tags[f]+'_change_h'
    return tags[f]+'_same'


def data():
    cut=pd.Timestamp('2023-07-01',tz='UTC').value
    means=np.zeros((4,4,2));labels={}
    for j,h in enumerate(HS):
        by=[[] for _ in ROLES]
        for y in range(2020,2024):
            d=DEST/f'labels-{y}-{h}'
            assert file_sha(d/'rows.npz')==read(d/'DONE.json')['sha']
            with np.load(d/'rows.npz') as z:
                for r,role in enumerate(ROLES):
                    ids=z[role+'_id'];v=z[role][:,3]
                    by[r].append(v[ids[:,2]<cut])
                    if y==2023 and r<2:
                        use=ids[:,1]>=cut
                        labels[r,j]=dict(zip(ids[use,0],v[use]))
                        assert len(labels[r,j])==sum(use)
        for r in range(4):
            a=np.concatenate(by[r]);means[r,j]=(a[a<=0].mean(),a[a>0].mean())
    support=sorted(set.intersection(*(set(v) for v in labels.values())))
    actual=np.array([[[labels[r,j][i] for j in range(4)] for r in range(2)] for i in support])
    assert file_sha(DEST/'market.npz')==read(DEST/'CACHE.json')['sha']
    with np.load(DEST/'market.npz') as z:panel=torch.from_numpy(z['x'][support])
    return means,actual,panel


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(1);means,true,panel=data()
    bundle=ROOT/'model/61/bundle'
    _,m=assemble(PARENT,bundle,ROOT/'history397-61-r2')
    ins=m.inputs;ix=np.flatnonzero(ins.valid&(ins.period=='VALIDATION'))
    index=np.full(len(ins.valid),-1);index[ix]=np.arange(len(ix))
    src,_=load_bundle(bundle,'cpu')
    with torch.inference_mode(),src.loss.actor_network_params.to_module(src.actor):
        enc=src.actor_net.encoder.eval()
        bank=torch.cat([enc({'features':torch.from_numpy(ins.windows[i])}) for i in np.array_split(ix,np.arange(256,len(ix),256))])
    risk=read(PARENT/'RISK.json')
    ref=read(BASE/'WEIGHT_FORENSICS.json')['accounts'];report={}
    for si,seed in enumerate((7,19,43)):
        def nets(arm,features,roles):
            arr=[]
            for role in roles:
                d=BASE/arm/f'cls-{role}-4-{seed}';p=d/'expert.pt'
                assert file_sha(p)==ref[arm+'/'+arm][si]['weights'][role]==read(d/'RESULT.json')['weights']
                assert read(d/'CONTRACT.json')['encoder']==file_sha(bundle/'model.pt')
                net=Expert(208,role.startswith('CLOSE'),'cls').eval()
                net.load_state_dict(torch.load(p,weights_only=True))
                with torch.inference_mode():arr.append(net.net(features).reshape(len(features),4,net.n).numpy())
            return arr
        def opens(arm,features):
            logits=np.stack(nets(arm,features,ROLES[:2]),axis=1)[...,0].astype(float)
            prob=1/(1+np.exp(-np.clip(logits,-50,50)))
            lo,hi=means[:2,:].transpose(2,0,1)
            # Match the sealed router's float32 difference of its TRAIN means.
            return lo+prob*(hi.astype('f')-lo.astype('f')),prob
        rv={};routes=[];scores=[];internal={}
        for arm in ('fine','explore'):
            s,_=opens(arm,bank);scores.append(s);routes.append(decisions(s))
            ev,pr=opens(arm,panel);act,h=decisions(ev);chosen=act>0
            idx=np.nonzero(chosen)[0];rr=act[chosen]-1;hh=h[chosen]
            pred=ev[idx,rr,hh];obs=true[idx,rr,hh]
            internal[arm]=dict(ticks=len(panel),chosen=len(idx),predicted_bp=float(pred.mean()),actual_bp=float(obs.mean()),
                brier=float(((pr-(true>0))**2).mean()),sell_h24_brier=float(((pr[:,1,3]-(true[:,1,3]>0))**2).mean()),
                chosen_sell=int((rr==1).sum()),positive_cells=float((ev>0).mean()))
        rv['internal_2023H2_common_filled']=internal
        cl=nets('fine',bank,ROLES[2:])
        envs=[environment(m,risk,h,['2024-01-01','2025-01-01'],.99) for h in HS]
        e=envs[0].engine
        for en in envs:en.engine=e
        e.reset()
        bars=[b for b in range(e.first,e.last) if envs[0].valid[b] and e.entry_gate(b)[0]]
        assert len(bars)==4912
        def close_cmd(engine,b,active):
            if not envs[0].valid[b]:return 0
            i=index[envs[0].rows[b]];assert i>=0
            p=engine.pos;r=0 if p.direction==1 else 1
            px=engine.quote_price(b,engine.c[b],-p.direction,int(engine.v[b])+HOUR)
            scale=abs(p.entry_price-p.sl)
            g=np.maximum(0.,[p.direction*(px-p.sl)/scale,p.direction*(p.tp-px)/scale,
                            (engine.ff[engine.seg_of[b]]-int(engine.v[b])-HOUR)/HOUR])
            raw=lookup(cl[r][i,active],g/(g+[1.,1.,24.]))
            lo,hi=means[r+2,active]
            return 3 if lo+float(np.float32(hi)-np.float32(lo))/(1+np.exp(-np.clip(raw,-50,50)))>0 else 0
        opportunity=Counter();hchanges=Counter();margins=[];only24=0
        paired=[];refused=Counter()
        with gzip.open(out/f'flat-{seed}.csv.gz','wt') as stream:
            w=csv.writer(stream);w.writerow(['available_utc_ns','fine','h_fine','explore','h_explore']+
                [f'{arm}_{role}_{h}_bp' for arm in ('fine','explore') for role in ROLES[:2] for h in HS])
            for b in bars:
                i=int(index[envs[0].rows[b]]);assert i>=0
                (f,j),(a,k)=[(cmd[i],h[i]) for cmd,h in routes]
                opportunity[classes(f,j,a,k)]+=1
                if f==a and f and j!=k:hchanges[f'{tags[a]}_{HS[j]}->{HS[k]}']+=1
                if (f,a)==(0,2):
                    diff=scores[1][i,1,k]-max(0.,scores[1][i,0].max())
                    margins.append([diff,scores[1][i,1,k]-scores[0][i,1,k],scores[1][i,1,k]])
                    only24+=int(k==3 and scores[1][i,:,:3].max()<=0)
                    start=clone(e);start.next_decision=b
                    en=envs[k];en.bar=b
                    info=start.step(b,2,entry_plan=en.entry_plan(-1))
                    if not info['entry']:
                        refused[str(info['refused'])]+=1
                    else:
                        pay=[]
                        for discretionary in (False,True):
                            trial=clone(start);nxt=b+1
                            while trial.pos is not None:
                                assert nxt<trial.last
                                trial.step(nxt,close_cmd(trial,nxt,k) if discretionary else 0);nxt+=1
                            assert len(trial.trades)==1
                            pnl=trial.trades[0]['net_usd']
                            assert abs(trial.balance-trial.cfg.capital-pnl)<1e-6
                            pay.append(pnl*1e4/trial.cfg.capital)
                        paired.append([int(ins.row_ts_ns[envs[0].rows[b]]),HS[k],scores[1][i,1,k],*pay])
                w.writerow([ins.row_ts_ns[envs[0].rows[b]],tags[f],HS[j] if f else '',tags[a],HS[k] if a else '']+
                           [v for s in scores for v in s[i].ravel()])
        rv.update(opportunities=opportunity,horizon_changes=hchanges,only_h24_positive_new_sell=only24,
            new_sell_median_margin_delta_score_bp=np.median(margins,axis=0).tolist() if margins else None)
        assert len(paired)+sum(refused.values())==len(margins)
        with gzip.open(out/f'new-sell-pairs-{seed}.csv.gz','wt') as stream:
            w=csv.writer(stream);w.writerow(['origin_ns','h','predicted_advantage_bp','hold_net_bp','fine_close_net_bp']);w.writerows(paired)
        rv['independent_new_sells']=dict(filled=len(paired),refused=refused,
            means_predicted_hold_close_bp=np.mean([p[2:] for p in paired],axis=0).tolist())
        for arm in ('fine','explore','veto_new_sell','keep_h','veto_all_new'):
            e.reset();active=cat=None;fills=defaultdict(list);ledger=[]
            for b in range(e.first,e.last):
                cmd=0;plan=None
                if envs[0].valid[b]:
                    i=index[envs[0].rows[b]];assert i>=0
                    (f,j),(a,k)=[(cmds[i],hs[i]) for cmds,hs in routes]
                    if e.pos is None and e.entry_gate(b)[0]:
                        cat=classes(f,j,a,k);cmd,active=(f,j) if arm=='fine' else (a,k)
                        if f==0 and ((arm=='veto_new_sell' and a==2) or arm=='veto_all_new'):cmd=0
                        if arm=='keep_h' and f==a and f:active=j
                        if cmd:
                            opening=int(ins.row_ts_ns[envs[0].rows[b]])
                            en=envs[active];en.bar=b;plan=en.entry_plan(1 if cmd==1 else -1)
                    elif e.pos is not None:
                        cmd=close_cmd(e,b,active)
                before=len(e.trades)
                info=e.step(b,int(cmd),entry_plan=plan)
                if len(e.trades)>before:
                    assert len(e.trades)==before+1 and cat is not None
                    t=e.trades[-1];fills[cat].append(t['net_usd'])
                    ledger.append((opening,cat,t['direction'],HS[active],info['close'],t['net_usd']))
            assert e.pos is None
            net=sum(t['net_usd'] for t in e.trades);assert abs(e.balance-e.cfg.capital-net)<1e-6
            if arm in ('fine','explore'):
                orig=ref[arm+'/fine'][si]
                assert abs(net-orig['net_usd'])<1e-6 and len(ledger)==sum(t['trades'] for t in orig['sides'].values())
                with gzip.open(out/f'trades-{seed}-{arm}.csv.gz','wt') as stream:
                    w=csv.writer(stream);w.writerow(['origin_ns','transition','direction','h','close','net_usd']);w.writerows(ledger)
            rv[arm]=dict(trades=len(ledger),net_usd=net,return_pct=100*net/e.cfg.capital,
                groups={key:dict(n=len(vals),wins=sum(v>0 for v in vals),net=sum(vals)) for key,vals in fills.items()})
        report[seed]=rv
    (out/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);main(p.parse_args().out)
