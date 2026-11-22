"""Managed opening value, one common reference opportunity, chronological calibration."""
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from tools.role404_data import DEST,ROOT,PARENT,HS,ROLES,HOUR,assemble,environment,clone,file_sha,save
from tools.role404_fit import Expert
from tools.role404_account import lookup
from tools.role406_compare import ROOT as SOURCE
from tools.role407_openings import decisions
from tools.role408_folds import OUT,SEEDS
from app.structural_refit import load_bundle

read=lambda p:json.loads(p.read_text())
ns=lambda date:pd.Timestamp(date,tz='UTC').value


def calibrate(blocks,score_year):
    if not blocks or any(not (b['year']<score_year and b['end']<ns(f'{score_year}-01-01')) for b in blocks):
        raise ValueError('future calibration outcome')
    p=np.concatenate([b['p'] for b in blocks]);y=np.concatenate([b['y'] for b in blocks])
    if not np.isfinite(p).all():raise ValueError('nonfinite probability')
    fits=np.empty((2,4,4))
    for r,j in np.ndindex(2,4):
        ok=np.isfinite(y[:,r,j]);x=p[ok,r,j];v=y[ok,r,j]
        if not len(x):raise ValueError('no filled targets')
        dx=x-x.mean();den=dx@dx
        slope=max(0.,(dx@(v-v.mean()))/den) if den>0 else 0.
        fits[r,j]=[v.mean()-slope*x.mean(),slope,x.min(),x.max()]
    return fits


def apply_map(p,fits):
    a,b,lo,hi=fits.transpose(2,0,1)
    return a+b*np.clip(p,lo,hi)


def terminal(cash,next_free,anchor,extra):
    return cash+extra(anchor,cash) if anchor>=next_free else cash


def class_means(year):
    groups=[[[] for j in HS] for r in ROLES]
    bound=ns(f'{year-1}-07-01')
    for y in range(year-4,year):
        for j,h in enumerate(HS):
            d=DEST/f'labels-{y}-{h}'; f=d/'rows.npz'
            if file_sha(f)!=read(d/'DONE.json')['sha']:raise ValueError('labels digest')
            with np.load(f) as z:
                for r,role in enumerate(ROLES):
                    groups[r][j].append(z[role][z[role+'_id'][:,2]<bound,3])
    out=np.empty((4,4,2),np.float32)
    for r,j in np.ndindex(4,4):
        v=np.concatenate(groups[r][j]);out[r,j]=[v[v<=0].mean(),v[v>0].mean()]
    assert np.isfinite(out).all()
    return out


class Panel:
    def __init__(self,market,year,seed,bank,ix,means,risk):
        self.year=year;self.means=means;self.hashes={}
        root=SOURCE/'fine' if year==2024 else OUT/str(year)/'champion'
        raw=[]
        for role in ROLES:
            d=root/f'cls-{role}-4-{seed}';p=d/'expert.pt';rec=read(d/'CONTRACT.json')
            assert rec['encoder']==file_sha(ROOT/'model/61/bundle/model.pt') and rec['seed']==seed and rec['role']==role
            if year<2024:assert rec['score_year']==year and rec['monitor_label_end_max']<ns(f'{year}-01-01')
            model=Expert(208,role.startswith('CLOSE'),'cls').eval()
            self.hashes[role]=file_sha(p);assert self.hashes[role]==read(d/'RESULT.json')['weights']
            model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
            with torch.inference_mode():raw.append(model.net(bank).reshape(len(ix),4,model.n).numpy())
        self.cl=raw[2:]
        self.envs=[environment(market,risk,h,[f'{year}-01-01',f'{year+1}-01-01'],.99) for h in HS]
        self.e=e=self.envs[0].engine
        for env in self.envs:env.engine=e
        self.index=np.full(len(market.inputs.valid),-1);self.index[ix]=np.arange(len(ix))
        assert np.all(self.index[self.envs[0].rows[e.first:e.last][self.envs[0].valid[e.first:e.last]]]>=0)
        self.bars=np.array([b for b in range(e.first,e.last) if self.envs[0].valid[b] and e.entry_gate(b)[0]])
        self.ixbar={int(b):i for i,b in enumerate(self.bars)}
        i=self.index[self.envs[0].rows[self.bars]]
        logits=np.stack([x[i,:,0] for x in raw[:2]],axis=1).astype(float)
        self.p=1/(1+np.exp(-np.clip(logits,-50,50)))
        lo,hi=means[:2].transpose(2,0,1);self.raw=lo+self.p*(hi-lo)
        self.cutoff=ns(f'{year+1}-01-01')

    def close(self,e,b,j):
        if not self.envs[0].valid[b]:return 0
        p=e.pos;r=int(p.direction<0);i=self.index[self.envs[0].rows[b]]
        assert i>=0
        px=e.quote_price(b,e.c[b],-p.direction,int(e.v[b])+HOUR)
        sc=abs(p.entry_price-p.sl)
        g=np.maximum(0.,[p.direction*(px-p.sl)/sc,p.direction*(p.tp-px)/sc,(e.ff[e.seg_of[b]]-int(e.v[b])-HOUR)/HOUR])
        z=lookup(self.cl[r][i,j],g/(g+[1.,1.,24.]));lo,hi=self.means[2+r,j]
        return 3 if lo+(hi-lo)/(1+np.exp(-np.clip(z,-50,50)))>0 else 0

    def isolated(self,b,side,j,cash):
        t=clone(self.e);assert not t.trades and t.pos is None
        t.next_decision=b;t.balance=cash
        env=self.envs[j];env.bar=b
        entry=t.step(b,side,entry_plan=env.entry_plan(1 if side==1 else -1))['entry']
        k=b+1
        while t.pos is not None:
            assert k<t.last
            t.step(k,self.close(t,k,j));k+=1
        assert len(t.trades)==int(entry)
        assert abs(t.balance-cash-sum(v['net_usd'] for v in t.trades))<1e-6
        return entry,t.balance,k

    def targets(self):
        e=self.e;e.reset();cap=e.cfg.capital
        anchors={}
        for b in self.bars:
            seg=e.seg_of[b]
            if b>=e.first+3 and e.v[b]+HOUR>=e.op[seg]+36*HOUR:anchors.setdefault(seg,b)
        @lru_cache(4096)
        def extra(anchor,cash):
            side=np.sign(e.c[anchor]-e.c[anchor-3])
            if not side:return 0.
            return self.isolated(anchor,1 if side>0 else 2,HS.index(6),cash)[1]-cash
        y=np.full(self.raw.shape,np.nan);ends=[]
        for i,b in enumerate(self.bars):
            seg=e.seg_of[b];limit=e.ff[seg]
            if limit>=self.cutoff or limit>e.v[e.last]+HOUR:continue
            a=anchors.get(seg,-1);wait=terminal(cap,b+1,a,extra)
            for r,j in np.ndindex(2,4):
                entry,cash,k=self.isolated(b,r+1,j,cap)
                if entry:
                    assert e.v[k-1]+HOUR<=limit+HOUR
                    y[i,r,j]=1e4*(terminal(cash,k,a,extra)-wait)/cap
            ends.append(int(limit))
        return dict(year=self.year,end=max(ends),p=self.p,y=y)

    def account(self,scores):
        dirs,hs=decisions(scores);e=self.e;e.reset();j=None;eq=[e.cfg.capital];ticks=[]
        for b in range(e.first,e.last):
            cmd=0;plan=None
            if self.envs[0].valid[b]:
                if e.pos is None and e.entry_gate(b)[0]:
                    i=self.ixbar[b];cmd=int(dirs[i])
                    if cmd:
                        j=int(hs[i]);env=self.envs[j];env.bar=b
                        plan=env.entry_plan(1 if cmd==1 else -1)
                elif e.pos is not None:cmd=self.close(e,b,j)
            e.step(b,cmd,entry_plan=plan)
            eq.append(e.equity_at(b+1));ticks.append(int(e.v[b+1])+HOUR)
        assert e.pos is None
        net=e.balance-e.cfg.capital;assert abs(net-sum(t['net_usd'] for t in e.trades))<1e-6
        eq=np.array(eq);ix=pd.to_datetime(ticks,utc=True)
        daily=pd.Series(eq[1:],index=ix).resample('D').last().dropna().pct_change().dropna()
        weeks=len(set(ix.to_period('W').astype(str)))
        sides={s:sum(t['direction']==d for t in e.trades) for s,d in [('buy',1),('sell',-1)]}
        return dict(net_usd=net,ret=100*net/e.cfg.capital,dd=100*float(np.max(1-eq/np.maximum.accumulate(eq))),
            sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else None,
            trades=len(e.trades),wins=sum(int(t['net_usd']>0) for t in e.trades),
            weeks=weeks,orders_week=len(e.trades)/weeks,active=len(e.trades)>2*weeks,**sides)


def opportunity(raw,cal,y):
    rf,rh=decisions(raw);cf,ch=decisions(cal)
    def value(d,h):
        return np.where(d==0,0.,y[np.arange(len(d)),np.maximum(d-1,0),np.maximum(h,0)])
    a,b=value(rf,rh),value(cf,ch);ok=np.isfinite(a)&np.isfinite(b)&np.isfinite(y).any((1,2))
    veto=ok&(rf>0)&(cf==0);new=ok&(rf==0)&(cf>0)
    pos=lambda x:float(x[x>0].sum())
    return dict(origins=len(y),joint_filled_or_nop=int(ok.sum()),both_targets_all8=int(np.isfinite(y).all((1,2)).sum()),
        vetoes=int(veto.sum()),wins_rejected=int((a[veto]>0).sum()),losses_avoided=int((a[veto]<0).sum()),
        gains_rejected_bp_sum=pos(a[veto]),losses_avoided_bp_sum=pos(-a[veto]),
        new=int(new.sum()),new_wins=int((b[new]>0).sum()),changed_side_or_h=int((ok&(rf>0)&(cf>0)&((rf!=cf)|(rh!=ch))).sum()),
        mean_raw_bp=float(a[ok].mean()),mean_cal_bp=float(b[ok].mean()))


def calibration_report(pn,cal,y):
    metrics={}
    for r,j in np.ndindex(2,4):
        mask=np.isfinite(y[:,r,j]);v=y[mask,r,j];p=pn.p[mask,r,j]
        lo,hi=pn.means[r,j];a=pn.raw[mask,r,j];b=cal[mask,r,j]
        metrics[f'{ROLES[r]}_{HS[j]}']=dict(n=len(v),p_mean=float(p.mean()),win_fraction=float((v>0).mean()),
            brier=float(((p-(v>0))**2).mean()),train_conditional_bp=[float(lo),float(hi)],
            managed_conditional_bp=[float(v[z].mean()) if z.any() else None for z in (v<=0,v>0)],
            raw_mean=float(a.mean()),cal_mean=float(b.mean()),managed_mean=float(v.mean()),
            raw_rmse=float(np.mean((a-v)**2)**.5),cal_rmse=float(np.mean((b-v)**2)**.5))
    for name,scores in [('winner_raw',pn.raw),('winner_cal',cal)]:
        d,h=decisions(scores);i=np.flatnonzero(d);r=d[i]-1;j=h[i];v=y[i,r,j];ok=np.isfinite(v)
        metrics[name]=dict(chosen=len(i),observed=int(ok.sum()),
            mean_bid=float(scores[i,r,j][ok].mean()) if ok.any() else None,
            mean_realized=float(v[ok].mean()) if ok.any() else None)
    return metrics


def evaluate_seed(seed):
    torch.set_num_threads(1)
    bundle=ROOT/'model/61/bundle';risk=read(PARENT/'RISK.json')
    joined,orig=assemble(PARENT,bundle,ROOT/'history397-61-r2')
    assert file_sha(DEST/'market.npz')==read(DEST/'CACHE.json')['sha']
    src,_=load_bundle(bundle,'cpu');blocks=[];res={}
    for year in (2022,2023,2024):
        m=orig if year==2024 else joined;ins=m.inputs
        ix=np.flatnonzero(ins.valid&(ins.row_ts_ns>=ns(f'{year}-01-01'))&(ins.row_ts_ns<ns(f'{year+1}-01-01')))
        if year<2024:
            with np.load(DEST/'market.npz') as z:
                assert np.array_equal(ins.row_ts_ns,z['t']) and np.array_equal(ins.valid,z['valid'])
                bank=torch.from_numpy(z['x'][ix])
        else:
            with torch.inference_mode(),src.loss.actor_network_params.to_module(src.actor):
                enc=src.actor_net.encoder.eval()
                bank=torch.cat([enc({'features':torch.from_numpy(ins.windows[ix[k:k+256]])}) for k in range(0,len(ix),256)])
        pn=Panel(m,year,seed,bank,ix,class_means(year),risk)
        fits=calibrate(blocks,year) if blocks else None
        dest=OUT/f'score-{seed}-{year}';dest.mkdir(exist_ok=True)
        if fits is not None:
            # Freeze the map before observing any outcome in the scored year.
            seal=dict(fit_years=[b['year'] for b in blocks],fit_sha=[b['sha'] for b in blocks],coefficients=fits.tolist())
            if (dest/'MAP.json').exists():assert seal==read(dest/'MAP.json')
            save(dest/'MAP.json',seal)
        f=dest/'labels.npz';meta=dest/'RECEIPT.json'
        spec=dict(heads=pn.hashes,year=year,code=file_sha(__file__),seed=seed)
        if meta.exists():
            rc=read(meta);assert rc['spec']==spec and rc['sha']==file_sha(f)
            with np.load(f) as z:
                assert np.array_equal(z['p'],pn.p) and np.array_equal(z['bars'],pn.bars)
                b=dict(year=year,end=rc['end'],p=pn.p,y=z['y'])
        else:
            b=pn.targets()
            tmp=dest/'building.npz';np.savez_compressed(tmp,p=pn.p,y=b['y'],bars=pn.bars,origin_ns=pn.e.v[pn.bars]+HOUR);tmp.replace(f)
            rc=dict(spec=spec,sha=file_sha(f),end=b['end']);save(meta,rc)
        b['sha']=rc['sha']
        if fits is not None:
            cal=apply_map(pn.p,fits);raw_acct=pn.account(pn.raw);cal_acct=pn.account(cal)
            if year==2024:
                reference=read(SOURCE/'fine'/'accounts'/f'cls-4-{seed}.json')
                assert abs(raw_acct['net_usd']-reference['net_usd'])<1e-6
                assert all(raw_acct[s]==reference['sides'][s]['trades'] for s in ('buy','sell'))
            res[year]=dict(raw=raw_acct,cal=cal_acct,opportunities=opportunity(pn.raw,cal,b['y']),
                scores=calibration_report(pn,cal,b['y']))
            save(dest/'ACCOUNT.json',res[year])
        if year<2024:blocks.append(b)
    return res


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    OUT.mkdir(exist_ok=True)
    with (OUT/'scores.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results={}
        for seed in SEEDS:
            paths=[OUT/str(y)/'champion'/f'cls-{r}-4-{seed}'/'RESULT.json' for y in (2022,2023) for r in ROLES]
            while not all(p.exists() for p in paths):
                import subprocess
                state=subprocess.run(['systemctl','--user','is-active','role408-folds.service'],capture_output=True,text=True)
                if state.stdout.strip() not in ('active','activating'):raise RuntimeError('missing fold fits: '+state.stdout.strip())
                time.sleep(5)
            results[seed]=evaluate_seed(seed);save(OUT/'COMPARISON.json',results)
        save(OUT/'DONE.json',dict(seeds=list(SEEDS),promoted=False))


if __name__=='__main__':main()
