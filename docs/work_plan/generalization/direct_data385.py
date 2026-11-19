"""Native direct-role labels. CPU only; snapshots never alter the source account."""
import argparse
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from app.decision_attribution import fork_env
from app.economic_roles385 import best_open
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment
from docs.work_plan.generalization.role_data384 import PERIODS
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import read,write

def context(a):
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    old=read(a.old/'CONFIG.json');verify_done(a.old,old['identity']);verify_done(a.cache,old['diagnostic'])
    if file_sha(a.bundle/'model.pt')!=old['source_sha256']:raise ValueError('source checkpoint')
    model,_=load_bundle(a.bundle,'cpu');m=market_for(a.parent,spec,model.recipe)
    if support_hash(m.inputs)!=old['support'] or spec['design']['horizon']!=6:raise ValueError('support/profile')
    with np.load(a.cache/'SIGNALS.npz') as f:sig=f['source'].copy()
    risk=read(a.parent/'RISK.json')
    def make(start,end):
        e=environment(m,risk,6,[start,end],model.effective['gamma']);e.reset();return e
    return model,m,sig,make,old

def period(p):
    start,end,split=p
    e=MAKE(start,end);rows=[];decisions=[];counts=Counter()
    def after(b,cmd):
        v=fork_env(b);n=v.engine.counts['entries'];v.step(cmd)
        return v,v.engine.counts['entries']>n
    def endpoint(first):
        b=fork_env(first)
        for _ in range(23):
            if b.finished:break
            b.step(int(SIG[b.rows[b.bar]]))
        return float(b.engine.equity_at(b.bar)),b.bar
    while not e.finished:
        ix=int(e.rows[e.bar]);decisions.append(ix)
        if e.inputs.period[ix]!='TRAIN' or e.engine.pos:raise ValueError('flat TRAIN')
        if e.engine.entry_gate(e.bar)[0]:
            base,clk=endpoint(after(e,0)[0]);adv=[]
            for cmd,name in ((1,'long'),(2,'short')):
                b,filled=after(e,cmd);cash,k=endpoint(b)
                if k!=clk:raise ValueError('entry clock')
                adv.append(1e4*(cash-base)/e.engine.equity_at(e.bar) if filled else None)
                if not filled or b.finished or not b.engine.pos:
                    counts['not_intact_'+name]+=1;continue
                if b.engine.pos.direction!=(1 if cmd==1 else -1):raise ValueError('direction')
                hold,kh=endpoint(after(b,0)[0]);close,kc=endpoint(after(b,3-cmd)[0])
                if kc!=kh:raise ValueError('exit clock')
                gain=1e4*(close-hold)/b.engine.equity_at(b.bar)
                rows.append((name,int(b.rows[b.bar]),int(gain>1e-8),0.,gain,0.,split))
            if None not in adv:
                y=best_open(*adv)
                if y>=0:rows.append(('flat',ix,y,0.,*adv,split))
                else:counts['ambiguous']+=1
            else:counts['unpaired']+=1
        e.step(0)
    print(end,len(rows),dict(counts),flush=True)
    return rows,decisions,dict(counts)

def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU required')
    global MAKE,SIG
    model,m,SIG,MAKE,old=context(a)
    with mp.get_context('fork').Pool(3) as pool:parts=pool.map(period,PERIODS)
    rows=[r for rr,_,_ in parts for r in rr];ix=np.unique([i for _,d,_ in parts for i in d])
    latent=[]
    def hook(module,args):latent.append(args[0].detach().numpy().copy())
    h=model.actor_net.head.register_forward_pre_hook(hook)
    with torch.inference_mode():
        for batch in np.array_split(ix,(len(ix)+255)//256):
            x=torch.from_numpy(np.stack([m.inputs.observation(int(i))['features'] for i in batch]))
            logits=model.surfaces(x)
            with model.loss.actor_network_params.to_module(model.actor):
                torch.testing.assert_close(model.actor_net.head(torch.from_numpy(latent[-1])),logits,rtol=0,atol=0)
            latent.pop()
    h.remove();X=np.concatenate(latent)
    if len(ix)!=len(X) or not np.isfinite(X).all():raise ValueError('encoder')
    a.out.mkdir(exist_ok=False,parents=True)
    np.savez_compressed(a.out/'market.npz',ix=ix,x=X,t=m.inputs.row_ts_ns[ix])
    for name in ('flat','long','short'):
        r=np.array([r[1:] for r in rows if r[0]==name]);idx=np.searchsorted(ix,r[:,0])
        if not np.array_equal(ix[idx],r[:,0]):raise ValueError('support')
        np.savez_compressed(a.out/(name+'.npz'),idx=idx,y=r[:,1].astype(int),bp=r[:,2:5],split=r[:,5].astype(int))
    write(a.out/'DATA.json',dict(source=old['source_sha256'],support=old['support'],periods=PERIODS,
        exclusions=[v for _,_,v in parts],files={p.name:file_sha(p) for p in a.out.glob('*.npz')}))

def arguments():
    p=argparse.ArgumentParser()
    for name in ('parent','bundle','old','cache','out'):p.add_argument('--'+name,type=Path,required=True)
    return p
if __name__=='__main__':run(arguments().parse_args())
