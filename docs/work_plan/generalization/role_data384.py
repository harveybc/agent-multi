"""Build offline role labels; emit market latents and external lookup coordinates separately."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from app.decision_attribution import fork_env
from app.frozen_recipe_inputs import file_sha
from app.role_window384 import windows
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment
from app.zigzag_opportunities import load_tape,engine,HOUR
from docs.work_plan.generalization.role_surface379 import query
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import read,write
from tools.zigzag_opportunity_deap import authenticated

PERIODS=[('2020-01-01','2021-01-01',0),('2021-01-01','2022-01-01',0),
         ('2022-01-01','2022-07-01',1),('2022-07-01','2023-01-01',2),('2023-01-01','2024-01-01',3)]
ROLES=('open_buy','open_sell','close_buy','close_sell')

def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU required')
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    old=read(a.old/'CONFIG.json');verify_done(a.old,old['identity']);verify_done(a.cache,old['diagnostic'])
    if file_sha(a.bundle/'model.pt')!=old['source_sha256']:raise ValueError('source')
    model,_=load_bundle(a.bundle,'cpu');m=market_for(a.parent,spec,model.recipe)
    if support_hash(m.inputs)!=old['support']:raise ValueError('support')
    zz=read(a.zig/'SPEC.private.json')
    ident=hashlib.sha256(json.dumps(zz,sort_keys=True).encode()).hexdigest()
    report=authenticated(a.zig/'RESULTS.json',ident)
    h=spec['design']['horizon']
    par=next(r['params'] for r in report['rows'] if r['horizon']==h)
    tape=load_tape(zz);z,_,_=engine(zz['engine_root'])
    expected=zz['engine_sha256']['app/short_horizon/zigzag.py']
    if file_sha(z.__file__)!=expected:
        src=subprocess.check_output(['git','-C',zz['engine_root'],'cat-file','blob','1aaea5cb2262130cf4ce8ff7c87a745ccbe82290'])
        if hashlib.sha256(src).hexdigest()!=expected:raise ValueError('pivot implementation')
        z=types.ModuleType('app.short_horizon.zigzag384_pinned')
        z.__package__='app.short_horizon';sys.modules[z.__name__]=z
        exec(compile(src,'verified_historical_zigzag','exec'),z.__dict__)
    with np.load(a.cache/'SIGNALS.npz') as f:sig=f['source'].copy()
    risk=read(a.parent/'RISK.json')
    rows={name:[] for name in ROLES}
    for start,end,split in PERIODS:
        lo,hi=[pd.Timestamp(t,tz='UTC').value for t in (start,end)]
        yy=np.zeros((len(tape.open),2),bool);known=np.zeros(len(yy),bool)
        # available_at, not midnight bar-open time, defines the split.
        l,r=np.searchsorted(tape.times+HOUR,[lo,hi])
        for beg,stop in tape.chunks:
            b,k=max(l,beg),min(r,stop)
            if b>=k:continue
            piv=z.hindsight_pivots(tape.high[b:k],tape.low[b:k],z.Params(**par))
            yy[b:k],known[b:k]=windows(k-b,piv,a.left,a.right)
        e=environment(m,risk,h,[start,end],model.effective['gamma']);e.reset()
        def branch(cmd):
            v=fork_env(e);count=v.engine.counts['entries']
            v.step(cmd);filled=v.engine.counts['entries']>count
            for _ in range(23):
                if v.finished:break
                v.step(int(sig[v.rows[v.bar]]))
            return float(v.engine.equity_at(v.bar)),filled,v.bar
        while not e.finished:
            row=int(e.rows[e.bar]);p=e.engine.pos
            if e.inputs.period[row]!='TRAIN':raise ValueError('period')
            stamp=int(e.inputs.row_ts_ns[row])
            if not lo<=stamp<hi:raise ValueError('clock')
            q=np.zeros(3) if p is None else query(e)
            if p:
                tasks=[(2 if p.direction==1 else 3,2 if p.direction==1 else 1,None)]
            elif e.engine.entry_gate(e.bar)[0]:
                fill=int(e.engine.v[e.bar+1]);i=np.searchsorted(tape.times,fill)
                tasks=[(j,j+1,yy[i,j]) for j in range(2)] if i<len(yy) and tape.times[i]==fill and known[i] else []
            else:tasks=[]
            if tasks:
                base,_,clock=branch(0)
                for j,cmd,tag in tasks:
                    cash,filled,other=branch(cmd)
                    if other!=clock:raise ValueError('unpaired clock')
                    if p is None and not filled:continue
                    bp=10000*(cash-base)/e.engine.equity_at(e.bar)
                    rows[ROLES[j]].append((row,stamp,split,q,float(bp>1e-8 if tag is None else tag),bp))
            e.step(int(sig[row]))
        print(end,[len(v) for v in rows.values()],flush=True)
    ix=np.unique([r[0] for v in rows.values() for r in v]);latent=[]
    # Intercept the actual head input, with TorchRL functional parameters bound.
    def hook(module,args):
        latent.append(args[0].detach().cpu().numpy().copy())
    handle=model.actor_net.head.register_forward_pre_hook(hook)
    with torch.inference_mode():
        for batch in np.array_split(ix,max(1,(len(ix)+255)//256)):
            x=torch.from_numpy(np.stack([m.inputs.observation(int(i))['features'] for i in batch]))
            logits=model.surfaces(x)
            with model.loss.actor_network_params.to_module(model.actor):
                torch.testing.assert_close(model.actor_net.head(torch.from_numpy(latent[-1])),logits,rtol=0,atol=0)
            latent.pop()  # the parity call fired the same hook
    handle.remove();x=np.concatenate(latent)
    if len(x)!=len(ix) or not np.isfinite(x).all():raise ValueError('latents')
    a.out.mkdir(parents=True,exist_ok=False)
    for role,v in rows.items():
        row,t,fold,q,y,bp=map(np.asarray,zip(*v))
        np.savez_compressed(a.out/(role+'.npz'),x=x[np.searchsorted(ix,row)],t=t,fold=fold,q=q,y=y,bp=bp)
    write(a.out/'DATA.json',dict(source=old['source_sha256'],support=old['support'],zigzag=ident,
        params=par,horizon=h,left=a.left,right=a.right,latent_dim=x.shape[1],periods=PERIODS,
        files={p.name:file_sha(p) for p in sorted(a.out.glob('*.npz'))}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('bundle','parent','cache','zig','old','out'):p.add_argument('--'+k,type=Path,required=True)
    for k,d in [('left',.25),('right',.5)]:p.add_argument('--'+k,type=float,default=d)
    run(p.parse_args())
