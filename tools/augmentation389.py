"""Matched online SAC with optional fixed native synthetic replay banks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from app.frozen_recipe_inputs import file_sha
from app.g3_replay import restore_buffer
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment,emit
from tools.audit_g3_producers import write
from tools.prepare_g3_joint import synthetic_market
from tools.run_discrete_curve_probe import configure,transition
from tools.run_structural_sac_refit import assess_full,fit_epochs
from app.g3_producer_audit import identity

def read(p):return json.loads(p.read_text())
def checked(a):
    manifest=read(a.generated/'COHORTS.json')
    if not manifest['release_for_training'] or manifest['source']!=file_sha(a.bundle/'model.pt'):
        raise ValueError('389_SOURCE')
    spec=read(a.parent/'SPEC.json')
    verify_files(a.parent,{k:v for k,v in spec['files'].items() if k.startswith(('data/','engine/')) or k in ('RISK.json','SUPPORT_RECIPE.json')})
    return manifest,spec

def pack(a):
    manifest,spec=checked(a)
    model,_=load_bundle(a.bundle,'cpu')
    market=market_for(a.parent,spec,model.recipe)
    risk=read(a.parent/'RISK.json')
    for seed in manifest['seeds']['16']:
        path=a.generated/str(seed);out=a.banks/str(seed)
        if file_sha(path/'RESULT.json')!=manifest['receipts'][str(seed)]:raise ValueError('389_PATH')
        report=read(path/'RESULT.json')
        verify_files(path,report['sha256'])
        if (out/'DONE.json').exists():
            rec=read(out/'DONE.json');verify_files(out,rec['files'])
            if rec['scenario']!=manifest['receipts'][str(seed)]:raise ValueError('389_CACHE')
            continue
        out.mkdir(parents=True,exist_ok=True)
        m=synthetic_market(market,pd.read_parquet(path/'OHLC.parquet'),pd.read_parquet(path/'FEATURES71.parquet'))
        logits,_=emit(model,m.inputs)
        env=environment(m,risk,6,model.recipe['split']['periods']['TRAIN'],model.effective['gamma'])
        rng=np.random.default_rng(389);obs=env.reset();records=[];windows=[]
        while not env.finished:
            row=env.rows[env.bar];raw=logits[row].astype(float)
            p=np.exp(raw-raw.max());act=int(rng.choice(3,p=p/p.sum()))
            nxt,reward,done,duration=env.step(act)
            windows.append(obs['features']);records.append((act,reward,duration,done))
            if not obs['mask'].all() or not nxt['mask'].all():raise ValueError('mask')
            if done:windows.append(nxt['features'])
            obs=nxt
        np.save(out/'X.npy',np.stack(windows))
        np.save(out/'META.npy',np.asarray(records))
        if sum(r[3] for r in records)!=1 or not records[-1][3]:raise ValueError('episode')
        write(out/'DONE.json',dict(scenario=manifest['receipts'][str(seed)],source=manifest['source'],
            decisions=len(records),trades=len(env.engine.trades),files={n:file_sha(out/n) for n in ('X.npy','META.npy')}))
        print('packed',seed,len(records),len(env.engine.trades),flush=True)

class PairedReplay:
    _storage=None
    def __init__(self,real,banks,seed,batch):
        self.real,self.banks,self.rng,self.n=real,banks,np.random.default_rng(seed),batch
        self.end=np.cumsum([len(v) for _,v in banks]);self.updates=self.syn=0
    def __len__(self):return len(self.real)
    def extend(self,item):return self.real.extend(item)
    def sample(self):
        n=self.n;r=self.rng.integers(len(self.real),size=n)
        td=self.real._storage.get(torch.as_tensor(r)).clone()
        # Use identical real-row RNG in all arms, even in the zero-dose control.
        u=self.rng.random(n);order=self.rng.permutation(n)
        k=((self.updates+1)*n//10-self.updates*n//10) if self.banks else 0
        if k:
            picks=(u[:k]*self.end[-1]).astype(int);which=np.searchsorted(self.end,picks,side='right')
            for j,(x,v) in enumerate(self.banks):
                sel=np.flatnonzero(which==j)
                if not len(sel):continue
                ix=picks[sel]-(self.end[j-1] if j else 0);dest=order[sel]
                act,reward,steps,done=v[ix].T
                td['features'][dest]=torch.from_numpy(x[ix])
                td['next','features'][dest]=torch.from_numpy(x[ix+1])
                for key,value in [('action',act),('steps_to_next_obs',steps),(('next','reward'),reward),
                                  (('next','done'),done),(('next','terminated'),done)]:
                    dst=td[key]
                    dst[dest]=torch.as_tensor(value.copy(),dtype=dst.dtype).reshape((len(dest),*dst.shape[1:]))
        self.updates+=1;self.syn+=k
        return td
    def receipt(self):
        n=self.updates*self.n
        return dict(updates=self.updates,samples=n,synthetic_samples=self.syn,
                    synthetic_fraction=self.syn/n if n else 0)
    def state_dict(self):
        return dict(real=self.real.state_dict(),rng=self.rng.bit_generator.state,updates=self.updates,syn=self.syn)
    def load_state_dict(self,state):
        restore_buffer(self.real,state['real'])
        self.rng.bit_generator.state=state['rng'];self.updates=state['updates'];self.syn=state['syn']

def train(a):
    manifest,spec=checked(a);seed=a.seed;years=a.years
    if not torch.cuda.is_available():raise ValueError('389_CUDA')
    configure(seed);torch.backends.cudnn.allow_tf32=False
    model,_=load_bundle(a.bundle,'cuda')
    trainable=[]
    for name,p in model.loss.named_parameters():
        yes=not name.startswith('target_') and name.startswith(('actor_network_params.','qvalue_network_params.')) and any(
            s in name for s in ('.head.','.encoder.core.','.encoder.post_fusion.'))
        p.requires_grad_(yes)
        if yes:trainable.append(name)
    if not all(any(n.startswith(k) and '.encoder.core.' in n for n in trainable) for k in ('actor','qvalue')):
        raise ValueError(('389_TRAINABILITY',trainable))
    market=market_for(a.parent,spec,model.recipe);risk=read(a.parent/'RISK.json')
    ids=manifest['seeds'][str(years)] if years else [];banks=[];bank_ids=[]
    for s in ids:
        root=a.banks/str(s);rec=read(root/'DONE.json');verify_files(root,rec['files'])
        if rec['source']!=manifest['source'] or rec['scenario']!=manifest['receipts'][str(s)]:raise ValueError('389_BANK_ID')
        x,v=(np.load(root/n,mmap_mode='r') for n in ('X.npy','META.npy'))
        if len(x)!=len(v)+1 or v[:,3].sum()!=1 or not v[-1,3]:raise ValueError('389_BANK_TERMINAL')
        banks.append((x,v));bank_ids.append(file_sha(root/'DONE.json'))
    d=dict(spec['design'],head_only=False,max_epochs=80,matched_budget=True,initialize_epoch_zero=True)
    binding=dict(source=manifest['source'],recipe=file_sha(a.bundle/'RECIPE.json'),seed=seed,years=years,
        banks=bank_ids,trainable=trainable,design=d,code=file_sha(__file__),risk=file_sha(a.parent/'RISK.json'))
    key=identity(binding)
    out=a.out/str(years)/str(seed);out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'SPEC.json').exists() and read(out/'SPEC.json')!=binding:raise ValueError('389_RESUME_ID')
        if (out/'RESULT.json').exists():return
        write(out/'SPEC.json',binding)
        splits=model.recipe['split']['periods'];assert set(splits)=={'TRAIN','VALIDATION'}
        ix=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
        probe=market.inputs.windows[ix[np.linspace(0,len(ix)-1,32,dtype=int)]].copy()
        def assess(m):
            metrics=assess_full(m,market,risk,d)
            if not (out/'EPOCH_ZERO.json').exists():
                for split,n,pct in [('TRAIN',745,16.05822355),('VALIDATION',207,18.36994575)]:
                    if metrics[split]['trades']!=n or abs(metrics[split]['return_pct']-pct)>1e-5:raise ValueError(('389_BASELINE',split,metrics[split]))
            return metrics
        state=fit_epochs(model,lambda:environment(market,risk,6,splits['TRAIN'],model.effective['gamma']),
            assess,d,seed,out,key,probe,replay_factory=lambda real:PairedReplay(real,banks,seed+200,d['batch_size']))
        write(out/'RESULT.json',dict(identity=key,epochs=state['epoch'],updates=state['updates'],
              restored=state['best_bundle'],metrics=assess(model),promoted=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack','train'])
    for k in ('parent','bundle','generated','banks','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--years',type=int,choices=[0,4,8,16],default=0)
    a=p.parse_args();(pack if a.mode=='pack' else train)(a)
