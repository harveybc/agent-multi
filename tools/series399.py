"""Historical 61-input SAC extension. Separate directories, accounts, replay and RNG per arm."""
import argparse
import copy
import fcntl
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools import pilot391 as p
from tools.account395 import take
from tools.account397 import assemble
from tools.loss_cache397 import Bank, fingerprint, gate
from app.frozen_signal_masks import prepared_signals
from app.three_signal_learning import evaluate


def run(root, years, seed, out):
    parent=Path('/var/tmp/sac70-warm-v3-20261108')
    bundle=root/'model/61/bundle'
    recorded=p.read(root/'out/CONTRACT.json')
    proof=p.read(root/'account397-61/RESULT.json')
    prior=root/'cache-d'
    assert p.read(prior/'EPOCH_PARITY.json')['pass_gate']
    for path,key in [(bundle/'model.pt','model'),(parent/'SPEC.json','spec'),(parent/'RISK.json','risk'),
                     (root/'history397-61-r2/RESULT.json','data'),(root/'account397-61/RESULT.json','proof')]:
        if p.file_sha(path)!=recorded[key]: raise ValueError(key)
    spec=p.read(parent/'SPEC.json')
    p.verify_files(parent,{k:v for k,v in spec['files'].items() if k.startswith(('data/','engine/')) or k=='SUPPORT_RECIPE.json'})
    d=dict(spec['design'],matched_budget=False,head_only=False,allow_ineligible_epoch_zero=True,
           initialize_epoch_zero=True)
    p.configure(seed)
    torch.backends.cudnn.allow_tf32=False
    m,_=p.load_bundle(bundle,'cuda')
    train,val=assemble(parent,bundle,root/'history397-61-r2')
    if years==4:
        train=copy.copy(val)
        hi=pd.Timestamp('2024-01-01',tz='UTC')
        mask=val.bars.index+pd.Timedelta(hours=1)<hi
        train.bars=val.bars[mask]
        train.sigma=val.sigma[mask]
        train.sigma_rule=val.sigma_rule[mask]
        train.inputs=take(val.inputs,val.inputs.row_ts_ns<hi.value)
    tr=[f'{2024-years}-01-01','2024-01-01']
    vr=m.recipe['split']['periods']['VALIDATION']
    assert vr==['2024-01-01','2025-01-01']
    risk=p.read(parent/'RISK.json')
    def env(market,period):
        return p.environment(market,risk,d['horizon'],period,m.effective['gamma'])
    def make(): return env(train,tr)
    def assess(model):
        result={}
        for key,market,period in [('TRAIN',train,tr),('VALIDATION',val,vr)]:
            logits,_=p.emit(model,market.inputs)
            result[key]=evaluate(env(market,period),logits,prepared=prepared_signals(logits))
        return result
    start=time.monotonic()
    bank=Bank(m,train.inputs)
    delta=gate(bundle,train,bank,make)
    zero=assess(m)
    for s,ref in [('TRAIN',proof['control' if years==4 else 'eight']),
                  ('VALIDATION',p.read(root/'model/61/READY.json')['initial']['VALIDATION'])]:
        if zero[s]['trades']!=ref['trades'] or abs(zero[s]['return_pct']-ref['return_pct'])>1e-5:
            raise ValueError(('epoch-zero mismatch',s,zero[s]['return_pct']))
    inherit=(years,seed)==(8,7)
    contract=dict(model=recorded['model'],spec=recorded['spec'],risk=recorded['risk'],data=recorded['data'],
                  years=years,seed=seed,train=tr,validation=vr,design=d,frozen_sha=bank.before,
                  code=p.file_sha(__file__),cache_code=p.file_sha(Path(__file__).with_name('loss_cache397.py')),
                  origin='cache-d/epoch_0001' if inherit else 'common_frozen_start')
    dest=out/'CONTRACT.json'
    if dest.exists() and p.read(dest)!=contract: raise ValueError('contract changed')
    p.write(dest,contract)
    ident=p.identity(contract)
    if inherit:
        old=p.read(prior/'GATE.json')
        ident=p.identity(old)
        if not (old['model_sha256']==recorded['model'] and old['rows']==len(bank.ids)):
            raise ValueError('incompatible row replay')
        if not (out/'RESUME.json').is_file():
            for f in ['recovery','bundles']:
                shutil.copytree(prior/f,out/f,dirs_exist_ok=True)
            for f in ['EPOCH_ZERO.json','RESUME.json']:
                shutil.copy2(prior/f,out/f)
    p.write(out/'ADMISSION.json',dict(gpu=torch.cuda.get_device_name(),paired_max=delta,
            rows=len(bank.ids),seconds=time.monotonic()-start,epoch_zero={s:{k:z[k] for k in ('trades','return_pct')} for s,z in zero.items()}))
    np.random.seed(seed)
    torch.manual_seed(seed)
    m,_=p.load_bundle(bundle,'cuda')
    if seed!=7: torch.manual_seed(seed)
    refs=p.read(root/'out/HISTORY.json') if inherit else []
    def check(model,row):
        if fingerprint(model)!=bank.before: raise ValueError('changed frozen prefix')
        if inherit and row['epoch']<=len(refs):
            ref=refs[row['epoch']-1]
            keys=('decisions','steps','epoch_updates','updates','metrics','mean_update','last_update')
            bad=[k for k in keys if row[k]!=ref[k]]
            p.write(out/f"PARITY_{row['epoch']:04d}.json",dict(differences=bad,source='history397'))
            if bad: raise ValueError(('historical epoch diverged',row['epoch'],bad))
    ix=np.flatnonzero(train.inputs.valid)[::761]
    with bank.use(m):
        state=p.fit_epochs(m,lambda:bank.row_env(make()),assess,d,seed,out,ident,
                           train.inputs.windows[ix].copy(),epoch_hook=check)
    p.write(out/'RESULT.json',dict(epoch=state['epoch'],updates=state['updates'],training=p.read(out/'TRAINING.json'),
                                   promoted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--years',type=int,choices=[4,8],required=True)
    parser.add_argument('--seed',type=int,choices=[7,19],required=True)
    a=parser.parse_args()
    out=a.root/'runs'/f'{a.years}y-s{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (out/'RESULT.json').exists(): run(a.root,a.years,a.seed,out)
