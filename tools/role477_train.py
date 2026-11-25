"""Restartable fold-local seasonal regressors. Never load a pretrained encoder."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from agent_plugins.three_signal_sac import ThreeLogits


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(obj,path):
    tmp=path.with_suffix('.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)


def masks(t, ok, h, start, monitor, test, stop):
    maturity=t+h*3600*10**9
    return [(t>=lo)&(maturity<hi)&ok for lo,hi in
            ((start,monitor),(monitor,test),(test,stop))]


def train(root,bundle,task):
    ready=json.loads((root/'ready.json').read_text())
    assert all(sha(root/n)==ready[k] for k,n in [('batch','batch.npz'),('audit','audit.json')])
    audit=json.loads((root/'audit.json').read_text()); y=audit['year']
    bounds=[pd.Timestamp(audit[k]).value for k in ('start','monitor','eval_start')]
    stop=pd.Timestamp(f'{y+1}-01-01',tz='UTC').value
    dest=root/task;dest.mkdir(exist_ok=True)
    key=[ready,sha(Path(__file__)),sha(bundle/'RECIPE.json'),sha(bundle/'MODEL_CONFIG.json')]
    if (dest/'done.json').exists():
        done=json.loads((dest/'done.json').read_text())
        if done['key']!=key or done['predictions']!=sha(dest/'pred.npz'):
            raise ValueError('finished artifacts changed')
        return
    torch.manual_seed(123); rng=np.random.default_rng(20261122)
    with np.load(root/'batch.npz') as d:
        x=torch.from_numpy(d['x']).cuda()
        ts=d['ts']; targets=d['y_'+task]
        tr,va,te=masks(ts,d['ok_'+task],6 if task=='short' else 48,*bounds,stop)
    assert tr.any() and va.any() and te.any() and np.isfinite(targets[tr|va|te]).all()
    recipe=json.loads((bundle/'RECIPE.json').read_text())
    arch=json.loads((bundle/'MODEL_CONFIG.json').read_text())['architecture']
    enc=ThreeLogits(recipe,arch).encoder.cuda()
    assert enc.features_dim==208 and len(enc.branches)==22
    head=nn.Linear(208,targets.shape[1]).cuda()
    nn.init.normal_(head.weight,std=.05);nn.init.zeros_(head.bias)
    net=nn.ModuleList([enc,head])
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.004)
    ix=np.flatnonzero(tr)
    mu=targets[tr].mean(0,dtype=np.float64);sd=targets[tr].std(0,dtype=np.float64)
    assert (sd>1e-8).all()
    labels=torch.tensor((targets-mu)/sd,dtype=torch.float32,device='cuda')
    enc.eval()
    with torch.no_grad():
        scale=(sum(float(enc({'features':x[take]}).double().square().sum())
                   for take in np.array_split(ix,(len(ix)+255)//256))/len(ix))**.5
    def predict(indices):
        net.eval()
        with torch.no_grad():
            return torch.cat([head(enc({'features':x[j]})/scale).cpu()
                              for j in np.array_split(indices,(len(indices)+255)//256)]).numpy()*sd+mu
    ep=0;best=float('inf');wait=lrwait=0;rate=3e-4;bestep=0
    def restore(ck):
        net.load_state_dict(ck['weights']);opt.load_state_dict(ck['opt'])
    resume=dest/'resume.pt'; champ=dest/'champ.pt'
    if resume.exists():
        ck=torch.load(resume,weights_only=False)
        assert ck['key']==key
        restore(ck);ep,best,wait,lrwait,rate,bestep=ck['counters']
        rng.bit_generator.state=ck['rng'];torch.cuda.set_rng_state(ck['cuda'])
        # Best is in the same atomic transaction as last; no split-brain rollback.
        best_st=ck['best']
    else:
        best_st=None
    while ep<2000 and wait<80:
        ep+=1;net.train()
        perm=rng.permutation(ix)
        total=0.
        for j in range(0,len(ix),256):
            take=perm[j:j+256];opt.zero_grad(set_to_none=True)
            loss=(head(enc({'features':x[take]})/scale)-labels[take]).abs().mean()
            loss.backward();opt.step();total+=float(loss.detach())*len(take)
        mae=float(np.abs(predict(np.flatnonzero(va))-targets[va]).mean())
        if not np.isfinite(mae):raise ValueError('nonfinite monitor')
        rollback=False
        if mae<best-1e-8:
            best=mae;bestep=ep;wait=lrwait=0
            best_st=copy.deepcopy(dict(weights=net.state_dict(),opt=opt.state_dict()))
        else:
            wait+=int(ep>=100);lrwait+=int(ep>=20)
            if lrwait>=20 and rate>4.8e-7:
                rate=max(4.8e-7,rate/5);lrwait=0;restore(best_st)
                for g in opt.param_groups:g['lr']=rate
                rollback=True
        status=dict(epoch=ep,champion=bestep,monitor_mae=mae,train_scaled_l1=total/len(ix),
                    best=best,patience=wait,lr_wait=lrwait,lr=rate,rollback=rollback,
                    train_rows=len(ix),monitor_rows=int(va.sum()),device=torch.cuda.get_device_name())
        save(dict(weights=net.state_dict(),opt=opt.state_dict(),best=best_st,key=key,
                  counters=(ep,best,wait,lrwait,rate,bestep),rng=rng.bit_generator.state,
                  cuda=torch.cuda.get_rng_state()),resume)
        (dest/'status.json').write_text(json.dumps(status)+'\n')
        print(json.dumps(status),flush=True)
    restore(best_st)
    save(dict(weights=best_st['weights'],key=key,scale=scale,mean=mu.tolist(),std=sd.tolist(),
              epoch=bestep),champ)
    take=np.flatnonzero(ts>=bounds[1]);pred=predict(take)
    np.savez(dest/'pred.npz',ts=ts[take],pred=pred)
    np.testing.assert_array_equal(ts[take][te[take]],ts[te])
    result=dict(key=key,year=y,task=task,epoch=ep,best_epoch=bestep,
                holdout_rows=int(te.sum()),stop='patience' if wait>=80 else 'cap',
                mae=np.abs(targets[te]-pred[te[take]]).mean(0).tolist(),
                persistence_mae=np.abs(targets[te]).mean(0).tolist(),
                predictions=sha(dest/'pred.npz'),checkpoint=sha(champ))
    tmp=dest/'done.tmp';tmp.write_text(json.dumps(result)+'\n');tmp.replace(dest/'done.json')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path);p.add_argument('bundle',type=Path)
    a=p.parse_args()
    for task in ('short','long'):train(a.root,a.bundle,task)
