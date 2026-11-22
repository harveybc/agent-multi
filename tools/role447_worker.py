"""Continual SELL-head fit: only current semester can enter the optimizer."""
import copy
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import role405_plateau as plateau
import role429_worker as w


def quota(weeks):
    return [np.flatnonzero(weeks==i) for i in np.unique(weeks)]


@torch.inference_mode()
def evaluate(net,block):
    x,y,groups = block
    net.eval()
    pred = w.heads(net(x))[-1].cpu().numpy()
    vals = [y[i[np.argsort(-pred[i],axis=0,kind='stable')[:max(1,len(i)//10)]],np.arange(4)]
            for i in groups]
    return float(np.concatenate(vals).mean(dtype=np.float64))


def monitor(valid,old,mixed):
    return float((valid+np.mean(old))/2) if mixed and old else valid


def train(path,sha):
    with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==sha
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    with np.load(path) as d:
        sem,seed=d['sem'],int(d['seed'])
        x,y=torch.from_numpy(d['x']).cuda(),torch.from_numpy(d['y']).cuda()
        valid=torch.from_numpy(d['v']).cuda(),d['vy'],quota(d['vw'])
        raw,week=d['raw'],d['week']
    assert np.array_equal(np.unique(sem),np.arange(sem.max()+1))
    blocks=[(x[sem==s],raw[sem==s],quota(week[sem==s])) for s in range(sem.max()+1)]
    torch.manual_seed(seed)
    net=w.model().cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.001)
    initial=(w.snapshot(net),plateau.snapshot(opt))
    for s,block in enumerate(blocks):
        ids=np.flatnonzero(sem==s)
        # The only tensors accessible to the minibatch sampler are this semester's.
        tx,ty=block[0],y[ids]
        for arm in (('common',) if s==0 else ('validation','retention')):
            dst=path.with_suffix(f'.s{s}.{arm}.pt')
            if dst.exists():
                assert torch.load(dst,map_location='cpu',weights_only=False)['sha']==sha
                continue
            if s==0:
                weights,moments=initial
            else:
                parent=path.with_suffix(f'.s{s-1}.{"common" if s==1 else arm}.pt')
                saved=torch.load(parent,map_location='cpu',weights_only=False)
                assert saved['sha']==sha
                weights,moments=saved['weights'],saved['optimizer']
            net.load_state_dict(weights)
            opt.load_state_dict(copy.deepcopy(moments))
            assert all(torch.equal(weights[k],v.cpu()) for k,v in net.state_dict().items())
            def read():
                v=evaluate(net,valid)
                old=[evaluate(net,b) for b in blocks[:s]]
                return v,old,monitor(v,old,arm=='retention')
            v,old,best=read()
            assert np.isfinite(best)
            champ=w.snapshot(net); mom=plateau.snapshot(opt)
            best_epoch=bad=miss=0
            rng=np.random.default_rng([447,seed,s])
            hist=[(0,v,old,best,0,opt.param_groups[0]['lr'],False)]
            for epoch in range(1,2001):
                net.train()
                order=rng.permutation(len(ids))
                assert np.all(sem[ids[order]]==s)
                for i in range(0,len(order),512):
                    k=order[i:i+512]
                    opt.zero_grad(set_to_none=True)
                    w.severity(net(tx[k]),ty[k]).backward()
                    opt.step()
                v,old,metric=read()
                assert np.isfinite([v,*old,metric]).all()
                improved=metric>best+1e-8
                if improved:
                    best,best_epoch,bad=metric,epoch,0
                    champ=w.snapshot(net);mom=plateau.snapshot(opt)
                elif epoch>=100:
                    bad+=1
                miss,lr=plateau.observe(epoch,improved,miss,opt.param_groups[0]['lr'],bad>=80)
                if lr is not None: plateau.restore(net,opt,champ,mom,lr)
                hist.append((epoch,v,old,metric,bad,opt.param_groups[0]['lr'],lr is not None))
                if epoch==1 or epoch%50==0:
                    print(path.stem,s,arm,epoch,'best',best_epoch,'patience',bad,'lr',hist[-1][5],flush=True)
                if bad>=80:break
            net.load_state_dict(champ)
            assert abs(read()[-1]-best)<1e-8
            w.save(dict(sha=sha,stage=s,arm=arm,weights=champ,optimizer=mom,history=hist,
                        best_epoch=best_epoch,rows=len(ids),epochs=epoch,
                        prior_rows_optimized=0,exposures=epoch*len(ids),
                        stop='patience' if bad>=80 else 'cap'),dst)


if __name__=='__main__':train(Path(sys.argv[1]),sys.argv[2])
