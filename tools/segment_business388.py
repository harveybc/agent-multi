"""TRAIN price-path behavior; standalone GPU worker, with CPU test mode."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

NAMES=('sl_first','tp_first','ambiguous','timeout','duration','mae_pips',
       'close_gain_pips','close_helpful','sl_time','tp_time')

def path_metrics(v,side,sl,tp):
    # v: batch x horizon x OHLC. All calculations below are signed pips.
    o,h,l,c=((v[:,:,j]-v[:,:1,0])*10000*side for j in range(4))
    hi=torch.maximum(h,l);lo=torch.minimum(h,l)
    og=o>=tp;ob=o<=-sl
    s=(lo<=-sl)&~og;t=(hi>=tp)&~ob
    hit=s|t
    first=hit.int().argmax(1)
    H=o.shape[1];anyhit=hit.any(1)
    idx=torch.where(anyhit,first,H-1)
    i=torch.arange(len(v),device=v.device)
    both=anyhit&s[i,idx]&t[i,idx]
    bad=anyhit&s[i,idx]&~both;good=anyhit&t[i,idx]&~both
    payoff=torch.where(anyhit,torch.where(og[i,idx]|ob[i,idx],o[i,idx],
                       torch.where(good,torch.full_like(o[:,0],tp),torch.full_like(o[:,0],-sl))),c[:,-1])
    step=torch.arange(H,device=v.device)[None,:]
    # No claim about within-bar drawdown on the terminal fill bar.
    minima=torch.where((step<idx[:,None])|(~anyhit[:,None]),lo,0.).amin(1)
    at_risk=(step>0)&(step<=idx[:,None])&~og&~ob&~both[:,None]
    gain=o-payoff[:,None];count=at_risk.sum(1)
    nan=torch.full_like(payoff,float('nan'))
    avg=lambda x:torch.where(count>0,torch.where(at_risk,x,0.).sum(1)/count.clamp_min(1),nan)
    duration=(idx+1).double()
    return torch.stack([bad,good,both,~anyhit,duration,-minima,avg(gain),
                        avg((gain>0).double()),torch.where(bad,duration,nan),
                        torch.where(good,duration,nan)],1)

def stress(device):
    # First two bars identical; after divergence, next open can be acted upon.
    paths=np.array([[[0,2,-2,0],[0,2,-2,0],[0,2,-3,-2],[-2,-1,-12,-10]],
                    [[0,2,-2,0],[0,2,-2,0],[0,3,-2,2],[2,12,1,10]]],dtype=float)
    np.testing.assert_array_equal(paths[0,:2],paths[1,:2])
    def run(p,d):
        return path_metrics(torch.tensor(1+p/10000,device=device),d,9.5,9.5).cpu().numpy()
    for direction in (1,-1):
        p=paths*direction
        if direction<0:p=p[:,:,[0,2,1,3]]
        m=run(p,direction)
        np.testing.assert_array_equal(m[:,:2],[[1,0],[0,1]])
        assert m[0,6]>0 and m[1,6]<0
        # No bar-0 liquidation is counted. Verify the post-divergence tick separately.
        np.testing.assert_allclose(p[:,3,0]*direction-np.array([-9.5,9.5]),[7.5,-7.5])
    p=np.array([[[0,2,-2,0],[0,12,-12,0]],[[0,2,-2,0],[-15,12,-16,0]]],float)
    m=run(p,1);assert m[0,2]==1 and m[1,0]==1 and np.isnan(m[1,6])

def run(a):
    device=torch.device(a.device)
    stress(device)
    if a.test:return
    with np.load(a.data) as f:data=dict(f)
    results={};ticks=data['ts'];q=data['quarter'];group=data['group']
    for name in ('real','syn'):
        v=torch.tensor(data[name],device=device)
        for H,side,sl,tp in data['risk']:
            H=int(H);side=int(side);ix=np.arange(len(v)-H+1)
            ix=ix[(ticks[ix+H-1]-ticks[ix]==(H-1)*3600*10**9)&(q[ix+H-1]==q[ix])&(group[ix]>=0)]
            t=torch.tensor(ix,device=device)
            m=path_metrics(v[t[:,None]+torch.arange(H,device=device)],side,float(sl),float(tp)).cpu().numpy()
            key=f'{name}_{H}_{side}'
            results[key]=m;results[key+'_idx']=ix
    results.update(quarter=q,group=group,ts=ticks)
    np.savez_compressed(a.out,**results)
    print(json.dumps(dict(device=torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',
        sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),stress='passed')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--test',action='store_true');p.add_argument('--device',default='cuda')
    run(p.parse_args())
