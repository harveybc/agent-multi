"""CPU comparators for 384: fit-only prevalence and median, pooled or by UTC hour."""
import argparse
import json
from pathlib import Path
import numpy as np

def compare(data):
    with np.load(data) as d:
        t,y,bp,split=[d[k] for k in ('t','y','bp','fold')]
    hour=(t//(3600*10**9))%24
    train=split==0;held=split==3
    prior=y[train].mean();median=np.median(bp[train])
    def measure(p,b):
        p=np.clip(p,1e-6,1-1e-6)
        return dict(brier=float(np.mean((p-y[held])**2)),
            log_loss=float(-np.mean(y[held]*np.log(p)+(1-y[held])*np.log1p(-p))),
            mae_bp=float(np.mean(abs(b-bp[held]))))
    probs=[];med=[]
    for h in range(24):
        mask=train & (hour==h)
        probs.append((y[mask].sum()+prior)/(mask.sum()+1))
        med.append(np.median(bp[mask]) if mask.any() else median)
    return dict(n_fit=int(train.sum()),n_scored=int(held.sum()),constant=measure(prior,median),
        clock=measure(np.array(probs)[hour[held]],np.array(med)[hour[held]]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);a=p.parse_args()
    print(json.dumps({f.stem:compare(f) for f in sorted(a.data.glob('*.npz'))},indent=2))
