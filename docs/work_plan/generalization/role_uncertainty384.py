"""Cold CPU replay of each deployed head's posterior on diagnostic 2023 rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_role384 import Head,weights

def score(data,folder):
    with np.load(data) as f:
        ix=f['fold']==3;x,q,y=[f[k][ix] for k in ('x','q','y')]
    with np.load(folder/'apply.npz') as f:
        x=(x-f['xmu'])/f['xsd'];T=float(f['T'])
        w=weights(q,f['bounds']) if folder.name.startswith('close') else np.ones((len(x),1))
    head=Head(x.shape[1],w.shape[1])
    head.load_state_dict(torch.load(folder/'best.pt',weights_only=True,map_location='cpu'))
    head.eval()
    with torch.no_grad():
        m,s,_=head(torch.from_numpy(x).float(),torch.from_numpy(w).float())
    z,v=np.polynomial.hermite.hermgauss(16);v/=np.sqrt(np.pi)
    p=1/(1+np.exp(-((m[:,:1].numpy()+s[:,:1].numpy()*z*np.sqrt(2))/T).clip(-50,50)))
    mean=p@v;sd=np.sqrt(((p-mean[:,None])**2)@v)
    bins=np.minimum((mean*10).astype(int),9);error=(mean-y)**2
    rec=json.loads((folder/'result.json').read_text())
    if abs(float(error.mean())-rec['calibrated_2023']['brier'])>1e-5:raise ValueError('cold replay')
    # Equal-count descriptive strata. Do not fit an abstention threshold here.
    low,_,__,high=np.array_split(np.argsort(sd,kind='stable'),4)
    return dict(brier=float(error.mean()),sd_mean=float(sd.mean()),
        ece=float(sum(abs(np.sum((mean-y)[bins==i])) for i in np.unique(bins))/len(y)),
        low_sd_brier=float(error[low].mean()),high_sd_brier=float(error[high].mean()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('results',type=Path)
    a=p.parse_args()
    print(json.dumps({d.stem:score(d,a.results/d.stem) for d in sorted(a.data.glob('*.npz'))},indent=2))
