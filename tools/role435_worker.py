"""Within-week fixed ridge probes; no held labels sent to this worker."""
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch


def run(path,sha):
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha
    torch.use_deterministic_algorithms(True)
    with np.load(path) as z:
        tensor=lambda k:torch.as_tensor(z[k],device='cuda',dtype=torch.float64)
        x,y,u=tensor('x'),tensor('y'),tensor('u')
        wk=z['week']
    mu=x.mean(0);sd=x.std(0,unbiased=False).clamp_min(1e-6)
    x=(x-mu)/sd;u=(u-mu)/sd
    groups=[np.flatnonzero(wk==q) for q in np.unique(wk)]
    def demean(v):
        a=v.clone()
        for i in groups:a[i]-=v[i].mean(0)
        return a
    rng=np.random.default_rng(435)
    shift=np.arange(len(x))
    for i in groups:
        if len(i)>1:shift[i]=np.roll(i,rng.integers(1,len(i)))
    assert (wk==wk[shift]).all()
    gen=torch.Generator(device='cuda').manual_seed(435)
    proj=torch.randn(x.shape[1],256,generator=gen,device='cuda',dtype=x.dtype)/x.shape[1]**.5
    phase=2*np.pi*torch.rand(256,generator=gen,device='cuda',dtype=x.dtype)
    dx=demean(x);col=dx.var(0).argmax();plant=dx[:,col:col+1]
    out={}
    for name,f,e in [('linear',x,u),('nonlinear',torch.cat((x,torch.cos(x@proj+phase)),1),
                                     torch.cat((u,torch.cos(u@proj+phase)),1))]:
        df=demean(f)
        target=torch.cat((demean(y),demean(y[shift]),plant),1)
        coef=torch.linalg.solve(df.T@df+.01*len(x)*torch.eye(df.shape[1],device='cuda',dtype=f.dtype),df.T@target)
        corr=torch.corrcoef(torch.stack(((df@coef)[:,-1],plant[:,0])))[0,1].item()
        assert corr>.95
        pred=((e-f.mean(0))@coef).cpu().numpy()
        for j,k in enumerate((name,name+'_shift')):out[k]=pred[:,4*j:4*j+4]
        out[name+'_plant_corr']=corr
    file=path.with_suffix('.scores.npz')
    tmp=file.with_suffix('.tmp.npz')
    np.savez(tmp,**out,sha=sha);tmp.replace(file)


if __name__=='__main__':run(Path(sys.argv[1]),sys.argv[2])
