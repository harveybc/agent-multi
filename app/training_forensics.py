"""Stateful optimizer replay and functional diagnostics for frozen critic heads."""
import hashlib
import random

import numpy as np
import torch


def tree_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous().numpy();visit(a)
        elif isinstance(x,np.ndarray):
            h.update(str((x.dtype.str,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=repr):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update(repr((type(x).__name__,x)).encode())
    visit(value);return h.hexdigest()


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy=np.random.get_state(),python=random.getstate())


def restore_rng(s):
    torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])
    np.random.set_state(s['numpy']);random.setstate(s['python'])


def mse_components(q,y):
    error=q-y.unsqueeze(0)
    level=error.mean(-1,keepdim=True)
    return level.square().mean(),(error-level).square().mean()


def gradient_geometry(model,z,y):
    params=list(model.parameters());q=model(z).mean(2)
    level,gap=mse_components(q,y)
    a=torch.cat([v.reshape(-1) for v in torch.autograd.grad(level,params,retain_graph=True)])
    b=torch.cat([v.reshape(-1) for v in torch.autograd.grad(gap,params)])
    na=a.norm();nb=b.norm()
    return dict(level_mse=float(level.detach()),gap_mse=float(gap.detach()),
        level_gradient_norm=float(na),gap_gradient_norm=float(nb),
        gradient_cosine=float(a.dot(b)/(na*nb).clamp_min(1e-30)))


def sinusoid_lags(n,dimension=208):
    if n<1 or dimension<1:raise ValueError('FORENSIC_SIGNAL_SHAPE')
    t=np.arange(n)[:,None]-np.arange(dimension)[None,:]
    return (np.sin(2*np.pi*t/13)+.5*np.sin(2*np.pi*t/37+.7)+.2*np.cos(2*np.pi*t/71)).astype(np.float32)


def balance_teacher(teacher,z,fit):
    """Three exact linear envelopes of one hidden activation, thresholds FIT-only."""
    with torch.no_grad():
        head=teacher.twins[0][0]
        s=head[1](head[0](z[0,fit]))[:,0]
        low,high=torch.quantile(s,torch.tensor([1/3,2/3],device=s.device))
        if not torch.isfinite(s).all() or high-low<=1e-6:raise ValueError('FORENSIC_TEACHER_DEGENERATE')
        scale=1/(high-low)
        for twin in teacher.twins:
            twin[0].load_state_dict(head.state_dict())
            last=twin[0][2];last.weight.zero_();last.bias.zero_()
            last.weight[0,0]=-scale;last.weight[2,0]=scale
            last.bias[0]=low*scale;last.bias[2]=-high*scale
        counts=teacher(z[:,fit]).mean(2)[0].argmax(-1).bincount(minlength=3)
        if (counts<float(len(fit))*.2).any():raise ValueError('FORENSIC_TEACHER_ACTION_COVERAGE')
    return dict(low=float(low),high=float(high),scale=float(scale),fit_counts=counts.cpu().tolist())
