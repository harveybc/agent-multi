"""Market-only close/hold fields, queried externally at native barrier geometry."""
import numpy as np
import torch
from torch import nn

NODES={'scalar':1,'mean81':81,'progress9':9,'surface81':81}
EPS=1e-8


def geometry(direction,liquidation,sl,tp,sigma):
    if (direction not in (-1,1) or not np.isfinite([liquidation,sl,tp,sigma]).all()
            or sigma<=0 or direction*(tp-sl)<=0):raise ValueError('SURFACE_GEOMETRY')
    return np.array([direction*(liquidation-sl)/sigma,direction*(tp-liquidation)/sigma])


def linear_weights(values,axis):
    x=np.asarray(values,dtype=float);a=np.asarray(axis,dtype=float)
    if (x.ndim!=1 or a.ndim!=1 or len(a)<2 or not np.isfinite(x).all()
            or not np.isfinite(a).all() or not (np.diff(a)>0).all()):raise ValueError('SURFACE_AXIS')
    low=np.clip(np.searchsorted(a,x,side='right')-1,0,len(a)-2)
    fraction=(x-a[low])/(a[low+1]-a[low]);w=np.zeros((len(x),len(a)))
    w[np.arange(len(x)),low]=1-fraction;w[np.arange(len(x)),low+1]=fraction
    return w


def fit_axes(coords,directions,fit):
    x=np.asarray(coords,dtype=float);d=np.asarray(directions);f=np.asarray(fit)
    if x.shape!=(len(d),2) or f.shape!=d.shape or f.dtype!=bool:raise ValueError('SURFACE_FIT_ROWS')
    axes={}
    for value,name in enumerate(('long','short')):
        selected=x[f & (d==value)]
        if not len(selected) or not np.isfinite(selected).all():raise ValueError('SURFACE_FIT_SUPPORT')
        lo=selected.min(0);hi=selected.max(0)
        if not (hi>lo).all():raise ValueError('SURFACE_CONSTANT_AXIS')
        axes[name]=[np.linspace(a,b,9).tolist() for a,b in zip(lo,hi)]
    return axes


def query_weights(coords,directions,axes,mode):
    x=np.asarray(coords,dtype=float);d=np.asarray(directions)
    if (mode not in NODES or x.ndim!=2 or x.shape!=(len(d),2)
            or not np.isfinite(x).all() or not np.isin(d,[0,1]).all()
            or not (x.sum(1)>0).all()):raise ValueError('SURFACE_QUERY')
    if mode in ('scalar','mean81'):
        return np.full((len(x),NODES[mode]),1/NODES[mode]),np.zeros(len(x),dtype=bool)
    result=np.zeros((len(x),NODES[mode]));outside=np.zeros(len(x),dtype=bool)
    for value,name in enumerate(('long','short')):
        ix=np.flatnonzero(d==value)
        if mode=='progress9':
            z=x[ix,0]/x[ix].sum(1);result[ix]=linear_weights(z,np.linspace(0,1,9))
            outside[ix]=(z<0)|(z>1)
        else:
            a,b=[np.asarray(v) for v in axes[name]]
            result[ix]=(linear_weights(x[ix,0],a)[:,:,None]*linear_weights(x[ix,1],b)[:,None,:]).reshape(len(ix),81)
            outside[ix]=(x[ix,0]<a[0])|(x[ix,0]>a[-1])|(x[ix,1]<b[0])|(x[ix,1]>b[-1])
    return result,outside


class GapSurfaceHeads(nn.Module):
    def __init__(self,features,nodes,seed,device='cpu'):
        super().__init__()
        if nodes not in (1,9,81):raise ValueError('SURFACE_NODES')
        self.nodes=nodes;device=torch.device(device)
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(seed)
            self.twins=nn.ModuleList([nn.Sequential(nn.Linear(features,8,device=device),nn.SiLU(),
                nn.Linear(8,2*nodes,device=device)) for _ in range(2)])

    def forward(self,embeddings):
        if embeddings.ndim!=3 or embeddings.shape[0]!=2:raise ValueError('SURFACE_EMBEDDING')
        return torch.stack([head(embeddings[t]).reshape(-1,2,self.nodes) for t,head in enumerate(self.twins)])


def evaluate_surface(outputs,directions,weights):
    if (outputs.ndim!=4 or outputs.shape[0]!=2 or outputs.shape[2]!=2
            or directions.shape!=(outputs.shape[1],) or directions.dtype!=torch.long
            or weights.shape!=(outputs.shape[1],outputs.shape[3])
            or not ((directions==0)|(directions==1)).all() or not torch.isfinite(weights).all()):
        raise ValueError('SURFACE_CONTROLLER')
    selected=outputs.gather(2,directions[None,:,None,None].expand(2,-1,1,outputs.shape[3])).squeeze(2)
    return (selected*weights[None]).sum(-1)


def oracle_fit(weights,gaps,device='cpu'):
    w=np.asarray(weights,dtype=float);y=np.asarray(gaps,dtype=float)
    if w.ndim!=2 or y.shape!=(len(w),) or not np.isfinite(w).all() or not np.isfinite(y).all():
        raise ValueError('SURFACE_ORACLE_INPUT')
    nodes,_,rank,s=np.linalg.lstsq(w,y,rcond=None);pred=w@nodes
    decision=pred>EPS;regret=np.maximum(y,0)-decision*y
    with torch.no_grad():
        field=torch.tensor(nodes,dtype=torch.float32,device=device)[None,None,None,:].expand(2,len(w),2,-1)
        decoded=evaluate_surface(field,torch.zeros(len(w),dtype=torch.long,device=device),
            torch.tensor(w,dtype=torch.float32,device=device))[0].cpu().numpy().astype(float)
    runtime_regret=np.maximum(y,0)-(decoded>EPS)*y
    nonzero=s[s>np.finfo(float).eps*max(w.shape)*s[0]] if len(s) else []
    return dict(rank=int(rank),rows=len(w),max_error=float(np.max(np.abs(pred-y))),
        max_node_magnitude=float(np.max(np.abs(nodes))),
        nonzero_condition=float(nonzero[0]/nonzero[-1]) if len(nonzero) else None,
        wrong_actions=int((regret>EPS).sum()),mean_regret=float(regret.mean()),
        predictions=pred.tolist(),float32_predictions=decoded.tolist(),
        float32_max_error=float(np.max(np.abs(decoded-y))),
        float32_wrong_actions=int((runtime_regret>EPS).sum()),future_label_oracle_only=True)


def decision_metrics(prediction,target,local_delta,effective,banks):
    p=np.asarray(prediction,float);y=np.asarray(target,float);local=np.asarray(local_delta,float);effective=np.asarray(effective,bool)
    if (not len(y) or p.shape!=y.shape or local.shape!=y.shape or effective.shape!=y.shape
            or not np.isfinite([p,y,local]).all()):raise ValueError('SURFACE_METRICS')
    close=p>EPS;regret=np.maximum(y,0)-close*y;informative=np.abs(y)>EPS;actual=close & effective
    return dict(n=len(y),gap_rmse=float(np.sqrt(np.mean((p-y)**2))),
        sign_accuracy=float(((close==(y>EPS))[informative]).mean()) if informative.any() else None,
        mean_regret=float(regret.mean()),bank_regret=float(np.mean([regret[b].mean() for b in banks])),
        requested_closes=int(close.sum()),effective_closes=int(actual.sum()),
        harmful_effective_closes=int((actual & (local < -EPS)).sum()),
        beneficial_effective_closes=int((actual & (local > EPS)).sum()),
        neutral_effective_closes=int((actual & (np.abs(local)<=EPS)).sum()),
        harmful_account_closes=int((close & (y < -EPS)).sum()),
        missed_account_beneficial_closes=int((~close & (y > EPS)).sum()),
        local_account_sign_disagreements=int(((np.sign(local)!=np.sign(y)) & (np.abs(local)>EPS) & informative).sum()))
