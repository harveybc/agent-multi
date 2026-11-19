"""Market-only linear heads and an external three-coordinate position lookup."""
from itertools import product
import numpy as np

CORNERS = np.array(list(product((0,1), repeat=3)), bool)

def coordinates(side, quote, sl, tp, entry, hours):
    risk = abs(entry-sl)
    if side not in (1,-1) or risk<=0 or side*(tp-sl)<=0 or hours<0:
        raise ValueError('invalid position for lookup')
    q = np.array([side*(quote-sl)/risk, side*(tp-quote)/risk, hours])
    if not np.isfinite(q).all():
        raise ValueError('nonfinite position')
    return q

def weights(q, bounds):
    span = bounds[1]-bounds[0]
    z = np.clip((q-bounds[0])/np.where(span>0,span,1),0,1)
    z = np.where(span>0,z,0.5)
    return np.prod(np.where(CORNERS,z[...,None,:],1-z[...,None,:]),axis=-1)

def ridge(x,y):
    mean=x.mean(0)
    xc=x-mean
    yc=y-y.mean()
    lam=.1*len(y)
    if len(x)<x.shape[1]:
        beta=xc.T @ np.linalg.solve(xc @ xc.T + lam*np.eye(len(x)),yc)
    else:
        beta=np.linalg.solve(xc.T @ xc + lam*np.eye(x.shape[1]),xc.T @ yc)
    return beta,y.mean()-mean @ beta

def fit(x,q,y):
    mu=x.mean(0)
    sd=x.std(0)
    sd=np.where(sd>1e-12,sd,1.)
    z=(x-mu)/sd
    bounds=np.quantile(q,[.1,.9],axis=0)
    phi=np.column_stack((np.ones(len(x)),z))
    wc=weights(q,bounds)-1/8
    design=np.column_stack((z,(wc[:,:,None]*phi[:,None,:]).reshape(len(x),-1)))
    b,i=ridge(z,y)
    bs,isurf=ridge(design,y)
    k=z.shape[1]
    corners=bs[k:].reshape(8,-1)
    # Centering over corners makes the scalar component the common prediction.
    table=np.r_[isurf,bs[:k]]+corners-corners.mean(0)
    return dict(mu=mu,sd=sd,bounds=bounds,scalar=np.r_[i,b],table=table)

def scores(model,x,q):
    z=(x-model['mu'])/model['sd']
    phi=np.column_stack((np.ones(len(x)),z))
    reports=phi @ model['table'].T
    return phi @ model['scalar'], (reports*weights(q,model['bounds'])).sum(1)
