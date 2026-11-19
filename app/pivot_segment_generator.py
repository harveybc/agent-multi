"""Conditional surrogate paths in log-return space; the controller never sees the controls."""
import numpy as np


def controls(y):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('segment')
    t = np.arange(len(y)) - (len(y)-1)/2
    return np.array([y.mean(), y.std(), t @ y / (t @ t) if len(y)>1 else 0.])


def draw(y, noise, target):
    """Rotate residuals while preserving the constant/linear components and scale."""
    y = np.asarray(y, dtype=float)
    mu, sd, accel = target
    if not np.isfinite(target).all() or sd < 0: raise ValueError('target')
    t = np.arange(len(y)) - (len(y)-1)/2
    slope0, _, accel0 = controls(y)
    energy = sd**2 - accel**2 * np.mean(t*t)
    if energy < -1e-16 or (len(y)==1 and accel!=0): raise ValueError('infeasible curvature')
    base = mu + accel*t
    r = y-slope0-accel0*t
    if energy <= 1e-24: return base
    if len(y)<3 or np.linalg.norm(r)<1e-15: raise ValueError('infeasible residual')
    for v in noise:
        v = v-v.mean()
        v = v-t*((t@v)/(t@t))
        n = np.linalg.norm(v)
        if n>1e-10:
            u=v/n
            r=r-2*(r@u)*u
    return base+r*np.sqrt(energy/np.mean(r*r))


def chunks(index):
    gaps = np.flatnonzero(np.diff(index.as_unit('ns').asi8)!=3600*10**9)+1
    return zip(np.r_[0,gaps],np.r_[gaps,len(index)])


def pivots_by_chunk(bars, zz, params):
    result=[]
    for lo,hi in chunks(bars.index):
        p=zz.hindsight_pivots(bars.HIGH.values[lo:hi],bars.LOW.values[lo:hi],params)
        result.append([(lo+i,k) for i,k,_ in p])
    return result


def match(reference, candidate, tolerance):
    count=0
    for ref,syn in zip(reference,candidate,strict=True):
        for kind in (-1,1):
            a=[i for i,k in ref if k==kind]; b=[i for i,k in syn if k==kind]
            j=0
            for i in a:
                while j<len(b) and b[j]<i-tolerance:j+=1
                if j<len(b) and b[j]<=i+tolerance:count+=1;j+=1
    return count


def generate(bars, crosses, reference, seed, warmup=256):
    """Rotate the joint interior innovations; keep pivots, closures and missingness."""
    import pandas as pd
    if not bars.index.equals(crosses.index): raise ValueError('unaligned crosses')
    prices=np.column_stack([bars.CLOSE,crosses])
    log=np.log(prices)
    out=log.copy()
    rng=np.random.default_rng(seed)
    covered=np.zeros(len(log),bool)
    records=[]
    for pivots in reference:
        for (a,_),(b,_) in zip(pivots,pivots[1:]):
            if a<warmup:continue
            covered[a+1:b+1]=True
            full=np.diff(log[a:b+1,0])
            target=controls(full)
            n=b-a-1
            if n>=3:
                # Fix both anchors and the last increment (hence original pivot OHLC).
                t=np.arange(n)-(n-1)/2
                def unit(v):
                    v=v-v.mean();v-=t*(t@v)/(t@t)
                    return v/np.linalg.norm(v)
                u=unit(rng.normal(size=n))
                if n==3:
                    reflect=[u] if rng.random()<.5 else []
                else:
                    v=unit(rng.normal(size=n));v-=u*(u@v);v/=np.linalg.norm(v)
                    angle=rng.uniform(.20,.48)
                    reflect=[u,u*np.cos(angle)+v*np.sin(angle)]
                for j in range(prices.shape[1]):
                    piece=log[a:b,j]
                    if not np.isfinite(piece).all():continue
                    r=np.diff(piece)
                    moved=draw(r,reflect,controls(r)) if r.std()>1e-14 else r
                    out[a+1:b-1,j]=piece[0]+np.cumsum(moved)[:-1]
            actual=controls(np.diff(out[a:b+1,0]))
            records.append([int(a),int(b),*target,*actual])
    mult=np.exp(out[:,0]-log[:,0])
    cl=bars.CLOSE.values*mult;prev=np.r_[1.,mult[:-1]]
    op=bars.OPEN.values*prev
    high=np.maximum(cl,op)*(bars.HIGH.values/np.maximum(bars.OPEN,bars.CLOSE))
    low=np.minimum(cl,op)*(bars.LOW.values/np.minimum(bars.OPEN,bars.CLOSE))
    unchanged=(mult==1)&(prev==1)
    high=np.where(unchanged,bars.HIGH,high)
    low=np.where(unchanged,bars.LOW,low)
    generated=pd.DataFrame(dict(OPEN=op,HIGH=high,LOW=low,CLOSE=cl),index=bars.index)
    fx=pd.DataFrame(prices[:,1:]*np.exp(out[:,1:]-log[:,1:]),columns=crosses.columns,index=bars.index)
    return generated,fx,covered,np.asarray(records)
