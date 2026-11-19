import numpy as np

def windows(n,pivots,left=.25,right=.5):
    """Boolean buy/sell zones and a separate known-support mask."""
    if not all(np.isfinite(f) and 0<=f<=1 for f in (left,right)):
        raise ValueError('fractions')
    out=np.zeros((n,2),bool);known=np.zeros(n,bool)
    stamps=[int(i) for i,_,_ in pivots]
    if any(not 0<=i<n for i in stamps) or any(a>=b for a,b in zip(stamps,stamps[1:])):
        raise ValueError('unordered pivots')
    kinds=[s for _,s,_ in pivots]
    if any(s not in (-1,1) for s in kinds) or any(a==b for a,b in zip(kinds,kinds[1:])):
        raise ValueError('pivot directions')
    if len(stamps)>=3:
        known[stamps[1]:stamps[-2]+1]=True
    for j in range(1,len(stamps)-1):
        t=stamps[j]
        lo=t-int(np.floor(left*(t-stamps[j-1])))
        hi=t+int(np.floor(right*(stamps[j+1]-t)))
        out[lo:hi+1,int(kinds[j]==1)]=True
    return out,known
