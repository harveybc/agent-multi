"""Pure dynamic allocation and finite-panel Pareto helpers; no market access."""
from bisect import bisect_right
import math

import numpy as np

HORIZONS=(3,6,12,24)
GENES=('weight_3','weight_6','weight_12','weight_24','trend_weight',
       'uncertainty_penalty','time_weight','entry_threshold','size_multiplier',
       'close_margin','close_persistence')


def bounds():
    return [(.25,2.)]*4+[(-.5,.5),(0.,2.),(0.,1.),(0.,1.),(.25,1.),(0.,.2),(1.,3.)]


def decode(values):
    if len(values)!=11 or not all(math.isfinite(v) for v in values):
        raise ValueError('ARBITER_GENOME')
    out=[round(min(hi,max(lo,float(v))),3) for v,(lo,hi) in zip(values,bounds())]
    out[-1]=int(round(out[-1]))
    return tuple(out)


def choose(bids,genes,past_efficiency):
    if not math.isfinite(past_efficiency) or abs(past_efficiency)>1+1e-12:
        raise ValueError('PAST_EFFICIENCY')
    best=None; best_score=0.; tie=False
    for b in bids:
        h=b['horizon'];d=b['direction']
        if (h not in HORIZONS or d not in (-1,1)
                or not all(math.isfinite(b[k]) for k in ('gain_risk','uncertainty_risk','duration'))
                or min(b['uncertainty_risk'],b['duration'])<0):
            raise ValueError('BID_CONTRACT')
        score=(genes[HORIZONS.index(h)]*b['gain_risk']-genes[5]*b['uncertainty_risk']
               +genes[4]*d*past_efficiency*h/24-genes[6]*b['duration']/24-genes[7])
        if score>best_score+1e-12:
            best=dict(b,score=score);best_score=score;tie=False
        elif best is not None and abs(score-best_score)<=1e-12:
            tie=True
    return None if tie else best


def objectives(equity,initial):
    x=np.asarray(equity,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not math.isfinite(initial) or initial<=0:
        raise ValueError('ACCOUNT_EQUITY')
    x=np.r_[initial,x]
    high=np.maximum.accumulate(x)
    return float(100*(x[-1]/initial-1)),float(100*np.max((high-x)/high))


def frontier(vectors):
    if any(len(v)!=2 or not np.isfinite(v).all() for v in vectors.values()):
        raise ValueError('PARETO_VECTORS')
    # Descending objective1, then objective2; retain equal nondominated vectors.
    ordered=sorted(vectors,key=lambda k:(-vectors[k][0],-vectors[k][1],k))
    result=[]; best_y=-math.inf; previous=None
    for key in ordered:
        x,y=vectors[key]
        if y>best_y or (x,y)==previous:
            result.append(key);best_y=y;previous=(x,y)
    return result


def interval_best(intervals):
    rows=sorted(intervals,key=lambda r:r[1]);ends=[r[1] for r in rows]
    dp=[0.];chosen=[[]]
    for i,(start,end,gain) in enumerate(rows):
        if end<=start or not all(math.isfinite(x) for x in (start,end,gain)):
            raise ValueError('INTERVAL')
        prev=bisect_right(ends,start,hi=i)
        value=dp[prev]+gain
        if value>dp[-1]:
            dp.append(value);chosen.append(chosen[prev]+[(start,end,gain)])
        else:
            dp.append(dp[-1]);chosen.append(chosen[-1])
    return dp[-1],chosen[-1]
