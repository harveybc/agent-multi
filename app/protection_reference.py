"""Independent synthetic-future reference; never a financial feature or policy."""
import math

import numpy as np


def protection_exit(direction, tp, sl, opens, highs, lows, closes, through):
    """Exit-side quotes, conservative OHLC order; deadline at last completed close."""
    o,h,l,c=(np.asarray(x,dtype=float) for x in (opens,highs,lows,closes))
    if (direction not in (-1,1) or not all(math.isfinite(v) for v in (tp,sl,through))
        or through<0 or direction*(tp-sl)<=0 or o.ndim!=1 or not len(o)
        or not (o.shape==h.shape==l.shape==c.shape)
        or not all(np.isfinite(x).all() for x in (o,h,l,c))
        or np.any(h<np.maximum(o,c)) or np.any(l>np.minimum(o,c))):
        raise ValueError('REFERENCE_QUOTE_PATH')
    for k in range(len(o)):
        if direction*(o[k]-sl)<=0:return k,float(o[k]),'SL'
        if direction*(o[k]-tp)>=0:return k,float(o[k]),'TP'
        adverse=l[k] if direction==1 else h[k]
        favorable=h[k] if direction==1 else l[k]
        if direction*(adverse-sl)<=0:return k,float(sl),'SL'
        if direction*(favorable-tp)>=through:return k,float(tp),'TP'
    return len(c)-1,float(c[-1]),'DEADLINE'


def entry_command(net_gains,losses_at_stop,minimum_gain_risk):
    """Diagnostic estimates for BUY and SELL; risk-qualified maximum, ties abstain."""
    g,r=np.asarray(net_gains,float),np.asarray(losses_at_stop,float)
    if (g.shape!=(2,) or r.shape!=(2,) or not np.isfinite(g).all()
        or not np.isfinite(r).all() or (r<=0).any()
        or not math.isfinite(minimum_gain_risk) or minimum_gain_risk<0):
        raise ValueError('REFERENCE_ENTRY_RISK')
    eligible=(g>0)&(g>=minimum_gain_risk*r)
    scores=np.where(eligible,g,-np.inf)
    best=np.max(scores)
    if not np.isfinite(best) or np.count_nonzero(scores==best)!=1:return 0
    return int(np.argmax(scores))+1


def prefer_close(close_net,hold_net,minimum_advantage):
    if not all(math.isfinite(x) for x in (close_net,hold_net,minimum_advantage)) or minimum_advantage<0:
        raise ValueError('REFERENCE_CLOSE_VALUES')
    return close_net-hold_net>minimum_advantage
