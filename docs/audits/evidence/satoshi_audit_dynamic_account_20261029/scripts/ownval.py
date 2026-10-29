"""Own first-passage valuation and arbiter (no import of value_paths/protection_exit/choose)."""
import math
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import numpy as np
H=3600*10**9
HZ=(3,6,12,24)
def levels(d,e,tp_ref,mae_ref,g,tick=1e-5):
    D=lambda x: Decimal(str(x))
    ds=D(g['a_sl'])*D(mae_ref)+D(g['b_sl'])*D(.001)+D(g['buffer_pips'])*D(.0001)
    dt=D(g['a_tp'])*D(tp_ref)-D(g['b_tp'])*D(.001)
    rnd=ROUND_FLOOR if d==1 else ROUND_CEILING
    r=lambda x: float((x/D(tick)).to_integral_value(rounding=rnd)*D(tick))
    return r(D(e)-D(d)*ds), r(D(e)+D(d)*dt)
def first_passage(d,sl,tp,q,through):
    """q: [bars,4] exit-side OHLC. returns (index, price, reason, at_open). Native ordering & native TP expression."""
    for j in range(len(q)):
        o,h,l,c=q[j]
        if d*(o-sl)<=0: return j,o,'SL',True
        if d*(o-tp)>=0: return j,o,'TP',True
        adv=l if d==1 else h
        if d*(adv-sl)<=0: return j,sl,'SL',False
        if (h>=tp+through) if d==1 else (l<=tp-through): return j,tp,'TP',False
    return len(q)-1,q[-1,3],'DEADLINE',False
def values(d,E,sl,tp,units,paths,spreads,fo,fe,prof):
    slm,sls,com=prof['slip_m'],prof['slip_s'],prof['comm_rt']
    Hn=paths.shape[1]; out=np.empty((len(paths),Hn+1)); meta=[]
    for i in range(len(paths)):
        q=paths[i]-d*spreads[i][:,None]/2
        at,X,reason,at_open=first_passage(d,sl,tp,q,prof['through'])
        fund=fo[at] if at_open else fe[at]
        xs=sls if reason=='SL' else 0. if reason=='TP' else slm
        passive=units*(d*(X-E)-slm-xs-com-fund)
        out[i,:]=passive
        n=at if at_open else at+1
        for j in range(n): out[i,j]=units*(d*(q[j,0]-E)-slm-slm-com-fo[j])
        meta.append((at,reason,at_open))
    return out,meta
def advantage(vals,w,z,risk):
    delta=vals[:,[0]]-vals[:,1:]; mean=w@delta; sd=np.sqrt(w@((delta-mean)**2))
    return float(np.min(mean-z*sd))/risk
def choose(bids,g,eff):
    best=None;bs=0.;tie=False
    for b in bids:
        s=(g[HZ.index(b['h'])]*b['gain_risk']-g[5]*b['unc']+g[4]*b['d']*eff*b['h']/24-g[6]*b['dur']/24-g[7])
        if s>bs+1e-12: best=dict(b,score=s);bs=s;tie=False
        elif best is not None and abs(s-bs)<=1e-12: tie=True
    return None if tie else best
