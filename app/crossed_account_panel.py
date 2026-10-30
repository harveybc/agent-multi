"""Versioned synthetic panel. Future-aware reports are diagnostics, never features."""
from dataclasses import replace
import itertools

import numpy as np
import pandas as pd

from app.close_hold_controller import PathReport
from app.dynamic_account_panel import World, HOUR

PATTERNS=('trend','reversal','overlap')
ENVELOPES=('constant','increasing','decreasing')
QUALITIES=('ideal','correlated','delayed','omitted_reversal','missing')


def panel_specs(bank,quality='correlated',draws=10):
    if bank not in ('calibration','confirmation') or quality not in QUALITIES or not 1<=draws<=10:
        raise ValueError('CROSSED_PANEL_SPEC')
    start=50100 if bank=='calibration' else 60100
    return [dict(bank=bank,seed=start+i,pattern=p,volatility=v,spread_pips=s,mirror=m,quality=quality)
            for i,p,v,s,m in itertools.product(range(draws),PATTERNS,ENVELOPES,(.5,3.),(1,-1))]


def execution_specs(bank,quality,smoke=False):
    rows=panel_specs(bank,quality)
    if not smoke: return rows
    return [dict(s,seed=s['seed']+20000,bank='smoke_'+bank) for s in rows[:2]]


def market(spec):
    p=PATTERNS.index(spec['pattern']);v=ENVELOPES.index(spec['volatility'])
    rng=np.random.default_rng(np.random.SeedSequence([spec['seed'],p,7401]))
    t=np.arange(120,dtype=float);strength=rng.uniform(.000025,.00010)
    if p==0:
        drift=np.full(120,strength)
        signal=np.cumsum(drift)+rng.uniform(.0002,.0007)*np.sin(t/rng.uniform(3,9))
    elif p==1:
        cuts=sorted(rng.choice(np.arange(25,96),2,replace=False))
        drift=np.full(120,strength);drift[cuts[0]:cuts[1]]=-strength*rng.uniform(.6,1.5)
        signal=np.cumsum(drift)
    else:
        signal=sum(rng.uniform(.0003,.0015)*np.sin(2*np.pi*t/period+rng.uniform(0,2*np.pi))
                   for period in (12,36,96))
    innovations=rng.normal(0,.00018,120)
    width=rng.uniform(.00003,.00016,120)
    envelope=(np.ones(120),np.linspace(.3,1.8,120),np.linspace(1.8,.3,120))[v]
    close=1.1+signal+np.cumsum(innovations*envelope)
    opens=np.r_[close[0],close[:-1]]
    for k in sorted(rng.choice(np.arange(15,104),3,replace=False)):
        jump=rng.choice((-1,1))*rng.uniform(.0003,.0018)
        close[k:]+=jump;opens[k:]+=jump
    high=np.maximum(opens,close)+width*envelope
    low=np.minimum(opens,close)-width*envelope
    if spec['mirror']==-1:
        opens,high,low,close=2.2-opens,2.2-low,2.2-high,2.2-close
    elif spec['mirror']!=1: raise ValueError('MIRROR')
    return np.column_stack((opens,high,low,close))


class CrossedWorld(World):
    def __init__(self,spec,risk_config):
        if spec['quality'] not in QUALITIES or spec['spread_pips'] not in (.5,3.):
            raise ValueError('CROSSED_WORLD_SPEC')
        # Reuse native calendar, fixed risk references and funding, not old prices.
        super().__init__(spec,risk_config)
        from app.short_horizon.expert_guided.absolute import AbsoluteBarEnv
        ts=pd.date_range('2020-01-05T22:00Z',periods=120,freq='h')
        bars=pd.DataFrame(market(spec),columns=['OPEN','HIGH','LOW','CLOSE'],index=ts)
        s=spec['spread_pips']*.0001
        profile=replace(self.env.cfg.profile,weekly_open_spread=s,
                        spread_schedule=((0,10,s),(10,11,2*s),(11,24,s)))
        cfg=replace(self.env.cfg,profile=profile)
        self.env=AbsoluteBarEnv(bars,self.env.cal,np.full(120,.001),np.full(120,.002),
                               ts[0],ts[-1]+pd.Timedelta(hours=1),cfg,tick=.00001)
        self.end=int(np.searchsorted(self.env.v,self.env.ff[0]))
        rng=self.rng(0,13)
        self.latent=np.zeros(120)
        for i in range(1,120): self.latent[i]=.6*self.latent[i-1]+rng.normal(0,.00022)
        self.available=self.rng(0,17).random(120)>=.2

    def rng(self,b,stream):
        return np.random.default_rng(np.random.SeedSequence([
            self.spec['seed'],PATTERNS.index(self.spec['pattern']),
            ENVELOPES.index(self.spec['volatility']),b,stream,7409]))

    def report_available(self,b):
        return self.spec['quality']!='missing' or bool(self.available[b])

    def report(self,b):
        self._ensure_cache_context()
        if b in self.reports: return self.reports[b]
        if not self.report_available(b): raise ValueError('REPORT_NOT_AVAILABLE')
        e=self.env;q=self.spec['quality'];future=np.column_stack((e.o,e.h,e.l,e.c))[b+1:self.end]
        h=len(future);rng=self.rng(b,23);side=self.spec['mirror']
        if h<1: raise ValueError('NO_REPORT_SUPPORT')
        if q=='ideal': paths=future[None].copy()
        elif q in ('correlated','missing'):
            error=np.zeros((3,h));state=np.full(3,self.latent[b])
            for j in range(h):
                state=.6*state+rng.normal(0,.00025,3)
                error[:,j]=(self.latent[b]+state)*np.sqrt(min(j+1,24))*side
            paths=future[None]+error[:,:,None]
        elif q=='delayed':
            # Delayed forecast SHAPE is freshly issued; stale timestamps are never relabeled.
            history=np.column_stack((e.o,e.h,e.l,e.c))
            indices=np.maximum(0,np.arange(b+1,self.end)-3)
            paths=np.repeat(history[indices][None],3,axis=0)
            paths+=rng.normal(0,.00008,(3,h,1))*side
        else:
            prefix=e.c[max(0,b-8):b+1]
            drift=float(np.median(np.diff(prefix))) if len(prefix)>1 else 0.
            center=e.c[b]+np.arange(1,h+1)*drift
            center=np.repeat(center[None],3,axis=0)+rng.normal(0,.00004,(3,h))*side
            opens=np.column_stack((np.full(3,e.c[b]),center[:,:-1]))
            width=max(.00001,float(np.std(np.diff(prefix)))*.2)
            paths=np.stack((opens,np.maximum(opens,center)+width,np.minimum(opens,center)-width,center),axis=2)
        if q!='ideal':
            paths[:,0,0]=e.c[b]
            paths[:,:,1]=np.maximum(paths[:,:,1],np.maximum(paths[:,:,0],paths[:,:,3]))
            paths[:,:,2]=np.minimum(paths[:,:,2],np.minimum(paths[:,:,0],paths[:,:,3]))
        times=e.v[b+1:self.end];opening=int(e.mo[0]);f=e.cfg.profile
        spreads=np.array([f.spread_at(int(t),opening) for t in times])
        n=len(paths);first=int(e.v[b])+HOUR
        r=PathReport(first,first,first,HOUR,int(e.ff[0]),paths,
            np.repeat(spreads[None],n,axis=0),np.full(n,1/n),
            'synthetic_ideal' if q=='ideal' else 'synthetic_noisy',
            np.full(n,f.spread_at(int(e.ff[0]),opening)))
        self.reports[b]=r
        return r

    def close_values(self,e,h,b):
        result=super().close_values(e,h,b)
        while len(self.cache)>128: self.cache.popitem(last=False)
        return result
