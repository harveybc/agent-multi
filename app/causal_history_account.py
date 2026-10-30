"""Prefix-only account experiments with controller-side matured-error penalties."""
import math

import numpy as np

from app.causal_report_baseline import sampled_prefix_report
from app.close_hold_controller import CloseConfig, CloseController, value_paths, bind_values
from app.crossed_account_panel import CrossedWorld, panel_specs
from app.dynamic_account_panel import BASE,HOUR,World
from app.dynamic_account_arbiter import bounds,decode,GENES
from app.report_track_record import ReportTrackRecord

HISTORY_GENES=GENES+('entry_history_weight','close_history_weight','close_uncertainty')


def history_bounds():
    out=bounds()+[(0.,2.),(0.,2.),(0.,2.)]
    out[8]=(1.,1.)
    return out


def base_genes(entry=0.,close=0.):
    return list(BASE)+[entry,close,0.]


def decode_history(g):
    if len(g)!=14 or not all(math.isfinite(x) for x in g):
        raise ValueError('HISTORY_GENOME')
    out=list(decode(g[:11]));out[8]=1.
    out.extend(round(min(2.,max(0.,float(x))),3) for x in g[11:])
    return tuple(out)


def specifications(bank,smoke=False):
    rows=panel_specs(bank,'omitted_reversal',draws=1 if bank=='calibration' else 3)
    offset=100000+(100000 if smoke else 0)
    return [dict(s,seed=s['seed']+offset,bank='history_'+bank) for s in (rows[:2] if smoke else rows)]


class HistoryWorld(CrossedWorld):
    def __init__(self,spec,config,mode):
        if mode not in ('median','zero'): raise ValueError('HISTORY_REPORTER_MODE')
        super().__init__(spec,config)
        self.mode=mode;self.active_genes=base_genes()
        self.quality={};self.through=1
        self.ledger=ReportTrackRecord('history-prefix-v1/'+mode)

    def _ensure_cache_context(self):
        old=self._cache_context
        super()._ensure_cache_context()
        if old is not None and old!=self._cache_context:
            self.quality.clear();self.through=1
            self.ledger=ReportTrackRecord('history-prefix-v1/'+self.mode)

    def report(self,b):
        self._ensure_cache_context()
        if b<2 or b+1>=self.end: raise ValueError('HISTORY_REPORT_SUPPORT')
        e=self.env
        for i in range(self.through+1,b+1):
            now=int(e.v[i])+HOUR
            self.ledger.observe(now,float(e.c[i]),now=now)
            self.quality[i]={h:self.ledger.diagnostic(h) for h in (1,3,6,12,24)}
            f=e.cfg.profile;times=e.v[i+1:self.end]
            spread=np.array([f.spread_at(int(t),int(e.mo[0])) for t in times])
            seed=np.random.SeedSequence([self.spec['seed'],i,1889])
            r=sampled_prefix_report(np.column_stack((e.o,e.h,e.l,e.c))[:i+1],now=now,
                step=HOUR,deadline=int(e.ff[0]),spreads=spread,
                terminal_spread=f.spread_at(int(e.ff[0]),int(e.mo[0])),seed=seed,drift=self.mode)
            self.ledger.issue(r,now=now,expected_deadline=int(e.ff[0]))
            self.reports[i]=r;self.through=i
        return self.reports[b]

    def history_error(self,b,h):
        self.report(b)
        remaining=self.end-b-1
        lead=max(x for x in (1,3,6,12,24) if x<=min(h,remaining))
        info=self.quality[b][lead]
        return info['mae'] if info['ready'] else None

    def candidate_bids(self,b):
        rows=super().candidate_bids(b)
        if self.active_genes[11]==0: return rows
        result=[];r=self.report(b)
        for row in rows:
            error=self.history_error(b,row['horizon'])
            if error is None: continue
            d=row['direction'];entry=float(r.weights@(r.paths[:,0,0]+d*r.spreads[:,0]/2))
            risk=self.position(row['horizon'],d,entry,b+1).risk
            result.append(dict(row,gain_risk=row['gain_risk']-self.active_genes[11]*error/risk))
        return result

    def make_controller(self,genes):
        return CloseController(CloseConfig(self.active_genes[13],genes[9],genes[10],.25))

    def close_values(self,e,h,b):
        return World.close_values(self,e,h,b)

    def close_decision(self,controller,unit,r,clock,deadline,values,binding,b,horizon):
        weight=self.active_genes[12];error=self.history_error(b,horizon) if weight else 0.
        if error is None:
            controller.streak=0;controller.position_id=unit.identity;controller.last_tick=clock
            return dict(action='HOLD',advantage_risk=0.,reason='UNMATURED_HISTORY')
        margin=weight*error*unit.units/unit.risk
        return controller.decide_causal_values(unit,r,clock,values,expected_deadline=deadline,
            binding=binding,history_margin_risk=margin)

    def verify_close_decision(self,direct,unit,r,clock,deadline,fo,fe,b,horizon):
        values=value_paths(unit,r,fo,fe)
        return self.close_decision(direct,unit,r,clock,deadline,values,bind_values(unit,r,values),b,horizon)

    def run(self,genes,enabled=None,verify=False,details=False,*,voluntary_close=True):
        self.active_genes=decode_history(genes)
        return super().run(self.active_genes[:11],enabled,verify,details,voluntary_close=voluntary_close)
