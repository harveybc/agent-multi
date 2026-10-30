"""Observational decomposition on frozen visited states, not policy tuning."""
from collections import Counter

import numpy as np

from app.causal_history_account import HistoryWorld
from app.crossed_account_panel import CrossedWorld
from app.dynamic_account_arbiter import choose


def close_stages(values, weights, risk, uncertainty, history, margin):
    delta=values[:,[0]]-values[:,1:]
    mean=weights@delta
    sd=np.sqrt(weights@((delta-mean)**2))
    raw=float(np.min(mean))/risk
    after_unc=float(np.min(mean-uncertainty*sd))/risk
    final=after_unc-history
    threshold=max(margin,1e-12)
    return dict(raw=raw,after_uncertainty=after_unc,final=final,
                raw_pass=raw>threshold,uncertainty_pass=after_unc>threshold,
                history_pass=final>threshold)


class DiagnosticWorld(HistoryWorld):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.diagnostic=Counter()

    def candidate_bids(self,b):
        raw=CrossedWorld.candidate_bids(self,b)
        adjusted=super().candidate_bids(b)
        c=self.diagnostic;c['flat_ticks']+=1
        c['raw_bids']+=len(raw);c['adjusted_bids']+=len(adjusted)
        if self.active_genes[11]:
            c['unmatured_bids']+=sum(self.history_error(b,x['horizon']) is None for x in raw)
        efficiency=self.past_efficiency(b)
        before=choose(raw,self.active_genes[:11],efficiency)
        after=choose(adjusted,self.active_genes[:11],efficiency)
        c['entry_raw_selected']+=before is not None
        c['entry_adjusted_selected']+=after is not None
        c['entry_to_nop_after_history']+=before is not None and after is None
        return adjusted

    def close_decision(self,controller,unit,r,clock,deadline,values,binding,b,horizon):
        c=self.diagnostic;c['occupied_decisions']+=1
        error=self.history_error(b,horizon) if self.active_genes[12] else 0.
        result=super().close_decision(controller,unit,r,clock,deadline,values,binding,b,horizon)
        if error is None:
            c['close_unmatured']+=1
            return result
        stages=close_stages(values,r.weights,unit.risk,self.active_genes[13],
            self.active_genes[12]*error*unit.units/unit.risk,self.active_genes[9])
        if abs(stages['final']-result['advantage_risk'])>1e-12:
            raise AssertionError('DIAGNOSTIC_FORMULA_PARITY')
        c['close_matured']+=1
        for name in ('raw_pass','uncertainty_pass','history_pass'):
            c['close_'+name]+=stages[name]
        c['close_persistence_wait']+=result['qualifies'] and result['action']=='HOLD'
        c['close_requests']+=result['action']=='CLOSE'
        return result
