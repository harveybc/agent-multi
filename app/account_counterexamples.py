"""Admission rules for finite, native TRAIN-account counterexamples."""
import numpy as np


def new_constraints(events, timestamps, periods, existing):
    result=[]
    for e in events:
        if not np.isfinite(e['contribution_usd']):raise ValueError('COUNTEREXAMPLE_NONFINITE')
        if e['contribution_usd']>=-1e-6:continue
        match=np.flatnonzero(timestamps==e['timestamp'])
        if len(match)!=1:raise ValueError('COUNTEREXAMPLE_TIMESTAMP')
        row=int(match[0])
        if periods[row]!='TRAIN':raise ValueError('COUNTEREXAMPLE_TRAIN_ONLY')
        old,new=e['old_signal'],e['new_signal']
        if old not in (0,1,2) or new not in (0,1,2) or old==new:
            raise ValueError('COUNTEREXAMPLE_ACTION')
        key=(row,old,new)
        if key not in existing and key not in result:result.append(key)
    return result


def account_qualifies(reference, candidate, events):
    return (np.isfinite([reference['net_profit_usd'],candidate['net_profit_usd'],candidate['orders_per_week']]).all()
        and candidate['orders_per_week']>2
        and candidate['net_profit_usd']>reference['net_profit_usd']+1e-6
        and all(np.isfinite(e['contribution_usd']) and e['contribution_usd']>=-1e-6 for e in events))
