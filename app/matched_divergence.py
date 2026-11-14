"""Endpoint-defined paired decision events and conditional factorial effects."""
import numpy as np


def locate_events(control,treatment,y,rows):
    c=np.asarray(control);t=np.asarray(treatment);y=np.asarray(y)
    if (c.shape!=t.shape or c.ndim!=2 or not len(c) or y.shape!=(c.shape[1],3)
            or c.dtype.kind not in 'iu' or t.dtype.kind not in 'iu'
            or (c<0).any() or (c>2).any() or (t<0).any() or (t>2).any()
            or not np.isfinite(y).all()):raise ValueError('DIVERGENCE_ACTIONS')
    events=[]
    for row in rows:
        ca,ta=int(c[-1,row]),int(t[-1,row]);adv=float(y[row,ca]-y[row,ta])
        if adv<=0:continue
        first=int(np.flatnonzero(c[:,row]!=t[:,row])[0])
        crossings=[i for i in range(1,len(c)) if c[i,row]==ca and y[row,c[i-1,row]]<y[row,ca]]
        events.append(dict(row=int(row),control_action=ca,treatment_action=ta,control_advantage=adv,
            first_disagreement=first,control_acquisition=crossings[0] if crossings else None))
    return events


def factorial_effects(cells):
    if set(cells)!={'CC','CT','TC','TT'} or not all(np.isfinite(v) for v in cells.values()):raise ValueError('DIVERGENCE_FACTORIAL')
    cc,ct,tc,tt=(float(cells[k]) for k in ('CC','CT','TC','TT'))
    return dict(optimizer_at_control_weights=ct-cc,optimizer_at_treatment_weights=tt-tc,
        weights_at_control_optimizer=tc-cc,weights_at_treatment_optimizer=tt-ct,interaction=tt-tc-ct+cc)
