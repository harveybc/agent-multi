"""Frozen TRAIN panel, categorical SAC derivatives and matched-state rankings."""
import numpy as np


def select_panel(traces, events, changed_timestamps):
    by_time={r['timestamp']:r for r in traces}
    if len(by_time)!=len(traces):raise ValueError('OPENING_DUPLICATE_TIME')
    changed=sorted({r['timestamp'] for r in events if r['isolated_position']==0})
    selected={t:dict(timestamp=t,kind='changed',matched_to=None) for t in changed}
    used=set(changed)
    for t in changed:
        anchor=by_time[t]
        if anchor['position']!=0:raise ValueError('OPENING_NOT_FLAT')
        candidates=[r for r in traces if r['timestamp'] not in used
                    and r['timestamp'] not in changed_timestamps and r['position']==0
                    and r['year']==anchor['year'] and r['signal']==anchor['signal']]
        if not candidates:raise ValueError('OPENING_NO_MATCHED_CONTROL')
        match=min(candidates,key=lambda r:(abs(r['timestamp']-t),r['timestamp']))['timestamp']
        selected[match]=dict(timestamp=match,kind='unchanged_control',matched_to=t)
        used.add(match)
    return [selected[t] for t in sorted(selected)]


def sac_gradients(logits,q,alpha):
    z,q=np.asarray(logits,dtype=float),np.asarray(q,dtype=float)
    if (z.ndim!=2 or z.shape[1]!=3 or q.shape!=z.shape or
            not np.isfinite([z,q]).all() or not np.isfinite(alpha) or alpha<0):
        raise ValueError('OPENING_GRADIENT_INPUT')
    z=z-z.max(1,keepdims=True);lp=z-np.log(np.exp(z).sum(1,keepdims=True));p=np.exp(lp)
    qg=-p*(q-(p*q).sum(1,keepdims=True))
    eg=alpha*p*(lp-(p*lp).sum(1,keepdims=True))
    return dict(q=qg,entropy=eg,total=qg+eg)


def project_gradient(gradient,hidden,index,old,new):
    g,h=np.asarray(gradient),np.asarray(hidden);x=h[index]
    weights=g.T@h/len(h);bias=g.mean(0)
    return dict(local=float(-(g[index,new]-g[index,old])*(x@x+1)),
                shared=float(-((weights[new]-weights[old])@x+bias[new]-bias[old])))


def compare_pair(q,flat,calibration,old,new):
    q,f,c=np.asarray(q),np.asarray(flat),np.asarray(calibration)
    if q.shape!=(3,) or f.shape!=(3,3) or c.shape!=(3,3) or not all(np.isfinite(x).all() for x in (q,f,c)):
        raise ValueError('OPENING_PAIR_INPUT')
    fg=f[:,new]-f[:,old];cg=c[:,new]-c[:,old];qgap=float(q[new]-q[old]);tiny=1e-8
    def conflict(g):return bool((qgap>tiny and (g < -tiny).all()) or (qgap < -tiny and (g>tiny).all()))
    flip=bool(((fg>tiny).all() and (cg < -tiny).all()) or ((fg < -tiny).all() and (cg>tiny).all()))
    return dict(q_gap=qgap,flat_gap=fg.tolist(),calibration_gap=cg.tolist(),
                flat_q_conflict=conflict(fg),calibration_q_conflict=conflict(cg),
                state_order_flip=flip,three_draws_not_exact_expectation=True)


def gradient_strata(gradient,hidden,labels):
    g,h,labels=np.asarray(gradient),np.asarray(hidden),np.asarray(labels)
    if len(g)!=len(h) or len(labels)!=len(h):raise ValueError('OPENING_STRATA_SHAPE')
    # Divide by the full support size, so strata sum to the original gradient.
    return {str(label):dict(rows=int((labels==label).sum()),
        weight=g[labels==label].T@h[labels==label]/len(h),
        bias=g[labels==label].sum(0)/len(h)) for label in np.unique(labels)}
