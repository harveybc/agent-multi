"""FIT-only candidate ranking for finite, decision-targeted critic corrections."""
import numpy as np
from app.optimal_fit_protection import optimal_support,complete_pair_jacobian


def correction_candidates(state,z,indices,y,views,excluded,radius,count=8):
    if not np.isfinite(radius) or radius<=0 or type(count) is not int or count<1:raise ValueError('TARGET_BUDGET')
    allowed,_=optimal_support(y,views);indices=np.asarray(indices,dtype=int)
    if indices.shape!=(len(y),) or len(set(indices.tolist()))!=len(indices):raise ValueError('TARGET_INDICES')
    qs=list(views.values());actions=[q.min(0).argmax(1) for q in qs];excluded=set(excluded)
    rows=[];eligible=0;too_far=0
    for pos,row in enumerate(indices):
        action=actions[0][pos]
        if row in excluded or allowed[pos,action] or any(a[pos]!=action for a in actions):continue
        eligible+=1;minimum=np.asarray(qs[0]).min(0)[pos]
        chosen=int(np.where(allowed[pos],minimum,-np.inf).argmax())
        pairs=[(int(row),chosen,int(b)) for b in np.flatnonzero(~allowed[pos])]
        margin=float(64*np.finfo(np.float32).eps*max(1.,*(float(np.max(abs(q[:,pos]))) for q in qs)))
        gaps,jac=complete_pair_jacobian(state,z,pairs);need=np.maximum(margin-gaps,0.);norms=np.linalg.norm(jac,axis=1)
        distances=np.divide(need,norms,out=np.full_like(need,np.inf),where=norms>1e-14);distances[need==0]=0
        bound=float(distances.max())
        if bound>radius:too_far+=1;continue
        rows.append(dict(row=int(row),position=pos,action=chosen,pairs=pairs,margin=margin,lower_bound=bound))
    rows.sort(key=lambda r:(r['lower_bound'],r['row']))
    return rows[:count],dict(eligible_wrong=eligible,too_far=too_far,within_radius=len(rows),examined=min(count,len(rows)))


def strict_correction(assessment,baseline,views,position,allowed):
    if not assessment['passed'] or set(baseline)!=set(views) or set(assessment['metrics'])!=set(views):return False
    return all(assessment['metrics'][k]['regret']<baseline[k]['regret'] and bool(allowed[q.min(0).argmax(1)[position]]) for k,q in views.items())
