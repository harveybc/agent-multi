"""FIT-only optimal-set protection with finite acceptance and complete twin pairs."""
import numpy as np
from app.primal_direction_projection import project_primal
from app.weight_decision_forensics import native_forward,cost_parts


def optimal_support(y,views):
    y=np.asarray(y,dtype=float)
    if y.ndim!=2 or y.shape[1]!=3 or not np.isfinite(y).all() or not views:raise ValueError('OPTIMAL_SUPPORT')
    allowed=y==y.max(1,keepdims=True);eligible=allowed.sum(1)<3
    for q in views.values():
        q=np.asarray(q)
        if q.shape!=(2,len(y),3) or not np.isfinite(q).all():raise ValueError('OPTIMAL_VIEW')
        eligible&=allowed[np.arange(len(y)),q.min(0).argmax(1)]
    return allowed,np.flatnonzero(eligible)


def optimal_pairs(q,indices,allowed,eligible):
    minimum=np.asarray(q).min(0);pairs=[]
    for p in eligible:
        a=int(np.where(allowed[p],minimum[indices[p]],-np.inf).argmax())
        pairs.extend((int(indices[p]),a,int(b)) for b in np.flatnonzero(~allowed[p]))
    return pairs


def complete_pair_jacobian(state,z,pairs):
    q,cache=native_forward(state,z);p=np.asarray(pairs,dtype=int)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or (p[:,0]<0).any() or (p[:,0]>=z.shape[1]).any() or (p[:,1:]<0).any() or (p[:,1:]>2).any() or (p[:,1]==p[:,2]).any():
        raise ValueError('OPTIMAL_PAIRS')
    rows=np.repeat(p[:,0],2);a=np.repeat(p[:,1],2);b=np.repeat(p[:,2],2)
    twins=np.tile([0,1],len(p));other=q[:,rows,b].argmin(0);n=len(rows);ix=np.arange(n)
    coefficients=np.zeros((2,n,3));coefficients[twins,ix,a]=1;coefficients[other,ix,b]-=1
    blocks=[]
    for t,c in enumerate(cache):
        routed=coefficients[t];du=(routed@c['output_weight'])*c['derivative'][rows]
        blocks.extend([(du[:,:,None]*c['z'][rows,None,:]).reshape(n,-1),du,
            (routed[:,:,None]*c['h'][rows,None,:]).reshape(n,-1),routed])
    return q[twins,rows,a]-q[other,rows,b],np.concatenate(blocks,axis=1)


def finite_quality(y,views,baseline,allowed,eligible,temperature,original_passed,motion,norm):
    if not views or set(views)!=set(baseline):raise ValueError('OPTIMAL_FINITE_VIEWS')
    metrics={};passed=bool(original_passed and 0<motion<=4*norm)
    for name,q in views.items():
        p=cost_parts(q,y,temperature);b=cost_parts(baseline[name],y,temperature)
        actions=np.asarray(q).min(0).argmax(1)
        lost=int(np.sum(~allowed[eligible,actions[eligible]]))
        cost=float(p['row_cost'].mean());regret=float(p['row_regret'].mean())
        good=lost==0 and cost<float(b['row_cost'].mean()) and regret<=float(b['row_regret'].mean())
        metrics[name]=dict(cost=cost,regret=regret,lost_optimal=lost,action_changes=int(np.sum(actions!=np.asarray(baseline[name]).min(0).argmax(1))),passed=bool(good))
        passed&=good
    return dict(passed=bool(passed),motion=float(motion),original_passed=bool(original_passed),metrics=metrics)


def bounded_repair(delta,linearize,evaluate):
    d=np.asarray(delta,dtype=float);norm=float(np.linalg.norm(d));center=np.zeros_like(d);repairs=[]
    if d.ndim!=1 or not np.isfinite(d).all():raise ValueError('OPTIMAL_DISPLACEMENT')
    if norm==0:return None,dict(accepted=False,reason='zero_motion',repairs=[])
    for iteration in range(3):
        gaps,jac,actual,threshold=linearize(center)
        if threshold.shape!=gaps.shape or not np.isfinite(threshold).all() or (threshold<0).any():raise ValueError('OPTIMAL_THRESHOLD')
        candidate,solver=project_primal(d,jac,threshold-gaps+jac@actual)
        ratio=float(np.linalg.norm(candidate)/norm);entry=dict(iteration=iteration,solver=solver,norm_ratio=ratio,trials=[]);repairs.append(entry)
        if not solver['feasible'] or not np.isfinite(ratio) or ratio>4:return None,dict(accepted=False,reason='solver_or_norm_cap',repairs=repairs)
        for exponent in range(8):
            scale=2.**(-exponent);result=evaluate(candidate*scale)
            entry['trials'].append(dict(scale=scale,**result))
            if result['passed'] and 0<result['motion']<=4*norm:return candidate*scale,dict(accepted=True,reason='finite_guard',repairs=repairs)
        center=candidate
    return None,dict(accepted=False,reason='finite_guard',repairs=repairs)
