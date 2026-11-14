"""SPD-whitened target repair with fixed FIT-only action-gap locality."""
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from app.primal_direction_projection import project_primal
from app.weight_decision_forensics import weighted_min_gradient

PAIRS=((0,1),(0,2),(1,2))


def gap_jacobian(state,z,indices,target):
    ix=np.asarray(indices,dtype=int)
    if len(set(ix))!=len(ix) or target not in ix:raise ValueError('LOCAL_SUPPORT')
    ix=ix[ix!=target]
    if not len(ix):raise ValueError('LOCAL_EMPTY_SUPPORT')
    x=z[:,ix];blocks=[]
    for a,b in PAIRS:
        c=np.zeros((len(ix),3));c[:,a]=1;c[:,b]=-1
        blocks.append(weighted_min_gradient(state,x,c))
    return np.concatenate(blocks),ix


def locality_factor(g):
    g=np.asarray(g,dtype=float)
    if g.ndim!=2 or min(g.shape)<1 or not np.isfinite(g).all():raise ValueError('LOCAL_METRIC')
    s=float(np.sum(g*g));p=g.shape[1]
    h=np.eye(p)
    if s:h+=(p/s)*(g.T@g)
    return cholesky(h,lower=True,check_finite=True)


def metric_projection(d,j,b,l):
    d=np.asarray(d,dtype=float);j=np.asarray(j,dtype=float);b=np.asarray(b,dtype=float)
    if l.shape!=(len(d),len(d)) or not np.isfinite(l).all():raise ValueError('LOCAL_FACTOR')
    # u=L^T delta; unconstrained u0=L^-1 d, constraints J L^-T u >= b.
    u0=solve_triangular(l,d,lower=True);jw=solve_triangular(l,j.T,lower=True).T
    u,receipt=project_primal(u0,jw,b)
    delta=solve_triangular(l.T,u,lower=False)
    norms=np.linalg.norm(j,axis=1);active=norms>1e-14;scale=float(np.linalg.norm(d))
    if scale<=0:raise ValueError('LOCAL_ZERO_MOTION')
    violation=float(max(0.,np.max((b[active]-j[active]@delta)/norms[active]/scale))) if active.any() else 0.
    passed=receipt['feasible'] and np.isfinite(delta).all() and violation<=1e-7
    return delta,dict(receipt,feasible=bool(passed),original_normalized_violation=violation)


def bounded_local_repair(d,linearize,evaluate,l):
    d=np.asarray(d,dtype=float);norm=float(np.linalg.norm(d));center=np.zeros_like(d);repairs=[]
    if d.ndim!=1 or not np.isfinite(d).all() or norm<=0:raise ValueError('LOCAL_DISPLACEMENT')
    # This mirrors the immutable344 loop; only the quadratic metric changes.
    for iteration in range(3):
        gaps,jac,actual,threshold=linearize(center)
        if threshold.shape!=gaps.shape or not np.isfinite(threshold).all() or (threshold<0).any():raise ValueError('LOCAL_THRESHOLD')
        candidate,solver=metric_projection(d,jac,threshold-gaps+jac@actual,l)
        ratio=float(np.linalg.norm(candidate)/norm);entry=dict(iteration=iteration,solver=solver,norm_ratio=ratio,trials=[]);repairs.append(entry)
        if not solver['feasible'] or not np.isfinite(ratio) or ratio>4:return None,dict(accepted=False,reason='solver_or_norm_cap',repairs=repairs)
        for exponent in range(8):
            scale=2.**(-exponent);result=evaluate(candidate*scale);entry['trials'].append(dict(scale=scale,**result))
            if result['passed'] and 0<result['motion']<=4*norm:return candidate*scale,dict(accepted=True,reason='finite_guard',repairs=repairs)
        center=candidate
    return None,dict(accepted=False,reason='finite_guard',repairs=repairs)


def gap_drift(before,after,position):
    before=np.asarray(before);after=np.asarray(after)
    if before.shape!=after.shape or before.ndim!=3 or before.shape[0]!=2 or before.shape[2]!=3 or not 0<=position<before.shape[1] or before.shape[1]<2:raise ValueError('LOCAL_DRIFT_SHAPE')
    if not np.isfinite(before).all() or not np.isfinite(after).all():raise ValueError('LOCAL_DRIFT_FINITE')
    old=before.min(0).astype(float);new=after.min(0).astype(float);keep=np.arange(len(old))!=position
    differences=np.stack([(new[keep,a]-new[keep,b])-(old[keep,a]-old[keep,b]) for a,b in PAIRS])
    return float(np.mean(differences**2))


def pairwise_qualified(assessment,control,drift,control_drift):
    if not assessment or not assessment['passed']:return False
    if not drift or set(drift)!=set(control_drift) or set(drift)!=set(control) or set(drift)!=set(assessment['metrics']):raise ValueError('LOCAL_PAIR_VIEWS')
    return all(np.isfinite(drift[k]) and drift[k]<control_drift[k] and assessment['metrics'][k]['regret']<=control[k]['regret'] for k in drift)
