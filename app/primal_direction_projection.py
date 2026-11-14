"""Independent reduced-primal projection with original-space and KKT checks."""
import numpy as np
from scipy.optimize import minimize,LinearConstraint


def project_primal(displacement,jacobian,required):
    d=np.asarray(displacement,dtype=float);j=np.asarray(jacobian,dtype=float);b=np.asarray(required,dtype=float)
    if (d.ndim!=1 or b.ndim!=1 or j.shape!=(len(b),len(d)) or not len(d) or not len(b)
            or not all(np.isfinite(x).all() for x in (d,j,b))):raise ValueError('PRIMAL_CONTRACT')
    scale=float(np.linalg.norm(d))
    if scale<=0:raise ValueError('PRIMAL_ZERO')
    norms=np.linalg.norm(j,axis=1);active=norms>1e-14
    if ((~active)&(b>0)).any():return d.copy(),dict(feasible=False,reason='zero_constraint_gradient')
    a=j[active]/norms[active,None];r=(b[active]-j[active]@d)/norms[active]/scale
    if not len(a) or (r<=0).all():
        return d.copy(),dict(feasible=True,reason='already_feasible',correction_norm=0.,iterations=0)
    basis,_=np.linalg.qr(a.T,mode='reduced');reduced=a@basis
    opt=minimize(lambda w:(float(.5*w@w),w),np.zeros(basis.shape[1]),jac=True,method='SLSQP',
        constraints=[LinearConstraint(reduced,r,np.inf)],options=dict(maxiter=2000,ftol=1e-14))
    candidate=d+scale*(basis@opt.x)
    violation=float(max(0.,np.max((b[active]-j[active]@candidate)/norms[active]/scale)))
    multipliers=np.asarray(getattr(opt,'multipliers',[]),dtype=float)
    kkt=None
    if multipliers.shape==r.shape and np.isfinite(multipliers).all():
        slack=reduced@opt.x-r
        kkt=dict(stationarity=float(np.max(np.abs(opt.x-reduced.T@multipliers))),
            dual_violation=float(max(0.,-multipliers.min())),complementarity=float(np.max(np.abs(multipliers*slack))))
    kkt_ok=kkt is not None and all(v<=1e-7 for v in kkt.values())
    passed=bool(opt.success and np.isfinite(candidate).all() and violation<=1e-7 and kkt_ok)
    return candidate,dict(feasible=passed,reason=str(opt.message),solver_success=bool(opt.success),
        iterations=int(opt.nit),normalized_violation=violation,kkt=kkt,
        constraints=len(a),reduced_dimensions=basis.shape[1],
        correction_norm=float(np.linalg.norm(candidate-d)))


def select_stalled_probes(trace,per_reason=8):
    if type(per_reason) is not int or per_reason<1:raise ValueError('PRIMAL_PROBE_COUNT')
    if [t['step'] for t in trace]!=list(range(1,len(trace)+1)):raise ValueError('PRIMAL_TRACE_SEQUENCE')
    accepted=[t['step'] for t in trace if t['accepted']]
    if not accepted:raise ValueError('PRIMAL_TRACE_NO_ACCEPTED_START')
    last=max(accepted);selected=[]
    for reason in ('solver_or_norm_cap','finite_guard'):
        rows=[t['step'] for t in trace if t['step']>last and not t['accepted'] and t['reason']==reason]
        if len(rows)<per_reason:raise ValueError('PRIMAL_PROBE_STRATUM_TOO_SMALL')
        selected.extend(rows[i] for i in np.linspace(0,len(rows)-1,per_reason,dtype=int))
    return last,sorted(selected)


def qualify_continuation(rows):
    if len(rows)!=16 or len({r['attempt'] for r in rows})!=16:return False
    valid=[r for r in rows if r['accepted'] and r['robust'] and r['geometry_agrees']
        and np.isfinite(r['before']) and np.isfinite(r['after']) and r['after']<=r['before']+1e-8]
    return len(valid)>=4 and any(r['after']<r['before']-1e-8 for r in valid)
