"""Local halfspace projection; finite nonlinear guards remain authoritative."""
import numpy as np
from scipy.optimize import minimize


def project_displacement(displacement,jacobian,required):
    d=np.asarray(displacement,dtype=float);j=np.asarray(jacobian,dtype=float);b=np.asarray(required,dtype=float)
    if (d.ndim!=1 or j.ndim!=2 or j.shape!=(len(b),len(d)) or b.ndim!=1
            or not d.size or not b.size or not all(np.isfinite(x).all() for x in (d,j,b))):
        raise ValueError('DIRECTION_SHAPE_NONFINITE')
    scale=float(np.linalg.norm(d))
    if scale<=0:raise ValueError('DIRECTION_ZERO')
    norms=np.linalg.norm(j,axis=1);active=norms>1e-14
    if ((~active)&(b>0)).any():
        return d.copy(),dict(feasible=False,reason='zero_constraint_gradient')
    a=j[active]/norms[active,None]
    residual=(b[active]-j[active]@d)/norms[active]/scale
    if not len(a) or (residual<=0).all():
        return d.copy(),dict(feasible=True,reason='already_feasible',correction_norm=0.,iterations=0,dual_success=True)
    gram=a@a.T
    # Dual of Euclidean projection onto J*delta >= required, normalized for scale.
    def objective(lam):
        product=gram@lam
        return float(.5*lam@product-residual@lam),product-residual
    opt=minimize(objective,np.zeros(len(a)),jac=True,method='L-BFGS-B',
                 bounds=[(0.,None)]*len(a),options=dict(maxiter=5000,ftol=1e-14,gtol=1e-10,maxls=40))
    p=d+scale*(a.T@opt.x)
    violation=float(max(0.,np.max((b[active]-j[active]@p)/norms[active]/scale)))
    valid=bool(opt.success and np.isfinite(p).all() and violation<=1e-7)
    return p,dict(feasible=valid,reason=str(opt.message),dual_success=bool(opt.success),
        iterations=int(opt.nit),normalized_violation=violation,correction_norm=float(np.linalg.norm(p-d)))


def decision_guard(q,actions,margin,y,limit):
    q=np.asarray(q,dtype=float);actions=np.asarray(actions);margin=np.asarray(margin,dtype=float);y=np.asarray(y,dtype=float)
    if (q.ndim!=2 or q.shape[1]!=3 or q.shape!=y.shape or not len(q)
            or actions.shape!=(len(q),) or actions.dtype.kind not in 'iu'
            or ((actions<0)|(actions>2)).any() or margin.shape!=(len(q),)
            or (margin<=0).any() or not all(np.isfinite(v).all() for v in (q,y,margin))
            or not np.isfinite(limit) or limit<0):raise ValueError('DIRECTION_GUARD_CONTRACT')
    chosen=q.argmax(1);rows=np.arange(len(q));gaps=q[rows,actions,None]-q
    gaps[rows,actions]=np.inf
    regret=float(np.mean(y.max(1)-y[rows,chosen]))
    minimum=float(np.min(gaps-margin[:,None]));equal=bool(np.array_equal(chosen,actions))
    return dict(passed=equal and minimum>=0 and regret<=limit+1e-8,
        actions=chosen.tolist(),same_actions=equal,min_margin_surplus=minimum,regret=regret)


def numpy_heads(state,z):
    z=np.asarray(z,dtype=float);out=[]
    for twin in range(2):
        prefix=f'twins.{twin}.0.'
        def arr(k):return state[prefix+k].detach().cpu().double().numpy()
        h=z[twin]@arr('0.weight').T+arr('0.bias')
        h=h*np.exp(-np.logaddexp(0.,-h))
        out.append(h@arr('2.weight').T+arr('2.bias'))
    return np.stack(out)
