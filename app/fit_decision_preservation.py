"""FIT-only source-action preservation and finite constrained step projection."""
import numpy as np
from scipy.optimize import minimize
import torch


def validate_fresh_calibration(source, diagnosis):
    if (source.get('schema')!='selected_input_critic_calibration.v1'
            or source.get('actor_updates')!=0 or source.get('modes')!=[]
            or diagnosis.get('schema')!='selected_calibrated_boundary_diagnosis.v1'
            or diagnosis.get('source')!=source.get('identity')
            or diagnosis.get('source_sha256')!=source.get('source_sha256')
            or diagnosis.get('support')!=source.get('support_sha256')
            or diagnosis.get('actor_updates')!=0 or diagnosis.get('posthoc_timestamps')!=[]):
        raise ValueError('PRESERVE_FRESH_CALIBRATION_LINEAGE')


def protected_actions(logits, native, records):
    if logits.shape!=native.shape or logits.shape!=(len(records),3):
        raise ValueError('PRESERVE_SHAPE')
    if not torch.isfinite(logits).all() or not torch.isfinite(native).all():
        raise ValueError('PRESERVE_NONFINITE')
    actions=logits.argmax(-1)
    top=native.topk(2,dim=-1).values
    fit=torch.tensor([r['role']=='fit' for r in records],device=logits.device)
    keep=fit & (actions==native.argmax(-1)) & ((top[:,0]-top[:,1])>1e-8)
    rows=torch.nonzero(keep).flatten()
    return rows,actions[rows]


def margins(logits, rows, actions):
    other=torch.tensor([[b for b in range(3) if b!=int(a)] for a in actions],
                       device=logits.device,dtype=torch.long).reshape(-1,2)
    return logits[rows,actions,None]-logits[rows[:,None],other]


def interior_buffer(logits, rows, actions):
    """Fixed FIT-logit rounding cushion for proposal constraints, not acceptance."""
    if (logits.dtype not in (torch.float32,torch.float64) or logits.ndim!=2 or logits.shape[1]!=3
            or rows.ndim!=1 or not len(rows) or actions.shape!=rows.shape
            or rows.dtype!=torch.long or actions.dtype!=torch.long
            or (rows<0).any() or (rows>=len(logits)).any()
            or (actions<0).any() or (actions>2).any()):
        raise ValueError('PRESERVE_INTERIOR_INPUT')
    selected=logits[rows].detach().double()
    if not torch.isfinite(selected).all():raise ValueError('PRESERVE_INTERIOR_NONFINITE')
    others=torch.tensor([[b for b in range(3) if b!=int(a)] for a in actions],device=logits.device)
    preferred=selected.gather(1,actions[:,None]).abs()
    return 8*torch.finfo(logits.dtype).eps*(1+preferred+selected.gather(1,others).abs())


def preservation(logits, source_margins, rows, actions, tolerance=1e-7):
    if not len(rows):raise ValueError('PRESERVE_EMPTY')
    delta=margins(logits.double(),rows,actions)-source_margins.double()
    finite=bool(torch.isfinite(delta).all())
    changes=int((logits[rows].argmax(-1)!=actions).sum())
    return dict(passed=finite and changes==0 and bool((delta>=-tolerance).all()),
                minimum_margin_change=float(delta.min()),action_changes=changes,
                violations=int((delta < -tolerance).sum()),protected_rows=len(rows))


def project_step(direction, matrix, lower):
    d=np.asarray(direction,dtype=np.float64); a=np.asarray(matrix,dtype=np.float64)
    b=np.asarray(lower,dtype=np.float64)
    if (d.ndim!=1 or a.ndim!=2 or a.shape[1]!=len(d) or b.shape!=(len(a),)
            or not all(np.isfinite(x).all() for x in (d,a,b))):
        raise ValueError('PRESERVE_PROJECTION_INPUT')
    zero=np.zeros_like(d); size=float(np.linalg.norm(d)); norms=np.linalg.norm(a,axis=1)
    if np.any((norms==0)&(b>0)):
        return zero,dict(success=False,nonzero=False,reason='infeasible_zero_constraint')
    if size==0:return zero,dict(success=bool((b<=0).all()),nonzero=False,reason='zero_proposal')
    keep=norms>0; an=a[keep]/norms[keep,None]; bn=b[keep]/norms[keep]/size
    unit=d/size
    if not len(an):return d.copy(),dict(success=True,nonzero=True,rank=0,iterations=0)
    _, s, vh=np.linalg.svd(an,full_matrices=False)
    rank=int((s>s[0]*1e-10).sum()); basis=vh[:rank]
    y0=basis@unit; null=unit-basis.T@y0; reduced=an@basis.T; rhs=bn-an@null
    fit=minimize(lambda y:.5*np.sum((y-y0)**2),y0,jac=lambda y:y-y0,
        constraints={'type':'ineq','fun':lambda y:reduced@y-rhs,'jac':lambda y:reduced},
        method='SLSQP',options={'ftol':1e-12,'maxiter':300})
    proposed=(null+basis.T@fit.x)*size
    residual=an@(proposed/size)-bn
    success=bool(fit.success and np.isfinite(proposed).all() and residual.min()>=-2e-8)
    return (proposed if success else zero),dict(success=success,nonzero=success and
        float(np.linalg.norm(proposed))>max(1e-12,size*1e-8),rank=rank,
        iterations=int(fit.nit),solver_status=int(fit.status),message=str(fit.message),
        minimum_normalized_slack=float(residual.min()),
        relative_projection_change=float(np.linalg.norm(proposed-d)/size))
