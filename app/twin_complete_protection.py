"""Sufficient smooth inner constraints for min-of-twin decision protection."""
import numpy as np
import torch


def twin_complete_gaps(q,actions):
    actions=np.asarray(actions)
    if (q.ndim!=3 or q.shape[0]!=2 or q.shape[2]!=3 or not q.shape[1]
            or actions.shape!=(q.shape[1],) or actions.dtype.kind not in 'iu'
            or ((actions<0)|(actions>2)).any() or not torch.isfinite(q).all()):
        raise ValueError('TWIN_COMPLETE_CONTRACT')
    rows=torch.arange(len(actions),device=q.device)
    alternatives=torch.tensor([[b for b in range(3) if b!=a] for a in actions],device=q.device)
    chosen=q[:,rows,torch.tensor(actions,device=q.device)].T
    # Alternative minimum fixes a branch at this center; BOTH chosen twins are constrained.
    other=q.min(0).values.gather(1,alternatives)
    return (chosen[:,None,:]-other[:,:,None]).reshape(-1)


def twin_complete_jacobian(head,z,old,actions):
    gaps=twin_complete_gaps(head(z[:,:239]).mean(2)[:,old],actions)
    parameters=list(head.parameters());rows=[]
    for gap in gaps:
        grad=torch.autograd.grad(gap,parameters,retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in grad]).detach().cpu().numpy())
    return gaps.detach().cpu().numpy(),np.stack(rows)


def qualify_twin_continuation(rows):
    return len(rows)==16 and sum(r['accepted'] and r['robust'] and r['geometry_agrees']
        and np.isfinite([r['before'],r['after'],r['cost_before'],r['cost_after']]).all()
        and r['after']<=r['before'] and r['cost_after']<r['cost_before'] for r in rows)>=12
