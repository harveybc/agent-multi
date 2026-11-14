"""Bounded interior projection and transactional actual-AdamW refinement."""
from copy import deepcopy
import numpy as np
import torch

from app.native_head_warm_start import fit_step
from app.robust_adamw_direction import project_displacement


def repair_direction(delta,linearize,evaluate,margin,factor=2.,max_repairs=3):
    d=np.asarray(delta,dtype=float);margin=np.asarray(margin,dtype=float)
    norm=float(np.linalg.norm(d))
    if (d.ndim!=1 or not np.isfinite(d).all() or norm<=0 or margin.ndim!=1 or not len(margin)
            or not np.isfinite(margin).all() or (margin<=0).any()
            or not np.isfinite(factor) or factor<1 or type(max_repairs) is not int or not 1<=max_repairs<=3):
        raise ValueError('INTERIOR_CONTRACT')
    center=np.zeros_like(d);repairs=[]
    for iteration in range(max_repairs):
        gaps,jac,actual_center=linearize(center)
        required=factor*margin-gaps+jac@actual_center
        candidate,solver=project_displacement(d,jac,required)
        ratio=float(np.linalg.norm(candidate)/norm)
        entry=dict(iteration=iteration,solver=solver,norm_ratio=ratio,trials=[]);repairs.append(entry)
        if not solver['feasible'] or not np.isfinite(ratio) or ratio>4.:
            return dict(accepted=False,reason='solver_or_norm_cap',repairs=repairs,delta=None)
        for scale in [2.**(-i) for i in range(8)]:
            value=evaluate(candidate*scale)
            if type(value['passed']) is not bool or not np.isfinite(value['motion']):raise ValueError('INTERIOR_EVALUATION')
            entry['trials'].append(dict(scale=scale,**value))
            if value['passed'] and 0<value['motion']<=4.*norm:
                return dict(accepted=True,reason='finite_guard',repairs=repairs,delta=candidate*scale)
        center=candidate
    return dict(accepted=False,reason='finite_guard',repairs=repairs,delta=None)


def transactional_step(head,optimizer,z,y,repair):
    before=deepcopy(head.state_dict());old_opt=deepcopy(optimizer.state_dict())
    try:
        clipped=fit_step(head,optimizer,z,y)
        proposed=deepcopy(head.state_dict());state,record=repair(before,proposed)
        if bool(record['accepted'])!=(state is not None):raise ValueError('INTERIOR_STATE_CONTRACT')
        if state is None:
            head.load_state_dict(before);optimizer.load_state_dict(old_opt)
        else:
            if set(state)!=set(before):raise ValueError('INTERIOR_STATE_KEYS')
            for key,value in state.items():
                if value.shape!=before[key].shape or value.dtype!=before[key].dtype or not torch.isfinite(value).all():
                    raise ValueError('INTERIOR_STATE_FINITE')
            head.load_state_dict(state)
            if all(torch.equal(v,before[k]) for k,v in head.state_dict().items()):raise ValueError('INTERIOR_ZERO_MOTION')
        return dict(record,clipped=clipped)
    except BaseException:
        head.load_state_dict(before);optimizer.load_state_dict(old_opt);raise


def extension_gate(arms):
    return (len(arms)==4 and all(a.get('robust') is True and a.get('learning') is True
            and a.get('accepted',0)>=32 for a in arms)
            and sum(a.get('regret_improved') is True for a in arms)>=3)
