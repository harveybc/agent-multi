"""Descriptive tensor and frozen-panel statistics, never a training decision."""
import numpy as np
import torch

from app.sac_training_reliability import module_key


def selected_epochs(history):
    if not history:return []
    values=[int(row['epoch']) for row in history]
    return sorted({values[0],values[-1],*(e for e in values if e%20==0)})


def matrix_ranks(tensor):
    if tensor.ndim<2:return []
    matrices=tensor.detach().double().reshape(-1,*tensor.shape[-2:])
    singular=torch.linalg.svdvals(matrices)
    energy=singular.square().sum(-1);largest=singular[:,0].square()
    return torch.where(largest>0,energy/largest.clamp_min(1e-300),0.).tolist()


def component_stats(model,anchor):
    groups={}
    with torch.no_grad():
        for name,p in model.loss.named_parameters():
            row=groups.setdefault(module_key(name),dict(norm2=0.,delta2=0.,anchor2=0.,ranks=[]))
            row['norm2']+=float(p.double().square().sum())
            row['delta2']+=float((p.double()-anchor[name].double()).square().sum())
            row['anchor2']+=float(anchor[name].double().square().sum())
            row['ranks'].extend(matrix_ranks(p))
    for row in groups.values():
        ranks=row.pop('ranks')
        row['mean_matrix_stable_rank']=float(np.mean(ranks)) if ranks else None
        row['matrices']=len(ranks)
        row['relative_drift']=(row['delta2']/max(row['anchor2'],1e-30))**.5
    return groups


def policy_stats(logits,q,reference_actions):
    with torch.no_grad():
        probabilities=logits.softmax(-1);top=probabilities.topk(2,dim=-1).values
        actions=logits.argmax(-1);critic=q.min(0).values.argmax(-1)
        entropy=-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1)
        margin=top[:,0]-top[:,1]
    return dict(action_flip_fraction=float((actions!=reference_actions).float().mean()),
        probability_margin_median=float(margin.median()),near_tie_fraction=float((margin<.01).float().mean()),
        entropy_mean=float(entropy.mean()),critic_actor_disagreement=float((critic!=actions).float().mean()),
        twin_gap_mean=float((q[0]-q[1]).abs().mean()),q_abs_mean=float(q.abs().mean()),
        signals={str(i):int((actions==i).sum()) for i in range(3)})
