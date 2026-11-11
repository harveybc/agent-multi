"""Fixed own-policy FIT errors and advantage-weighted corrective actor objective."""
import torch


def error_targets(logits,native,records):
    if (logits.shape!=native.shape or logits.shape!=(len(records),3)
            or not torch.isfinite(logits).all() or not torch.isfinite(native).all()):
        raise ValueError('FIT_ERROR_INPUT')
    source=logits.argmax(-1);best=native.argmax(-1);top=native.topk(2,dim=-1).values
    fit=torch.tensor([r['role']=='fit' for r in records],device=logits.device)
    keep=fit & (source!=best) & ((top[:,0]-top[:,1])>1e-8)
    targets=torch.full_like(best,-1);targets[keep]=best[keep]
    weights=torch.zeros(len(logits),device=logits.device,dtype=torch.double)
    advantage=native.gather(1,best[:,None]).squeeze(1)-native.gather(1,source[:,None]).squeeze(1)
    weights[keep]=advantage[keep].double()
    return targets,weights


def error_batch(targets,weights,indices):
    if (targets.shape!=weights.shape or not torch.isfinite(weights).all() or (weights<0).any()
            or ((weights>0)&((targets<0)|(targets>2))).any()):
        raise ValueError('FIT_ERROR_TARGET')
    selected=indices[weights[indices]>0]
    if not len(selected):raise ValueError('FIT_ERROR_EMPTY_BATCH')
    return selected


def correction_loss(logits,targets,weights,indices):
    selected=error_batch(targets,weights,indices)
    logp=logits[selected].log_softmax(-1)
    return -(weights[selected]*logp.gather(1,targets[selected,None]).squeeze(1)).sum()/weights[selected].sum()


def requested_margins(logits,targets,weights,indices):
    selected=error_batch(targets,weights,indices)
    unique,count=torch.unique(selected,sorted=True,return_counts=True)
    mass=weights[unique]*count;mass=mass/mass.sum()
    grad=logits[unique].double().softmax(-1)
    grad[torch.arange(len(unique),device=logits.device),targets[unique]]-=1
    grad*=mass[:,None]
    others=torch.tensor([[b for b in range(3) if b!=int(a)] for a in targets[unique]],device=logits.device)
    desired=-(grad[torch.arange(len(unique),device=logits.device),targets[unique],None]-grad.gather(1,others))
    return unique,desired


def error_progress(source,after,native,records):
    out={}
    source=source.double();after=after.double();native=native.double()
    chosen=source.argmax(-1);best=native.argmax(-1);top=native.topk(2,dim=-1).values
    unique=(top[:,0]-top[:,1])>1e-8
    for role in ('fit','monitor'):
        mask=torch.tensor([r['role']==role for r in records],device=source.device)&unique&(chosen!=best)
        ix=torch.nonzero(mask).flatten()
        if not len(ix):out[role]=dict(rows=0);continue
        p0=source[ix].softmax(-1).gather(1,best[ix,None]).squeeze(1)
        p1=after[ix].softmax(-1).gather(1,best[ix,None]).squeeze(1)
        delta=(after[ix,best[ix]]-after[ix,chosen[ix]])-(source[ix,best[ix]]-source[ix,chosen[ix]])
        out[role]=dict(rows=len(ix),corrected=int((after[ix].argmax(-1)==best[ix]).sum()),
            increased_preferred_probability=int(((p1-p0)>1e-10).sum()),
            decreased_preferred_probability=int(((p1-p0)<-1e-10).sum()),
            mean_preferred_probability_change=float((p1-p0).mean()),
            improved_target_vs_source_margin=int((delta>1e-10).sum()),
            mean_target_vs_source_margin_change=float(delta.mean()))
    return out
