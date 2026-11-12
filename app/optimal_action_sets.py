"""Versioned FIT correction for equivalent native actions, including HOLD aliases."""
import torch


def optimal_sets(logits, native, records):
    if (logits.shape!=native.shape or logits.shape!=(len(records),3)
            or not torch.isfinite(logits).all() or not torch.isfinite(native).all()
            or any(r['role'] not in ('fit','monitor') for r in records)):
        raise ValueError('OPTIMAL_SET_INPUT')
    return (native.double()>=native.double().max(-1,keepdim=True).values-1e-8).detach()


class SetGuard:
    def __init__(self, logits, native, records):
        self.optimal=optimal_sets(logits,native,records)
        source=logits.argmax(-1)
        fit=torch.tensor([r['role']=='fit' for r in records],device=logits.device)
        good=self.optimal.gather(1,source[:,None]).squeeze(1)
        self.rows=torch.nonzero(fit & good & ~self.optimal.all(-1)).flatten()
        if not len(self.rows):raise ValueError('OPTIMAL_SET_EMPTY_GUARD')
        self.actions=source[self.rows]
        local,bad=torch.nonzero(~self.optimal[self.rows],as_tuple=True)
        self.constraint_rows=self.rows[local]
        self.preferred=self.actions[local]
        self.bad=bad
        self.floor=self.margins(logits.double()).detach()
        z=logits.detach().double()
        self.buffer=8*torch.finfo(logits.dtype).eps*(1+
            z[self.constraint_rows,self.preferred].abs()+z[self.constraint_rows,self.bad].abs())

    def margins(self, logits):
        return logits[self.constraint_rows,self.preferred]-logits[self.constraint_rows,self.bad]

    def check(self, logits, tolerance=1e-7):
        delta=self.margins(logits.double())-self.floor
        chosen=logits[self.rows].argmax(-1)
        incorrect=int((~self.optimal[self.rows].gather(1,chosen[:,None])).sum())
        finite=bool(torch.isfinite(logits[self.rows]).all())
        return dict(passed=finite and incorrect==0 and bool((delta>=-tolerance).all()),
            minimum_margin_change=float(delta.min()),action_changes=int((chosen!=self.actions).sum()),
            suboptimal_actions=incorrect,violations=int((delta< -tolerance).sum()),protected_rows=len(self.rows))


class SourceValueGuard(SetGuard):
    """Freeze a native FIT value floor even for initially incorrect decisions."""
    def __init__(self, logits, native, records):
        self.optimal=optimal_sets(logits,native,records)
        source=logits.argmax(-1)
        self.fit=torch.tensor([r['role']=='fit' for r in records],device=logits.device)
        self.native=native.detach().double().clone()
        self.source_values=self.native.gather(1,source[:,None]).squeeze(1).clone()
        self.allowed=self.native>=self.source_values[:,None]-1e-8
        self.rows=torch.nonzero(self.fit & ~self.allowed.all(-1)).flatten()
        if not len(self.rows):raise ValueError('SOURCE_VALUE_EMPTY_GUARD')
        self.actions=source[self.rows]
        local,bad=torch.nonzero(~self.allowed[self.rows],as_tuple=True)
        self.constraint_rows=self.rows[local]
        self.preferred=self.actions[local]
        self.bad=bad
        self.floor=self.margins(logits.double()).detach()
        z=logits.detach().double()
        self.buffer=8*torch.finfo(logits.dtype).eps*(1+
            z[self.constraint_rows,self.preferred].abs()+z[self.constraint_rows,self.bad].abs())

    def check(self, logits, tolerance=1e-7):
        delta=self.margins(logits.double())-self.floor
        chosen=logits.argmax(-1)
        reward_delta=self.native.gather(1,chosen[:,None]).squeeze(1)-self.source_values
        regressions=int((self.fit & (reward_delta< -1e-8)).sum())
        suboptimal=int((self.fit & ~self.optimal.gather(1,chosen[:,None]).squeeze(1)).sum())
        finite=bool(torch.isfinite(logits[self.fit]).all())
        return dict(passed=finite and regressions==0 and bool((delta>=-tolerance).all()),
            minimum_margin_change=float(delta.min()),
            action_changes=int((chosen[self.rows]!=self.actions).sum()),
            suboptimal_actions=suboptimal,source_value_regressions=regressions,
            minimum_native_value_change=float(reward_delta[self.fit].min()),
            violations=int((delta< -tolerance).sum()),protected_rows=len(self.rows),
            native_checked_fit_rows=int(self.fit.sum()))


def error_targets(logits, native, records):
    mask=optimal_sets(logits,native,records)
    chosen=logits.argmax(-1)
    fit=torch.tensor([r['role']=='fit' for r in records],device=logits.device)
    keep=fit & ~mask.gather(1,chosen[:,None]).squeeze(1)
    regret=native.double().max(-1).values-native.double().gather(1,chosen[:,None]).squeeze(1)
    return mask,torch.where(keep,regret,torch.zeros_like(regret)).detach()


def error_batch(targets, weights, indices):
    if (targets.dtype!=torch.bool or targets.shape!=(len(weights),3)
            or not targets.any(-1).all() or not torch.isfinite(weights).all()
            or (weights<0).any() or (targets.all(-1)&(weights>0)).any()):
        raise ValueError('OPTIMAL_SET_TARGET')
    selected=indices[weights[indices]>0]
    if not len(selected):raise ValueError('OPTIMAL_SET_EMPTY_BATCH')
    return selected


def correction_loss(logits, targets, weights, indices):
    selected=error_batch(targets,weights,indices)
    z=logits[selected]
    logmass=z.masked_fill(~targets[selected],-torch.inf).logsumexp(-1)-z.logsumexp(-1)
    return -(weights[selected]*logmass).sum()/weights[selected].sum()


def set_margin(logits, rows, mask):
    z=logits[rows].double()
    return z.masked_fill(~mask,-torch.inf).logsumexp(-1)-z.masked_fill(mask,-torch.inf).logsumexp(-1)


def requested_margins(logits, targets, weights, indices):
    selected=error_batch(targets,weights,indices)
    rows,count=torch.unique(selected,sorted=True,return_counts=True)
    mass=weights[rows]*count;mass=mass/mass.sum()
    z=logits[rows].double();mask=targets[rows]
    good=z.masked_fill(~mask,-torch.inf).softmax(-1)
    bad=z.masked_fill(mask,-torch.inf).softmax(-1)
    gradient=(z.softmax(-1)-good)*mass[:,None]
    return rows,(-(good-bad)*gradient).sum(-1)


def error_progress(source, after, native, records):
    mask=optimal_sets(source,native,records)
    chosen=source.argmax(-1);out={}
    for role in ('fit','monitor'):
        eligible=torch.tensor([r['role']==role for r in records],device=source.device)
        rows=torch.nonzero(eligible & ~mask.gather(1,chosen[:,None]).squeeze(1)).flatten()
        if not len(rows):out[role]=dict(rows=0);continue
        p0=(source[rows].double().softmax(-1)*mask[rows]).sum(-1)
        p1=(after[rows].double().softmax(-1)*mask[rows]).sum(-1)
        delta=set_margin(after,rows,mask[rows])-set_margin(source,rows,mask[rows])
        out[role]=dict(rows=len(rows),corrected=int(mask[rows].gather(1,after[rows].argmax(-1)[:,None]).sum()),
            increased_preferred_probability=int(((p1-p0)>1e-10).sum()),
            decreased_preferred_probability=int(((p1-p0)< -1e-10).sum()),
            mean_preferred_probability_change=float((p1-p0).mean()),
            improved_target_vs_source_margin=int((delta>1e-10).sum()),
            mean_target_vs_source_margin_change=float(delta.mean()))
    return out
