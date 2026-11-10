"""Frozen donor-assisted restoration; never silently discard historical candidates."""
import copy
import math

import torch
from torch import nn


def cumulative_pass(universe, blocks, initial, evaluate, progress=None):
    if len(universe)!=len(set(universe)) or sorted(sum(blocks,[]))!=sorted(universe):
        raise ValueError('CUMULATIVE_UNIVERSE')
    if len(initial)!=len(set(initial)) or not set(initial)<=set(universe):
        raise ValueError('CUMULATIVE_INITIAL')
    if any(set(b)&set(initial) and not set(b)<=set(initial) for b in blocks):
        raise ValueError('CUMULATIVE_PARTIAL_BLOCK')
    current=[c for c in universe if c in initial]; cache={};trace=[];after=None
    def measure(cols):
        key=tuple(cols)
        if key not in cache:
            value=evaluate(cols)
            if not math.isfinite(value['score']) or type(value['eligible']) is not bool:
                raise ValueError('CUMULATIVE_SCORE')
            cache[key]=value
        return cache[key]
    if not measure(current)['eligible']: raise ValueError('CUMULATIVE_INACTIVE_BASELINE')
    def snapshot(stage):
        result=dict(selected=list(current),trace=list(trace),stage=stage,
            counts=dict(initial=len(initial),after_deletion=after,final=len(current),universe=len(universe)),
            evaluations=len(cache),globally_optimal=False,
            termination='TWO_PASSES_COMPLETE' if stage=='COMPLETE' else None)
        if progress: progress(result)
        return result
    for phase in ('DELETE','REINSERT'):
        candidates=[b for b in blocks if (set(b)<=set(current) if phase=='DELETE' else not set(b)&set(current))]
        for block in candidates:
            before=list(current)
            wanted=set(current)-set(block) if phase=='DELETE' else set(current)|set(block)
            if not wanted: continue
            candidate=[c for c in universe if c in wanted]
            old,new=measure(current),measure(candidate)
            accept=new['eligible'] and (new['score']>old['score'] or
                    (phase=='DELETE' and new['score']==old['score']))
            if accept: current=candidate
            trace.append(dict(phase=phase,block=block,before=before,candidate=candidate,
                after=list(current),accepted=accept,before_score=old['score'],
                candidate_score=new['score'],candidate_eligible=new['eligible']))
            snapshot(phase)
        if phase=='DELETE': after=len(current);snapshot('DELETION_COMPLETE')
    return snapshot('COMPLETE')


def scale_table(recipe):
    s=recipe['scaler']; names=s['features']
    if len(names)!=len(s['mean']) or len(names)!=len(s['std']) or len(set(names))!=len(names):
        raise ValueError('RESTORE_SCALER')
    values={c:(float(m),float(v)) for c,m,v in zip(names,s['mean'],s['std'])}
    if any(not math.isfinite(m) or not math.isfinite(v) or v<=0 for m,v in values.values()):
        raise ValueError('RESTORE_SCALER')
    return values


class CenteredDonor(nn.Module):
    """Retain donor signal response without importing its neutral-group offset."""
    def __init__(self,branch,ratios,offsets,observations):
        super().__init__();self.branch=copy.deepcopy(branch).eval()
        weight=next(branch.parameters())
        self.register_buffer('ratios',weight.new_tensor(ratios).view(1,-1,1))
        self.register_buffer('offsets',weight.new_tensor(offsets).view(1,-1,1))
        with torch.no_grad():
            zero=weight.new_zeros(1,len(ratios),observations)
            self.register_buffer('neutral',self.branch(zero+self.offsets).detach().clone())
    def forward(self,x):
        return self.branch(x*self.ratios+self.offsets)-self.neutral


def expand_actor(base,base_recipe,universe,donors):
    """Only removed connections/groups come from donors; common weights stay fixed."""
    names=universe['list']['columns'];active=base_recipe['list']['columns']
    if not set(active)<=set(names): raise ValueError('RESTORE_UNIVERSE')
    us,bs=scale_table(universe),scale_table(base_recipe)
    if any(us[c]!=bs[c] for c in active): raise ValueError('RESTORE_ACTIVE_SCALER')
    ug=universe['representation']['resolved']; bg=base_recipe['representation']['resolved']
    group_columns={g:[names[j] for j in ix] for g,ix in zip(ug['names'],ug['indices'])}
    if getattr(base.encoder,'context',None) is not None: raise ValueError('RESTORE_LONG_CONTEXT')
    result=copy.deepcopy(base).eval().requires_grad_(False);enc=result.encoder
    originals=copy.deepcopy(base.encoder.fusion.conv)
    branch_width=originals.in_channels//len(bg['names'])
    if branch_width*len(bg['names'])!=originals.in_channels or originals.kernel_size!=(1,):
        raise ValueError('RESTORE_FUSION')
    records=[];extra=[];order=list(bg['names'])
    def donor_for(group,columns):
        for actor,recipe,label in donors:
            dg=recipe['representation']['resolved']; dn=recipe['list']['columns']
            if group not in dg['names']: continue
            i=dg['names'].index(group);cols=[dn[j] for j in dg['indices'][i]]
            if not set(columns)<=set(cols): continue
            if dg['kinds'][i]!=ug['kinds'][ug['names'].index(group)]:
                continue
            return actor,recipe,label,i,cols
        raise ValueError('RESTORE_DONOR_MISSING:'+group+':'+','.join(columns))
    with torch.no_grad():
        for i,g in enumerate(bg['names']):
            if g not in group_columns: raise ValueError('RESTORE_GROUP')
            oldcols=[active[j] for j in bg['indices'][i]]; cols=group_columns[g]
            first=enc.branches[i][0].conv;oldweight=first.weight.detach().clone()
            if first.kernel_size!=(1,) or first.groups!=1: raise ValueError('RESTORE_INPUT_CONV')
            weight=oldweight.new_empty(oldweight.shape[0],len(cols),1)
            for j,c in enumerate(cols):
                if c in oldcols:
                    weight[:,j]=oldweight[:,oldcols.index(c)]
                else:
                    actor,recipe,label,di,dcols=donor_for(g,[c]); ds=scale_table(recipe)
                    value=actor.encoder.branches[di][0].conv.weight[:,dcols.index(c)]
                    if value.shape!=weight[:,j].shape or not torch.isfinite(value).all() or not value.count_nonzero():
                        raise ValueError('RESTORE_DONOR_WEIGHT:'+c)
                    ratio=us[c][1]/ds[c][1];weight[:,j]=value*ratio
                    records.append(dict(column=c,group=g,donor=label,std_ratio=ratio,mode='input_weight'))
            first.weight=nn.Parameter(weight,requires_grad=False);first.in_channels=len(cols)
            setattr(enc,f'indices_{i}',torch.tensor([names.index(c) for c in cols],device=weight.device))
        for g in ug['names']:
            if g in bg['names']: continue
            cols=group_columns[g]
            actor,recipe,label,di,dcols=donor_for(g,cols)
            if set(cols)!=set(dcols): raise ValueError('RESTORE_DONOR_GROUP_COLUMNS')
            # Donor order is authoritative inside this restored branch.
            ds=scale_table(recipe);ratios=[us[c][1]/ds[c][1] for c in dcols]
            offsets=[(us[c][0]-ds[c][0])/ds[c][1] for c in dcols]
            branch=actor.encoder.branches[di]
            if branch[0].conv.out_channels!=branch_width: raise ValueError('RESTORE_DONOR_WIDTH')
            enc.branches.append(CenteredDonor(branch,ratios,offsets,base.encoder.shape[0]))
            enc.register_buffer(f'indices_{len(order)}',torch.tensor([names.index(c) for c in dcols],
                device=originals.weight.device));order.append(g)
            value=actor.encoder.fusion.conv.weight[:,di*branch_width:(di+1)*branch_width]
            if value.shape!=(originals.out_channels,branch_width,1) or not torch.isfinite(value).all() or not value.count_nonzero():
                raise ValueError('RESTORE_DONOR_FUSION')
            extra.append(value.detach().clone())
            records.extend(dict(column=c,group=g,donor=label,std_ratio=ratios[j],
                mean_offset=offsets[j],mode='centered_group') for j,c in enumerate(dcols))
        if extra:
            enc.fusion.conv.weight=nn.Parameter(torch.cat([originals.weight,*extra],1),requires_grad=False)
            enc.fusion.conv.in_channels=enc.fusion.conv.weight.shape[1]
        enc.shape=(enc.shape[0],len(names));enc.transport_shape=enc.shape
    if sorted(x['column'] for x in records)!=sorted(set(names)-set(active)):
        raise ValueError('RESTORE_INVENTORY')
    for p in result.parameters():
        if not torch.isfinite(p).all(): raise ValueError('RESTORE_NONFINITE')
    return result.eval().requires_grad_(False),dict(restored_columns=[c for c in names if c not in active],
        records=records,group_order=order,new_groups=order[len(bg['names']):],
        scope='DONOR_ASSISTED_FROZEN_READMISSION; no joint refit or proof of feature irrelevance')
