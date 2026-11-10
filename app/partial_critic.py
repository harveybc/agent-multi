"""Pinned last-core critic trainability and isolated weight interventions."""
from collections import defaultdict

import torch

PREFIX='qvalue_network_params.module.'
EXPECTED={PREFIX+f'encoder.core.1.{conv}.conv.{kind}' for conv in ('first','second') for kind in ('weight','bias')}
EXPECTED|={PREFIX+f'head.{layer}.{kind}' for layer in (0,2) for kind in ('weight','bias')}


def module_name(name):
    if not name.startswith(PREFIX):return None
    part=name[len(PREFIX):].split('.')
    if part[0]=='head':return 'head'
    if part[:2]==['encoder','branches']:return 'branch_'+part[2]
    if part[:2]==['encoder','core']:return 'core_'+part[2]
    if part[:2]==['encoder','fusion']:return 'fusion'
    if part[:2]==['encoder','post_fusion']:return 'attention'
    raise ValueError('PARTIAL_UNKNOWN_MODULE:'+name)


def scope_map(model):
    if model.effective['architecture']['core_blocks']!=2:
        raise ValueError('PARTIAL_ARCHITECTURE')
    params=dict(model.loss.named_parameters())
    selected={n for n in params if n.startswith(PREFIX+'encoder.core.1.') or n.startswith(PREFIX+'head.')}
    core_ids={n.split('.core.')[1].split('.')[0] for n in params if n.startswith(PREFIX+'encoder.core.')}
    if selected!=EXPECTED or core_ids!={'0','1'}:raise ValueError('PARTIAL_SCOPE')
    return {n:dict(shape=list(params[n].shape),elements=params[n].numel()) for n in sorted(selected)}


def set_partial_scope(model):
    mapping=scope_map(model)
    for n,p in model.loss.named_parameters():p.requires_grad_(n in mapping)
    for group in model.optimizer.param_groups:group['lr']=model.effective['learning_rate']
    return mapping


def restore_component(model,values,component):
    params={n:p for n,p in model.loss.named_parameters() if module_name(n)==component}
    if not params or any(n not in values or values[n].shape!=p.shape for n,p in params.items()):
        raise ValueError('PARTIAL_RESTORE_COMPONENT')
    with torch.no_grad():
        for n,p in params.items():p.copy_(values[n])


def weight_drift(initial,current):
    groups=defaultdict(lambda:dict(tensors=0,elements=0,initial_sq=0.,delta_sq=0.))
    if set(initial)!=set(current):raise ValueError('PARTIAL_DRIFT_KEYS')
    for n,p in current.items():
        component=module_name(n)
        if component is None:continue
        old=initial[n].detach().double().cpu();value=p.detach().double().cpu()
        if old.shape!=value.shape or not torch.isfinite(value).all() or not torch.isfinite(old).all():
            raise ValueError('PARTIAL_DRIFT_SHAPE_OR_FINITE')
        g=groups[component];g['tensors']+=1;g['elements']+=p.numel()
        g['initial_sq']+=float(old.square().sum());g['delta_sq']+=float((value-old).square().sum())
    return {k:dict(tensors=g['tensors'],elements=g['elements'],initial_l2=g['initial_sq']**.5,
        delta_l2=g['delta_sq']**.5,relative_l2=(g['delta_sq']/g['initial_sq'])**.5 if g['initial_sq'] else None)
        for k,g in groups.items()}
