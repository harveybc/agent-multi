"""Controlled native-head weight warm start; frozen components stay exact."""
import math

import torch

from app.critic_target_contract import contract_loss
from app.head_component_recombination import combine_heads


def configure_scope(head,scope,lr,weight_decay):
    if scope not in ('output_only','full_head'):
        raise ValueError('WARM_SCOPE')
    if not math.isfinite(lr) or lr<=0 or not math.isfinite(weight_decay) or weight_decay<0:
        raise ValueError('WARM_OPTIMIZER')
    combine_heads(head.state_dict(),head.state_dict())
    names=[];parameters=[]
    for name,p in head.named_parameters():
        active=scope=='full_head' or name.split('.')[3]=='2'
        p.requires_grad_(active);p.grad=None
        if active:names.append(name);parameters.append(p)
    expected=54 if scope=='output_only' else 3398
    if sum(p.numel() for p in parameters)!=expected:raise ValueError('WARM_PARAMETER_COUNT')
    return torch.optim.AdamW(parameters,lr=lr,weight_decay=weight_decay),names


def check_frozen(head,initial,trainable):
    current=head.state_dict()
    if set(current)!=set(initial):raise ValueError('WARM_STATE_SCHEMA')
    for name,value in current.items():
        if not torch.isfinite(value).all():raise ValueError('WARM_NONFINITE')
        if name not in trainable and not torch.equal(value,initial[name].to(value.device)):
            raise ValueError('WARM_FROZEN_CHANGED')


def fit_step(head,optimizer,z,y):
    parameters=[p for p in head.parameters() if p.requires_grad]
    if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in parameters}:
        raise ValueError('WARM_OPTIMIZER_SCOPE')
    optimizer.zero_grad(set_to_none=True)
    loss=contract_loss(head(z).mean(2),y,'mse')
    if not torch.isfinite(loss):raise ValueError('WARM_LOSS_NONFINITE')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
    optimizer.step()
    return float(norm)>10
