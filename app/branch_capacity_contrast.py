"""Paired branch reductions with named, exact retained SAC/Adam state transfer."""
from copy import deepcopy

import torch

from app.structural_refit import build_model
from app.training_forensics import tree_digest


def reduce_model(source, arm, seed):
    if arm not in ('control', 'A', 'D'):
        raise ValueError('CAPACITY_ARM')
    effective = deepcopy(source.effective)
    effective['learning_rate'] = 3e-6
    architecture = effective['architecture']
    if architecture.get('group_configs') is not None:
        raise ValueError('CAPACITY_EXPECTED_UNIFORM_SOURCE_BRANCHES')
    if architecture['branch_blocks'] != 2 or architecture['core_blocks'] != 2:
        raise ValueError('CAPACITY_SOURCE_DEPTH')
    if arm == 'A':
        architecture['branch_blocks'] = 1
    elif arm == 'D':
        architecture['group_configs'] = [dict(kind='dense', blocks=0, kernel=1, dilations=[],
            width=architecture['branch_width'], activation='silu')
            for _ in source.recipe['representation']['resolved']['indices']]
    model = build_model(source.recipe, effective, seed, str(source.device))
    old, current = source.loss.state_dict(), model.loss.state_dict()
    for name, value in current.items():
        if name not in old or old[name].shape != value.shape:
            raise ValueError('CAPACITY_RETAINED_TENSOR:' + name)
    removed = sorted(set(old)-set(current))
    if any('.encoder.branches.' not in n for n in removed):
        raise ValueError('CAPACITY_NON_BRANCH_REMOVAL')
    model.loss.load_state_dict({n: old[n] for n in current}, strict=True)
    old_params = dict(source.loss.named_parameters())
    new_params = dict(model.loss.named_parameters())
    # Parameter order changes when blocks disappear; matching integer Adam IDs is unsafe.
    if len(source.optimizer.param_groups) != 1:
        raise ValueError('CAPACITY_SOURCE_OPTIMIZER_LAYOUT')
    options = {k: deepcopy(v) for k, v in source.optimizer.param_groups[0].items() if k != 'params'}
    options['lr'] = 3e-6
    model.optimizer = torch.optim.AdamW([dict(params=list(new_params.values()), **options)])
    for name, p in new_params.items():
        old_p = old_params[name]
        if old_p in source.optimizer.state:
            model.optimizer.state[p] = deepcopy(source.optimizer.state[old_p])
    exact = all(torch.equal(v, old[n]) for n, v in model.loss.state_dict().items())
    moments = all(tree_digest(model.optimizer.state.get(p, {})) ==
        tree_digest(source.optimizer.state.get(old_params[n], {})) for n, p in new_params.items())
    if not exact or not moments:
        raise ValueError('CAPACITY_TRANSFER_PARITY')
    receipt = dict(arm=arm, retained_tensors=len(current), removed_tensors=removed,
        all_retained_tensors_exact=exact, optimizer_moments_exact=moments,
        retained_state_sha256=tree_digest(model.loss.state_dict()),
        actor_parameters=sum(p.numel() for p in model.actor_net.parameters()),
        trainable_parameters=sum(p.numel() for p in model.loss.parameters() if p.requires_grad),
        resolved=model.actor_net.encoder.resolved_architecture,
        optimizer_lr=3e-6, source_state_sha256=tree_digest(old))
    return model, receipt
