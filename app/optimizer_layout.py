"""Persist optimizer group membership by parameter name, not transient IDs."""
import torch


def optimizer_layout(model):
    names={id(p):n for n,p in model.loss.named_parameters()}
    return [[names[id(p)] for p in g['params']] for g in model.optimizer.param_groups]


def restore_optimizer_layout(model,layout,state):
    params=dict(model.loss.named_parameters())
    expected={id(p) for g in model.optimizer.param_groups for p in g['params']}
    if not isinstance(layout,list) or len(layout)!=len(state['param_groups']):
        raise ValueError('OPTIMIZER_LAYOUT_GROUPS')
    if any(not isinstance(g,list) or not g or any(not isinstance(n,str) for n in g) for g in layout):
        raise ValueError('OPTIMIZER_LAYOUT_NAMES')
    flat=[n for group in layout for n in group]
    if len(flat)!=len(set(flat)) or any(n not in params for n in flat):
        raise ValueError('OPTIMIZER_LAYOUT_NAMES')
    if {id(params[n]) for n in flat}!=expected:
        raise ValueError('OPTIMIZER_LAYOUT_COVERAGE')
    if any(len(group)!=len(saved['params']) for group,saved in zip(layout,state['param_groups'])):
        raise ValueError('OPTIMIZER_LAYOUT_LENGTH')
    model.optimizer=torch.optim.AdamW([{'params':[params[n] for n in group]} for group in layout])
    model.optimizer.load_state_dict(state)
