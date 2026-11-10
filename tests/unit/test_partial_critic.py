from types import SimpleNamespace

import pytest
import torch

from app.partial_critic import scope_map, set_partial_scope, module_name, restore_component, weight_drift


def fake_model():
    params={}
    for b in (0,1):
        for conv in ('first','second'):
            for kind in ('weight','bias'):
                params[f'qvalue_network_params.module.encoder.core.{b}.{conv}.conv.{kind}']=torch.nn.Parameter(torch.ones(2))
    for b in (0,2):
        for kind in ('weight','bias'):
            params[f'qvalue_network_params.module.head.{b}.{kind}']=torch.nn.Parameter(torch.ones(2))
    for n in ('actor_network_params.module.head.0.weight','target_qvalue_network_params.module.head.0.weight',
              'qvalue_network_params.module.encoder.branches.0.0.conv.weight','log_alpha'):
        params[n]=torch.nn.Parameter(torch.ones(2))
    return SimpleNamespace(loss=SimpleNamespace(named_parameters=lambda:params.items()),
        effective={'architecture':{'core_blocks':2},'learning_rate':3e-5},
        optimizer=SimpleNamespace(param_groups=[{'lr':1.}]))


def test_exact_last_core_scope_and_fail_closed():
    m=fake_model();mapping=set_partial_scope(m)
    assert len(mapping)==8
    assert {n for n,p in m.loss.named_parameters() if p.requires_grad}==set(mapping)
    assert m.optimizer.param_groups[0]['lr']==3e-5
    m.effective['architecture']['core_blocks']=3
    with pytest.raises(ValueError,match='ARCHITECTURE'):scope_map(m)
    m=fake_model();original=m.loss.named_parameters
    m.loss.named_parameters=lambda:[(n,p) for n,p in original() if '.core.1.second.conv.bias' not in n]
    with pytest.raises(ValueError,match='SCOPE'):scope_map(m)


def test_component_mapping_and_restore_roundtrip():
    m=fake_model();params=dict(m.loss.named_parameters());before={n:p.detach().clone() for n,p in params.items()}
    initial={n:torch.zeros_like(p) for n,p in params.items()}
    assert module_name('qvalue_network_params.module.encoder.branches.12.1.first.conv.bias')=='branch_12'
    assert module_name('actor_network_params.module.head.0.weight') is None
    with pytest.raises(ValueError):module_name('qvalue_network_params.module.unknown.weight')
    restore_component(m,initial,'core_1')
    assert all(torch.equal(p,initial[n] if module_name(n)=='core_1' else before[n]) for n,p in params.items())
    restore_component(m,before,'core_1')
    assert all(torch.equal(p,before[n]) for n,p in params.items())
    with pytest.raises(ValueError):restore_component(m,{},'core_1')
    with pytest.raises(ValueError):restore_component(m,initial,'missing')
    drift=weight_drift(initial,before)
    assert drift['core_1']['delta_l2']>0 and drift['core_1']['relative_l2'] is None
    assert drift['head']['tensors']==4
