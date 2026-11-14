from copy import deepcopy

import pytest
import torch
from torch import nn

from app.native_head_warm_start import configure_scope,check_frozen,fit_step
from app.role_routed_critic import RoleHeads


def head():
    return RoleHeads(nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,3)),1,19)


@pytest.mark.parametrize('scope,count',[('output_only',54),('full_head',3398)])
def test_scope_updates_and_preserves_initial_weights(scope,count):
    h=head();before=deepcopy(h.state_dict())
    opt,names=configure_scope(h,scope,3e-5,.01)
    assert all(torch.equal(v,before[k]) for k,v in h.state_dict().items())
    assert sum(p.numel() for p in h.parameters() if p.requires_grad)==count
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in h.parameters() if p.requires_grad}
    assert not opt.state
    z=torch.randn(2,32,208);y=torch.randn(32,3)
    for _ in range(4):fit_step(h,opt,z,y)
    check_frozen(h,before,names)
    for k,v in h.state_dict().items():
        if k in names:assert not torch.equal(v,before[k])
        else:assert torch.equal(v,before[k])


@pytest.mark.parametrize('scope',['output_only','full_head'])
def test_restart_exact(scope):
    torch.manual_seed(8);z=torch.randn(2,64,208);y=torch.randn(64,3)
    h=head();opt,names=configure_scope(h,scope,3e-5,.01)
    for i in range(6):fit_step(h,opt,z[:,i:i+32],y[i:i+32])
    state=deepcopy(h.state_dict());optimizer=deepcopy(opt.state_dict())
    resumed=head();resumed.load_state_dict(state)
    ropt,rnames=configure_scope(resumed,scope,3e-5,.01);ropt.load_state_dict(optimizer)
    assert rnames==names
    for i in range(6,12):
        assert fit_step(h,opt,z[:,i:i+32],y[i:i+32])==fit_step(resumed,ropt,z[:,i:i+32],y[i:i+32])
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in h.state_dict().items())


def test_invalid_scope_schema_and_frozen_mutation_rejected():
    h=head()
    with pytest.raises(ValueError):configure_scope(h,'unknown',3e-5,.01)
    before=deepcopy(h.state_dict());_,names=configure_scope(h,'output_only',3e-5,.01)
    with torch.no_grad():h.twins[0][0][0].bias.add_(1)
    with pytest.raises(ValueError):check_frozen(h,before,names)
    h=head();h.twins[0][0][0]=nn.Linear(207,8)
    with pytest.raises(ValueError):configure_scope(h,'full_head',3e-5,.01)


def test_nonfinite_update_rejected_before_mutation():
    h=head();opt,_=configure_scope(h,'full_head',3e-5,.01);before=deepcopy(h.state_dict())
    with pytest.raises((ValueError,RuntimeError)):
        fit_step(h,opt,torch.ones(2,8,208),torch.full((8,3),float('nan')))
    assert all(torch.equal(v,before[k]) for k,v in h.state_dict().items())
