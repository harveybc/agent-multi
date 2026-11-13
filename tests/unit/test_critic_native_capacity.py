import pytest
import torch
from torch import nn

from app.critic_native_capacity import native_from_ensemble
from app.role_routed_critic import RoleHeads


def test_native_heads_match_both_twins_without_rng_shift_or_aliasing():
    template=nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,3))
    ensemble=RoleHeads(template,3,19);native=native_from_ensemble(ensemble)
    assert sum(p.numel() for p in ensemble.parameters())==10194
    assert sum(p.numel() for p in native.parameters())==3398
    z=torch.randn(2,7,208)
    assert torch.equal(native(z).squeeze(2),ensemble(z)[:,:,0])
    for a,b in zip(native.twins,ensemble.twins):
        for p,q in zip(a[0].parameters(),b[0].parameters()):
            assert torch.equal(p,q) and p.data_ptr()!=q.data_ptr()
    native(z).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in native.parameters())
    assert all(p.grad is None for p in ensemble.parameters())
    with pytest.raises(ValueError):native_from_ensemble(native)


def test_naive_count_one_seed_would_change_twin_one():
    template=nn.Sequential(nn.Linear(5,8),nn.SiLU(),nn.Linear(8,3))
    ensemble=RoleHeads(template,3,43);naive=RoleHeads(template,1,43)
    assert torch.equal(naive.twins[0][0][0].weight,ensemble.twins[0][0][0].weight)
    assert not torch.equal(naive.twins[1][0][0].weight,ensemble.twins[1][0][0].weight)
