from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from app.fit_gradient_conflict import components,alignment,partition,inspect_gradients
from app.role_routed_critic import RoleHeads


def test_components_translation_and_pairwise_identity():
    torch.manual_seed(7);y=torch.randn(9,3,dtype=torch.float64)
    q=torch.randn(2,9,3,dtype=torch.float64,requires_grad=True)
    loss=components(q,y)
    assert torch.allclose(loss['total'],loss['level']+loss['gap'],atol=1e-14)
    pair=torch.stack([(q[:,:,a]-q[:,:,b]-(y[:,a]-y[:,b]))**2 for a,b in ((0,1),(0,2),(1,2))])
    assert torch.allclose(pair.mean(),3*loss['gap'],atol=1e-14)
    shift=components(q+4,y)
    assert torch.allclose(shift['gap'],loss['gap'],atol=1e-14)
    grads=[torch.autograd.grad(loss[k],q,retain_graph=True)[0] for k in ('total','level','gap')]
    assert torch.allclose(grads[0],grads[1]+grads[2],atol=1e-14)


def test_alignment_sign_and_degenerate_case():
    a=np.array([1.,2.]);b=-a
    s=alignment(a,b)
    assert s['cosine']==pytest.approx(-1) and s['conflict']
    assert s['left_derivative_descend_right_unit']>0
    assert not alignment(a,a)['conflict']
    assert alignment(a,np.zeros(2))['cosine'] is None
    assert alignment(a,np.zeros(2))['left_derivative_descend_right_unit'] is None
    with pytest.raises(ValueError):alignment(a,np.array([np.nan,1]))


def test_partition_is_strict_and_disjoint():
    old,new,expanded=partition([1,3],[1,2,3,4],5)
    assert old.tolist()==[1,3] and new.tolist()==[2,4]
    assert expanded.tolist()==[1,2,3,4]
    for a,b in (([1,1],[1,2]),([1],[1]),([4],[1,2]),([1],[1,7])):
        with pytest.raises(ValueError):partition(a,b,5)


@pytest.mark.parametrize('output_only',[False,True])
def test_gradients_identities_finite_difference_and_no_mutation(output_only):
    torch.manual_seed(12)
    h=RoleHeads(nn.Sequential(nn.Linear(5,4),nn.SiLU(),nn.Linear(4,3)),1,19).double()
    for name,p in h.named_parameters():p.requires_grad_(not output_only or name.split('.')[3]=='2')
    before=deepcopy(h.state_dict());z=torch.randn(2,16,5,dtype=torch.float64)
    y=torch.randn(16,3,dtype=torch.float64)
    r,v=inspect_gradients(h,z,y,np.arange(4),np.arange(4,16))
    assert r['original_weight']==.25
    assert r['gradient_union_error']<1e-12 and r['gradient_decomposition_error']<1e-12
    assert r['finite_difference_max_error']<1e-7
    assert all(torch.equal(x,before[k]) for k,x in h.state_dict().items())
    assert all(p.grad is None for p in h.parameters())
    assert set(v)=={f'{a}_{b}' for a in ('original','added','expanded') for b in ('total','level','gap')}
    if output_only:assert 'hidden' not in r['groups']


def test_invalid_shapes_and_overlapping_support_rejected():
    with pytest.raises(ValueError):components(torch.ones(2,3,3),torch.ones(4,3))
    h=RoleHeads(nn.Sequential(nn.Linear(5,4),nn.SiLU(),nn.Linear(4,3)),1,19).double()
    with pytest.raises(ValueError):inspect_gradients(h,torch.ones(2,4,5,dtype=torch.float64),torch.ones(4,3,dtype=torch.float64),[0,1],[1,2])


def test_failed_numerical_probe_restores_weights(monkeypatch):
    import app.fit_gradient_conflict as module
    h=RoleHeads(nn.Sequential(nn.Linear(5,4),nn.SiLU(),nn.Linear(4,3)),1,19).double()
    before=deepcopy(h.state_dict());original=module.components;calls=[]
    def fail(q,y):
        calls.append(1)
        if len(calls)==4:raise RuntimeError('injected finite-difference failure')
        return original(q,y)
    monkeypatch.setattr(module,'components',fail)
    with pytest.raises(RuntimeError):
        inspect_gradients(h,torch.ones(2,4,5,dtype=torch.float64),torch.zeros(4,3,dtype=torch.float64),[0],[1,2,3])
    assert all(torch.equal(v,before[k]) for k,v in h.state_dict().items())
