import numpy as np
import pytest
import torch
from torch import nn
from app.role_routed_critic import RoleHeads
from app.decision_cost_objective import cost_objective
from app.weight_decision_forensics import (native_forward,cost_parts,cost_gradients,weighted_min_gradient,
    hybrid_state,block_indices,exact_shapley,gradient_contributions,cosine)


def fixture():
    head=RoleHeads(nn.Sequential(nn.Linear(4,3),nn.SiLU(),nn.Linear(3,3)).double(),1,17)
    rng=np.random.default_rng(98);z=rng.normal(size=(2,7,4));y=rng.normal(size=(7,3))
    return head,z,y


def test_native_forward_and_row_cost_gradients_match_autograd():
    head,z,y=fixture();q,cache=native_forward(head.state_dict(),z)
    actual=head(torch.tensor(z)).mean(2);np.testing.assert_allclose(q,actual.detach().numpy(),atol=1e-13)
    g,parts=cost_gradients(head.state_dict(),z,y,.7)
    loss=cost_objective(actual,torch.tensor(y),'cost',.7)
    grad=torch.autograd.grad(loss,list(head.parameters()))
    np.testing.assert_allclose(g.mean(0),torch.cat([x.reshape(-1) for x in grad]).detach(),atol=1e-12)
    for row in range(len(y)):
        qr=head(torch.tensor(z[:,row:row+1])).mean(2)
        gr=torch.autograd.grad(cost_objective(qr,torch.tensor(y[row:row+1]),'cost',.7),list(head.parameters()))
        np.testing.assert_allclose(g[row],torch.cat([x.reshape(-1) for x in gr]).detach(),atol=1e-12)
    assert len(cache)==2
    np.testing.assert_allclose(parts['row_cost'].mean(),loss.item(),atol=1e-13)


def test_surrogate_shift_invariance_and_directional_finite_difference():
    head,z,y=fixture();q,_=native_forward(head.state_dict(),z);p=cost_parts(q,y,.7)
    np.testing.assert_allclose(cost_parts(q+10,y,.7)['row_cost'],p['row_cost'],atol=1e-12)
    np.testing.assert_allclose(p['logit_gradient'].sum(1),0,atol=1e-12)
    d=np.random.default_rng(5).normal(size=(len(y),3));eps=1e-6
    numerical=(cost_parts(q+eps*d[None],y,.7)['row_cost']-cost_parts(q-eps*d[None],y,.7)['row_cost'])/(2*eps)
    np.testing.assert_allclose(numerical,(p['logit_gradient']*d).sum(1),atol=1e-8)


def test_known_interaction_shapley_and_hybrid_no_mutation():
    values=np.array([[sum((i+1) for i in range(4) if mask&(1<<i))+10*int(mask&3==3)] for mask in range(16)])
    np.testing.assert_allclose(exact_shapley(values)[:,0],[6,7,3,4],atol=1e-13)
    head,z,y=fixture();before={k:v.clone() for k,v in head.state_dict().items()};after={k:v+1 for k,v in before.items()}
    first=hybrid_state(before,after,0);last=hybrid_state(before,after,15)
    assert all(torch.equal(first[k],before[k]) and torch.equal(last[k],after[k]) for k in before)
    assert all(v.data_ptr()!=before[k].data_ptr() for k,v in first.items())
    indices=block_indices(before);assert len(np.unique(np.concatenate(list(indices.values()))))==sum(v.numel() for v in before.values())


def test_conflicting_shared_examples_decompose_exactly():
    g=np.array([[-1.,0.],[3.,0.],[0.,4.]])
    r=gradient_contributions(np.array([1.,0.]),g,np.arange(3),0,['flat','flat','long'])
    assert r['own_descent']>0 and r['total_descent']<0
    assert r['total_descent']==pytest.approx(r['own_descent']+r['others_descent'])
    assert sum(r['roles'].values())==pytest.approx(r['total_descent'])
    assert cosine([0,0],[1,2]) is None


def test_independent_logits_can_lower_greedy_margin_for_better_action():
    q=np.array([[[20.,19.,18.]],[[21.,20.,19.]]]);y=np.array([[1.,0.,100.]])
    g=cost_parts(q,y,1.)['logit_gradient'][0]
    assert y[0,0]>y[0,1] and -(g[0]-g[1])<0


def test_weighted_margin_derivative_and_invalid_contracts():
    head,z,y=fixture();c=np.zeros((len(y),3));c[:,0]=1;c[:,2]=-1
    g=weighted_min_gradient(head.state_dict(),z,c)
    q=head(torch.tensor(z)).mean(2).min(0).values
    gt=torch.autograd.grad((q[:,0]-q[:,2]).sum(),list(head.parameters()))
    np.testing.assert_allclose(g.sum(0),torch.cat([x.reshape(-1) for x in gt]).detach(),atol=1e-12)
    with pytest.raises(ValueError):exact_shapley(np.zeros((15,1)))
    with pytest.raises(ValueError):hybrid_state(head.state_dict(),head.state_dict(),16)
    with pytest.raises(ValueError):cost_parts(np.zeros((2,7,3)),y,0)


def test_real_point_excludes_held_labels_from_gradients_and_preserves_weights():
    from tools.analyze_saved_weight_decisions import point
    from app.training_forensics import tree_digest
    head=RoleHeads(nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,3)),1,19)
    state=head.state_dict();initial=tree_digest(state)
    opt=dict(param_groups=[dict(params=list(range(len(state))),betas=(.9,.999),eps=1e-8)],
        state={i:dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(v),exp_avg_sq=torch.ones_like(v)) for i,v in enumerate(state.values())})
    rng=np.random.default_rng(72);z=rng.normal(size=(2,6,208));y=rng.normal(size=(6,3));fit=np.array([0,1,2])
    p=point(state,opt,z,y,fit,.7,19,'cpu',['flat']*6)
    changed=y.copy();changed[3:]=rng.normal(size=(3,3))*1e5
    q=point(state,opt,z,changed,fit,.7,19,'cpu',['flat']*6)
    np.testing.assert_array_equal(p['grad'],q['grad']);assert not p['grad'][3:].any()
    assert p['summary']==q['summary'] and tree_digest(state)==initial
