import numpy as np
import pytest
import torch
from torch import nn

from app.role_routed_critic import RoleHeads, select_role, matched_schedule, delayed_histories
from app.decision_attribution import state_view
from app.three_signal_learning import synthetic_env
from tests.unit.test_discrete_curve_native import curve_env
from tests.unit.test_double_tcn_probe import probe_recipe


def test_all_heads_market_only_and_external_routing():
    template=nn.Sequential(nn.Linear(5,8),nn.SiLU(),nn.Linear(8,3))
    heads=RoleHeads(template,3,19)
    z=torch.randn(2,4,5)
    q=heads(z)
    roles=torch.tensor([0,1,2,0])
    out=select_role(q,roles)
    assert q.shape==(2,4,3,3)
    for i,r in enumerate(roles):assert torch.equal(out[:,i],q[:,i,r])
    perm=torch.tensor([2,0,3,1])
    assert torch.equal(select_role(q[:,perm],roles[perm]),out[:,perm])
    for bad in (torch.tensor([0,1,3,0]),torch.tensor([0.,1.,2.,0.]),roles[:2]):
        with pytest.raises(ValueError):select_role(q,bad)


def test_paired_initialization_and_role_specific_gradient():
    template=nn.Sequential(nn.Linear(5,8),nn.SiLU(),nn.Linear(8,3))
    a=RoleHeads(template,3,43);b=RoleHeads(template,3,43)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    assert not torch.equal(a.twins[0][0][0].weight,a.twins[0][1][0].weight)
    z=torch.randn(2,3,5);roles=torch.zeros(3,dtype=torch.long)
    select_role(a(z),roles).sum().backward()
    for twin in a.twins:
        assert any(p.grad.abs().sum()>0 for p in twin[0].parameters())
        assert all(p.grad is None or p.grad.abs().sum()==0 for head in twin[1:] for p in head.parameters())


def test_schedule_shared_and_fit_only():
    banks=[[0,1,2],[3,4],[5,6,7]]
    a=matched_schedule(banks,25,16,19);b=matched_schedule(banks,25,16,19)
    assert np.array_equal(a,b) and a.shape==(25,16)
    assert set(a.ravel())==set(range(8))
    with pytest.raises(ValueError):matched_schedule([[]],1,2,19)


def test_positive_control_routing_can_fit_identical_market_opposed_actions():
    from app.action_ranking_contrast import advantage_loss
    template=nn.Sequential(nn.Linear(2,8),nn.SiLU(),nn.Linear(8,3))
    heads=RoleHeads(template,3,19)
    z=torch.ones(2,3,2);roles=torch.arange(3)
    target=torch.eye(3)
    opt=torch.optim.Adam(heads.parameters(),lr=.03)
    for _ in range(200):
        opt.zero_grad();loss=advantage_loss(select_role(heads(z),roles),target)
        loss.backward();opt.step()
    with torch.no_grad():
        assert torch.equal(select_role(heads(z),roles).min(0).values.argmax(1),roles)
        shared=heads(z).mean(2)
        assert torch.equal(shared,shared[:,:1].expand_as(shared))


def test_cached_native_encoder_head_parity():
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    from tools.run_role_routed_critic import cache_embeddings
    model=ThreeSignalSAC(probe_recipe(),device='cpu')
    x=torch.randn(7,probe_recipe()['tensor']['window']['observations'],probe_recipe()['list']['n'])
    before={k:v.clone() for k,v in model.loss.state_dict().items()}
    z,template,errors=cache_embeddings(model,x)
    assert z.shape[:2]==(2,7) and max(errors)<=1e-6
    assert template(z[0]).shape==(7,3)
    assert all(torch.equal(v,before[k]) for k,v in model.loss.state_dict().items())


def test_delayed_histories_native_only(curve_env):
    env,regime=synthetic_env(probe_recipe(),curve_env.risk_config,17,1,.99)
    env.reset()
    while not env.engine.entry_gate(env.bar)[0]:env.step(0)
    before=state_view(env)
    banks=list(delayed_histories(env))
    assert banks and state_view(env)==before
    for lag,states,info in banks:
        assert lag in (4,12) and info['observation_equal']
        for key,e in states.items():
            assert e.engine.pos.direction==({'long':1,'short':-1}[key.split(':')[0]])
            assert e.engine.counts['entries']==before['counts']['entries']+1
            assert e.bar==states[next(iter(states))].bar
