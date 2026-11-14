import numpy as np
import pytest
import torch
from torch import nn
from app.twin_complete_protection import twin_complete_gaps,twin_complete_jacobian,qualify_twin_continuation
from app.primal_direction_projection import project_primal


def test_active_minimum_crossing_counterexample_and_complete_projection():
    # At x=0 the first chosen twin wins; at x=.03 the ignored twin becomes unsafe.
    def outputs(x):
        return torch.stack([torch.stack([.02+x,x*0,x*0-1]),
                            torch.stack([.021-x,x*0,x*0-1])])[:,None,:]
    x=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    q=outputs(x);active=q.min(0).values[0,0]-q.min(0).values[0,1]
    derivative=torch.autograd.grad(active,x,retain_graph=True)[0]
    assert active.item()+.03*derivative.item()>.01
    assert outputs(torch.tensor(.03)).min(0).values[0,0]<0
    gaps=twin_complete_gaps(q,np.array([0]));j=np.array([[torch.autograd.grad(g,x,retain_graph=True)[0].item()] for g in gaps])
    p,r=project_primal(np.array([.03]),j,.01-gaps.detach().numpy())
    assert r['feasible'];np.testing.assert_allclose(p,[.011],atol=1e-12)
    assert outputs(torch.tensor(p[0])).min(0).values[0,0]>=.01-1e-12


def test_rows_alternatives_twins_order_and_sufficient_margin():
    rng=np.random.default_rng(91);center=torch.tensor(rng.normal(size=(2,100,3)))
    actions=center.min(0).values.argmax(1).numpy();g=twin_complete_gaps(center,actions)
    expected=[]
    for row,action in enumerate(actions):
        for alternative in range(3):
            if alternative==action:continue
            for twin in range(2):expected.append((center[twin,row,action]-center[:,row,alternative].min()).item())
    np.testing.assert_array_equal(g.detach(),expected)
    # Lowering either alternative changes its branch, but cannot break the implication.
    trial=center.clone();trial[0]-=.3
    for row,action in enumerate(actions):
        trial[:,row,action]=center[:,row,:].max()+.4
    for row,action in enumerate(actions):
        for alternative in range(3):
            if alternative!=action:
                branch=int(center[:,row,alternative].argmin())
                assert torch.all(trial[:,row,action]-trial[branch,row,alternative]>=.1)
                assert trial[:,row,action].min()-trial[:,row,alternative].min()>=.1


class SmallTwins(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.arange(18,dtype=torch.float64).reshape(2,3,3)/20)
    def forward(self,z):return torch.einsum('tni,tai->tna',z,self.weight)[:,:,None,:]


def test_complete_jacobian_central_differences():
    head=SmallTwins();z=torch.tensor(np.random.default_rng(7).normal(size=(2,5,3)))
    old=np.array([1,4]);actions=np.array([0,2]);g,j=twin_complete_jacobian(head,z,old,actions)
    assert g.shape==(8,) and j.shape==(8,18)
    original=head.weight.detach().clone();eps=1e-6
    for k in range(18):
        with torch.no_grad():head.weight.copy_(original);head.weight.reshape(-1)[k]+=eps
        gp=twin_complete_gaps(head(z).mean(2)[:,old],actions).detach().numpy()
        with torch.no_grad():head.weight.copy_(original);head.weight.reshape(-1)[k]-=eps
        gm=twin_complete_gaps(head(z).mean(2)[:,old],actions).detach().numpy()
        np.testing.assert_allclose(j[:,k],(gp-gm)/(2*eps),atol=1e-8)


@pytest.mark.parametrize('actions', [np.array([3]),np.array([0.]),np.array([0,1])])
def test_invalid_actions_rejected(actions):
    with pytest.raises(ValueError):twin_complete_gaps(torch.zeros((2,1,3)),actions)


def test_frozen_continuation_gate_requires_safety_and_cost_progress():
    rows=[dict(accepted=True,robust=True,geometry_agrees=True,before=.3,after=.3,cost_before=.2,cost_after=.19) for _ in range(16)]
    assert qualify_twin_continuation(rows)
    for r in rows[:5]:r['cost_after']=.2
    assert not qualify_twin_continuation(rows)
    for r in rows:r['cost_after']=.19;r['after']=.31
    assert not qualify_twin_continuation(rows)


def test_independent_manual_native_derivative():
    from app.role_routed_critic import RoleHeads
    from tools.verify_twin_complete_run import manual_complete_system
    head=RoleHeads(nn.Sequential(nn.Linear(4,3),nn.SiLU(),nn.Linear(3,3)).double(),1,71)
    z=torch.tensor(np.random.default_rng(11).normal(size=(2,5,4)))
    old=np.array([0,2,4]);actions=np.array([0,1,2])
    gaps,j=twin_complete_jacobian(head,z,old,actions)
    gm,jm=manual_complete_system(head.state_dict(),z.numpy()[:,old],actions)
    np.testing.assert_allclose(gaps,gm,atol=1e-13)
    np.testing.assert_allclose(j,jm,atol=1e-13)


def test_finite_evaluator_can_reject_step_before_checkpoint(monkeypatch):
    from tools.run_automated_robust_refinement import Context
    from app.interior_adamw_repair import transactional_step
    from app.training_forensics import tree_digest
    from copy import deepcopy
    ctx=object.__new__(Context);ctx.double=nn.Linear(1,1).double();ctx.zcpu=torch.zeros((2,1,1))
    ctx.old=np.array([0]);ctx.actions=np.array([0]);ctx.margin=np.array([1e-5])
    ctx.observe=lambda state:dict(passed=True)
    monkeypatch.setattr('tools.run_automated_robust_refinement.gap_jacobian',
        lambda *args:(np.ones(2),np.zeros((2,2))))
    head=nn.Linear(1,1);opt=torch.optim.AdamW(head.parameters(),lr=.01)
    def fake_fit(head,optimizer,z,y,loss_fn=None):
        optimizer.zero_grad();head(torch.ones((1,1))).sum().backward();optimizer.step();return False
    monkeypatch.setattr('app.interior_adamw_repair.fit_step',fake_fit)
    initial=deepcopy(head.state_dict());initial_opt=deepcopy(opt.state_dict());seen=[]
    def single(state):seen.append(1);return dict(passed=False)
    record=transactional_step(head,opt,None,None,
        lambda before,proposed:ctx.repair(before,proposed,'interior',finite_evaluate=single))
    assert not record['accepted'] and len(seen)==24
    assert tree_digest(head.state_dict())==tree_digest(initial)
    assert tree_digest(opt.state_dict())==tree_digest(initial_opt)


def test_automatic_verifier_wait_is_bounded_and_refuses_failure(tmp_path):
    from tools.verify_twin_replication import wait_complete
    with pytest.raises(ValueError,match='NOT_COMPLETE'):wait_complete(tmp_path,0)
    (tmp_path/'PROGRESS.json').write_text('{"stage":"FAILED"}')
    with pytest.raises(ValueError,match='PRODUCER_FAILED'):wait_complete(tmp_path,100)
    (tmp_path/'DONE.json').write_text('{}')
    wait_complete(tmp_path,0)
    with pytest.raises(ValueError,match='WAIT_BUDGET'):wait_complete(tmp_path,-1)
