import numpy as np
import pytest
import torch

from agent_plugins.hybrid_orders import OrderDistribution
from app.order_rewards import Reward
from agent_plugins.hybrid_orders import ElapsedRolloutBuffer
from gymnasium import spaces


def distribution(mask=(1,1,1,0,0), learned=True):
    logits = torch.zeros((1,5), requires_grad=True)
    raw = torch.zeros((1,2,6), requires_grad=True)
    return OrderDistribution(logits, raw, torch.tensor([mask], dtype=torch.bool), learned), logits, raw


def test_categorical_support_and_joint_density():
    d, logits, raw = distribution()
    for _ in range(50):
        a = d.sample()
        assert int(a[0,0]) in (0,1,2) and a[0,0] == int(a[0,0])
    a = torch.tensor([[1., .2,.4,.6]])
    expected = d.cat.log_prob(torch.tensor([1])) + d.risk.log_prob(a[:,None,1:]).sum(-1)[:,0]
    torch.testing.assert_close(d.log_prob(a), expected)
    d.log_prob(torch.tensor([[0.,.5,.5,.5]])).sum().backward()
    assert raw.grad is not None and torch.count_nonzero(raw.grad) == 0
    assert torch.isfinite(d.entropy()).all()


def test_illegal_commands_and_fractional_codes_rejected():
    d, _, _ = distribution((1,0,0,1,0))
    assert d.cat.probs[0,1] == 0
    with pytest.raises(ValueError): d.log_prob(torch.tensor([[.4,.5,.5,.5]]))
    with pytest.raises(ValueError): d.log_prob(torch.tensor([[2.,.5,.5,.5]]))
    with pytest.raises(ValueError): distribution((0,0,0,0,0))


def test_fixed_risk_has_only_categorical_likelihood():
    d, _, _ = distribution(learned=False)
    a = d.sample()
    torch.testing.assert_close(d.log_prob(a), d.cat.log_prob(a[:,0].long()))


def test_reward_telescope_and_drawdown():
    path = [100.,110.,99.,105.]
    equity = Reward('equity',100.)
    log = Reward('log',100.)
    dd = Reward('drawdown',100.,penalty=1.)
    er=lr=dr=0.
    for a,b in zip(path,path[1:]):
        er += equity.step(a,b); lr += log.step(a,b); dr += dd.step(a,b)
    assert er == pytest.approx(5.)
    assert lr == pytest.approx(100*np.log(1.05))
    assert dr == pytest.approx(100*(np.log(1.05)-.1))
    assert dd.step(105.,105.) == 0


def test_differential_sharpe_prefix_and_flat():
    a,b=Reward('dsr',100.),Reward('dsr',100.)
    assert all(a.step(100.,100.)==0 for _ in range(40))
    for before,after in [(100,101),(101,99),(99,102)]:
        assert np.isfinite(b.step(before,after))
    with pytest.raises(ValueError): b.step(100.,-1.)


def test_dsr_matches_formula_before_moment_update():
    r=Reward('dsr',100.,warmup=0)
    r.a,r.b=.002,.00002
    u=.01
    expected=(r.b*(u-r.a)-.5*r.a*(u*u-r.b))/max(r.b-r.a*r.a,r.floor)**1.5
    assert r.step(100.,101.)==pytest.approx(expected)


def test_duration_returns_match_two_step_analytical_case():
    obs=spaces.Dict({'features':spaces.Box(-1,1,(1,),np.float32)})
    buf=ElapsedRolloutBuffer(2,obs,spaces.Box(0,1,(4,),np.float32),device='cpu',gamma=.5,gae_lambda=1.)
    for duration,reward in [(2,2.),(1,4.)]:
        buf.pending_duration=np.array([duration])
        buf.add({'features':np.zeros((1,1),np.float32)},np.zeros((1,4)),np.array([reward]),
                np.array([0.]),torch.tensor([0.]),torch.tensor([0.]))
    buf.compute_returns_and_advantage(torch.tensor([0.]),np.array([True]))
    np.testing.assert_array_equal(buf.returns[:,0],[3.,4.])


@pytest.mark.parametrize('terminal,expected',[(False,3.),(True,2.)])
def test_rollout_boundary_bootstrap_vs_true_terminal(terminal,expected):
    obs=spaces.Dict({'features':spaces.Box(-1,1,(1,),np.float32)})
    buf=ElapsedRolloutBuffer(1,obs,spaces.Box(0,1,(4,),np.float32),device='cpu',gamma=.5,gae_lambda=.7)
    buf.pending_duration=np.array([2])
    buf.add({'features':np.zeros((1,1),np.float32)},np.zeros((1,4)),np.array([2.]),
            np.array([0.]),torch.tensor([7.]),torch.tensor([0.]))
    buf.compute_returns_and_advantage(torch.tensor([4.]),np.array([terminal]))
    assert buf.returns[0,0]==pytest.approx(expected)
