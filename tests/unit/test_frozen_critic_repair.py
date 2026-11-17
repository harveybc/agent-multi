from copy import deepcopy

import numpy as np
import pytest

from app.frozen_critic_repair import choose, passed, constrained_step


def test_selection_ignores_targets_and_purges_monitor_context():
    rows=[dict(row=i,year=2020+i//10,kind='open_long',window_start=1609459200000000000,
               timestamp=1609459200000000000,role='fit') for i in range(30)]
    rows.append(dict(row=31,year=2023,kind='open_long',window_start=0,timestamp=1672531200000000000,role='monitor'))
    a=choose(rows)
    assert len(a)==12 and all(r['row']!=31 for r in a)
    assert a==choose([dict(r,hidden_future=999) for r in rows])


def test_gate_does_not_allow_rmse_alone_or_ties():
    base=dict(mean_regret=.3,pairwise_accuracy=.5,rmse=.4)
    actor=dict(mean_regret=.2);nop=dict(mean_regret=.25)
    good=dict(mean_regret=.1,pairwise_accuracy=.6,rmse=.3)
    assert passed(good,base,actor,nop)
    assert not passed(dict(good,mean_regret=.21),base,actor,nop)
    assert not passed(dict(good,mean_regret=.3),base,actor,nop)
    assert not passed(dict(good,pairwise_accuracy=None),base,actor,nop)


def test_kl_rejection_restores_adam_and_weights():
    import torch
    p=torch.nn.Parameter(torch.zeros(1,3))
    opt=torch.optim.AdamW([p],lr=.1)
    p.sum().backward();opt.step();opt.zero_grad()
    before=p.detach().clone();state=deepcopy(opt.state_dict())
    reference=p.detach().clone()
    loss=-p[0,1]
    # A zero budget rejects every nonzero trial including roundoff.
    result=constrained_step(opt,[p],loss,lambda:p,reference,0.,max_backtracks=0)
    assert not result['accepted'] and torch.equal(p,before)
    for key,v in state['state'][0].items():
        assert torch.equal(v,opt.state_dict()['state'][0][key])


def test_kl_acceptance_and_nonfinite_refusal():
    import torch
    p=torch.nn.Parameter(torch.zeros(1,3));opt=torch.optim.AdamW([p],lr=.01)
    r=constrained_step(opt,[p],-p[0,1],lambda:p,torch.zeros(1,3),1e-3)
    assert r['accepted'] and 0<r['kl']<=1e-3
    with pytest.raises(ValueError):
        constrained_step(opt,[p],p.sum(),lambda:p,torch.zeros(1,3),float('nan'))


@pytest.mark.parametrize('fresh', [False, True])
def test_native_calibration_freezes_actor_encoders_and_temperature(fresh):
    import torch
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    from app.actor_learning_diagnostics import initialize_critic_heads, clear_optimizer_memory
    from app.critic_fit_probe import set_scope
    from app.action_ranking_contrast import differentiable_q, target_loss
    from tests.unit.test_double_tcn_probe import probe_recipe
    from tools.diagnose_warmstart_drift import frozen_state, verify_frozen_state
    torch.set_num_threads(1)
    m=ThreeSignalSAC(probe_recipe(),seed=7)
    x=torch.randn(8,13,2);z=m.surfaces(x).detach().clone()
    if fresh:initialize_critic_heads(m,7)
    set_scope(m,'heads');clear_optimizer_memory(m.optimizer)
    frozen=frozen_state(m);before=m.q_surfaces(x).detach().clone()
    m.optimizer.zero_grad(set_to_none=True)
    target_loss(differentiable_q(m,x),torch.ones(8,3)).backward();m.optimizer.step()
    verify_frozen_state(m,frozen)
    assert torch.equal(m.surfaces(x),z)
    assert not torch.equal(m.q_surfaces(x),before)


def test_cache_rejects_tampering(tmp_path):
    from tools.run_frozen_critic_repair import save, cached
    from tools.run_discrete_curve_probe import read, write
    p=tmp_path/'row.json';save(p,'id',dict(a=1))
    assert cached(p,'id')==dict(a=1)
    with pytest.raises(ValueError):cached(p,'other')
    d=read(p);d['value']['a']=2;write(p,d)
    with pytest.raises(ValueError):cached(p,'id')


def test_native_actor_trust_update_changes_only_head():
    import torch
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    from tests.unit.test_double_tcn_probe import probe_recipe
    from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
    m=ThreeSignalSAC(probe_recipe(),seed=7)
    set_arm(m,'actor_only');frozen=frozen_state(m)
    x=torch.randn(8,13,2);reference=m.surfaces(x).detach().clone()
    z=m.surfaces(x);lp=z.log_softmax(-1)
    with torch.no_grad():q=m.q_surfaces(x).min(0).values
    loss=(lp.exp()*(m.loss.log_alpha.detach().exp()*lp-q)).sum(-1).mean()
    r=constrained_step(m.optimizer,[p for p in m.loss.parameters() if p.requires_grad],
        loss,lambda:m.surfaces(x),reference,1e-3)
    assert r['accepted'] and r['kl']<=1e-3
    verify_frozen_state(m,frozen)
    assert not torch.equal(m.surfaces(x),reference)
