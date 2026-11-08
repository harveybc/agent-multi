import pytest
import torch
from tensordict import TensorDict
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tests.unit.test_double_tcn_probe import probe_recipe


def fixture_model():
    torch.set_num_threads(1)
    model = ThreeSignalSAC(probe_recipe(), critic_regularization={'kind': 'core_dropout', 'p': .1})
    x = torch.randn(8, 13, 2)
    batch = TensorDict({'features': x, 'mask': torch.ones(8, 3, dtype=torch.bool),
        'action': torch.arange(8) % 3, 'steps_to_next_obs': torch.arange(1, 9).reshape(-1, 1),
        'next': {'features': x * .9, 'mask': torch.ones(8, 3, dtype=torch.bool),
                 'reward': torch.linspace(-1, 1, 8).reshape(-1, 1),
                 'done': torch.tensor([False]*4+[True]*4).reshape(-1, 1),
                 'terminated': torch.tensor([False]*4+[True]*4).reshape(-1, 1)}}, [8])
    return model, batch


def test_target_loss_gradient_and_polyak_contract():
    from app.fixed_actor_critic_audit import configure_arm, algebra_audit, checked_update
    model, batch = fixture_model()
    configure_arm(model, 'production')
    audit = algebra_audit(model, batch)
    assert audit['target_max_error'] < 1e-5
    assert audit['loss_error'] < 1e-5
    assert audit['terminal_max_error'] == 0
    assert audit['actor_to_critic_gradient'] == 0
    assert audit['critic_to_actor_gradient'] == 0
    assert checked_update(model, batch, 'production')['polyak_max_error'] < 1e-6


@pytest.mark.parametrize('arm', ['no_update', 'production', 'no_core_dropout', 'no_weight_decay', 'fixed_target'])
def test_single_factor_and_frozen_actor(arm):
    from app.fixed_actor_critic_audit import configure_arm, checked_update
    model, batch = fixture_model()
    old_p = [m.p for m in model.q_net.encoder.core.modules() if isinstance(m, torch.nn.Dropout)]
    old_wd = model.optimizer.param_groups[0]['weight_decay']
    configure_arm(model, arm)
    actor = {n:p.clone() for n,p in model.loss.named_parameters() if n.startswith('actor_')}
    target = {k:v.clone() for k,v in model.loss.target_qvalue_network_params.items(True,True)}
    logits = model.surfaces(batch['features']).detach().clone()
    checked_update(model, batch, arm)
    assert all(torch.equal(v,dict(model.loss.named_parameters())[n]) for n,v in actor.items())
    assert torch.equal(logits,model.surfaces(batch['features']))
    if arm in ('fixed_target','no_update'):
        assert all(torch.equal(v,model.loss.target_qvalue_network_params[k]) for k,v in target.items())
    assert [m.p for m in model.q_net.encoder.core.modules() if isinstance(m,torch.nn.Dropout)] == ([0.]*len(old_p) if arm=='no_core_dropout' else old_p)
    assert model.optimizer.param_groups[0]['weight_decay'] == (0. if arm=='no_weight_decay' else old_wd)


def test_algebra_rejects_wrong_target(monkeypatch):
    from app.fixed_actor_critic_audit import algebra_audit
    model,batch=fixture_model()
    original=model.loss._compute_target
    monkeypatch.setattr(model.loss,'_compute_target',lambda td:original(td)+1.)
    with pytest.raises(AssertionError):algebra_audit(model,batch)


@pytest.mark.parametrize('dropout', [True, False])
def test_measured_mode_gap_and_rng_isolation(dropout):
    from app.fixed_actor_critic_audit import configure_arm
    from tools.audit_fixed_actor_critics import measure
    model,batch=fixture_model()
    configure_arm(model,'production' if dropout else 'no_core_dropout')
    panel=dict(features=batch['features'],action=batch['action'],returns=torch.zeros(8).numpy())
    rng=torch.get_rng_state().clone()
    result=measure(model,batch,panel)
    assert torch.equal(rng,torch.get_rng_state())
    assert (result['dropout_std']['max']>0) == dropout
    if not dropout:assert abs(result['train_minus_eval_q']['max'])<1e-6


def test_minimum_penalty_decomposition_terminal_and_entropy():
    from app.fixed_actor_critic_audit import backup_components
    from app.generalization_diagnostics import soft_td_target
    p=torch.tensor([[.2,.3,.5],[.5,.25,.25]])
    q=torch.tensor([[[1.,2.,3.],[1.,2.,3.]],[[3.,4.,5.],[5.,4.,3.]]])
    r=torch.tensor([.3,-.1]); done=torch.tensor([False,True]); d=torch.tensor([4,1]); online=torch.tensor([1.,1.])
    c=backup_components(r,done,d,p,q,online,.1,.99)
    expected=soft_td_target(r,done,d,p,q,.1,.99)-online
    torch.testing.assert_close(c['residual'],expected)
    torch.testing.assert_close(c['minimum_twin_penalty'],torch.tensor([-(.99**4),0.]))
    assert c['entropy_bonus'][1]==0


def test_mean_target_changes_only_bootstrap_and_preserves_terminal():
    from app.fixed_actor_critic_audit import configure_arm
    from app.selected_warm_start import target_surfaces
    model,batch=fixture_model()
    with torch.no_grad():
        native=model.loss._compute_target(batch.clone())
        p=model.surfaces(batch['next','features']).softmax(-1)
        q=target_surfaces(model,batch['next','features'])
        v=(p*(q.mean(0)-model.loss.log_alpha.exp()*p.log())).sum(-1)
        expected=batch['next','reward'].reshape(-1)+(.99**batch['steps_to_next_obs'].reshape(-1))*(~batch['next','terminated'].reshape(-1))*v
    configure_arm(model,'mean_target')
    actual=model.loss._compute_target(batch.clone())
    torch.testing.assert_close(actual,expected)
    assert torch.equal(actual[4:],native[4:])
    assert (actual[:4]>=native[:4]).all()


def test_residual_check_uses_target_scale_and_rejects_material_error():
    from app.fixed_actor_critic_audit import verify_backup_identity
    target=torch.tensor([-100.]);online=torch.tensor([-100.75])
    assert verify_backup_identity({'residual':torch.tensor([.7500534])},online,target)<6e-5
    with pytest.raises(AssertionError):
        verify_backup_identity({'residual':torch.tensor([1.7500534])},online,target)


def test_covering_schedule_visits_every_row_and_reproduces():
    import numpy as np
    from tools.audit_fixed_actor_critics import covering_schedule
    a=covering_schedule(23,5,8,7)
    assert a.shape==(5,8) and set(a.ravel())==set(range(23))
    assert np.array_equal(a,covering_schedule(23,5,8,7))
    with pytest.raises(ValueError):covering_schedule(100,5,8,7)


def test_native_full_replay_preserves_terminal_and_rejects_bad_receipt():
    import numpy as np
    from types import SimpleNamespace
    from tools.audit_fixed_actor_critics import collect_full_replay
    from env_plugins.discrete_curve_native import HOUR
    class Env:
        def __init__(self):
            self.engine=SimpleNamespace(v=np.arange(4)*HOUR,pos=None)
        def reset(self):
            self.bar=0;self.finished=False
            return self.obs()
        def obs(self):return dict(features=np.full((13,2),self.bar,dtype=np.float32),mask=np.ones(3,bool))
        def step(self,action):
            self.last_command=action;self.bar+=1;self.finished=self.bar==3
            return self.obs(),float(action),self.finished,1
    trace=[dict(timestamp=(i+1)*HOUR,position=0,signal=i,command=i,reward=float(i),duration=1) for i in range(3)]
    replay,receipt=collect_full_replay(Env(),trace)
    assert len(replay)==3 and receipt['terminal_count']==1
    assert replay['next','terminated'].reshape(-1).tolist()==[False,False,True]
    assert replay['next','features'][-1,0,0]==3
    bad=[dict(r) for r in trace];bad[1]['reward']=99
    with pytest.raises(ValueError,match='REPLAY_REWARD'):collect_full_replay(Env(),bad)
    with pytest.raises(ValueError,match='REPLAY_TERMINAL'):collect_full_replay(Env(),trace[:2])
