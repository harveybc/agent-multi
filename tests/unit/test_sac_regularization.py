import numpy as np
import pytest
import torch
from gymnasium import spaces

from agent_plugins.sac_agent import Plugin


def env():
    import gymnasium
    return gymnasium.make('Pendulum-v1')


def build(profile=None):
    cfg=dict(device='cpu', net_arch=[32,32], train_seed=7, buffer_size=64,
             learning_starts=4,batch_size=8,total_timesteps=24)
    if profile is not None: cfg['sac_regularization']=profile
    return Plugin().build(env(),cfg)


@pytest.mark.parametrize('name',['dropout005','l2_1e6','dropout005_l2_1e6','batchnorm'])
def test_regularization_only_critics_and_roundtrip(tmp_path,name):
    from agent_plugins.sac_regularization import PROFILES
    baseline=build()
    m=build(PROFILES[name])
    assert all(torch.equal(v,m.actor.state_dict()[k]) for k,v in baseline.actor.state_dict().items())
    assert not any(isinstance(n,(torch.nn.Dropout,torch.nn.BatchNorm1d)) for n in m.actor.modules())
    drops=[n for n in m.critic.modules() if isinstance(n,torch.nn.Dropout)]
    bns=[n for n in m.critic.modules() if isinstance(n,torch.nn.BatchNorm1d)]
    assert bool(drops)==bool(PROFILES[name]['dropout'])
    assert bool(bns)==PROFILES[name]['batch_norm']
    assert all(p['weight_decay']==PROFILES[name]['l2'] for p in m.critic.optimizer.param_groups)
    assert all(p['weight_decay']==0 for p in m.actor.optimizer.param_groups)
    m.learn(24)
    assert not m.critic_target.training
    assert all(torch.isfinite(p).all() for p in m.policy.parameters())
    m.policy.set_training_mode(False)
    obs=np.full((1,3),.2,dtype=np.float32)
    before=m.predict(obs,deterministic=True)[0]
    torch.manual_seed(99)
    assert np.array_equal(before,m.predict(obs,deterministic=True)[0])
    path=tmp_path/'policy.zip';m.save(path)
    from stable_baselines3 import SAC
    restored=SAC.load(path,device='cpu')
    assert np.array_equal(before,restored.predict(obs,deterministic=True)[0])
    assert all(torch.equal(v,restored.policy.state_dict()[k]) for k,v in m.policy.state_dict().items())
    assert restored.policy.regularization==PROFILES[name]


def test_critic_dropout_train_vs_eval():
    from agent_plugins.sac_regularization import PROFILES
    m=build(PROFILES['dropout005'])
    obs=torch.ones((32,3));action=torch.zeros((32,1))
    m.policy.set_training_mode(True)
    assert not torch.equal(m.critic(obs,action)[0],m.critic(obs,action)[0])
    assert not m.critic_target.training
    m.policy.set_training_mode(False)
    assert torch.equal(m.critic(obs,action)[0],m.critic(obs,action)[0])


@pytest.mark.parametrize('value',[{},None,{'dropout':.5},
    {'dropout':.05,'l2':0.,'batch_norm':True},
    {'dropout':float('nan'),'l2':0.,'batch_norm':False}])
def test_invalid_profiles_refused(value):
    from agent_plugins.sac_regularization import validate_profile
    with pytest.raises(ValueError):validate_profile(value)


def test_gpu_smoke_refuses_missing_assignment_without_learning(tmp_path,monkeypatch):
    import json
    from agent_plugins.sac_regularization import gpu_smoke,PROFILES
    (tmp_path/'CONTRACT.json').write_text(json.dumps(dict(
        common_config=dict(sac_regularization=PROFILES['dropout005']),
        execution=dict(gpu_evidence=dict(assigned_gpu_uuid='GPU-'+'a'*36)))))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(Plugin,'build',lambda *a,**k:pytest.fail('must not train'))
    assert gpu_smoke(tmp_path)==1
    assert not (tmp_path/'GPU_REGULARIZATION_SMOKE.json').exists()
