import copy
import json
from pathlib import Path

import pytest

from tests.unit.test_selected_rl_capacity_sequence import paired_request
from tests.unit.test_selected_rl_comparison import write

ARCH = dict(net_arch=[8], branch_filters=2, core_filters=4, kernel_size=3)


def test_temporal_request_changes_only_declared_architecture(tmp_path):
    from tools import selected_rl_comparison as selected, curated_rl_campaign as queue
    path, request = paired_request(tmp_path)
    old, _ = selected._build(path, tmp_path/'old')
    request.update(schema='selected_rl_comparison_request.v7', temporal_encoder=ARCH)
    write(path, request)
    root = tmp_path/'temporal'
    selected.materialize(path, root)
    body = queue.load_contract(root/'CONTRACT.json')
    expected = copy.deepcopy(old['common_config'])
    expected.update(net_arch=[8], temporal_branch_filters=2, temporal_core_filters=4)
    assert body['common_config'] == expected
    assert body['arms'] == ['SAC_PER_FEATURE_CONV1D']
    assert body['splits'] == old['splits']
    assert body['selected_input_binding'] == old['selected_input_binding']
    d=root/'task/attempt-1';d.mkdir(parents=True)
    cfg=queue._task_config(body,dict(arm='SAC_PER_FEATURE_CONV1D',split='train',seed=7,id='test',attempt=1),d,d/'policy.zip')
    assert cfg['observation_encoder'] == 'per_feature_conv1d'
    assert cfg['queue_arm'] == 'SAC_PER_FEATURE_CONV1D'
    selected.validate(root)


@pytest.mark.parametrize('field,value', [('net_arch',[16]),('branch_filters',8),('core_filters',8),('kernel_size',5)])
def test_undeclared_temporal_recipe_refused(tmp_path,field,value):
    from tools import selected_rl_comparison as selected
    path, request = paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v7',temporal_encoder={**ARCH,field:value})
    write(path,request)
    with pytest.raises(ValueError): selected._build(path,tmp_path/'out')


def test_minimal_extractor_shape_prefix_origin_and_native_roundtrip(tmp_path):
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from agent_plugins.sac_agent import Plugin, _policy_tensor_hash
    from agent_plugins.temporal_features import PerFeatureTemporalExtractor
    from tests.unit.test_temporal_sac_features import ToyEnv
    env=ToyEnv()
    env.observation_space=gym.spaces.Dict(dict(features=gym.spaces.Box(-10,10,(13,15),dtype=np.float32),
                                            position=gym.spaces.Box(-1,1,(1,),dtype=np.float32)))
    extractor=PerFeatureTemporalExtractor(env.observation_space,2,4,3)
    for p in extractor.parameters(): torch.nn.init.constant_(p,.1)
    x=torch.ones(1,13,15);y=extractor.temporal_forward(x)
    assert y.shape == (1,4,13) and extractor.features_dim == 53
    changed=x.clone();changed[:,8:,:]+=1
    assert torch.equal(y[:,:,:8],extractor.temporal_forward(changed)[:,:,:8])
    changed=x.clone();changed[:,-1,:]+=1
    assert not torch.equal(y[:,:,-1],extractor.temporal_forward(changed)[:,:,-1])
    cfg=dict(observation_encoder='per_feature_conv1d',net_arch=[8],temporal_branch_filters=2,
             temporal_core_filters=4,temporal_kernel_size=3,train_seed=7,device='cpu',
             buffer_size=64,batch_size=2,learning_starts=0)
    plugin=Plugin(cfg);model=plugin.build(env,cfg);model.learn(8)
    assert model._n_updates == 8
    before=_policy_tensor_hash(model.policy);path=tmp_path/'policy.zip';model.save(path)
    loaded=SAC.load(path,env=env,device='cpu')
    assert _policy_tensor_hash(loaded.policy)==before
    obs,_=env.reset()
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
    env.close()


def test_temporal_units_use_native_gate_and_own_publisher(tmp_path):
    from tools import selected_rl_temporal_minimal as temporal
    path,request=paired_request(tmp_path)
    root=tmp_path/'sequence'
    spec=temporal.prepare(path,root,tmp_path/'closed','f'*64)
    assert temporal.prepare(path,root,tmp_path/'closed','f'*64)==spec
    units=temporal.render(spec,root/'units',tmp_path/'private.env',root/'SEQUENCE.json')
    assert len(units)==4
    worker=(root/'units/rl-selected-temporal-minimal-conv1d2x4-n8.service').read_text()
    publisher=(root/'units/rl-selected-temporal-minimal-publication.service').read_text()
    assert 'tools.verify_rl_predecessor' in worker
    assert 'tools.selected_rl_temporal_minimal' in worker and '--smoke-root' in worker
    assert 'MemoryMax=6G' in worker
    assert 'tools.selected_rl_temporal_minimal' in publisher and 'publish' in publisher


def test_native_temporal_gpu_smoke_is_cpu_testable():
    from tools.selected_rl_temporal_minimal import smoke_model
    result=smoke_model('cpu')
    assert result['gradient_updates']==16 and result['policy_reload_equal']
    assert result['synthetic_actor_parameters']>0
