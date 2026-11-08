import numpy as np
import pytest
import torch
from tensordict import TensorDict
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tests.unit.test_double_tcn_probe import probe_recipe


@pytest.mark.parametrize('arm',['no_update','actor_only','critic_only','joint',
                               'joint_adaptive_alpha','joint_lr_div10'])
def test_operator_partition_and_frozen_parameters(arm):
    from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
    torch.set_num_threads(1)
    model=ThreeSignalSAC(probe_recipe(),critic_regularization={'kind':'core_dropout','p':.1})
    lr=model.effective['learning_rate'];set_arm(model,arm)
    enabled=[n for n,p in model.loss.named_parameters() if p.requires_grad]
    assert all('.head.' in n or n=='log_alpha' for n in enabled)
    if arm=='actor_only':assert all(n.startswith('actor_') for n in enabled)
    if arm=='critic_only':assert all(n.startswith('qvalue_') for n in enabled)
    assert ('log_alpha' in enabled)==(arm=='joint_adaptive_alpha')
    assert all(g['lr']==(lr/10 if arm=='joint_lr_div10' else lr) for g in model.optimizer.param_groups)
    frozen=frozen_state(model)
    before={k:v.clone() for k,v in model.loss.state_dict().items()}
    td=TensorDict({'features':torch.randn(8,13,2),'mask':torch.ones(8,3,dtype=torch.bool)},[8])
    td['action']=torch.arange(8)%3;td['next']=td.clone()
    td['next','reward']=torch.ones(8,1);td['next','done']=torch.zeros(8,1,dtype=torch.bool)
    td['next','terminated']=torch.zeros(8,1,dtype=torch.bool)
    if arm!='no_update':assert np.isfinite(list(model.update(td).values())).all()
    verify_frozen_state(model,frozen)
    after=model.loss.state_dict()
    if arm=='no_update':assert all(torch.equal(v,after[k]) for k,v in before.items())
    if arm=='critic_only':assert all(torch.equal(v,after[k]) for k,v in before.items() if k.startswith('actor_'))
    if arm=='actor_only':assert all(torch.equal(v,after[k]) for k,v in before.items() if k.startswith('qvalue_'))
    if arm!='no_update':assert any(not torch.equal(v,after[k]) for k,v in before.items() if '.head.' in k)


def test_arm_rejects_unknown():
    from tools.diagnose_warmstart_drift import set_arm
    with pytest.raises(ValueError,match='DIAGNOSTIC_ARM'):
        set_arm(ThreeSignalSAC(probe_recipe()),'other')


def test_diagnostic_probe_and_tensor_digest_are_read_only():
    from tools.diagnose_warmstart_drift import probes, tensor_digest
    from types import SimpleNamespace
    torch.set_num_threads(1)
    model=ThreeSignalSAC(probe_recipe(),critic_regularization={'kind':'core_dropout','p':.1})
    x=torch.randn(8,13,2)
    td=TensorDict({'features':x,'mask':torch.ones(8,3,dtype=torch.bool)},[8])
    td['action']=torch.arange(8)%3;td['steps_to_next_obs']=torch.ones(8,1,dtype=torch.long)
    td['next']=TensorDict({'features':x.clone(),'mask':torch.ones(8,3,dtype=torch.bool),
        'reward':torch.ones(8,1),'terminated':torch.zeros(8,1,dtype=torch.bool),
        'done':torch.zeros(8,1,dtype=torch.bool)},[8])
    inputs=SimpleNamespace(valid=np.ones(8,dtype=bool),period=np.array(['TRAIN']*4+['VALIDATION']*4),
                           windows=x.numpy())
    before={k:v.clone() for k,v in model.loss.state_dict().items()};stamp=tensor_digest(td)
    result=probes(model,SimpleNamespace(inputs=inputs),model,td)
    assert tensor_digest(td)==stamp
    assert all(torch.equal(v,model.loss.state_dict()[k]) for k,v in before.items())
    assert result['TD']['native_target_max_error']<1e-5
    assert result['TRAIN']['kl_from_initial']['max']==0.
