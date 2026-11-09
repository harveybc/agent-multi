import copy
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from app.actor_learning_diagnostics import (actor_terms, clear_optimizer_memory,
    matched_position_summary, actor_gradient_report, initialize_critic_heads, action_calibration)
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tests.unit.test_double_tcn_probe import probe_recipe


def test_gradient_parts_and_common_offset():
    torch.manual_seed(7)
    z=torch.randn(9,3,dtype=torch.float64,requires_grad=True)
    q=torch.randn(9,3,dtype=torch.float64)
    alpha=torch.tensor(.01)
    v,e,p=actor_terms(z,q,alpha)
    derivative=lambda x:torch.autograd.grad(x,z,retain_graph=True)[0]
    torch.testing.assert_close(derivative(v)+derivative(e),derivative(v+e))
    shifted,_,_=actor_terms(z,q+torch.randn(9,1,dtype=torch.float64)*100,alpha)
    torch.testing.assert_close(derivative(v),derivative(shifted))
    with pytest.raises(ValueError):actor_terms(z,q[:,:2],alpha)


def test_clear_adam_memory_preserves_weights_and_group_contract():
    p=torch.nn.Parameter(torch.ones(3))
    opt=torch.optim.AdamW([p],lr=.002,weight_decay=.07)
    p.sum().backward();opt.step()
    old=p.detach().clone()
    groups=copy.deepcopy(opt.state_dict()['param_groups'])
    assert clear_optimizer_memory(opt)['previous_states']==1
    assert opt.state_dict()['param_groups']==groups
    assert torch.equal(old,p) and not opt.state


def test_native_gradient_and_trainability_restoration():
    torch.set_num_threads(1)
    model=ThreeSignalSAC(probe_recipe(),critic_regularization={'kind':'core_dropout','p':.1})
    flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    before={n:p.detach().clone() for n,p in model.loss.named_parameters()}
    td=TensorDict(dict(features=torch.randn(8,13,2),mask=torch.ones(8,3,dtype=torch.bool)),[8])
    result=actor_gradient_report(model,td)
    assert result['gradient_max_error']<2e-5
    assert {n:p.requires_grad for n,p in model.loss.named_parameters()}==flags
    assert all(torch.equal(p,before[n]) for n,p in model.loss.named_parameters())


def test_position_matching_and_no_causal_claim():
    a=dict(timestamp=np.arange(4),signal=np.array([0,1,2,0]),position=np.zeros(4),
           command=np.zeros(4),reward=np.zeros(4),soft_return=np.zeros(4))
    b={k:v.copy() for k,v in a.items()}
    b['position'][1]=1;b['command'][1]=3;b['reward'][1]=2;b['soft_return'][1]=3
    out=matched_position_summary(a,b)
    assert out['strata']['different_position']==dict(rows=1,command_disagreements=1,reward_rmse=2.,return_rmse=3.)
    assert not out['causal_position_effect_established']
    b['timestamp'][0]=99
    with pytest.raises(ValueError):matched_position_summary(a,b)


def test_pair_rejects_different_learning_schedule():
    from tools.run_optimizer_memory_contrast import validate_pair
    keys=('source_identity','initial_sha256','replay_identity','return_identity',
          'snapshot_sha256','schedule_sha256','updates','batch_size','train_panel_rows',
          'replay_with_targets_digest','full_train_replay','sources',
          'optimizer_diagnostic_sha256','points','arms','device_name')
    a={k:'same' for k in keys};a['reset_optimizer']=False
    b=dict(a,reset_optimizer=True)
    validate_pair(a,b)
    b['schedule_sha256']='different'
    with pytest.raises(ValueError):validate_pair(a,b)


def test_wrong_gpu_refused_before_loading_data(monkeypatch):
    from types import SimpleNamespace
    from tools import audit_fixed_actor_critics as runner
    monkeypatch.setattr(runner,'configure',lambda seed:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'different GPU')
    args=SimpleNamespace(return_anchor=False,anchor_control=None,updates=1024,
                         expected_device_name='requested GPU')
    with pytest.raises(ValueError,match='WRONG_GPU'):
        runner.run(args)


def test_component_export_adds_weight_profile_without_changing_model(tmp_path):
    import json
    from app.selected_warm_start import export_components
    from app.frozen_recipe_inputs import file_sha
    model=ThreeSignalSAC(probe_recipe())
    before={k:v.clone() for k,v in model.loss.state_dict().items()}
    files=export_components(model,tmp_path)
    assert files['WEIGHT_PROFILE.json']==file_sha(tmp_path/'WEIGHT_PROFILE.json')
    profile=json.loads((tmp_path/'WEIGHT_PROFILE.json').read_text())
    assert profile['components']['actor_network_params/head']
    assert all(torch.equal(v,before[k]) for k,v in model.loss.state_dict().items())


def test_effective_critic_head_initialization_only_and_reproducible():
    torch.set_num_threads(1)
    model=ThreeSignalSAC(probe_recipe())
    before={k:v.clone() for k,v in model.loss.state_dict().items()}
    objects={n:id(p) for n,p in model.loss.named_parameters()}
    rng=torch.get_rng_state().clone()
    receipt=initialize_critic_heads(model,31)
    first={k:v.clone() for k,v in model.loss.state_dict().items()}
    assert torch.equal(rng,torch.get_rng_state())
    changed=[k for k in first if not torch.equal(first[k],before[k])]
    assert len(changed)==4 and all(k.startswith('qvalue_network_params.') and '.head.' in k for k in changed)
    assert receipt['twins']==2
    assert all(not torch.equal(first[k][0],first[k][1]) for k in changed)
    initialize_critic_heads(model,31)
    assert all(torch.equal(v,first[k]) for k,v in model.loss.state_dict().items())
    assert objects=={n:id(p) for n,p in model.loss.named_parameters()}
    initialize_critic_heads(model,43)
    assert any(not torch.equal(model.loss.state_dict()[k],first[k]) for k in changed)


def test_action_calibration_does_not_hide_missing_actions():
    result=action_calibration([1.,2.,0.],[1.,0.,0.],[0,1,0])
    assert result['NOP']['rows']==2 and result['NOP']['critic']['rmse']==0
    assert result['BUY']['rows']==1 and result['BUY']['critic']['rmse']==2
    assert result['SELL']==dict(rows=0,critic=None,zero=None)
    with pytest.raises(ValueError):action_calibration([0],[0],[3])


def test_head_contrast_requires_matched_optimizer_and_layout():
    from tools.run_optimizer_memory_contrast import validate_head_pair
    keys=('source_identity','initial_sha256','replay_identity','return_identity',
          'snapshot_sha256','schedule_sha256','updates','batch_size','train_panel_rows',
          'replay_with_targets_digest','full_train_replay','sources',
          'optimizer_diagnostic_sha256','points','arms','device_name')
    a={k:'same' for k in keys};a.update(reset_optimizer=True,head_init_seed=None)
    b=dict(a,head_init_seed=7)
    validate_head_pair(a,b)
    b['reset_optimizer']=False
    with pytest.raises(ValueError):validate_head_pair(a,b)
