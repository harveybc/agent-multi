import copy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from app.critic_refinement import begin_stage, sync_targets, fit_anchor
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tests.unit.test_double_tcn_probe import probe_recipe


@pytest.mark.parametrize('mode',['resume','inference','frozen_selection','fresh_fit'])
def test_non_stage_modes_do_not_change_weights_or_optimizer(mode):
    torch.set_num_threads(1)
    model=ThreeSignalSAC(probe_recipe())
    before={k:v.clone() for k,v in model.loss.state_dict().items()}
    opt=copy.deepcopy(model.optimizer.state_dict())
    assert not begin_stage(model,'test',mode=mode)['applied']
    assert all(torch.equal(v,before[k]) for k,v in model.loss.state_dict().items())
    assert opt==model.optimizer.state_dict()


def test_new_stage_once_and_target_sync():
    model=ThreeSignalSAC(probe_recipe());old={k:v.clone() for k,v in model.loss.state_dict().items()}
    begin_stage(model,'stage1',seed=31)
    changed=[k for k,v in model.loss.state_dict().items() if not torch.equal(v,old[k])]
    assert len(changed)==4 and all(k.startswith('qvalue_') and '.head.' in k for k in changed)
    with pytest.raises(ValueError,match='ALREADY'):begin_stage(model,'stage1')
    assert sync_targets(model)['exact']
    for k,v in model.loss.qvalue_network_params.items(True,True):
        assert torch.equal(v,model.loss.target_qvalue_network_params[k])


def test_anchor_restores_native_target_and_flags_on_exception():
    model=ThreeSignalSAC(probe_recipe())
    flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    native=model.loss._compute_target
    def fail():raise RuntimeError('measured failure')
    with pytest.raises(RuntimeError,match='measured failure'):
        fit_anchor(model,None,[],fail,lambda r:None)
    assert model.loss._compute_target==native and not hasattr(model,'_audit_return_target')
    assert flags=={n:p.requires_grad for n,p in model.loss.named_parameters()}


def test_anchor_updates_only_critic_heads_and_restores_native_target():
    torch.set_num_threads(1);model=ThreeSignalSAC(probe_recipe())
    begin_stage(model,'test')
    before={n:p.detach().clone() for n,p in model.loss.named_parameters()}
    native=model.loss._compute_target
    x=torch.randn(8,13,2);mask=torch.ones(8,3,dtype=torch.bool)
    batch=TensorDict(dict(features=x,mask=mask,action=torch.arange(8)%3,
        steps_to_next_obs=torch.ones(8,1,dtype=torch.long),realized_soft_return=torch.ones(8,1),
        next=TensorDict(dict(features=x,mask=mask,reward=torch.ones(8,1),
            done=torch.ones(8,1,dtype=torch.bool),terminated=torch.ones(8,1,dtype=torch.bool)),[8])),[8])
    result=fit_anchor(model,batch,[torch.arange(8)],lambda:{'finite':True},lambda r:None)
    assert len(result)==2 and model.loss._compute_target==native
    changed=[n for n,p in model.loss.named_parameters() if not torch.equal(p,before[n])]
    assert changed and all(n.startswith('qvalue_') and '.head.' in n for n in changed)


def test_warm_resume_does_not_start_critic_calibration(tmp_path):
    from tools.run_selected_warm_start import qualify_new_critics
    (tmp_path/'RESUME.json').write_text('{}')
    assert qualify_new_critics(None,None,None,{},tmp_path,'test')==dict(mode='resume',reset=False)


def test_prepared_warm_refinement_adopts_policy_in_source():
    import ast,inspect
    from tools import run_selected_warm_start as runner
    tree=ast.parse(inspect.getsource(runner.prepare))
    assert any(isinstance(n,ast.Constant) and n.value=='critic_refinement' for n in ast.walk(tree))
