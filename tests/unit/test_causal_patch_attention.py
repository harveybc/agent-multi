import copy

import numpy as np
import pytest
import torch

from agent_plugins.causal_patch_attention import CausalPatchAttention
from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.structural_refit import export_bundle, load_bundle
from app.rl_subset import subset_recipe
from tests.unit.test_double_tcn_probe import probe_recipe


ARCH=dict(temporal_attention=dict(kind='causal_patch',patch_len=3,embedding=8,heads=2,feedforward=16,gate_init=.1))


@pytest.mark.parametrize('training',[True,False])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_prefix_causality_positions_and_channel_independence(training,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    block=CausalPatchAttention(13).to(device).train(training)
    x=torch.randn(3,4,13,device=device); y=x.clone(); y[:,:,7:]+=100
    with torch.no_grad():
        a,b=block(x),block(y)
        torch.testing.assert_close(a[:,:,:7],b[:,:,:7],atol=1e-6,rtol=1e-6)
        y=x.clone(); y[:,1,:]+=100
        torch.testing.assert_close(block(x)[:,0,:],block(y)[:,0,:],atol=1e-6,rtol=1e-6)
        assert a.shape==x.shape and not torch.equal(a,x)
        block.gate.zero_(); assert torch.equal(block(x),x)
    assert not torch.equal(block.positions[:,0],block.positions[:,1])
    assert block.causal_mask[0,1] and not block.causal_mask[1,0]
    assert all(m.p==0 for m in block.modules() if isinstance(m,torch.nn.Dropout))
    assert not any(isinstance(m,torch.nn.BatchNorm1d) for m in block.modules())


@pytest.mark.parametrize('kwargs',[dict(embedding=7),dict(heads=3),dict(patch_len=14),dict(gate_init=float('nan'))])
def test_bad_adapter_parameters_rejected(kwargs):
    with pytest.raises(ValueError,match='CONFIG'): CausalPatchAttention(13,**kwargs)


def test_boundary_placement_and_existing_actor_initialization():
    base=ThreeSignalSAC(probe_recipe(),seed=7)
    model=ThreeSignalSAC(probe_recipe(),seed=7,architecture=ARCH)
    with base.loss.actor_network_params.to_module(base.actor):
        base_state={k:v.clone() for k,v in base.actor_net.state_dict().items()}
    with model.loss.actor_network_params.to_module(model.actor):
        for k,v in base_state.items(): assert torch.equal(v,model.actor_net.state_dict()[k])
        with torch.no_grad(): model.actor_net.encoder.post_fusion.gate.zero_()
    x=torch.randn(3,13,2)
    with torch.no_grad(): assert torch.equal(base.surfaces(x),model.surfaces(x))
    seen=[]; enc=model.actor_net.encoder
    handles=[module.register_forward_hook(lambda m,i,o,name=name:seen.append(name))
             for name,module in [('fusion',enc.fusion),('attention',enc.post_fusion),('core',enc.core)]]
    with torch.no_grad(): model.surfaces(x)
    for h in handles: h.remove()
    assert seen==['fusion','attention','core']


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sac_update_vectorized_twins_and_checkpoint(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from tensordict import TensorDict
    model=ThreeSignalSAC(probe_recipe(),architecture=ARCH,device=device)
    obs=TensorDict(dict(features=torch.randn(4,13,2),mask=torch.ones(4,3,dtype=torch.bool)),[4])
    batch=obs.clone(); batch['action']=torch.zeros(4,dtype=torch.long); batch['next']=obs.clone()
    batch['next','reward']=torch.ones(4,1); batch['next','done']=torch.zeros(4,1,dtype=torch.bool)
    batch['next','terminated']=torch.zeros(4,1,dtype=torch.bool)
    before={k:v.clone() for k,v in model.loss.state_dict().items() if 'post_fusion' in k}
    values=model.update(batch.to(device))
    assert np.isfinite(list(values.values())).all()
    assert any(not torch.equal(v,model.loss.state_dict()[k]) for k,v in before.items())
    probe=np.random.default_rng(1).normal(size=(5,13,2)).astype('float32')
    export_bundle(tmp_path/'bundle',model,probe,dict(epoch=1))
    restored,_=load_bundle(tmp_path/'bundle',device)
    with torch.no_grad():
        x=torch.tensor(probe,device=device)
        assert torch.equal(model.surfaces(x),restored.surfaces(x))
        assert torch.equal(model.q_surfaces(x),restored.q_surfaces(x))


def test_compact_receipt_controls_refit_recipe():
    from tools.run_structural_sac_refit import compact_selected_recipe
    source=probe_recipe(); reduced=subset_recipe(source,['x'])
    result=dict(identity='id',result=dict(selected=['x'],termination='TWO_PASSES_COMPLETE'),
        test_read=False,training=False,source_unchanged=True,original_support_preserved=True,
        full_reload_exact=True,compact_parity=dict(qualified=True),final=dict(eligible=True),
        ledgers={p:dict(actions=True,trades=True,equity=True) for p in ('TRAIN','VALIDATION')})
    close=dict(identity='id',termination='TWO_PASSES_COMPLETE')
    config=dict(source_recipe=source,input_recipe=reduced,lineage=dict(selection_identity='id'))
    assert compact_selected_recipe(source,result,close,config)==reduced
    bad=copy.deepcopy(result); bad['ledgers']['VALIDATION']['actions']=False
    with pytest.raises(ValueError,match='RECEIPT'): compact_selected_recipe(source,bad,close,config)
