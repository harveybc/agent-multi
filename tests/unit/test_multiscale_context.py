from dataclasses import replace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from tests.unit.test_double_tcn_probe import probe_recipe
from tests.unit.test_causal_patch_attention import ARCH
from app.frozen_recipe_inputs import FrozenRecipeInputs
from app.multiscale_context import pack_context, context_config
from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.structural_refit import export_bundle, load_bundle


CONTEXT = dict(observations=13,stride_rows=6,branch_width=4,branch_blocks=2)


def make_inputs():
    r=probe_recipe(); n=100
    times=np.arange(n,dtype=np.int64)*3600*10**9
    values=np.arange(n*2,dtype=np.float32).reshape(n,2)
    slots=np.maximum(0,np.arange(n)[:,None]-np.arange(12,-1,-1)*2)
    return FrozenRecipeInputs(r,times,times[slots],values[slots],np.arange(n)>=24,np.full(n,'TRAIN'))


def test_pack_preserves_support_and_never_reads_future():
    base=make_inputs(); out=pack_context(base,CONTEXT)
    assert np.array_equal(out.windows[:,:13],base.windows)
    assert np.array_equal(out.valid,base.valid) and np.array_equal(out.slot_ts_ns,base.slot_ts_ns)
    assert np.isnan(out.windows[:72,13:]).all()
    assert np.array_equal(out.windows[72,13:],base.windows[np.arange(0,73,6),-1])
    changed=base.windows.copy(); changed[81:,-1]+=10000
    other=pack_context(replace(base,windows=changed),CONTEXT)
    assert np.array_equal(out.windows[:81],other.windows[:81],equal_nan=True)
    changed=base.windows.copy(); changed[78,-1,0]=np.nan
    other=pack_context(replace(base,windows=changed),CONTEXT)
    assert np.isnan(other.windows[84,13:]).all() and other.valid[84]
    assert np.array_equal(other.windows[84,:13],base.windows[84])


def test_configuration_rejects_unsupported_or_hidden_parameters():
    with pytest.raises(ValueError): context_config(dict(CONTEXT,unknown=1),13,2)
    with pytest.raises(ValueError): context_config(dict(CONTEXT,stride_rows=1),13,2)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_baseline_identity_context_training_and_bundle(device,tmp_path):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    base=ThreeSignalSAC(probe_recipe(),architecture=ARCH,device=device,seed=7)
    model=ThreeSignalSAC(probe_recipe(),architecture=dict(ARCH,long_context=CONTEXT),device=device,seed=7)
    x=torch.randn(8,13,2,device=device)
    long=torch.randn_like(x); packed=torch.cat((x,long),1)
    absent=torch.cat((x,torch.full_like(x,float('nan'))),1)
    with torch.no_grad():
        assert torch.equal(base.surfaces(x),model.surfaces(packed))
        assert torch.equal(base.surfaces(x),model.surfaces(absent))
        assert torch.equal(base.q_surfaces(x),model.q_surfaces(packed))
    td=TensorDict(dict(features=packed,mask=torch.ones(8,3,dtype=torch.bool,device=device)),[8])
    td['action']=torch.arange(8,device=device)%3;td['steps_to_next_obs']=torch.ones(8,1,device=device,dtype=torch.int64)
    td['next']=td.select('features','mask').clone()
    td['next','reward']=torch.arange(8,device=device,dtype=torch.float32)[:,None]/8
    td['next','done']=torch.zeros(8,1,device=device,dtype=torch.bool)
    td['next','terminated']=td['next','done'].clone()
    before={k:v.clone() for k,v in model.loss.named_parameters() if '.context.' in k}
    for _ in range(3):
        values=model.update(td.clone())
        assert np.isfinite(list(values.values())).all()
    after=dict(model.loss.named_parameters())
    assert before and any(not torch.equal(v,after[k]) for k,v in before.items())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for k,p in after.items() if '.context.branches.' in k)
    export_bundle(tmp_path/'bundle',model,packed.cpu().numpy(),dict(epoch=1))
    restored,_=load_bundle(tmp_path/'bundle',device)
    with torch.no_grad(): assert torch.equal(model.surfaces(packed),restored.surfaces(packed))


def test_context_latent_alignment_and_prefix_causality():
    model=ThreeSignalSAC(probe_recipe(),architecture=dict(ARCH,long_context=CONTEXT))
    enc=model.actor_net.encoder
    assert enc.context.alignment.tolist()==[8,8,8,9,9,9,10,10,10,11,11,11,12]
    with torch.no_grad(): enc.context.projection.conv.weight.fill_(.1)
    x=torch.randn(8,26,2); other=x.clone()
    other[:,7:13]+=100;other[:,24:]+=100
    with torch.no_grad():
        a=enc.sequence({'features':x});b=enc.sequence({'features':other})
    torch.testing.assert_close(a[:,:,:7],b[:,:,:7],rtol=0,atol=0)
    assert not torch.equal(a[:,:,7:],b[:,:,7:])


@pytest.mark.parametrize('backend',['memory','memmap'])
def test_multiscale_interrupted_replay_matches_uninterrupted(tmp_path,backend):
    from types import SimpleNamespace
    from tools.run_structural_sac_refit import fit_epochs
    torch.set_num_threads(1)
    class Env:
        def reset(self):
            self.bar=0;self.finished=False;self.engine=SimpleNamespace(last=12)
            return self.observation()
        def observation(self):
            features=np.full((26,2),self.bar/12,np.float32)
            if self.bar%2: features[13:]=np.nan
            return dict(features=features,mask=np.ones(3,bool))
        def step(self,action):
            self.bar+=1;self.finished=self.bar==12
            return self.observation(),float(action==1),self.finished,1
    def evaluator():
        n=0
        def score(model):
            nonlocal n
            n+=1
            return {'TRAIN':{'return_per_week':n},'VALIDATION':{'return_per_week':n}}
        return score
    design=dict(replay_size=32,batch_size=8,warmup=8,update_every=4,max_epochs=2,
        patience=20,min_delta=1e-8,monitor_mode='validation',monitor_start_epoch=1,
        select_during_warmup=True)
    create=lambda:ThreeSignalSAC(probe_recipe(),architecture=dict(ARCH,long_context=CONTEXT),seed=7)
    probe=np.zeros((2,26,2),np.float32);probe[0,13:]=np.nan
    a=create();first=fit_epochs(a,Env,evaluator(),design,7,tmp_path/'whole','same',probe)
    b=create();score=evaluator()
    fit_epochs(b,Env,score,dict(design,max_epochs=1,replay_backend=backend),7,tmp_path/'restart','same',probe)
    b=create();second=fit_epochs(b,Env,score,dict(design,replay_backend=backend),7,tmp_path/'restart','same',probe)
    assert first['stopping']==second['stopping']
    assert all(torch.equal(v,b.loss.state_dict()[k]) for k,v in a.loss.state_dict().items())
