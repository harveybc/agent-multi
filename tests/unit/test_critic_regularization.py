import numpy as np
import pytest
import torch
from tensordict import TensorDict

from agent_plugins.three_signal_sac import ThreeSignalSAC
from agent_plugins.sac_critic_regularization import module_mode
from app.structural_refit import EpochPatience, export_bundle, load_bundle
from tests.unit.test_causal_patch_attention import ARCH
from tests.unit.test_double_tcn_probe import probe_recipe


def batch(device):
    obs=TensorDict(dict(features=torch.randn(8,13,2),mask=torch.ones(8,3,dtype=torch.bool)),[8])
    out=obs.clone(); out['action']=torch.arange(8)%3; out['next']=obs.clone()
    out['next','reward']=torch.ones(8,1)
    out['next','done']=torch.zeros(8,1,dtype=torch.bool)
    out['next','terminated']=torch.zeros(8,1,dtype=torch.bool)
    return out.to(device)


@pytest.mark.parametrize('kind',['none','core_dropout','head_dropout','head_batchnorm'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_twins_update_modes_buffers_and_reload(kind,device,tmp_path):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    config=dict(kind=kind)
    if 'dropout' in kind: config['p']=.1
    model=ThreeSignalSAC(probe_recipe(),architecture=ARCH,device=device,critic_regularization=config)
    assert not any(isinstance(m,torch.nn.Dropout) and m.p for m in model.actor_net.modules())
    values=model.update(batch(device))
    assert np.isfinite(list(values.values())).all()
    state={k:v.clone() for k,v in model.loss.state_dict().items()}
    if kind=='head_batchnorm':
        keys=[k for k in state if 'num_batches_tracked' in k and not k.startswith('target_')]
        assert keys and all(torch.equal(state[k],torch.ones_like(state[k])) for k in keys)
        for k,v in state.items():
            if k.startswith('qvalue_network_params') and any(n in k for n in ('running_mean','running_var','num_batches_tracked')):
                assert torch.equal(v,state['target_'+k])
    x=torch.randn(5,13,2,device=device)
    rng=torch.get_rng_state().clone()
    crng=torch.cuda.get_rng_state_all() if device=='cuda' else []
    with torch.no_grad():
        a=model.q_surfaces(x); assert torch.equal(a,model.q_surfaces(x))
        torch.testing.assert_close(a[:,:1],model.q_surfaces(x[:1]),atol=2e-6,rtol=2e-6)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(crng,torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v,model.loss.state_dict()[k]) for k,v in state.items())
    probe=x.cpu().numpy()
    export_bundle(tmp_path/'bundle',model,probe,dict(epoch=1))
    restored,_=load_bundle(tmp_path/'bundle',device)
    with torch.no_grad():
        assert torch.equal(model.surfaces(x),restored.surfaces(x))
        assert torch.equal(model.q_surfaces(x),restored.q_surfaces(x))


def test_modes_restore_on_error_and_dropout_is_real():
    model=ThreeSignalSAC(probe_recipe(),critic_regularization=dict(kind='core_dropout',p=.1))
    net=model.q_net; net.train(); net.head.eval()
    modes=[m.training for m in net.modules()]
    with pytest.raises(RuntimeError):
        with module_mode(net,False):
            assert not any(m.training for m in net.modules())
            raise RuntimeError()
    assert modes==[m.training for m in net.modules()]
    with model.loss.qvalue_network_params[0].to_module(model.loss.qvalue_network):
        x=torch.randn(5,13,2)
        assert not torch.equal(net(x),net(x))
        with module_mode(net,False): assert torch.equal(net(x),net(x))


def test_best_from_epoch_one_patience_only_after_twenty():
    monitor=EpochPatience(20,1e-8,'validation',20,select_during_warmup=True)
    assert monitor.observe(1,10)['improved']
    for _ in range(19):
        row=monitor.observe(100,0)
        assert row['bad']==0 and row['best']==10 and not row['stop']
    assert monitor.observe(100,0)['bad']==1
    assert monitor.observe(100,10+2e-8)['improved']
    assert monitor.bad==0


@pytest.mark.parametrize('kind',['none','core_dropout','head_dropout','head_batchnorm'])
def test_common_initial_weights_and_effective_metadata(kind):
    control=ThreeSignalSAC(probe_recipe(),architecture=ARCH,seed=7)
    candidate=ThreeSignalSAC(probe_recipe(),architecture=ARCH,seed=7,
                             critic_regularization=dict(kind=kind))
    old=dict(control.loss.named_parameters())
    new=dict(candidate.loss.named_parameters())
    assert old.keys() <= new.keys()
    assert all(torch.equal(v,new[k]) for k,v in old.items())
    assert candidate.effective['batch_norm']==(kind=='head_batchnorm')
    assert candidate.effective['dropout']==(.1 if 'dropout' in kind else 0.)


def test_warmup_monitor_ties_and_patience_limit():
    p=EpochPatience(20,1e-8,'validation',20,True)
    for _ in range(20): assert not p.observe(0,1)['stop']
    for i in range(1,20):
        row=p.observe(1000,1)
        assert row['bad']==i and not row['stop']
    assert p.observe(1000,1)['stop']


@pytest.mark.parametrize('kind',['none','core_dropout','head_dropout','head_batchnorm'])
def test_regularized_epoch_recovery_and_separate_late_bundle(kind,tmp_path):
    from types import SimpleNamespace
    from tools.run_structural_sac_refit import fit_epochs
    torch.set_num_threads(1)

    class TinyEnv:
        def reset(self):
            self.bar=0; self.finished=False; self.engine=SimpleNamespace(last=12)
            return self.observation()
        def observation(self):
            return dict(features=np.full((13,2),self.bar/12,np.float32),mask=np.ones(3,bool))
        def step(self,action):
            self.bar+=1; self.finished=self.bar==12
            return self.observation(),float(action==1),self.finished,1

    def evaluator():
        count=0
        def assess(model):
            nonlocal count
            count+=1
            return {'TRAIN':{'return_per_week':count},'VALIDATION':{'return_per_week':3-count}}
        return assess

    d=dict(replay_size=32,batch_size=8,warmup=8,update_every=4,max_epochs=2,
           patience=20,min_delta=1e-8,monitor_mode='validation',monitor_start_epoch=2,
           select_during_warmup=True)
    probe=np.zeros((2,13,2),np.float32)
    make=lambda:ThreeSignalSAC(probe_recipe(),architecture=ARCH,seed=7,critic_regularization=dict(kind=kind))
    a=make(); one=fit_epochs(a,TinyEnv,evaluator(),d,7,tmp_path/'one','paired',probe)
    b=make(); score=evaluator()
    fit_epochs(b,TinyEnv,score,dict(d,max_epochs=1),7,tmp_path/'resumed','paired',probe)
    b=make(); two=fit_epochs(b,TinyEnv,score,d,7,tmp_path/'resumed','paired',probe)
    assert one['stopping']==two['stopping']
    assert one['best_bundle']['relative_path']=='bundles/epoch_0001'
    assert two['best_after_warmup']['relative_path']=='bundles/epoch_0002'
    assert all(torch.equal(v,b.loss.state_dict()[k]) for k,v in a.loss.state_dict().items())
    assert (tmp_path/'resumed'/'BEST_AFTER_WARMUP.json').exists()
