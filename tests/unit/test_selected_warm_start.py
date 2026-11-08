import numpy as np
import pytest
import torch
from tensordict import TensorDict
from tests.unit.test_double_tcn_probe import probe_recipe
from app.rl_subset import subset_recipe
from app.structural_refit import EpochPatience, build_model, export_bundle, load_bundle
from agent_plugins.three_signal_sac import ThreeSignalSAC


def test_reseed_never_reconfigures_live_thread_pool(monkeypatch):
    from app.selected_warm_start import reset_learning_rng
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda *a:pytest.fail('thread pool reconfigured'))
    reset_learning_rng(7);a=torch.rand(4);b=np.random.rand(4)
    reset_learning_rng(7)
    assert torch.equal(a,torch.rand(4)) and np.array_equal(b,np.random.rand(4))


def test_inactive_checkpoint_never_becomes_best_or_resets_patience():
    p=EpochPatience(patience=2,mode='validation')
    assert not p.observe(0.,0.,eligible=False)['improved']
    assert p.best is None and p.bad==1
    assert p.observe(-2.,-1.,eligible=True)['improved']
    assert not p.observe(0.,0.,eligible=False)['improved']
    assert p.best==-1. and p.bad==1
    assert p.observe(0.,100.,eligible=False)['stop']


def test_projection_preserves_actor_twins_targets_and_retained_tensors(tmp_path):
    from app.selected_warm_start import transfer_selected, configure_head_only, target_surfaces
    torch.set_num_threads(1)
    source=ThreeSignalSAC(probe_recipe(),critic_regularization={'kind':'core_dropout','p':.1})
    recipe=subset_recipe(source.recipe,['y'])
    dest=build_model(recipe,dict(source.effective,learning_rate=3e-5))
    report=transfer_selected(source,dest)
    assert report['sliced_tensors']==3 and report['retained_tensors_exact']
    x=torch.randn(8,13,2);x[:,:,0]=0
    with torch.no_grad():
        torch.testing.assert_close(source.surfaces(x),dest.surfaces(x[:,:,1:]),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(source.q_surfaces(x),dest.q_surfaces(x[:,:,1:]),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(target_surfaces(source,x),target_surfaces(dest,x[:,:,1:]),rtol=1e-5,atol=1e-6)
    configure_head_only(dest)
    before={k:v.clone() for k,v in dest.loss.state_dict().items()}
    td=TensorDict({'features':x[:,:,1:],'mask':torch.ones(8,3,dtype=torch.bool)},[8])
    td['action']=torch.arange(8)%3;td['next']=td.clone()
    td['next','reward']=torch.ones(8,1);td['next','done']=torch.zeros(8,1,dtype=torch.bool)
    td['next','terminated']=torch.zeros(8,1,dtype=torch.bool)
    loss=dest.update(td)
    assert np.isfinite(list(loss.values())).all()
    after=dest.loss.state_dict()
    for k,v in before.items():
        if '.encoder.' in k and not k.startswith('target_'):
            assert torch.equal(v,after[k]),k
    assert torch.equal(before['log_alpha'],after['log_alpha'])
    assert any(not torch.equal(v,after[k]) for k,v in before.items() if '.head.' in k)
    export_bundle(tmp_path/'bundle',dest,x[:,:,1:].numpy(),{'epoch':0})
    restored,_=load_bundle(tmp_path/'bundle')
    with torch.no_grad(): assert torch.equal(restored.surfaces(x[:,:,1:]),dest.surfaces(x[:,:,1:]))


def test_warmstart_epoch_zero_survives_inactive_training(tmp_path):
    from tools.run_structural_sac_refit import fit_epochs
    from types import SimpleNamespace
    r=probe_recipe();model=ThreeSignalSAC(r)
    class Env:
        def __init__(self):
            self.bar=0;self.finished=False;self.engine=SimpleNamespace(last=8)
        def reset(self):
            return {'features':np.ones((13,2),dtype=np.float32),'mask':np.ones(3,dtype=bool)}
        def step(self,action):
            self.bar+=1;self.finished=self.bar==8
            return self.reset(),float(action==1),self.finished,1
    factory=Env
    metrics=lambda activity,score:{p:dict(return_per_week=score,orders_per_week=activity)
                                    for p in ('TRAIN','VALIDATION')}
    calls=[]
    def assess(m):
        calls.append(1);return metrics(3.,1.) if len(calls)==1 else metrics(0.,0.)
    design=dict(replay_size=64,batch_size=8,warmup=8,update_every=4,max_epochs=1,
                patience=20,min_delta=1e-8,monitor_mode='validation',monitor_start_epoch=20,
                select_during_warmup=True,require_activity=True,initialize_epoch_zero=True,
                warmup_policy=True)
    state=fit_epochs(model,factory,assess,design,7,tmp_path,'test',np.ones((4,13,2),dtype=np.float32))
    assert state['best_bundle']['relative_path']=='bundles/epoch_0000'
    assert state['stopping']['best']==1.
    assert state['stopping']['epoch']==1
