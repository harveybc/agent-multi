import copy
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer,LazyTensorStorage,LazyMemmapStorage
from app.g3_replay import JointReplay
from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401


def buf(path=None):
    storage=LazyTensorStorage(16) if path is None else LazyMemmapStorage(16,scratch_dir=str(path))
    return TensorDictReplayBuffer(storage=storage,batch_size=32)


@pytest.mark.parametrize('disk',[False,True])
def test_mixture_ring_and_cold_recovery(tmp_path,disk):
    a,b=buf(tmp_path/'a' if disk else None),buf(tmp_path/'b' if disk else None)
    r=JointReplay(a,b,32,7)
    for i in range(35):
        a.extend(TensorDict(dict(value=torch.tensor([i]),domain=torch.tensor([0])),batch_size=[1]))
        b.extend(TensorDict(dict(value=torch.tensor([i]),domain=torch.tensor([1])),batch_size=[1]))
    count=0
    for _ in range(10):
        batch=r.sample();count+=int(batch['domain'].sum())
        assert batch['value'].min()>=19
    assert count==32 and r.receipt()['synthetic_fraction']==.1
    state=copy.deepcopy(r.state_dict())
    saved=tmp_path/'state.pt';torch.save(state,saved)
    r2=JointReplay(buf(tmp_path/'c' if disk else None),buf(tmp_path/'d' if disk else None),32,99)
    r2.load_state_dict(torch.load(saved,weights_only=False))
    assert torch.equal(r.sample()['value'],r2.sample()['value'])
    assert r.receipt()==r2.receipt()


def test_control_no_synthetic_and_torch_rng_unchanged():
    a=buf();a.extend(TensorDict(dict(value=torch.arange(16)),batch_size=[16]))
    r=JointReplay(a,None,32,7)
    before=torch.get_rng_state().clone()
    r.sample()
    assert torch.equal(before,torch.get_rng_state())
    assert r.receipt()['synthetic_samples']==0
    with pytest.raises(ValueError):JointReplay(a,None,3,7)


def test_native_collector_epoch_recovery_and_matched_budget(tmp_path,request):
    from app.g3_replay import NativeCollector
    from app.three_signal_learning import synthetic_env
    from tools.run_structural_sac_refit import fit_epochs
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    from tests.unit.test_double_tcn_probe import probe_recipe
    torch.set_num_threads(1)
    recipe=probe_recipe();base=request.getfixturevalue('curve_env')
    real=lambda:synthetic_env(recipe,base.risk_config,101,1,.99)[0]
    syn=lambda:synthetic_env(recipe,base.risk_config,102,1,.99)[0]
    probe=real().inputs.windows[30:34].copy()
    design=dict(replay_size=64,batch_size=32,warmup=32,update_every=4,max_epochs=3,
                patience=1,min_delta=1e-8,monitor_mode='validation',matched_budget=True)
    assess=lambda m:{p:dict(return_per_week=0.,orders_per_week=3.) for p in ('TRAIN','VALIDATION')}
    def train(path,epochs):
        model=ThreeSignalSAC(recipe,seed=7)
        collector=NativeCollector(model,syn,99)
        extra=TensorDictReplayBuffer(storage=LazyTensorStorage(64),batch_size=32)
        factory=lambda r:JointReplay(r,extra,32,88,collector)
        state=fit_epochs(model,real,assess,dict(design,max_epochs=epochs),7,path,'g3-test',probe,replay_factory=factory)
        return model,state
    full,a=train(tmp_path/'a',3)
    train(tmp_path/'b',1)
    resumed,b=train(tmp_path/'b',3)
    assert a['epoch']==b['epoch']==3 and a['stopping']['bad']==2
    assert a['history'][-1]['replay_mixture']==b['history'][-1]['replay_mixture']
    for k,v in full.loss.state_dict().items():assert torch.equal(v,resumed.loss.state_dict()[k])
    assert np.isfinite(a['history'][-1]['last_update']['actor'])
