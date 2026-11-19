import numpy as np
import torch
from torchrl.data import TensorDictReplayBuffer,LazyTensorStorage
from tools.augmentation389 import PairedReplay
from tools.run_discrete_curve_probe import transition

def test_native_batch_and_rng_recovery():
    def obs(v):return dict(features=np.full((13,71),v,np.float32),mask=np.ones(3,bool))
    bank=np.stack([obs(20+i)['features'] for i in range(5)])
    meta=np.array([[i%3,i+1,1+i,i==3] for i in range(4)])
    def factory():
        b=PairedReplay(TensorDictReplayBuffer(storage=LazyTensorStorage(8)),[(bank,meta)],19,32)
        b.extend(transition(obs(0),2,obs(1),0,False,1))
        return b
    b=factory()
    for _ in range(5):
        t=b.sample();v=t['features'][:,0,0];idx=v[v>=20].long()-20
        assert len(idx) in (3,4)
        assert torch.equal(t['next','features'][v>=20,0,0],(21+idx).float())
        assert torch.equal(t['action'][v>=20],idx%3)
        assert torch.equal(t['next','done'][v>=20,0],idx==3)
        assert torch.equal(t['next','reward'][v>=20,0],(idx+1).float())
        assert torch.equal(t['steps_to_next_obs'][v>=20,0],idx+1)
    assert b.syn==16
    other=factory();other.load_state_dict(b.state_dict())
    assert (b.sample()==other.sample()).all()
