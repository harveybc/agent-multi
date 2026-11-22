from types import SimpleNamespace
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer,LazyTensorStorage
from tools.full392 import TwoAccounts,Balanced


class Account:
    def __init__(self,start,h):
        self.start=start
        self.engine=SimpleNamespace(last=h)
        self.valid=np.ones(h,bool)

    def reset(self):
        self.bar,self.finished=0,False
        return self.start

    def step(self,a):
        self.bar+=1
        self.finished=self.bar==self.engine.last
        return self.start+self.bar,1,self.finished,1


def test_complete_accounts_and_native_successors():
    env=TwoAccounts(Account(10,3),Account(70,4));obs=env.reset();pairs=[]
    while not env.finished:
        nxt,_,done,_=env.step(0);pairs.append((obs,nxt,done));obs=env.collection_observation()
    assert pairs==[(10,11,False),(70,71,False),(11,12,False),(71,72,False),
                   (12,13,True),(72,73,False),(73,74,True)]
    assert env.n==[3,4]


def test_exact_halves_after_wrap_and_recovery():
    tag=SimpleNamespace(last_source=0)
    def new():
        return Balanced(TensorDictReplayBuffer(storage=LazyTensorStorage(8),batch_size=4),8,4,lambda:tag)
    b=new()
    for j in range(13):
        tag.last_source=int(j%3==0)
        b.extend(TensorDict({'x':torch.tensor([tag.last_source])},batch_size=[1]))
    for i in range(5):
        assert b.sample()['x'].sum()==2
    c=new();c.load_state_dict(b.state_dict())
    assert torch.equal(c.sample()['x'],b.sample()['x'])
    assert c.receipt()==b.receipt()
