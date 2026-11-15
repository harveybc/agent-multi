"""Separate native accounts and exactly counted synthetic replay samples."""
import copy
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer


def restore_buffer(buffer,state):
    if isinstance(buffer._storage,LazyMemmapStorage) and not buffer._storage.initialized:
        saved=TensorDict().load_state_dict(state['_storage']['_storage'],strict=False)
        buffer.extend(saved[:1])
    buffer.load_state_dict(state)


class JointReplay:
    # Recovery owns the two independent storages, not a merged account stream.
    _storage = None

    def __init__(self,real,synthetic,batch_size,seed,collector=None):
        if batch_size < 10 or not isinstance(real,TensorDictReplayBuffer):
            raise ValueError('G3_REPLAY_CONFIG')
        self.real,self.synthetic,self.batch_size = real,synthetic,batch_size
        self.rng=np.random.default_rng(seed)
        self.collector=collector
        self.samples=self.synthetic_samples=self.updates=0

    def __len__(self):
        return len(self.real)+(len(self.synthetic) if self.synthetic is not None else 0)

    def start_epoch(self):
        if self.collector is not None:self.collector.reset()

    def extend(self,item):
        self.real.extend(item)
        if self.synthetic is not None and self.collector is not None:
            self.synthetic.extend(self.collector.next())

    def sample(self):
        n=self.batch_size
        k=((self.updates+1)*n//10-self.updates*n//10) if self.synthetic is not None else 0
        if len(self.real)==0 or (k and len(self.synthetic)==0):raise ValueError('G3_EMPTY_REPLAY')
        real=self.real._storage.get(torch.as_tensor(self.rng.integers(len(self.real),size=n-k))).clone()
        if k:
            syn=self.synthetic._storage.get(torch.as_tensor(self.rng.integers(len(self.synthetic),size=k))).clone()
            batch=torch.cat([real,syn],0)
        else:batch=real
        batch=batch[torch.as_tensor(self.rng.permutation(n))]
        self.samples+=n;self.synthetic_samples+=k;self.updates+=1
        return batch

    def receipt(self):
        return dict(samples=self.samples,synthetic_samples=self.synthetic_samples,updates=self.updates,
                    synthetic_fraction=self.synthetic_samples/self.samples if self.samples else 0.)

    def state_dict(self):
        return dict(real=self.real.state_dict(),synthetic=None if self.synthetic is None else self.synthetic.state_dict(),
                    rng=copy.deepcopy(self.rng.bit_generator.state),counts=self.receipt(),
                    collector_rng=None if self.collector is None else copy.deepcopy(self.collector.rng.bit_generator.state))

    def load_state_dict(self,state):
        if (self.synthetic is None)!=(state['synthetic'] is None):raise ValueError('G3_REPLAY_ARM')
        restore_buffer(self.real,state['real'])
        if self.synthetic is not None:restore_buffer(self.synthetic,state['synthetic'])
        self.rng.bit_generator.state=state['rng']
        self.samples=state['counts']['samples'];self.synthetic_samples=state['counts']['synthetic_samples']
        self.updates=state['counts']['updates']
        if self.collector is not None:self.collector.rng.bit_generator.state=state['collector_rng']


class NativeCollector:
    def __init__(self,model,env_factory,seed):
        self.model,self.env_factory=model,env_factory
        self.rng=np.random.default_rng(seed)

    def reset(self):
        self.env=self.env_factory();self.obs=self.env.reset()

    def next(self):
        from agent_plugins.discrete_curve_sac import tensor_observation
        from tools.run_discrete_curve_probe import transition
        if self.env.finished:self.reset()
        with torch.no_grad():
            # Keep auxiliary inference from consuming the real policy's RNG.
            devices=[self.model.device.index or 0] if self.model.device.type=='cuda' else []
            with torch.random.fork_rng(devices=devices):
                probs=self.model.distribution(tensor_observation(self.obs,self.model.device)).probs[0].cpu().numpy()
        probs=probs.astype(float);probs/=probs.sum()
        action=int(self.rng.choice(3,p=probs))
        nxt,reward,done,duration=self.env.step(action)
        item=transition(self.obs,action,nxt,reward,done,duration)
        self.obs=nxt
        return item
