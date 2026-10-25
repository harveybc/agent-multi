import numpy as np
import gymnasium as gym
import torch
import pytest

from tools.recipe_hybrid_queue import make_model,callback
from tests.unit.test_frozen_recipe_inputs import recipe


@pytest.mark.parametrize('best_command',[0,1,2])
def test_categorical_learner_recovers_known_profitable_command(best_command):
    torch.set_num_threads(1)
    class Signal(gym.Env):
        observation_space=gym.spaces.Dict({'features':gym.spaces.Box(-2,2,(3,2),np.float32),
                                           'action_mask':gym.spaces.Box(0,1,(5,),np.float32)})
        action_space=gym.spaces.Box(np.array([0,0,0,0],np.float32),np.array([4,1,1,1],np.float32))
        def obs(self):
            return {'features':np.ones((3,2),np.float32),'action_mask':np.array([1,1,1,0,0],np.float32)}
        def reset(self,seed=None,options=None):
            super().reset(seed=seed);self.n=0
            return self.obs(),{}
        def step(self,a):
            self.n+=1
            return self.obs(),float(1 if int(a[0])==best_command else -1),self.n==16,False,{'elapsed_bars':1}
    e=Signal()
    m=make_model(e,recipe(),7,'cpu',False,n_steps=64,n_epochs=5)
    m.learn(8192,callback=callback(None,'synthetic',0))
    obs,_=e.reset()
    assert int(m.predict(obs,deterministic=True)[0][0])==best_command
    t,_=m.policy.obs_to_tensor(obs)
    assert m.policy.get_distribution(t).cat.probs[0,best_command]>.6
