"""Cached native exit transitions for isolated synthetic learning qualification."""
import gymnasium as gym
import numpy as np

from app.optimal_exit import targets,outcome


class OptimalExitDiagnostic(gym.Env):
    def __init__(self,cases,observations,reward_mode='dense_regret'):
        if reward_mode not in ('dense_regret','terminal_regret'):
            raise ValueError('EXIT_REWARD_MODE')
        self.cases,self.observations,self.reward_mode=cases,observations,reward_mode
        shape=observations[0].shape[1:]
        self.observation_space=gym.spaces.Dict({
            'features':gym.spaces.Box(-np.inf,np.inf,shape,np.float32),
            'action_mask':gym.spaces.Box(0,1,(5,),np.float32)})
        self.action_space=gym.spaces.Box(np.array([0,0,0,0],np.float32),
                                       np.array([4,1,1,1],np.float32))

    def observation(self):
        mask=np.array([1,0,0,0,0],np.float32);mask[self.close_command]=1
        return dict(features=self.observations[self.case_index][self.i].copy(),action_mask=mask)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.case_index=int(options['case_index']) if options and 'case_index' in options else int(self.np_random.integers(len(self.cases)))
        self.case=self.cases[self.case_index];self.i=0;self.done=False;self.actions=[]
        self.close_command=3 if self.case['direction']==1 else 4
        self.target=targets(self.case['close_net'],self.case['terminal_net'],self.case['risk_usd'])
        return self.observation(),{}

    def step(self,action):
        cmd=float(np.asarray(action)[0])
        if self.done or cmd not in (0,self.close_command):raise ValueError('EXIT_DIAGNOSTIC_ACTION')
        closing=cmd==self.close_command;self.actions.append(int(closing))
        last=self.i==len(self.case['close_net'])-1
        reward=float(self.target['reward_close' if closing else 'reward_hold'][self.i])
        self.done=closing or last
        info=dict(elapsed_bars=1,scope='SYNTHETIC_ONLY')
        if self.done:
            actions=self.actions+[0]*(len(self.case['close_net'])-len(self.actions))
            info['outcome']=outcome(self.case,actions)
        if self.reward_mode=='terminal_regret':
            reward=-info['outcome']['regret_risk'] if self.done else 0.
        if not self.done:self.i+=1
        return self.observation(),reward,self.done,False,info
