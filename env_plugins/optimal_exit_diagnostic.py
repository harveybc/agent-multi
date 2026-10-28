"""Cached native exit transitions for isolated synthetic learning qualification."""
import gymnasium as gym
import numpy as np

from app.optimal_exit import targets,outcome


class OptimalExitDiagnostic(gym.Env):
    def __init__(self,cases,observations,reward_mode='dense_regret',start_mode='first'):
        if reward_mode not in ('dense_regret','terminal_regret'):
            raise ValueError('EXIT_REWARD_MODE')
        if start_mode not in ('first','uniform'):
            raise ValueError('EXIT_START_MODE')
        if not cases or len(cases)!=len(observations):
            raise ValueError('EXIT_CASE_OBSERVATION_COUNT')
        self.cases,self.observations,self.reward_mode=cases,observations,reward_mode
        self.start_mode=start_mode
        shape=observations[0].shape[1:]
        for case,x in zip(cases,observations):
            if x.shape!=(len(case['close_net']),*shape) or not np.isfinite(x).all():
                raise ValueError('EXIT_OBSERVATION_CONTRACT')
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
        self.case=self.cases[self.case_index]
        n=len(self.case['close_net'])
        self.start_index=(options or {}).get('start_index')
        if self.start_index is None:
            self.start_index=int(self.np_random.integers(n)) if self.start_mode=='uniform' else 0
        if type(self.start_index) is not int or not 0<=self.start_index<n:
            raise ValueError('EXIT_START_INDEX')
        self.i=self.start_index;self.done=False;self.actions=[]
        self.close_command=3 if self.case['direction']==1 else 4
        # Rebase the optimum to the current decision: never penalize a missed past peak.
        self.suffix=dict(self.case,close_net=self.case['close_net'][self.start_index:])
        self.target=targets(self.suffix['close_net'],self.case['terminal_net'],self.case['risk_usd'])
        return self.observation(),dict(start_index=self.start_index)

    def step(self,action):
        cmd=float(np.asarray(action)[0])
        if self.done or cmd not in (0,self.close_command):raise ValueError('EXIT_DIAGNOSTIC_ACTION')
        closing=cmd==self.close_command;self.actions.append(int(closing))
        last=self.i==len(self.case['close_net'])-1
        offset=self.i-self.start_index
        reward=float(self.target['reward_close' if closing else 'reward_hold'][offset])
        self.done=closing or last
        info=dict(elapsed_bars=1,scope='SYNTHETIC_ONLY',start_index=self.start_index,
                  decision_index=self.i,executed_command=int(cmd),
                  oracle_close_here=bool(self.target['oracle_close'][offset]),
                  direction=self.case['direction'])
        if self.done:
            actions=self.actions+[0]*(len(self.suffix['close_net'])-len(self.actions))
            info['outcome']=outcome(self.suffix,actions)
            if info['outcome']['exit_index'] is not None:
                info['outcome']['exit_index']+=self.start_index
        if self.reward_mode=='terminal_regret':
            reward=-info['outcome']['regret_risk'] if self.done else 0.
        if not self.done:self.i+=1
        return self.observation(),reward,self.done,False,info
