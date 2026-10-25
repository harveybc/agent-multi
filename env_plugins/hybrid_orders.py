"""Typed categorical orders over the pinned business engine; no expert or feedback vector."""
import copy
from dataclasses import replace
import numpy as np
from gymnasium import spaces

from app.order_rewards import Reward
from app.frozen_recipe_inputs import require
from env_plugins.recipe_native import RecipeNativeEnv, equity


COMMANDS = ('NOP','OPEN_LONG','OPEN_SHORT','CLOSE_LONG','CLOSE_SHORT')


class FillSizing:
    """Size an already-decided market order at its execution price, not an earlier close."""
    def __init__(self,base,fill_price):
        self.base,self.fill_price=base,float(fill_price)

    def __getattr__(self,name):
        return getattr(self.base,name)

    def order_units(self,balance,price,percentual_order_size=1.,risk_accepts_minimum=False):
        return self.base.order_units(balance,self.fill_price,percentual_order_size,risk_accepts_minimum)


class HybridOrderEnv(RecipeNativeEnv):
    def __init__(self,engine,inputs,gamma=.99,learned_risk=True,reward_mode='equity',penalty=0.):
        engine.cfg=copy.deepcopy(engine.cfg)
        self.learned_risk,self.reward_mode,self.penalty=learned_risk,reward_mode,penalty
        super().__init__(engine,inputs,gamma)
        # Storage codec only: command is sampled from Categorical, never a Gaussian.
        self.action_space=spaces.Box(np.array([0,0,0,0],np.float32),np.array([4,1,1,1],np.float32))
        self.observation_space=spaces.Dict({'features':self.observation_space['features'],
                                           'action_mask':spaces.Box(0,1,(5,),np.float32)})
        self.contract={'version':'categorical_conditional_beta.v1','learned_risk':learned_risk,
                       'reward':reward_mode,'penalty':penalty,'feedback':'ROUTING_MASK_ONLY'}

    def feasible_bounds(self):
        e=self.engine
        sigma=e.decision_sigma(self.bar)
        sr=e.sigma_rule[min(self.bar+1,len(e.sigma_rule)-1)]
        if not (np.isfinite(sigma) and sigma>0 and np.isfinite(sr) and sr>0):
            return None
        # Intersect network ranges with the engine's exact executable distances.
        lo=e.cfg.rules.min_distance*(1+2e-9)/sigma
        hi=e.cfg.rules.effectiveness_c*sr*(1-2e-9)/sigma
        bounds=[(max(a,lo),min(b,hi)) for a,b in (e.cfg.tp_bounds,e.cfg.sl_bounds)]
        return bounds if all(b>a for a,b in bounds) else None

    def mask(self):
        e=self.engine
        mask=np.array([1,0,0,0,0],np.float32)
        if e.pos is not None:
            mask[3 if e.pos.direction>0 else 4]=1
        elif not self.finished and e.entry_gate(self.bar)[0] and self.feasible_bounds() is not None:
            if e.cfg.sizing.order_units(e.balance,float(e.c[self.bar]),1.) is not None:
                mask[1:3]=1
        return mask

    def _observe(self):
        obs=super()._observe()
        obs['action_mask']=self.mask()
        return obs

    def reset(self,**kwargs):
        self.order_counts={}
        self.reward=Reward(self.reward_mode,self.engine.cfg.capital,self.penalty)
        return super().reset(**kwargs)

    def decode(self,action):
        x=np.asarray(action)
        require(x.shape==(4,) and np.isfinite(x).all(),'ACTION_CODEC')
        cmd=int(x[0])
        require(x[0]==cmd and 0<=cmd<5 and self.mask()[cmd]==1,'ILLEGAL_CATEGORICAL_COMMAND')
        require(((x[1:]>0)&(x[1:]<1)).all(),'RISK_CODEC')
        e=self.engine
        if cmd in (1,2):
            bounds=self.feasible_bounds()
            tp,sl=[a+float(v)*(b-a) for (a,b),v in zip(bounds,x[1:3])]
            capacity=e.cfg.sizing.units_max(e.balance,float(e.c[self.bar]))
            qmin=min(1.,(e.cfg.sizing.min_units+1e-6)/capacity)
            q=qmin+(1-qmin)*float(x[3]) if self.learned_risk else 1.
            if not self.learned_risk:
                tp,sl=[(a+b)/2 for a,b in bounds]
            e.cfg=replace(e.cfg,percentual_order_size=q)
            return cmd,tp,sl,COMMANDS[cmd]
        label=COMMANDS[cmd]
        if cmd==0 and e.pos is not None:
            label='HOLD_LONG' if e.pos.direction>0 else 'HOLD_SHORT'
        return (0 if cmd==0 else 3),1.5,2.,label

    def step(self,action):
        require(not self.finished,'STEP_AFTER_TERMINAL')
        cmd,tp,sl,label=self.decode(action)
        start=self.bar
        total,n=0.,0
        self.coverage['policy_decisions']+=1
        self.order_counts[label]=self.order_counts.get(label,0)+1
        while True:
            before=equity(self.engine,self.bar)
            cfg=self.engine.cfg
            if cmd in (1,2):
                self.engine.cfg=replace(cfg,sizing=FillSizing(cfg.sizing,self.engine.o[self.bar+1]))
            try:
                executed=self.engine.step(self.bar,cmd,tp,sl)
            finally:
                self.engine.cfg=cfg
            self.bar+=1
            after=equity(self.engine,self.bar)
            total+=self.gamma**n*self.reward.step(before,after)
            n+=1
            self.coverage['engine_bars']+=1
            if self.bar>=self.engine.last:
                self.finished=True
                break
            if self.valid[self.bar]:
                break
            cmd,tp,sl=0,1.5,2.
            self.coverage['automatic_hold_bars']+=1
        placeholder=self.finished and not self.valid[self.bar]
        if placeholder:
            obs={k:v.copy() for k,v in self.last_observation.items()}
            obs['action_mask']=np.array([1,0,0,0,0],np.float32)
        else:
            obs=self._observe()
        return obs,total,self.finished,False,{'elapsed_bars':n,'discount':self.gamma**n,
            'decision_bar':start,'bar':self.bar,'command':label,'engine_last_event':executed,
            'terminal_observation_placeholder':placeholder}
