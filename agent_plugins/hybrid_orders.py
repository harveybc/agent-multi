"""Categorical commands with conditional Beta risk heads using the SB3 PPO learner.

The Box is a lossless storage codec, NOT a continuous command distribution:
slot zero is an exact categorical integer, the other slots are conditional risk.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Categorical
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agent_plugins.recipe_features import RecipeTemporalExtractor


class OrderDistribution:
    def __init__(self, logits, raw, mask, learned=True):
        if not mask.any(-1).all() or mask.shape != logits.shape:
            raise ValueError('EMPTY_ACTION_MASK')
        self.cat = Categorical(logits=logits.masked_fill(~mask.bool(), -torch.inf))
        self.mask = mask.bool()
        concentration = 1 + nn.functional.softplus(raw)
        self.risk = Beta(concentration[...,:3], concentration[...,3:])
        self.learned = learned

    def get_actions(self, deterministic=False):
        cmd = self.cat.probs.argmax(-1) if deterministic else self.cat.sample()
        risks = self.risk.mean if deterministic else self.risk.sample()
        row = torch.arange(len(cmd), device=cmd.device)
        selected = risks[row, (cmd-1).clamp(0,1)]
        active = ((cmd==1)|(cmd==2)) & self.learned
        selected = torch.where(active[:,None], selected, torch.full_like(selected,.5))
        return torch.cat([cmd[:,None].to(selected.dtype), selected], -1)

    def sample(self):
        return self.get_actions()

    def log_prob(self, actions):
        cmd = actions[:,0].long()
        if not (torch.isfinite(actions).all() and (actions[:,0]==cmd).all()
                and ((cmd>=0)&(cmd<5)).all()):
            raise ValueError('COMMAND_CODEC')
        row = torch.arange(len(cmd),device=cmd.device)
        if not self.mask[row,cmd].all():
            raise ValueError('ILLEGAL_COMMAND')
        if not ((actions[:,1:]>0)&(actions[:,1:]<1)).all():
            raise ValueError('RISK_CODEC')
        risk = self.risk.log_prob(actions[:,None,1:]).sum(-1)
        active = (((cmd==1)|(cmd==2)) & self.learned).to(risk.dtype)
        return self.cat.log_prob(cmd) + active*risk[row,(cmd-1).clamp(0,1)]

    def entropy(self):
        conditional = (self.cat.probs[:,1:3]*self.risk.entropy().sum(-1)).sum(-1)
        return self.cat.entropy() + conditional*float(self.learned)


class MarketOnlyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, recipe):
        market = spaces.Dict({'features':observation_space['features']})
        width = recipe['representation']['model']['core'] or recipe['representation']['model']['units']
        super().__init__(observation_space,width)
        self.encoder = RecipeTemporalExtractor(market,recipe)

    def forward(self, obs):
        return self.encoder({'features':obs['features']})


class OrderPolicy(ActorCriticPolicy):
    def __init__(self,*args,recipe,learned_risk=True,**kwargs):
        self.recipe, self.learned_risk = recipe,learned_risk
        kwargs.update(features_extractor_class=MarketOnlyExtractor,
                      features_extractor_kwargs={'recipe':recipe},
                      optimizer_class=torch.optim.AdamW,
                      optimizer_kwargs={'weight_decay':recipe['regularization']['wd']})
        super().__init__(*args,**kwargs)

    def _build(self, lr_schedule):
        width = self.recipe['representation']['model']['head']
        self.head = nn.Sequential(nn.Linear(self.features_dim,width),nn.Tanh())
        self.commands = nn.Linear(width,5)
        self.risks = nn.Linear(width,12)
        self.values = nn.Linear(width,3)
        nn.init.zeros_(self.commands.weight)
        nn.init.zeros_(self.commands.bias)
        self.optimizer = self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)
        self.action_dist = None
        self.features_extractor.eval()

    def set_training_mode(self, mode):
        super().set_training_mode(mode)
        # PPO must evaluate the SAME conditional likelihood as collection.
        # Dropout is explicitly inactive in this new algorithm's control.
        self.features_extractor.eval()

    def _parts(self, obs):
        x = self.head(self.features_extractor(obs))
        mask = obs['action_mask'].bool()
        dist = OrderDistribution(self.commands(x),self.risks(x).reshape(-1,2,6),mask,self.learned_risk)
        route = mask[:,3].long()+2*mask[:,4].long()
        value = self.values(x).gather(1,route[:,None])
        return dist,value

    def forward(self,obs,deterministic=False):
        d,v=self._parts(obs)
        a=d.get_actions(deterministic)
        return a,v,d.log_prob(a)

    def evaluate_actions(self,obs,actions):
        d,v=self._parts(obs)
        return v,d.log_prob(actions),d.entropy()

    def get_distribution(self,obs):
        return self._parts(obs)[0]

    def _predict(self,obs,deterministic=False):
        return self.get_distribution(obs).get_actions(deterministic)

    def predict_values(self,obs):
        return self._parts(obs)[1]

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(),'recipe':self.recipe,'learned_risk':self.learned_risk}


class ElapsedRolloutBuffer(DictRolloutBuffer):
    def reset(self):
        super().reset()
        self.durations = np.ones((self.buffer_size,self.n_envs),np.int32)
        self.pending_duration = None

    def add(self,*args,**kwargs):
        if self.pending_duration is None or (self.pending_duration<1).any():
            raise ValueError('ELAPSED_DURATION_REQUIRED')
        self.durations[self.pos] = self.pending_duration
        self.pending_duration = None
        super().add(*args,**kwargs)

    def compute_returns_and_advantage(self,last_values,dones):
        last_values=last_values.detach().cpu().numpy().flatten()
        gae=0.
        for i in reversed(range(self.buffer_size)):
            live=1.-(dones.astype(np.float32) if i==self.buffer_size-1 else self.episode_starts[i+1])
            value=last_values if i==self.buffer_size-1 else self.values[i+1]
            discount=self.gamma**self.durations[i]
            delta=self.rewards[i]+discount*value*live-self.values[i]
            gae=delta+discount*self.gae_lambda**self.durations[i]*live*gae
            self.advantages[i]=gae
        self.returns=self.advantages+self.values
