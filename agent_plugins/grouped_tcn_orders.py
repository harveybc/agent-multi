"""Small residual causal core; shared grouped market encoding, categorical actions."""
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from app.frozen_recipe_inputs import validate_recipe, require
from agent_plugins.temporal_features import CausalConv1d
from agent_plugins.hybrid_orders import SeparateHeadOrderPolicy


class ResidualCausalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.first=CausalConv1d(width,width,3,dilation)
        self.second=CausalConv1d(width,width,3,dilation)

    def forward(self,x):
        return F.silu(x+self.second(F.silu(self.first(x))))


class GroupedTCNExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space,recipe,fusion_units=8):
        validate_recipe(recipe)
        self.shape=(recipe['tensor']['window']['observations'],recipe['list']['n'])
        require(observation_space['features'].shape==self.shape,'OBSERVATION_SHAPE')
        require(type(fusion_units) is int and fusion_units>0,'FUSION_UNITS')
        super().__init__(observation_space,self.shape[0]*fusion_units)
        g=recipe['representation']['resolved']
        self.branches=nn.ModuleList()
        for i,(idx,u,k) in enumerate(zip(g['indices'],g['units'],g['kinds'])):
            self.register_buffer(f'gidx{i}',torch.tensor(idx,dtype=torch.long),persistent=False)
            self.branches.append(CausalConv1d(len(idx),u,3 if k=='conv' else 1))
        self.fusion=CausalConv1d(sum(g['units']),fusion_units,1)
        self.core=nn.Sequential(ResidualCausalBlock(fusion_units,1),ResidualCausalBlock(fusion_units,2))

    def sequence(self,obs):
        x=obs['features']
        require(x.ndim==3 and tuple(x.shape[1:])==self.shape,'OBSERVATION_SHAPE')
        x=x.transpose(1,2)
        z=torch.cat([F.silu(branch(x.index_select(1,getattr(self,f'gidx{i}'))))
                     for i,branch in enumerate(self.branches)],dim=1)
        return self.core(F.silu(self.fusion(z)))

    def forward(self,obs):
        return self.sequence(obs).flatten(1)


class GroupedTCNPolicy(SeparateHeadOrderPolicy):
    def __init__(self,*args,recipe,learned_risk=True,fusion_units=8,**kwargs):
        self.recipe,self.learned_risk,self.fusion_units=recipe,learned_risk,fusion_units
        kwargs.update(features_extractor_class=GroupedTCNExtractor,
                      features_extractor_kwargs=dict(recipe=recipe,fusion_units=fusion_units),
                      optimizer_class=torch.optim.AdamW,
                      optimizer_kwargs={'weight_decay':recipe['regularization']['wd']})
        ActorCriticPolicy.__init__(self,*args,**kwargs)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(),'fusion_units':self.fusion_units}
