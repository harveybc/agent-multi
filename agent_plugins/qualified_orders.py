"""Shallow grouped temporal representation for conditional order qualification."""
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from app.frozen_recipe_inputs import validate_recipe, require
from agent_plugins.temporal_features import CausalConv1d
from agent_plugins.hybrid_orders import SeparateHeadOrderPolicy


class GroupedSequenceExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, recipe, fusion_units=8):
        validate_recipe(recipe)
        window = recipe['tensor']['window']['observations']
        columns = recipe['list']['n']
        require(observation_space['features'].shape == (window, columns), 'OBSERVATION_SHAPE')
        require(type(fusion_units) is int and fusion_units > 0, 'FUSION_UNITS')
        super().__init__(observation_space, window * fusion_units)
        self.shape = (window, columns)
        self.branches = nn.ModuleList()
        groups = recipe['representation']['resolved']
        for i, (indices, units, kind) in enumerate(zip(groups['indices'], groups['units'], groups['kinds'])):
            self.register_buffer(f'gidx{i}', torch.tensor(indices, dtype=torch.long), persistent=False)
            self.branches.append(CausalConv1d(len(indices), units, 3 if kind == 'conv' else 1))
        self.fusion = CausalConv1d(sum(groups['units']), fusion_units, 3)

    def forward(self, obs):
        x = obs['features']
        require(x.ndim == 3 and tuple(x.shape[1:]) == self.shape, 'OBSERVATION_SHAPE')
        x = x.transpose(1, 2)
        branches = torch.cat([conv(x.index_select(1, getattr(self, f'gidx{i}')))
                              for i, conv in enumerate(self.branches)], dim=1)
        return F.silu(self.fusion(F.silu(branches))).flatten(1)


class QualifiedOrderPolicy(SeparateHeadOrderPolicy):
    def __init__(self, *args, recipe, learned_risk=True, fusion_units=8, **kwargs):
        self.recipe, self.learned_risk, self.fusion_units = recipe, learned_risk, fusion_units
        kwargs.update(features_extractor_class=GroupedSequenceExtractor,
                      features_extractor_kwargs=dict(recipe=recipe, fusion_units=fusion_units),
                      optimizer_class=torch.optim.AdamW,
                      optimizer_kwargs={'weight_decay': recipe['regularization']['wd']})
        ActorCriticPolicy.__init__(self, *args, **kwargs)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'fusion_units': self.fusion_units}
