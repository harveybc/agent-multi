"""Small market-only grouped encoder for the discrete SAC probe."""
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from app.frozen_recipe_inputs import require, validate_recipe
from agent_plugins.temporal_features import CausalConv1d


class ResidualTCN(nn.Module):
    def __init__(self, width, dilation, kernel=3):
        super().__init__()
        self.first = CausalConv1d(width, width, kernel, dilation)
        self.second = CausalConv1d(width, width, kernel, dilation)

    def forward(self, x):
        return F.silu(x + self.second(F.silu(self.first(x))))


class DoubleTCNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, recipe, branch_width=8, core_width=16):
        validate_recipe(recipe)
        self.shape = (recipe['tensor']['window']['observations'], recipe['list']['n'])
        require(observation_space['features'].shape == self.shape, 'OBSERVATION_SHAPE')
        require(all(type(v) is int and v > 0 for v in (branch_width, core_width)), 'WIDTH')
        super().__init__(observation_space, self.shape[0] * core_width)
        groups = recipe['representation']['resolved']
        self.branches = nn.ModuleList()
        for i, (indices, kind) in enumerate(zip(groups['indices'], groups['kinds'])):
            self.register_buffer(f'indices_{i}', torch.tensor(indices, dtype=torch.long))
            kernel = 3 if kind == 'conv' else 1
            self.branches.append(nn.Sequential(
                CausalConv1d(len(indices), branch_width, 1),
                ResidualTCN(branch_width, 1, kernel),
                ResidualTCN(branch_width, 2, kernel)))
        self.fusion = CausalConv1d(len(self.branches) * branch_width, core_width, 1)
        self.core = nn.Sequential(ResidualTCN(core_width, 1), ResidualTCN(core_width, 2))

    def sequence(self, observations):
        x = observations['features']
        require(x.ndim == 3 and tuple(x.shape[1:]) == self.shape, 'OBSERVATION_SHAPE')
        x = x.transpose(1, 2)
        grouped = [branch(x.index_select(1, getattr(self, f'indices_{i}')))
                   for i, branch in enumerate(self.branches)]
        return self.core(F.silu(self.fusion(torch.cat(grouped, dim=1))))

    def forward(self, observations):
        return self.sequence(observations).flatten(1)


