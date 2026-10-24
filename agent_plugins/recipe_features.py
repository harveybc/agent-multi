"""SB3 adapter of the measured C group topology, without expert policy heads."""
import copy

from gymnasium import spaces
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from app.frozen_recipe_inputs import validate_recipe, require
from agent_plugins.temporal_features import CausalConv1d


class RecipeTemporalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, recipe):
        validate_recipe(recipe)
        require(isinstance(observation_space, spaces.Dict)
                and set(observation_space.spaces) == {"features"}, "FEATURES_ONLY")
        n = recipe["list"]["n"]
        win = recipe["tensor"]["window"]["observations"]
        require(observation_space["features"].shape == (win, n), "OBSERVATION_SHAPE")
        m, g = recipe["representation"]["model"], recipe["representation"]["resolved"]
        width = m["units"] if m["core"] is None else m["core"]
        require(type(width) is int and width > 0 and type(m["depth"]) is int and m["depth"] > 0, "CORE_SHAPE")
        super().__init__(observation_space, width)
        self.recipe = copy.deepcopy(recipe)
        self.shape = (win, n)
        self.branches = nn.ModuleList()
        for i, (indices, units, kind) in enumerate(zip(g["indices"], g["units"], g["kinds"])):
            self.register_buffer(f"gidx{i}", torch.tensor(indices, dtype=torch.long), persistent=False)
            self.branches.append(CausalConv1d(len(indices), units, 3 if kind == "conv" else 1))
        self.drop = nn.Dropout(float(recipe["regularization"]["dropout"]))
        self.fuse = CausalConv1d(sum(g["units"]), width, 3, dilation=2)
        self.core = nn.ModuleList([CausalConv1d(width, width, 3, dilation=4 * 2**i) for i in range(m["depth"])])

    def forward(self, observations):
        require(set(observations) == {"features"}, "FEATURES_ONLY")
        w = observations["features"]
        require(w.ndim == 3 and tuple(w.shape[1:]) == self.shape, "OBSERVATION_SHAPE")
        x = w.transpose(1, 2)
        x = torch.cat([conv(x.index_select(1, getattr(self, f"gidx{i}")))
                       for i, conv in enumerate(self.branches)], dim=1)
        x = self.drop(F.gelu(x))
        x = F.gelu(self.fuse(x))
        for conv in self.core:
            x = F.gelu(conv(x))
        return x[:, :, -1]
