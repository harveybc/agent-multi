"""Small SB3 feature extractor retaining time until the policy-head boundary."""
from __future__ import annotations

import math

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.left_padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(inputs, outputs, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class PerFeatureTemporalExtractor(BaseFeaturesExtractor):
    """Dict `features` is B,T,F; auxiliary state is appended after flattening.

    No learned temporal compression, pooling, right padding or global statistics.
    Actor and critic receive independently trainable copies under SB3 defaults.
    """

    def __init__(self, observation_space, branch_filters=4, core_filters=8,
                 kernel_size=3, core_layers=1):
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("TEMPORAL_DICT_OBSERVATION_REQUIRED")
        if "features" not in observation_space.spaces:
            raise ValueError("TEMPORAL_FEATURES_MISSING")
        shape = observation_space.spaces["features"].shape
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("TEMPORAL_EXPECTED_TIME_FEATURE_MATRIX")
        for value in (branch_filters, core_filters, kernel_size):
            if type(value) is not int or value < 1:
                raise ValueError("TEMPORAL_POSITIVE_INTEGER_REQUIRED")
        if type(core_layers) is not int or core_layers not in (1,2):
            raise ValueError("TEMPORAL_CORE_DEPTH_ONE_OR_TWO_REQUIRED")
        auxiliary_keys = tuple(k for k in observation_space.spaces if k != "features")
        if any(not isinstance(observation_space.spaces[k], spaces.Box)
               for k in observation_space.spaces):
            raise ValueError("TEMPORAL_BOX_COMPONENTS_REQUIRED")
        auxiliary_dim = sum(math.prod(observation_space.spaces[k].shape)
                            for k in auxiliary_keys)
        time, features = shape
        super().__init__(observation_space, time * core_filters + auxiliary_dim)
        self.auxiliary_keys = auxiliary_keys
        self.time_size, self.feature_count = time, features
        self.branches = nn.ModuleList([
            CausalConv1d(1, branch_filters, kernel_size) for _ in range(features)
        ])
        self.core = CausalConv1d(features * branch_filters, core_filters, kernel_size)
        self.extra_core = CausalConv1d(core_filters,core_filters,kernel_size) if core_layers==2 else None

    def temporal_forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (self.time_size, self.feature_count):
            raise ValueError("TEMPORAL_OBSERVATION_SHAPE_MISMATCH")
        channels = features.transpose(1, 2)
        branches = [F.silu(branch(channels[:, i:i + 1, :]))
                    for i, branch in enumerate(self.branches)]
        encoded=F.silu(self.core(torch.cat(branches, dim=1)))
        return F.silu(self.extra_core(encoded)) if self.extra_core is not None else encoded

    def forward(self, observations):
        temporal = self.temporal_forward(observations["features"])
        outputs = [temporal.flatten(start_dim=1)]
        outputs.extend(observations[k].flatten(start_dim=1) for k in self.auxiliary_keys)
        return torch.cat(outputs, dim=1)
