"""Small market-only grouped encoder for the discrete SAC probe."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from app.frozen_recipe_inputs import require, validate_recipe
from agent_plugins.temporal_features import CausalConv1d
from agent_plugins.temporal_stage_config import activation, stage_config


class ResidualTCN(nn.Module):
    def __init__(self, width, dilation, kernel=3, dropout=0., activation_name='silu'):
        super().__init__()
        self.first = CausalConv1d(width, width, kernel, dilation)
        self.second = CausalConv1d(width, width, kernel, dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation(activation_name)

    def forward(self, x, first_norm=None):
        h = self.first(x)
        if first_norm is not None:
            h = first_norm(h)
        return self.activation(x + self.dropout2(self.second(self.dropout1(self.activation(h)))))


class DoubleTCNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, recipe, branch_width=8, core_width=16,
                 branch_blocks=2, core_blocks=2, temporal_attention=None, core_dropout=0.,long_context=None,
                 group_configs=None, core_config=None, fusion_activation='silu'):
        validate_recipe(recipe)
        self.shape = (recipe['tensor']['window']['observations'], recipe['list']['n'])
        self.transport_shape=(self.shape[0]*(2 if long_context is not None else 1),self.shape[1])
        require(observation_space['features'].shape == self.transport_shape, 'OBSERVATION_SHAPE')
        require(all(type(v) is int and v > 0 for v in
                    (branch_width, core_width, branch_blocks, core_blocks)), 'WIDTH')
        groups = recipe['representation']['resolved']
        if group_configs is None:
            group_configs = [None]*len(groups['indices'])
        require(isinstance(group_configs,list) and len(group_configs)==len(groups['indices']), 'GROUP_CONFIG_COUNT')
        specs = [stage_config(raw, width=branch_width, blocks=branch_blocks,
                  kind='tcn' if kind=='conv' else 'pointwise')
                 for raw,kind in zip(group_configs,groups['kinds'])]
        core = stage_config(core_config, width=core_width, blocks=core_blocks)
        core_width = core['width']
        super().__init__(observation_space, self.shape[0] * core_width)
        self.resolved_architecture = dict(groups=specs, core=core, fusion_activation=fusion_activation,
            token_receptive_fields_without_attention=[s['receptive_field']+core['receptive_field']-1 for s in specs],
            attention_full_causal_prefix=temporal_attention is not None,
            head_observations=self.shape[0], stride_rows=recipe['tensor']['window']['stride_rows'])
        self.fusion_activation = activation(fusion_activation)
        self.branches = nn.ModuleList()
        for i, (indices, spec) in enumerate(zip(groups['indices'], specs)):
            self.register_buffer(f'indices_{i}', torch.tensor(indices, dtype=torch.long))
            self.branches.append(nn.Sequential(
                CausalConv1d(len(indices), spec['width'], 1),
                *[ResidualTCN(spec['width'], d, spec['kernel'], activation_name=spec['activation'])
                  for d in spec['dilations']]))
        self.fusion = CausalConv1d(sum(s['width'] for s in specs), core_width, 1)
        self.core = nn.Sequential(*[ResidualTCN(core_width, d, core['kernel'], dropout=core_dropout,
                                    activation_name=core['activation']) for d in core['dilations']])
        self.post_fusion = nn.Identity()
        if temporal_attention is not None:
            from agent_plugins.causal_patch_attention import CausalPatchAttention
            require(isinstance(temporal_attention,dict)
                    and temporal_attention.get('kind')=='causal_patch', 'TEMPORAL_ATTENTION')
            params={k:v for k,v in temporal_attention.items() if k!='kind'}
            # Keep the random initialization of all existing modules unchanged.
            with torch.random.fork_rng(devices=[]):
                self.post_fusion=CausalPatchAttention(self.shape[0],**params)
        self.context=None
        if long_context is not None:
            from app.multiscale_context import context_config
            from agent_plugins.multiscale_context import LongContext
            config=context_config(long_context,self.shape[0],recipe['tensor']['window']['stride_rows'])
            with torch.random.fork_rng(devices=[]):
                self.context=LongContext(recipe,config,core_width,ResidualTCN)

    def sequence(self, observations, core_first_norm=None):
        x = observations['features']
        require(x.ndim == 3 and tuple(x.shape[1:]) == self.transport_shape, 'OBSERVATION_SHAPE')
        extra=None
        if self.context is not None:
            extra=self.context(x[:,self.shape[0]:])
            x=x[:,:self.shape[0]]
        x = x.transpose(1, 2)
        grouped = [branch(x.index_select(1, getattr(self, f'indices_{i}')))
                   for i, branch in enumerate(self.branches)]
        x = self.fusion_activation(self.fusion(torch.cat(grouped, dim=1)))
        if extra is not None: x=x+extra
        x = self.post_fusion(x)
        if core_first_norm is None:
            return self.core(x)
        for i, block in enumerate(self.core):
            x = block(x, first_norm=core_first_norm if i == 0 else None)
        return x

    def forward(self, observations):
        return self.sequence(observations).flatten(1)


