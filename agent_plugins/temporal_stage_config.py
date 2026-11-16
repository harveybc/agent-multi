"""Strict, backwards-compatible component specifications for temporal search."""
import copy

from torch import nn


ACTIVATIONS = {'silu': nn.SiLU, 'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}


def activation(name):
    if not isinstance(name, str) or name not in ACTIVATIONS:
        raise ValueError('TEMPORAL_ACTIVATION')
    return ACTIVATIONS[name]()


def stage_config(raw, *, width, blocks, kind='tcn'):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw)-{'width','blocks','kind','kernel','dilations','dilation_base','activation'}:
        raise ValueError('TEMPORAL_STAGE_FIELDS')
    raw = copy.deepcopy(raw)
    kind = raw.get('kind', kind)
    if kind not in ('tcn', 'pointwise', 'dense'):
        raise ValueError('TEMPORAL_STAGE_KIND')
    width, blocks = raw.get('width', width), raw.get('blocks', blocks)
    kernel = raw.get('kernel', 3 if kind == 'tcn' else 1)
    if (any(type(x) is not int or x < 1 for x in (width, kernel))
            or type(blocks) is not int or blocks < (0 if kind == 'dense' else 1)):
        raise ValueError('TEMPORAL_STAGE_DIMENSION')
    if kind == 'dense' and (blocks != 0 or kernel != 1):
        raise ValueError('TEMPORAL_DENSE_DIMENSION')
    if kind == 'pointwise' and kernel != 1:
        raise ValueError('TEMPORAL_POINTWISE_KERNEL')
    dilation_base = raw.get('dilation_base', 2)
    if type(dilation_base) is not int or dilation_base < 1 or ('dilations' in raw and 'dilation_base' in raw):
        raise ValueError('TEMPORAL_DILATION_POLICY')
    dilations = raw.get('dilations', [dilation_base**i for i in range(blocks)])
    if (not isinstance(dilations, list) or len(dilations) != blocks
            or any(type(x) is not int or x < 1 for x in dilations)):
        raise ValueError('TEMPORAL_DILATIONS')
    name = raw.get('activation', 'silu')
    activation(name)
    return dict(width=width, blocks=blocks, kind=kind, kernel=kernel,
                dilations=dilations, activation=name,
                receptive_field=1+2*(kernel-1)*sum(dilations))
