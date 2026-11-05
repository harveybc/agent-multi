"""Explicit online-Q training and frozen target/actor-loss critic evaluations."""
from contextlib import contextmanager

from torch import nn
from torchrl.objectives import DiscreteSACLoss


class PositionBatchNorm(nn.Module):
    """Batch statistics per channel and window position, never across time."""
    def __init__(self, channels, steps):
        super().__init__()
        self.shape = (channels, steps)
        self.bn = nn.BatchNorm1d(channels * steps, eps=1e-5, momentum=.1,
                               affine=True, track_running_stats=True)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != self.shape:
            raise ValueError('POSITION_BN_SHAPE')
        return self.bn(x.flatten(1)).reshape_as(x)


def regularization_config(config):
    config = dict(config)
    kind = config.get('kind')
    allowed = {'kind', 'p'} if kind in ('core_dropout', 'head_dropout') else {'kind'}
    if kind not in ('none', 'core_dropout', 'head_dropout', 'head_batchnorm', 'core_first_batchnorm') or set(config)-allowed:
        raise ValueError('CRITIC_REGULARIZATION_CONFIG')
    if 'dropout' in kind:
        p = config.get('p', .1)
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p < 1:
            raise ValueError('CRITIC_DROPOUT_PROBABILITY')
        config['p'] = float(p)
    return config


@contextmanager
def module_mode(module, training):
    states = [(m, m.training) for m in module.modules()]
    module.train(training)
    try:
        yield
    finally:
        for item, old in states:
            item.training = old


class RegularizedDiscreteSACLoss(DiscreteSACLoss):
    def qvalue_loss(self, tensordict):
        with module_mode(self.qvalue_network, True):
            return super().qvalue_loss(tensordict)

    def _compute_target(self, tensordict):
        with module_mode(self.qvalue_network, False), module_mode(self.actor_network, False):
            return super()._compute_target(tensordict)

    def actor_loss(self, tensordict):
        with module_mode(self.qvalue_network, False), module_mode(self.actor_network, False):
            return super().actor_loss(tensordict)


def sync_target_statistics(loss):
    # TorchRL SoftUpdate filters non-gradient buffers; copy statistics explicitly.
    for key, source in loss.qvalue_network_params.items(True, True):
        if key[-1] in ('running_mean', 'running_var', 'num_batches_tracked'):
            loss.target_qvalue_network_params[key].copy_(source)
