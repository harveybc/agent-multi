"""Per-critic residual moments for an explicitly scoped calibration contrast."""
from contextlib import contextmanager
import math
from types import MethodType

import torch

from agent_plugins.sac_critic_regularization import module_mode


def residual_moment_loss(prediction, target, variance_weight):
    if (not math.isfinite(variance_weight) or variance_weight <= 0
            or prediction.ndim != 2 or prediction.shape[1] < 1
            or prediction.shape != target.shape):
        raise ValueError('CRITIC_MOMENT_CONTRACT')
    error=prediction-target.detach()
    mean=error.mean(dim=1)
    variance=(error-mean[:,None]).square().mean(dim=1)
    return (mean.square()+variance_weight*variance).sum()


@contextmanager
def moment_critic_objective(model, variance_weight=None):
    """Leave native loss untouched unless this invocation explicitly opts in."""
    if variance_weight is None:
        yield
        return
    if not math.isfinite(variance_weight) or variance_weight <= 0:
        raise ValueError('CRITIC_VARIANCE_WEIGHT')
    loss=model.loss
    if loss.reduction!='mean' or loss.action_space!='categorical':
        raise ValueError('CRITIC_MOMENT_NATIVE_CONTRACT')
    original=loss.qvalue_loss
    def qvalue_loss(self, batch):
        if self._maybe_get_priority_weight(batch) is not None:
            raise ValueError('CRITIC_MOMENT_UNWEIGHTED_REPLAY_REQUIRED')
        target=self._compute_target(batch).detach()
        with module_mode(self.qvalue_network,True):
            q=self._vmap_qnetworkN0(batch.select(*self.qvalue_network.in_keys,strict=False),
                                   self.qvalue_network_params)[self.tensor_keys.action_value]
        action=batch[self.tensor_keys.action].reshape(1,-1,1).expand(q.shape[0],-1,1)
        chosen=q.gather(-1,action).squeeze(-1)
        value=residual_moment_loss(chosen,target.expand_as(chosen),variance_weight)
        return value,dict(td_error=(chosen-target).detach().abs().max(0).values)
    loss.qvalue_loss=MethodType(qvalue_loss,loss)
    try:
        yield
    finally:
        loss.qvalue_loss=original
