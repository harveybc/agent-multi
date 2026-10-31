"""TorchRL discrete SAC; market-only surfaces and external position routing."""
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from torchrl.objectives import DiscreteSACLoss, SoftUpdate
from torchrl.objectives.utils import ValueEstimators

from agent_plugins.double_tcn_probe import DoubleTCNExtractor

KNOTS = 9
OBS_KEYS = ['features', 'route', 'progress', 'mask']


def route_surface(surface, route, progress):
    """Interpolate logits/Q values, not prices; learned layers never see routing."""
    opening = surface[..., :3]
    curves = surface[..., 3:].reshape(*surface.shape[:-1], 2, KNOTS, 2)
    # Beyond a protection (e.g. spread change), extend the endpoint segment.
    # The coordinate is not clipped; native protections still preempt the action.
    t = progress.squeeze(-1) * (KNOTS - 1)
    low = t.floor().long().clamp(0, KNOTS - 2)
    frac = t - low
    direction = (route.squeeze(-1) - 1).clamp(0, 1).long()
    select_direction = direction[..., None, None, None].expand(*direction.shape, 1, KNOTS, 2)
    curve = curves.gather(-3, select_direction).squeeze(-3)
    a = curve.gather(-2, low[..., None, None].expand(*low.shape, 1, 2)).squeeze(-2)
    b = curve.gather(-2, (low + 1)[..., None, None].expand(*low.shape, 1, 2)).squeeze(-2)
    close = a + frac[..., None] * (b - a)
    routed = torch.cat((close, torch.zeros_like(close[..., :1])), -1)
    return torch.where((route == 0).expand_as(opening), opening, routed)


class MarketSurface(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        shape = (recipe['tensor']['window']['observations'], recipe['list']['n'])
        space = gym.spaces.Dict({'features': gym.spaces.Box(-np.inf, np.inf, shape, np.float32)})
        self.encoder = DoubleTCNExtractor(space, recipe)
        self.head = nn.Sequential(nn.Linear(self.encoder.features_dim, 8), nn.SiLU(),
                                  nn.Linear(8, 3 + 2 * KNOTS * 2))

    def surface(self, features):
        return self.head(self.encoder({'features': features}))

    def forward(self, features, route, progress):
        return route_surface(self.surface(features), route, progress)


class CurveSAC:
    def __init__(self, recipe, device='cpu', seed=7, gamma=.99, alpha=.01, lr=3e-4):
        torch.manual_seed(seed)
        self.recipe, self.device = recipe, torch.device(device)
        actor_net = MarketSurface(recipe).to(device)
        q_net = MarketSurface(recipe).to(device)
        actor_module = TensorDictModule(actor_net, in_keys=OBS_KEYS[:3], out_keys=['logits'])
        self.actor = ProbabilisticActor(actor_module, in_keys={'logits': 'logits', 'mask': 'mask'},
            out_keys=['action'], distribution_class=MaskedCategorical,
            distribution_kwargs={'neg_inf': -1e9})
        q_module = TensorDictModule(q_net, in_keys=OBS_KEYS[:3], out_keys=['action_value'])
        self.loss = DiscreteSACLoss(self.actor, q_module, action_space='categorical',
            num_actions=3, num_qvalue_nets=2, alpha_init=alpha, fixed_alpha=True,
            delay_qvalue=True).to(device)
        self.loss.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
        self.updater = SoftUpdate(self.loss, tau=.005)
        self.optimizer = torch.optim.AdamW(self.loss.parameters(), lr=lr,
                                           weight_decay=recipe['regularization']['wd'])
        self.actor_net = actor_net
        self.q_net = q_net

    def distribution(self, obs):
        with self.loss.actor_network_params.to_module(self.actor):
            return self.actor.get_dist(obs)

    def surfaces(self, features):
        with self.loss.actor_network_params.to_module(self.actor):
            return self.actor_net.surface(features)

    def q_surfaces(self, features):
        values = []
        for i in range(2):
            with self.loss.qvalue_network_params[i].to_module(self.loss.qvalue_network):
                values.append(self.q_net.surface(features))
        return torch.stack(values)

    def update(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        values = self.loss(batch)
        total = values['loss_actor'] + values['loss_qvalue']
        if not torch.isfinite(total):
            raise ValueError('NONFINITE_SAC_LOSS')
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.loss.parameters(), 10., error_if_nonfinite=True)
        self.optimizer.step()
        self.updater.step()
        return {'actor': float(values['loss_actor'].detach()),
                'critic': float(values['loss_qvalue'].detach()), 'gradient_norm': float(grad)}

    def save(self, path):
        torch.save({'loss': self.loss.state_dict(), 'optimizer': self.optimizer.state_dict(),
                    'recipe_identity': self.recipe['identity']}, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state['recipe_identity'] != self.recipe['identity']:
            raise ValueError('CHECKPOINT_RECIPE_MISMATCH')
        self.loss.load_state_dict(state['loss'])
        self.optimizer.load_state_dict(state['optimizer'])


def tensor_observation(obs, device='cpu'):
    return TensorDict({key: torch.as_tensor(value, device=device).unsqueeze(0)
                       for key, value in obs.items()}, batch_size=[1])
