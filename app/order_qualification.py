"""Observable synthetic controls; never financial or expert training data."""
import gymnasium as gym
import numpy as np


class OrderSignal(gym.Env):
    """Randomized flat/long/short routing with a current or past market signal."""
    def __init__(self, shape, lag=0, route='mixed'):
        if not 0 <= lag < shape[0] or route not in ('mixed', 'flat', 'long', 'short'):
            raise ValueError('QUALIFICATION_DESIGN')
        self.shape, self.lag, self.route_mode = shape, lag, route
        self.observation_space = gym.spaces.Dict({
            'features': gym.spaces.Box(-2, 2, shape, np.float32),
            'action_mask': gym.spaces.Box(0, 1, (5,), np.float32)})
        self.action_space = gym.spaces.Box(np.array([0, 0, 0, 0], np.float32),
                                         np.array([4, 1, 1, 1], np.float32))

    def observation(self):
        x = np.zeros(self.shape, np.float32)
        x[-1-self.lag, :] = self.signal
        masks = {'flat': [1, 1, 1, 0, 0], 'long': [1, 0, 0, 1, 0],
                 'short': [1, 0, 0, 0, 1]}
        return dict(features=x, action_mask=np.array(masks[self.route], np.float32))

    def expected(self):
        if self.route == 'flat':
            return {-1: 2, 0: 0, 1: 1}[self.signal]
        if self.route == 'long':
            return 3 if self.signal == -1 else 0
        return 4 if self.signal == 1 else 0

    def sample(self):
        self.signal = int(self.np_random.integers(-1, 2))
        self.route = str(self.np_random.choice(['flat', 'long', 'short'])) if self.route_mode == 'mixed' else self.route_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.n = 0
        self.sample()
        return self.observation(), {}

    def step(self, action):
        reward = 1. if int(action[0]) == self.expected() else -1.
        self.n += 1
        self.sample()
        return self.observation(), reward, self.n == 32, False, {'elapsed_bars': 1}


def conditional_probe(model, env):
    import torch
    rows = []
    for route in ('flat', 'long', 'short'):
        for signal in (-1, 0, 1):
            env.route, env.signal = route, signal
            obs, _ = model.policy.obs_to_tensor(env.observation())
            with torch.no_grad():
                d = model.policy.get_distribution(obs)
                probabilities = d.cat.probs[0].cpu().tolist()
                rows.append(dict(route=route, signal=signal, expected=env.expected(),
                    chosen=int(d.get_actions(True)[0, 0]), probabilities=probabilities))
    return rows


def passes(rows):
    if len(rows) != 9 or {(r['route'], r['signal']) for r in rows} != {
            (route, s) for route in ('flat', 'long', 'short') for s in (-1, 0, 1)}:
        return False
    env = OrderSignal((3, 2))
    for r in rows:
        env.route, env.signal = r['route'], r['signal']
        p = np.asarray(r['probabilities'])
        if (p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any()
                or not np.isclose(p.sum(), 1) or r['expected'] != env.expected()
                or r['chosen'] != env.expected() or p.argmax() != env.expected()
                or p[env.expected()] <= .6
                or p[env.observation()['action_mask'] == 0].any()):
            return False
    return True
