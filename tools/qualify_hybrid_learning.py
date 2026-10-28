"""Synthetic conditional-learning diagnosis. Not financial performance."""
import argparse
import copy
import math
from pathlib import Path
import time
import types

import gymnasium as gym
import numpy as np
import torch

from tools.recipe_hybrid_queue import callback, make_model
from tools.recipe_sac_queue import write
from tests.unit.test_frozen_recipe_inputs import recipe, seal


class ObservableSignal(gym.Env):
    observation_space = gym.spaces.Dict({
        'features': gym.spaces.Box(-2, 2, (3, 2), np.float32),
        'action_mask': gym.spaces.Box(0, 1, (5,), np.float32)})
    action_space = gym.spaces.Box(np.array([0, 0, 0, 0], np.float32),
                                  np.array([4, 1, 1, 1], np.float32))

    def __init__(self, route='flat'):
        self.route = route

    def obs(self):
        masks = {'flat': [1, 1, 1, 0, 0], 'long': [1, 0, 0, 1, 0],
                 'short': [1, 0, 0, 0, 1]}
        return {'features': np.full((3, 2), self.signal, np.float32),
                'action_mask': np.array(masks[self.route], np.float32)}

    def best(self):
        if self.route == 'flat':
            return {-1: 2, 0: 0, 1: 1}[self.signal]
        return (3 if self.route == 'long' else 4) if self.signal == (-1 if self.route == 'long' else 1) else 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.n = 0
        self.signal = int(self.np_random.integers(-1, 2))
        return self.obs(), {}

    def step(self, action):
        reward = float(1 if int(action[0]) == self.best() else -1)
        self.n += 1
        self.signal = int(self.np_random.integers(-1, 2))
        return self.obs(), reward, self.n == 32, False, {'elapsed_bars': 1}


def initialize(policy, mode):
    if mode in ('baseline', 'immediate_return_control', 'no_value_loss', 'separate_value', 'supervised_control', 'gae0', 'gae05', 'residual_core', 'sb3_reference', 'core4', 'core8', 'standard_heads', 'input_skip', 'qualified'):
        return
    if mode == 'last_values_control':
        class LastValues(torch.nn.Module):
            def forward(self, obs):
                return obs['features'][:, -1, :]
        policy.features_extractor.encoder = LastValues()
        return
    for module in policy.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if mode == 'current_tap_init' and isinstance(module, torch.nn.Conv1d):
                torch.nn.init.zeros_(module.weight)
                torch.nn.init.orthogonal_(module.weight[:, :, -1], gain=1.)
    torch.nn.init.orthogonal_(policy.commands.weight, gain=.01)
    torch.nn.init.orthogonal_(policy.risks.weight, gain=.01)
    torch.nn.init.orthogonal_(policy.values.weight, gain=1.)
    if mode == 'orthogonal_zero_commands':
        torch.nn.init.zeros_(policy.commands.weight)


def probe(model, env):
    rows = []
    for signal in (-1, 0, 1):
        env.signal = signal
        obs = env.obs()
        tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            dist = model.policy.get_distribution(tensor)
            discrete = isinstance(model.action_space, gym.spaces.Discrete)
            probabilities = dist.distribution.probs if discrete else dist.cat.probs
            rows.append(dict(signal=signal, expected=env.best(),
                chosen=int(dist.get_actions(True).reshape(-1)[0]),
                probabilities=probabilities[0].cpu().tolist()))
    return rows


def run(mode, route, steps, seed, core=None):
    torch.set_num_threads(1)
    env = ObservableSignal(route)
    rec = recipe()
    if mode in ('core4', 'core8'):
        rec['representation']['model']['core'] = int(mode[4:])
        rec = seal(rec)
    if core is not None:
        rec['representation']['model']['core'] = core
        rec = seal(rec)
    model = make_model(env, rec, seed, 'cpu', False, n_steps=64, n_epochs=5)
    if mode in ('standard_heads', 'input_skip', 'qualified'):
        from agent_plugins.hybrid_orders import SeparateHeadOrderPolicy, ElapsedRolloutBuffer
        from agent_plugins.qualified_orders import QualifiedOrderPolicy
        policy_type = QualifiedOrderPolicy if mode == 'qualified' else SeparateHeadOrderPolicy
        model = type(model)(policy_type,env,policy_kwargs=dict(recipe=rec,learned_risk=False),
                    rollout_buffer_class=ElapsedRolloutBuffer,device='cpu',seed=seed,
                    n_steps=64,batch_size=64,n_epochs=5,learning_rate=.0003,gamma=.99,
                    gae_lambda=.95,ent_coef=.01,clip_range=.2,vf_coef=.5,max_grad_norm=.5)
    if mode == 'sb3_reference':
        from stable_baselines3 import PPO
        class DiscreteSignal(gym.ActionWrapper):
            action_space = gym.spaces.Discrete(3)
            def __init__(self, env):
                super().__init__(env)
                self.action_space = gym.spaces.Discrete(3)
            def action(self, a):
                return np.array([a, .5, .5, .5], np.float32)
        model = PPO('MultiInputPolicy', DiscreteSignal(env), seed=seed, device='cpu',
                    n_steps=64, batch_size=64, n_epochs=5, learning_rate=.0003,
                    ent_coef=.01, policy_kwargs=dict(net_arch=dict(pi=[16], vf=[16]),
                    optimizer_class=torch.optim.AdamW, optimizer_kwargs={'weight_decay':.01}))
    initialize(model.policy, mode)
    if mode == 'input_skip':
        ext = model.policy.features_extractor
        ext.skip = torch.nn.Linear(2, ext.features_dim, bias=False)
        def with_skip(self, obs):
            return self.encoder({'features':obs['features']}) + self.skip(obs['features'][:, -1, :])
        ext.forward = types.MethodType(with_skip, ext)
        policy = model.policy
        policy.optimizer = policy.optimizer_class(policy.parameters(), lr=.0003, **policy.optimizer_kwargs)
    if mode == 'immediate_return_control':
        model.gamma = model.rollout_buffer.gamma = 0.
    if mode == 'no_value_loss':
        model.vf_coef = 0.
    if mode in ('gae0', 'gae05'):
        model.gae_lambda = model.rollout_buffer.gae_lambda = (0. if mode == 'gae0' else .5)
    if mode == 'residual_core':
        def residual(self, observations):
            x = observations['features'].transpose(1, 2)
            x = torch.cat([conv(x.index_select(1, getattr(self, f'gidx{i}')))
                           for i, conv in enumerate(self.branches)], dim=1)
            x = self.drop(torch.nn.functional.gelu(x))
            x = torch.nn.functional.gelu(self.fuse(x))
            for conv in self.core:
                x = x + torch.nn.functional.gelu(conv(x))
            return x[:, :, -1]
        encoder = model.policy.features_extractor.encoder
        encoder.forward = types.MethodType(residual, encoder)
    if mode in ('separate_value', 'separate_orthogonal'):
        policy = model.policy
        policy.critic_encoder = copy.deepcopy(policy.features_extractor)
        policy.critic_head = copy.deepcopy(policy.head)
        original_parts = policy._parts
        def separated(self, obs):
            distribution, _ = original_parts(obs)
            self.critic_encoder.eval()
            x = self.critic_head(self.critic_encoder(obs))
            mask = obs['action_mask'].bool()
            route = mask[:, 3].long() + 2 * mask[:, 4].long()
            return distribution, self.values(x).gather(1, route[:, None])
        policy._parts = types.MethodType(separated, policy)
        policy.optimizer = policy.optimizer_class(policy.parameters(), lr=.0003, **policy.optimizer_kwargs)
    before = probe(model, env)
    start = time.monotonic()
    if mode == 'supervised_control':
        samples = []
        labels = []
        for signal in (-1, 0, 1):
            env.signal = signal
            samples.append(env.obs())
            labels.append(env.best())
        obs = {k: np.stack([x[k] for x in samples]) for k in samples[0]}
        tensor, _ = model.policy.obs_to_tensor(obs)
        for _ in range(steps):
            dist = model.policy.get_distribution(tensor)
            loss = -dist.cat.log_prob(torch.tensor(labels)).mean()
            model.policy.optimizer.zero_grad()
            loss.backward()
            model.policy.optimizer.step()
    else:
        model.learn(steps, callback=None if mode == 'sb3_reference' else callback(None, 'synthetic_conditional', start))
    after = probe(model, env)
    return dict(mode=mode, route=route, seed=seed, core=core, decisions=steps,
        seconds=time.monotonic()-start, before=before, after=after,
        passed=all(r['chosen'] == r['expected'] and r['probabilities'][r['expected']] > .6 for r in after))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', default=['baseline', 'orthogonal_zero_commands', 'orthogonal'])
    p.add_argument('--routes', nargs='+', default=['flat'])
    p.add_argument('--steps', type=int, default=16384)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--core', type=int)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('NEW_DIAGNOSTIC_OUTPUT_REQUIRED')
    reports = []
    for route in args.routes:
        for mode in args.modes:
            report = run(mode, route, args.steps, args.seed, args.core)
            reports.append(report)
            write(args.output, dict(scope='SYNTHETIC_DIAGNOSTIC_NOT_FINANCIAL', reports=reports))
            print(mode, route, report['passed'], report['seconds'], flush=True)


if __name__ == '__main__':
    main()
