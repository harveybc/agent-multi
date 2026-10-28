"""Diagnostics of actual PPO actions and optimizer updates, not training inputs."""
from collections import Counter, defaultdict
import math
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from tools.recipe_sac_queue import now, write


def gradient_group(name):
    for key in ('branches', 'fusion', 'core', 'policy_net', 'value_net', 'commands', 'values', 'risks'):
        if key in name.split('.'):
            return key
    return 'other'


def gradient_norms(policy):
    squares = defaultdict(float)
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().double().square().sum().item())
        if not math.isfinite(value):
            raise ValueError('EXIT_NONFINITE_GRADIENT')
        squares[gradient_group(name)] += value
    return {key: math.sqrt(value) for key, value in squares.items()}


class ExitLearningTrace(BaseCallback):
    def __init__(self, root, unit, budget):
        super().__init__()
        self.root, self.unit, self.budget = root, unit, budget
        self.visits, self.commands = Counter(), Counter()
        self.updates, self.rollouts = [], []
        self.started = time.monotonic()
        self.handle = None

    def _on_training_start(self):
        self.handle = self.model.policy.optimizer.register_step_pre_hook(
            lambda *_: self.updates.append(gradient_norms(self.model.policy)))

    def _on_step(self):
        infos = self.locals['infos']
        self.model.rollout_buffer.pending_duration = np.asarray([i['elapsed_bars'] for i in infos], np.int32)
        for action, info in zip(self.locals['actions'], infos):
            command = info['executed_command']
            if float(action[0]) != command:
                raise ValueError('EXIT_SAMPLED_EXECUTED_MISMATCH')
            self.commands[str(command)] += 1
            key = f"{info['direction']}:{int(info['oracle_close_here'])}"
            self.visits[key] += 1
        if self.num_timesteps % 2048 == 0:
            write(self.root / 'PROGRESS.json', dict(state='TRAINING', unit=self.unit,
                steps=self.num_timesteps, budget=self.budget, at=now(),
                elapsed_seconds=time.monotonic()-self.started, commands=dict(self.commands),
                visits_by_direction_oracle_close=dict(self.visits)))
        return True

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        actions = buffer.actions[..., 0].reshape(-1)
        advantages = buffer.advantages.reshape(-1)
        if not np.isfinite(advantages).all():
            raise ValueError('EXIT_NONFINITE_ADVANTAGE')
        self.rollouts.append(dict(step=self.num_timesteps,
            advantages={str(int(c)):dict(n=int((actions==c).sum()),
                mean=float(advantages[actions==c].mean()), std=float(advantages[actions==c].std()))
                for c in np.unique(actions)}))

    def _on_training_end(self):
        if self.handle is not None:
            self.handle.remove()

    def report(self):
        keys = sorted({key for row in self.updates for key in row})
        return dict(commands=dict(self.commands), visits_by_direction_oracle_close=dict(self.visits),
            optimizer_updates=len(self.updates),
            gradients={key:dict(mean=float(np.mean([r.get(key,0.) for r in self.updates])),
                               max=max(r.get(key,0.) for r in self.updates)) for key in keys},
            rollouts=self.rollouts,
            final_losses={k:float(v) for k,v in self.model.logger.name_to_value.items()
                          if k.startswith('train/') and np.isscalar(v)})
