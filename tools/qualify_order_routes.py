"""Bounded synthetic mixed-route learning, not a financial result."""
import argparse
import time
from pathlib import Path
import torch
from stable_baselines3 import PPO
from agent_plugins.hybrid_orders import ElapsedRolloutBuffer
from agent_plugins.qualified_orders import QualifiedOrderPolicy
from app.order_qualification import OrderSignal, conditional_probe, passes
from app.frozen_recipe_inputs import file_sha, require
from tools.recipe_sac_queue import read, write
from tools.recipe_hybrid_queue import callback, configure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=int, default=131072)
    p.add_argument('--lag', type=int, default=0)
    a = p.parse_args()
    require(not a.output.exists(), 'NEW_QUALIFICATION_REQUIRED')
    configure()
    rec = read(a.recipe)
    env = OrderSignal((rec['tensor']['window']['observations'], rec['list']['n']), lag=a.lag)
    model = PPO(QualifiedOrderPolicy, env, policy_kwargs=dict(recipe=rec, learned_risk=False),
                rollout_buffer_class=ElapsedRolloutBuffer, device=a.device, seed=a.seed,
                n_steps=64, batch_size=64, n_epochs=5, learning_rate=.0003, gamma=.99,
                gae_lambda=.95, ent_coef=.01, clip_range=.2, vf_coef=.5, max_grad_norm=.5)
    start = time.monotonic()
    from stable_baselines3.common.callbacks import BaseCallback
    from tools.recipe_sac_queue import now
    class Heartbeat(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 4096 == 0:
                write(a.output.with_suffix('.progress.json'),dict(state='QUALIFYING',seed=a.seed,
                    lag=a.lag,step=self.num_timesteps,budget=a.steps,heartbeat_at=now(),
                    seconds=time.monotonic()-start))
            return True
    model.learn(a.steps, callback=[callback(None, 'synthetic_routes', start),Heartbeat()])
    rows = conditional_probe(model, env)
    path = a.output.with_suffix('.zip')
    model.save(path)
    loaded = PPO.load(path, device=a.device)
    replay = conditional_probe(loaded, env)
    report = dict(scope='SYNTHETIC_NOT_FINANCIAL', recipe_sha256=file_sha(a.recipe),
                  seed=a.seed, lag=a.lag, steps=a.steps, device=a.device,
                  seconds=time.monotonic()-start, rows=rows, reload_exact=rows == replay,
                  learner=dict(n_steps=64,batch_size=64,n_epochs=5,learning_rate=.0003),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  policy_class=type(model.policy).__name__,
                  passed=passes(rows) and rows == replay, model_sha256=file_sha(path))
    write(a.output, report)
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
