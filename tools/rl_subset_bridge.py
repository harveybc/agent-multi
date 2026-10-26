"""Actual categorical PPO training behind feature_selector's RL receipt contract."""
import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from app.frozen_recipe_inputs import file_sha, require
from app.rl_subset import digest, subset_recipe, project_inputs, support_hash, groups_for_selector
from tools.recipe_sac_queue import read, write, now, evaluate, tensor_hash, preserve_rng, clean


def build_model(env, recipe, seed, device, budget, family='tcn'):
    from stable_baselines3 import PPO
    from agent_plugins.hybrid_orders import ElapsedRolloutBuffer
    from agent_plugins.grouped_tcn_orders import GroupedTCNPolicy
    from agent_plugins.qualified_orders import QualifiedOrderPolicy
    require(family in ('tcn', 'cnn'), 'POLICY_FAMILY')
    return PPO(GroupedTCNPolicy if family == 'tcn' else QualifiedOrderPolicy, env,
        policy_kwargs=dict(recipe=recipe, learned_risk=False), rollout_buffer_class=ElapsedRolloutBuffer,
        seed=seed, device=device, n_steps=budget['n_steps'], batch_size=budget['batch_size'],
        n_epochs=budget['n_epochs'], learning_rate=.0003, gamma=.99, gae_lambda=.95,
        ent_coef=.01, clip_range=.2, vf_coef=.5, max_grad_norm=.5, verbose=0)


def progress_callback(root, unit, budget):
    from stable_baselines3.common.callbacks import BaseCallback
    start = time.monotonic()
    class Progress(BaseCallback):
        def _on_step(self):
            self.model.rollout_buffer.pending_duration = np.asarray(
                [i['elapsed_bars'] for i in self.locals['infos']], dtype=np.int32)
            if self.num_timesteps % min(512, budget) == 0:
                import torch
                write(root / 'PROGRESS.json', dict(state='TRAINING', unit=unit,
                    step=self.num_timesteps, budget=budget, heartbeat_at=now(),
                    elapsed_seconds=time.monotonic() - start,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0))
            return True
    return Progress()


class SubsetMarket:
    def __init__(self, spec, columns, mode):
        from app.recipe_runtime import RecipeMarket
        require(mode in ('configured', 'zero_cost'), 'COST_MODE')
        self.base = RecipeMarket(spec)
        self.recipe = subset_recipe(self.base.recipe, columns)
        self.inputs = project_inputs(self.base.inputs, self.recipe)
        self.mode = mode

    def env(self, period):
        import pandas as pd
        from app.short_horizon.expert_guided.env import BarEnv
        from env_plugins.hybrid_orders import HybridOrderEnv
        lo, hi = (pd.Timestamp(t, tz='UTC') - pd.Timedelta(hours=1)
                  for t in self.recipe['split']['periods'][period])
        engine = BarEnv(self.base.bars, self.base.calendar, self.base.sigma, self.base.sigma_rule,
                        lo, hi, replace(self.base.config, cost_mode=self.mode))
        return HybridOrderEnv(engine, self.inputs, gamma=.99, learned_risk=False,
                              reward_mode='equity', penalty=0.)


def native_metrics(report, trades):
    costs = {k: float(report['cost_' + k + '_usd']) for k in ('spread', 'slippage', 'commission', 'financing')}
    def rate(items):
        return sum(t['net_usd'] > 0 for t in items) / len(items) if items else None
    buys = [t for t in trades if t['direction'] == 1]
    sells = [t for t in trades if t['direction'] == -1]
    require(len(buys) + len(sells) == len(trades), 'TRADE_DIRECTION')
    return clean(dict(profit=float(report['net_profit_usd']),
        sharpe=report['sharpe_daily_sqrt365'] if trades else None,
        max_drawdown=float(report['max_drawdown_pct']) / 100,
        orders=len(trades), orders_per_week=float(report['orders_per_week']),
        win_rate=rate(trades), buy_orders=len(buys), sell_orders=len(sells),
        buy_win_rate=rate(buys), sell_win_rate=rate(sells),
        early_closes=sum(t['close_reason'] in ('AGENT_CLOSE', 'REVERSAL') for t in trades),
        costs=sum(costs.values()))), costs


def run(spec, request, params, folder, progress_root, cross=False):
    from stable_baselines3 import PPO
    from tools.recipe_hybrid_queue import configure
    from feature_selector.evaluators import RLModelEvaluator
    configure()
    RLModelEvaluator().validate(params)
    market = SubsetMarket(spec, request['features'], params['cost_mode'])
    require(support_hash(market.inputs) == request['support_hash'], 'PAIRED_SUPPORT')
    require(digest(market.base.recipe['scaler']) == request['scaler_hash'], 'SCALER_PIN')
    require(groups_for_selector(market.recipe) == request['groups'], 'STRUCTURAL_GROUPS')
    require(request['splits'] == dict(train=market.recipe['split']['periods']['TRAIN'],
        selection=market.recipe['split']['periods']['INNER_A'], test_start=market.recipe['split']['test_closed_from']), 'SPLITS')
    require(request['constraints'] == dict(agent_state='none', scaling='fixed_train', clipping=False,
        fit_scope='train_only', test_access=False), 'OBSERVATION_CONTRACT')
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'REQUEST.json', request)
    write(folder / 'RECIPE.json', market.recipe)
    budget = params['training_budget']
    require(budget['total_timesteps'] % budget['n_steps'] == 0, 'EXACT_BUDGET')
    env = market.env('TRAIN')
    model = build_model(env, market.recipe, request['seed'], spec['device'], budget, params['model_id'])
    start = time.monotonic()
    initial = tensor_hash(model)
    model.learn(budget['total_timesteps'], callback=progress_callback(progress_root, request['unit'], budget['total_timesteps']))
    require(model.num_timesteps == budget['total_timesteps'] and tensor_hash(model) != initial, 'TRAINING_UPDATES')
    checkpoint = folder / 'model.zip'
    model.save(checkpoint)
    with preserve_rng():
        obs, _ = env.reset()
        loaded = PPO.load(checkpoint, device=spec['device'])
        require(tensor_hash(loaded) == tensor_hash(model), 'RELOAD_WEIGHTS')
        require(np.array_equal(model.predict(obs, deterministic=True)[0],
                               loaded.predict(obs, deterministic=True)[0]), 'RELOAD_ACTION')
        del loaded
    write(folder / 'TRAINING.json', clean(dict(initial_weights=initial, final_weights=tensor_hash(model),
        updates=model._n_updates, steps=model.num_timesteps, seed=request['seed'],
        parameters=sum(p.numel() for p in model.policy.parameters()),
        optimizer=dict(model.logger.name_to_value), dropout_effective=0.,
        agent_feedback='NONE; deterministic action-mask routing only', fresh_training=True)))
    metrics, components = {}, {}
    for label, period in [('train', 'TRAIN'), ('selection', 'INNER_A')]:
        write(progress_root / 'PROGRESS.json', dict(state='EVALUATING', unit=request['unit'],
            period=period, heartbeat_at=now(), elapsed_seconds=time.monotonic() - start))
        report = evaluate(model, market, period, folder)
        metrics[label], components[label] = native_metrics(report, read(folder / (period + '.trades.json')))
    if cross:
        original_mode = market.mode
        market.mode = 'zero_cost' if original_mode == 'configured' else 'configured'
        other = folder / ('replay_' + market.mode)
        other.mkdir()
        for period in ('TRAIN', 'INNER_A'):
            write(progress_root / 'PROGRESS.json', dict(state='CROSS_COST_REPLAY', unit=request['unit'],
                period=period, cost_mode=market.mode, heartbeat_at=now()))
            evaluate(model, market, period, other)
        market.mode = original_mode
    result = dict(schema='feature_selector.rl_result.v1', request_sha256=digest(request),
        **{k: request[k] for k in ('features', 'groups', 'seed', 'support_hash', 'scaler_hash', 'splits')},
        **{k: params[k] for k in ('model_id', 'model_sha256', 'engine_sha256', 'training_budget', 'cost_mode')},
        cost_profile_sha256=digest(params['cost_profile']), fresh_training=True, test_access=False,
        observation_contract=request['constraints'], parameter_count=sum(p.numel() for p in model.policy.parameters()),
        metrics=metrics, cost_components=components, seconds=time.monotonic() - start,
        finished_at=now(), artifacts=[dict(path=str(p.resolve()), sha256=file_sha(p))
            for p in sorted(folder.rglob('*')) if p.is_file()])
    RLModelEvaluator().verify(result, request, params, folder)
    write(folder / 'RESULT.json', result)
    env.close()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--attempt', type=int, required=True)
    a = p.parse_args()
    from tools.rl_selection_run import verify
    spec = verify(a.root)
    task = read(a.task)
    require(digest(task) == a.task.stem, 'TASK_PIN')
    run(spec, task['request'], task['params'], a.task.parent / (a.task.stem + f'-attempt-{a.attempt}'),
        a.root, task['cross'])


if __name__ == '__main__':
    main()
