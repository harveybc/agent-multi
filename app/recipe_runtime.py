"""Pinned label-free loading and native SAC construction for the shared recipe."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC

from app.frozen_recipe_inputs import FrozenRecipeInputs, file_sha, require
from agent_plugins.recipe_features import RecipeTemporalExtractor
from env_plugins.recipe_native import RecipeNativeEnv, DurationReplayBuffer


def attach_engine(root):
    import app
    path = str(Path(root).resolve() / 'app')
    if path not in app.__path__:
        app.__path__.append(path)


class RecipeMarket:
    def __init__(self, spec, order_contract=None):
        self.spec = spec
        self.order_contract = order_contract
        attach_engine(spec['engine_root'])
        from app.short_horizon.expert_guided.env import EnvConfig
        from app.short_horizon.expert_guided.data import calendar_from_closures
        from app.short_horizon import execution as X
        from app.short_horizon.zigzag import causal_sigma, causal_range_sigma

        paths = spec['inputs']
        require(file_sha(paths['recipe']) == spec['recipe_sha256'], 'RECIPE_PIN')
        self.recipe = rec = json.loads(Path(paths['recipe']).read_text())
        self.inputs = FrozenRecipeInputs.load(rec, paths['store'], paths['manifest'])
        calendar = rec['business']['calendar']
        require(file_sha(paths['bars']) == calendar['admitted_sha256'], 'BARS_PIN')
        require(file_sha(paths['closures']) == calendar['closures_sha256'], 'CLOSURES_PIN')
        end = pd.Timestamp(rec['split']['test_closed_from'], tz='UTC')
        frame = pd.read_parquet(paths['bars'], filters=[('available_at', '<', end)])
        start = pd.DatetimeIndex(frame.timestamp).as_unit('ns')
        available = pd.DatetimeIndex(frame.available_at).as_unit('ns')
        require(start.tz is not None and available.tz is not None
                and np.array_equal(available.asi8, start.asi8 + 3600 * 10**9)
                and (np.diff(start.asi8) > 0).all() and (available < end).all(), 'BARS_CLOCK')
        self.bars = frame.rename(columns={'BidOpen': 'OPEN', 'BidHigh': 'HIGH',
                                         'BidLow': 'LOW', 'BidClose': 'CLOSE'})
        self.bars = self.bars.set_index(start)[['OPEN', 'HIGH', 'LOW', 'CLOSE']]
        b = self.bars
        require(np.isfinite(b.to_numpy()).all() and (b.to_numpy() > 0).all()
                and (b.HIGH >= b[['OPEN', 'CLOSE', 'LOW']].max(axis=1)).all()
                and (b.LOW <= b[['OPEN', 'CLOSE', 'HIGH']].min(axis=1)).all(), 'OHLC')
        self.calendar = calendar_from_closures(json.loads(Path(paths['closures']).read_text()),
            delay=calendar['entry_delay_after_open_hours'], flat=calendar['force_flat_before_close_hours'],
            cutoff=calendar['entry_cutoff_before_flat_hours'], calendar_id=calendar['id'])
        require((self.calendar.assign(b.index) >= 0).all(), 'BAR_OUTSIDE_CALENDAR')
        self.sigma = causal_sigma(b)
        self.sigma_rule = causal_range_sigma(b)
        cfg = copy.deepcopy(rec['business']['evaluation_and_policy'])
        cfg['sizing'] = X.Sizing(**cfg['sizing'])
        cfg['profile'] = X.RealisticProfile(**cfg['profile'])
        cfg['rules'] = X.OrderRules(**cfg['rules'])
        self.config = EnvConfig(**cfg)
        # JSON round-trip normalizes tuples only. No cost/default substitution.
        require(json.loads(json.dumps(asdict(self.config))) ==
                rec['business']['evaluation_and_policy'], 'ECONOMIC_CONFIG_CHANGED')

    def env(self, period, gamma=.99):
        from app.short_horizon.expert_guided.env import BarEnv
        require(period in self.recipe['split']['periods'], 'PERIOD')
        lo, hi = self.recipe['split']['periods'][period]
        # The split is by availability; the business engine indexes bar starts.
        lo, hi = (pd.Timestamp(t, tz='UTC') - pd.Timedelta(hours=1) for t in (lo, hi))
        engine = BarEnv(self.bars, self.calendar, self.sigma, self.sigma_rule, lo, hi, self.config)
        if self.order_contract is not None:
            from env_plugins.order_contract import OrderContractEnv
            return OrderContractEnv(engine, self.inputs, gamma, self.order_contract)
        return RecipeNativeEnv(engine, self.inputs, gamma)


def build_sac(env, recipe, design, arm, device):
    require(arm in ('GROUPED', 'FLAT'), 'ARM')
    kwargs = dict(net_arch=[recipe['representation']['model']['head']],
                  optimizer_class=torch.optim.AdamW,
                  optimizer_kwargs={'weight_decay': recipe['regularization']['wd']},
                  share_features_extractor=False)
    if arm == 'GROUPED':
        kwargs.update(features_extractor_class=RecipeTemporalExtractor,
                      features_extractor_kwargs={'recipe': recipe})
    return SAC('MultiInputPolicy', env, policy_kwargs=kwargs, device=device,
               seed=design['seed'], learning_rate=design['learning_rate'],
               buffer_size=design['buffer_size'], batch_size=design['batch_size'],
               learning_starts=design['learning_starts'], gamma=design['gamma'],
               tau=design['tau'], train_freq=1, gradient_steps=1, ent_coef='auto',
               replay_buffer_class=DurationReplayBuffer, verbose=0)
