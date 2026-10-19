"""Real feed, preprocessing and SAC construction; never learns or fits."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.unit.test_selected_rl_comparison import fixture
from tools import selected_rl_comparison as selected


def test_selected_information_is_identical_and_encoders_are_independent(tmp_path):
    gym_root = os.environ.get('GYM_FX_CHECKOUT')
    if not gym_root:
        pytest.skip('set GYM_FX_CHECKOUT to the matching gym-fx checkout')
    request, _, _ = fixture(tmp_path)
    contract = selected.materialize(request, tmp_path / 'comparison')
    code = r'''
import importlib.util
import json
from pathlib import Path
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import SAC
from agent_plugins.sac_agent import Plugin as Agent, _policy_tensor_hash
from agent_plugins.temporal_features import PerFeatureTemporalExtractor

def forbidden(*args, **kwargs):
    raise AssertionError('NO_LEARNING_ALLOWED')
SAC.learn = forbidden
root, contract = map(Path, sys.argv[1:])
def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == (root / relative).resolve()
    return module.Plugin
Feed = load('selected_feed', 'data_feed_plugins/default_data_feed.py')
Preprocessor = load('selected_preprocessor', 'preprocessor_plugins/feature_window_preprocessor.py')
body = json.loads(contract.read_text())
cfg = {**body['common_config'], **body['splits']['train'], 'max_rows': None,
       'device': 'cpu', 'train_seed': 7}
frame = Feed(cfg).load_data(cfg)
columns = cfg['feature_columns']
step = 35
def observe(data):
    return Preprocessor(cfg).make_observation(data=data, step=step, config=cfg,
        bridge_state={'initial_cash': 10000, 'equity': 10000, 'price': 1.15})
observation = observe(frame)
assert observation['features'].shape == (13, len(columns))
values = frame[columns].to_numpy(dtype=float)
indices = step - 1 - np.arange(12, -1, -1) * 2
history = values[:step]
std = np.where(history.std(axis=0) < 1e-8, 1., history.std(axis=0))
expected = np.clip(((values[indices] - history.mean(axis=0)) / std).astype(np.float32), -10, 10)
np.testing.assert_array_equal(observation['features'], expected)
future = frame.copy()
future.iloc[step:, future.columns.get_indexer(columns)] = 1e6
np.testing.assert_array_equal(observe(future)['features'], observation['features'])
prefix = frame.iloc[:step].copy()
np.testing.assert_array_equal(observe(prefix)['features'], observation['features'])
past = frame.copy()
past.iloc[step - 1, past.columns.get_loc(columns[0])] += 0.1
assert not np.array_equal(observe(past)['features'], observation['features'])

class InformationEnv(gym.Env):
    observation_space = spaces.Dict({k: spaces.Box(-np.inf, np.inf, v.shape, np.float32)
                                     for k, v in observation.items()})
    action_space = spaces.Box(-1, 1, (1,), np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {k: v.copy() for k, v in observation.items()}, {}
    def step(self, action):
        return self.reset()[0], 0., False, False, {}

for encoder in ('flat', 'per_feature_conv1d'):
    local = {**cfg, 'observation_encoder': encoder}
    env = InformationEnv()
    agent = Agent(local)
    wrapped = agent.wrap_env(env, local)
    try:
        obs, _ = wrapped.reset(seed=7)
        if encoder == 'flat':
            np.testing.assert_array_equal(obs, spaces.flatten(env.observation_space, observation))
        else:
            np.testing.assert_array_equal(obs['features'], observation['features'])
        model = agent.build(wrapped, local)
        assert model.num_timesteps == 0 and model.replay_buffer.size() == 0
        if encoder == 'per_feature_conv1d':
            actor = model.actor.features_extractor
            critic = model.critic.features_extractor
            assert isinstance(actor, PerFeatureTemporalExtractor)
            assert isinstance(critic, PerFeatureTemporalExtractor)
            assert actor is not critic
            assert len(actor.branches) == len(critic.branches) == len(columns)
            assert actor.core.conv.in_channels == len(columns) * cfg['temporal_branch_filters']
            actor_ids = {id(p) for p in actor.parameters()}
            assert actor_ids.isdisjoint(id(p) for p in critic.parameters())
            assert all(p.requires_grad for p in actor.parameters())
            assert all(p.requires_grad for p in critic.parameters())
            assert actor.temporal_forward(torch.as_tensor(obs['features'][None])).shape == (1, 8, 13)
        before = _policy_tensor_hash(model.policy)
        action, _ = model.predict(obs, deterministic=True)
        assert np.isfinite(action).all()
        assert _policy_tensor_hash(model.policy) == before
    finally:
        wrapped.close()
'''
    result = subprocess.run([sys.executable, '-c', code, str(Path(gym_root).resolve()), str(contract)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
             'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
