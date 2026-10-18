"""Cross-checkout CPU proof; explicit sibling path prevents testing installed old code."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("encoder", ["flat", "per_feature_conv1d"])
def test_real_gym_sac_static_inference(tmp_path, encoder):
    root = os.environ.get("GYM_FX_CHECKOUT")
    if not root:
        pytest.skip("set GYM_FX_CHECKOUT to the matching gym-fx revision")
    root = Path(root).resolve()
    code = r'''
import sys
from pathlib import Path
import app
gym_root, output, encoder = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
# Explicitly bind the sibling's shared app namespace in this isolated process.
app.__path__.insert(0, str(gym_root / "app"))
sys.path.insert(0, str(gym_root))
import numpy as np
import pandas as pd
import torch
import app.env
from app.env import GymFxEnv
from broker_plugins.default_broker import Plugin as Broker
from data_feed_plugins.default_data_feed import Plugin as Feed
from preprocessor_plugins.feature_window_preprocessor import Plugin as Preprocessor
from strategy_plugins.shared_execution_envelope import Plugin as Strategy
from reward_plugins.pnl_reward import Plugin as Reward
from metrics_plugins.trading_metrics import Plugin as Metrics
from agent_plugins.sac_agent import Plugin as Agent
assert Path(app.env.__file__).resolve() == gym_root / "app/env.py"
frame = pd.DataFrame({"DATE_TIME": pd.date_range("2024-01-08", periods=80, freq="1h"),
    "OPEN": 1.1, "HIGH": 1.1001, "LOW": 1.0999, "CLOSE": 1.1,
    "VOLUME": 1., "feature": np.linspace(-1., 1., 80)})
path = output / "synthetic.csv"
frame.to_csv(path, index=False)
cfg = {"input_data_file": str(path), "date_column": "DATE_TIME", "price_column": "CLOSE",
    "headers": True, "feature_columns": ["feature"], "window_size": 13,
    "observation_stride_bars": 2, "require_full_observation_window": True,
    "include_price_window": False, "include_agent_state": True,
    "agent_state_contract": "live_stationary_v2", "initial_cash": 10000.,
    "max_entries_per_day": 1, "action_space_mode": "continuous",
    "continuous_action_contract": "target_exposure_hysteresis_v2",
    "observation_encoder": encoder, "net_arch": [8], "device": "cpu",
    "buffer_size": 32, "learning_starts": 2, "batch_size": 2, "train_seed": 7}
env = GymFxEnv(cfg, Feed(cfg), Broker(cfg), Strategy(cfg), Preprocessor(cfg), Reward(cfg), Metrics(cfg))
agent = Agent(cfg)
wrapped = agent.wrap_env(env, cfg)
try:
    model = agent.build(wrapped, cfg)
    model.learn(total_timesteps=4)
    saved = output / "model.zip"
    agent.save(model, str(saved))
    policy = agent.load(str(saved), wrapped)
    weights = {key: value.clone() for key, value in policy.policy.state_dict().items()}
    obs, _ = wrapped.reset(seed=7)
    assert env.bridge.bar_index == 25
    for _ in range(8):
        action = agent.predict(policy, obs)
        obs, reward, done, truncated, info = wrapped.step(action)
        assert np.isfinite(reward)
        if done or truncated: break
    assert all(torch.equal(weights[key], value) for key, value in policy.policy.state_dict().items())
    assert env.summary()["max_entries_filled_in_day"] <= 1
finally:
    wrapped.close()
'''
    result = subprocess.run([sys.executable, "-c", code, str(root), str(tmp_path), encoder],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr
