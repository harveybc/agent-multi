"""CPU-only proof of native SAC replay with the real completed-bar engine."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_frozen_recipe_real_gym_sac(tmp_path):
    root = os.environ.get("GYM_FX_CHECKOUT")
    if not root:
        pytest.skip("set GYM_FX_CHECKOUT to the matching gym-fx revision")
    code = r'''
import sys
from pathlib import Path
import app
from app.frozen_recipe_inputs import FrozenRecipeInputs
from env_plugins.frozen_recipe import FrozenRecipeEnv
from tests.unit.test_frozen_recipe_inputs import recipe
gym_root, output = Path(sys.argv[1]), Path(sys.argv[2])
app.__path__.insert(0, str(gym_root / "app"))
sys.path.insert(0, str(gym_root))
import numpy as np
import pandas as pd
from app.env import GymFxEnv
from broker_plugins.default_broker import Plugin as Broker
from data_feed_plugins.default_data_feed import Plugin as Feed
from preprocessor_plugins.feature_window_preprocessor import Plugin as Preprocessor
from strategy_plugins.shared_execution_envelope import Plugin as Strategy
from reward_plugins.pnl_reward import Plugin as Reward
from metrics_plugins.trading_metrics import Plugin as Metrics
from agent_plugins.sac_agent import Plugin as Agent
ts = pd.date_range("2020-01-06", periods=40, freq="h", tz="UTC")
features = pd.DataFrame({"timestamp": ts, "x": np.arange(40.), "y": np.arange(40.) + 10})
data = FrozenRecipeInputs.from_frame(recipe(), features)
frame = pd.DataFrame({"DATE_TIME": ts + pd.Timedelta(hours=1),
    "OPEN": 1.1, "HIGH": 1.1001, "LOW": 1.0999, "CLOSE": 1.1, "VOLUME": 1.,
    "unused": 999999.})
path = output / "synthetic.csv"
frame.to_csv(path, index=False)
cfg = {"input_data_file": str(path), "date_column": "DATE_TIME", "price_column": "CLOSE",
    "timestamp_semantics": "completed_bar_available_at", "execution_transition": "next_open_v2", "timeframe": "1h",
    "headers": True, "feature_columns": ["unused"], "window_size": 3,
    "observation_stride_bars": 2, "require_full_observation_window": True,
    "include_price_window": False, "include_agent_state": False, "initial_cash": 10000.,
    "max_entries_per_day": 1, "action_space_mode": "continuous",
    "continuous_action_contract": "target_exposure_hysteresis_v2",
    "observation_encoder": "shared_recipe", "shared_recipe": recipe(),
    "net_arch": [16], "device": "cpu", "buffer_size": 32, "learning_starts": 2,
    "batch_size": 2, "train_seed": 7}
base = GymFxEnv(cfg, Feed(cfg), Broker(cfg), Strategy(cfg), Preprocessor(cfg), Reward(cfg), Metrics(cfg))
env = FrozenRecipeEnv(base, data)
agent = Agent(cfg)
try:
    model = agent.build(env, cfg)
    model.learn(total_timesteps=4)
    assert model.replay_buffer.pos == 4
    for i in range(4):
        np.testing.assert_array_equal(model.replay_buffer.observations["features"][i, 0], data.windows[4 + i])
        np.testing.assert_array_equal(model.replay_buffer.next_observations["features"][i, 0], data.windows[5 + i])
    obs, info = env.reset(seed=7)
    assert base.bridge.bar_index == 5
    steps = 0
    while True:
        before = base.bridge.bar_index
        obs, reward, done, truncated, info = env.step(np.array([.9], np.float32))
        assert set(obs) == {"features"} and np.isfinite(reward)
        np.testing.assert_array_equal(obs["features"], data.windows[base.bridge.bar_index - 1])
        assert base.bridge.bar_index >= before
        assert info["action_time"] == pd.Timestamp(data.row_ts_ns[before - 1], tz="UTC").isoformat()
        steps += 1
        if done or truncated:
            break
        assert steps <= 40
    assert base.summary()["max_entries_filled_in_day"] <= 1
finally:
    env.close()
'''
    result = subprocess.run([sys.executable, "-c", code, str(Path(root).resolve()), str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=90,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr
