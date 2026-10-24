import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from app.frozen_recipe_inputs import FrozenRecipeInputs
from tests.unit.test_frozen_recipe_inputs import recipe, frame


class Engine(gym.Env):
    def __init__(self, inputs):
        self.config = dict(timestamp_semantics="completed_bar_available_at",
                           execution_transition="next_open_v2", date_column="DATE_TIME")
        self.dataframe = pd.DataFrame({"DATE_TIME": pd.to_datetime(inputs.row_ts_ns, utc=True)})
        self.observation_space = gym.spaces.Dict({"account": gym.spaces.Box(-1., 1., (1,))})
        self.action_space = gym.spaces.Box(-1., 1., (1,))
        self.window_size = 3
        self.config.update(require_full_observation_window=True, observation_stride_bars=2)
        self.resets = 0
        self.risk_checks = 0

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self.k = 4
        return self._obs(), self._info()

    def _obs(self):
        obs = self.preprocessor_plugin.make_observation(
            data=self.dataframe, step=self.k + 1, bridge_state={"equity": 999}, config=self.config)
        return {**obs, "account": np.array([.5], np.float32)}

    def _info(self):
        return {"observation_available_at": self.dataframe.DATE_TIME.iloc[self.k].isoformat(),
                "timestamp_semantics": "completed_bar_available_at", "bar_index": self.k + 1}

    def step(self, action):
        self.risk_checks += 1
        self.k += 1
        return self._obs(), 2.5, self.k == 11, False, self._info()


def test_bridge_no_legacy_scaling_no_feedback_and_preserves_steps():
    from env_plugins.frozen_recipe import FrozenRecipeEnv
    data = FrozenRecipeInputs.from_frame(recipe(), frame())
    base = Engine(data)
    env = FrozenRecipeEnv(base, data)
    obs, info = env.reset()
    np.testing.assert_array_equal(obs["features"], data.windows[4])
    assert set(obs) == {"features"}
    for k in range(5, 12):
        obs, reward, done, truncated, info = env.step(np.array([.7]))
        np.testing.assert_array_equal(obs["features"], data.windows[k])
        assert reward == 2.5 and done == (k == 11) and not truncated
    assert base.risk_checks == 7
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("fault", ["missing_stamp", "duplicate", "invalid_window", "test", "clock"])
def test_bridge_refuses_before_engine_reset(fault):
    from env_plugins.frozen_recipe import FrozenRecipeEnv
    f = frame()
    if fault == "invalid_window":
        f.loc[8, "x"] = np.nan
    data = FrozenRecipeInputs.from_frame(recipe(), f)
    base = Engine(data)
    if fault == "missing_stamp":
        base.dataframe.loc[8, "DATE_TIME"] += pd.Timedelta(minutes=1)
    if fault == "duplicate":
        base.dataframe.loc[8, "DATE_TIME"] = base.dataframe.loc[7, "DATE_TIME"]
    if fault == "test":
        base.dataframe.loc[11, "DATE_TIME"] = pd.Timestamp("2025-01-01", tz="UTC")
    if fault == "clock":
        base.config["timestamp_semantics"] = "bar_start"
    with pytest.raises(ValueError, match="FROZEN_RECIPE_"):
        FrozenRecipeEnv(base, data)
    assert base.resets == 0


def test_bridge_does_not_accept_stale_engine_info():
    from env_plugins.frozen_recipe import FrozenRecipeEnv
    data = FrozenRecipeInputs.from_frame(recipe(), frame())
    base = Engine(data)
    env = FrozenRecipeEnv(base, data)
    old = base._info
    base._info = lambda: {**old(), "observation_available_at": "2020-01-01T00:00:00Z"}
    with pytest.raises(ValueError, match="ENGINE_INFO_CLOCK"):
        env.reset()
