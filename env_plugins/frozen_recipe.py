"""Bind frozen market tensors to gym-fx without changing execution or accounting."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd

from app.frozen_recipe_inputs import require, validate_recipe


class _FrozenPreprocessor:
    def __init__(self, inputs, frame, rows):
        self.inputs, self.frame, self.rows = inputs, frame, rows
        self.last_row = None

    def make_observation(self, *, data, step, bridge_state, config):
        require(data is self.frame and 0 < step <= len(self.rows), "ENGINE_FRAME_OR_STEP")
        row = int(self.rows[step - 1])
        observation = self.inputs.observation(row)
        self.last_row = row
        return observation


class FrozenRecipeEnv(gym.Wrapper):
    """Replace only the observation path of an unstarted, completed-bar engine.

    Preflight checks every emitted observation, including terminal observations.
    An invalid interior window is never filled or skipped. Business state stays
    inside the engine; the policy sees only the frozen selected market tensor.
    """

    def __init__(self, env, inputs):
        super().__init__(env)
        validate_recipe(inputs.recipe)
        base = env.unwrapped
        require(env is base and getattr(base, "bridge", None) is None, "UNSTARTED_BASE_ENGINE")
        cfg = base.config
        require(cfg.get("timestamp_semantics") == "completed_bar_available_at"
                and cfg.get("execution_transition") == "next_open_v2", "ENGINE_CLOCK")
        w = inputs.recipe["tensor"]["window"]
        require(base.window_size == w["observations"]
                and cfg.get("observation_stride_bars") == w["stride_rows"]
                and cfg.get("require_full_observation_window") is True, "ENGINE_WINDOW")
        frame = base.dataframe
        column = cfg.get("date_column", "DATE_TIME")
        require(column in frame or isinstance(frame.index, pd.DatetimeIndex), "ENGINE_DATE_COLUMN")
        clock = frame[column] if column in frame else frame.index
        stamps = pd.DatetimeIndex(pd.to_datetime(clock, utc=True)).as_unit("ns").asi8
        first = (w["observations"] - 1) * w["stride_rows"]
        require(len(stamps) > first and (np.diff(stamps) > 0).all(), "ENGINE_TIMESTAMPS")
        end = pd.Timestamp(inputs.recipe["split"]["test_closed_from"], tz="UTC").value
        require((stamps < end).all(), "ENGINE_TEST_ROWS")
        rows = np.searchsorted(inputs.row_ts_ns, stamps)
        bounded = np.minimum(rows, len(inputs.row_ts_ns) - 1)
        matched = (rows < len(inputs.row_ts_ns)) & (inputs.row_ts_ns[bounded] == stamps)
        require(matched[first:].all(), "ENGINE_TIMESTAMP_JOIN")
        require(inputs.valid[bounded[first:]].all(), "ENGINE_INVALID_WINDOW")
        require((inputs.period[bounded[first:]] != "").all(), "ENGINE_PERIOD")
        rows.flags.writeable = False
        self.inputs = inputs
        self.preprocessor = _FrozenPreprocessor(inputs, frame, rows)
        # Do not call the legacy scaler, clipping, NaN replacement or state builder.
        base.preprocessor_plugin = self.preprocessor
        self.observation_space = gym.spaces.Dict({"features": gym.spaces.Box(
            -np.inf, np.inf, inputs.windows.shape[1:], np.float32)})

    def _observation(self, observation, info):
        row = self.preprocessor.last_row
        require(row is not None, "ENGINE_NO_OBSERVATION")
        require(info.get("timestamp_semantics") == "completed_bar_available_at"
                and pd.Timestamp(info.get("observation_available_at")).value
                == self.inputs.row_ts_ns[row], "ENGINE_INFO_CLOCK")
        expected = self.inputs.windows[row]
        require("features" in observation and np.array_equal(observation["features"], expected),
                "ENGINE_TENSOR_CHANGED")
        # Session/quota fields emitted by the engine remain diagnostics, never policy inputs.
        return {"features": observation["features"].copy()}

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._observation(observation, info), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(observation, info), reward, terminated, truncated, info
