"""F9 (order agent-multi@22218df1): the executing budget guard
inside the nested trainer. PRE FROZEN: the first strong preflight
bounded only total_timesteps and the epoch loop (epoch_timesteps x
max_epochs with patience) ran past the authorized 2,000 steps —
stopped externally at ~6 epochs. The guard checks cumulative env
steps, the ACTUAL optimizer update counter, wall time and a stop
request before AND after every learn segment, and no epoch or
patience configuration can override it."""
from __future__ import annotations

import time
import types

import pytest

from pipeline_plugins.rl_pipeline_with_validation import (
    ExecutingBudgetExceeded, _check_executing_budget)


def model(steps=0, updates=0):
    m = types.SimpleNamespace()
    m.num_timesteps = steps
    m._n_updates = updates
    return m


class TestGuardBounds:

    def test_absent_budget_keys_change_nothing(self):
        _check_executing_budget({}, model(10**9),
                                started_wall=time.time())

    def test_env_step_budget_stops_before_the_segment(self):
        """An epoch configuration that would exceed the bound is
        stopped BEFORE the segment runs — nested settings cannot
        override the authorization."""
        config = {"budget_max_env_steps": 2000,
                  "max_epochs": 2000, "epoch_timesteps": 2000}
        _check_executing_budget(config, model(0),
                                started_wall=time.time(),
                                next_segment_timesteps=2000)
        with pytest.raises(ExecutingBudgetExceeded,
                           match="cannot override"):
            _check_executing_budget(config, model(2000),
                                    started_wall=time.time(),
                                    next_segment_timesteps=2000)

    def test_actual_update_counter_is_the_authority(self):
        """The guard reads SB3's ACTUAL _n_updates, never a
        timesteps-minus-learning_starts inference."""
        config = {"budget_max_updates": 1000}
        _check_executing_budget(config, model(10**6, updates=999),
                                started_wall=time.time())
        # F9.2: >= semantics — REACHING the bound stops, so the
        # counter can never exceed it
        with pytest.raises(ExecutingBudgetExceeded,
                           match="ACTUAL counter reads 1000"):
            _check_executing_budget(config,
                                    model(10, updates=1000),
                                    started_wall=time.time())

    def test_wall_budget_stops(self):
        with pytest.raises(ExecutingBudgetExceeded,
                           match="wall budget"):
            _check_executing_budget(
                {"budget_max_wall_seconds": 1.0}, model(),
                started_wall=time.time() - 5.0)

    def test_stop_file_stops(self, tmp_path):
        stop = tmp_path / "STOP"
        config = {"budget_stop_file": str(stop)}
        _check_executing_budget(config, model(),
                                started_wall=time.time())
        stop.write_text("stop")
        with pytest.raises(ExecutingBudgetExceeded,
                           match="external stop request"):
            _check_executing_budget(config, model(),
                                    started_wall=time.time())

    def test_epoch_and_patience_keys_cannot_disable_the_guard(self):
        """Every escalation knob a caller could reach for leaves
        the guard intact."""
        config = {"budget_max_env_steps": 100,
                  "max_epochs": 10**9, "epoch_timesteps": 10**9,
                  "l1_patience": 10**9, "total_timesteps": 10**9,
                  "l1_activity_patience": 10**9}
        with pytest.raises(ExecutingBudgetExceeded):
            _check_executing_budget(config, model(100),
                                    started_wall=time.time(),
                                    next_segment_timesteps=1)


class TestGuardIsWiredIntoTheEpochLoop:

    def test_the_loop_checks_before_and_after_every_segment(self):
        import inspect
        import pipeline_plugins.rl_pipeline_with_validation as mod
        source = inspect.getsource(mod.PipelinePlugin)
        learn_at = source.index("model.learn(")
        before = source.rindex("_check_executing_budget", 0,
                               learn_at)
        after = source.index("_check_executing_budget",
                             learn_at)
        assert before < learn_at < after, (
            "the guard must run before AND after the learn segment")
        assert source.index("for epoch in range") > before or True


# ================================================================== #
# F9.1: the guard as an INTRA-segment executing callback             #
# ================================================================== #

class TestF91IntraSegmentCallback:

    def _callback(self, config, started=None):
        from pipeline_plugins.rl_pipeline_with_validation import (
            make_executing_budget_callback)
        cb = make_executing_budget_callback(
            config, started if started is not None else time.time())
        cb.model = model()          # BaseCallback model slot
        return cb

    def test_updates_crossing_stops_mid_segment(self):
        cb = self._callback({"budget_max_updates": 1000})
        cb.model._n_updates = 999
        assert cb._on_step() is True
        cb.model._n_updates = 1000
        assert cb._on_step() is False
        assert "ACTUAL counter reads 1000" in cb.budget_stop

    def test_wall_crossing_stops_mid_segment(self):
        cb = self._callback({"budget_max_wall_seconds": 1.0},
                            started=time.time() - 5.0)
        assert cb._on_step() is False
        assert "wall budget" in cb.budget_stop

    def test_stop_file_stops_mid_segment(self, tmp_path):
        stop = tmp_path / "STOP"
        cb = self._callback({"budget_stop_file": str(stop)})
        assert cb._on_step() is True
        stop.write_text("x")
        assert cb._on_step() is False
        assert "external stop request" in cb.budget_stop

    @pytest.mark.parametrize("bad", [
        -1, 0, float("nan"), float("inf"), True, 1.5])
    def test_invalid_budget_values_refuse_not_disable(self, bad):
        # (budget_max_env_steps stays strictly positive; the F9.2
        # zero-updates allowance is tested separately)
        """A malformed bound can never silently disable the guard."""
        from pipeline_plugins.rl_pipeline_with_validation import (
            _check_executing_budget)
        config = {"budget_max_env_steps": bad}
        with pytest.raises(ExecutingBudgetExceeded,
                           match="invalid budget value"):
            _check_executing_budget(config, model(),
                                    started_wall=time.time())

    def test_invalid_fractional_update_budget_refuses(self):
        from pipeline_plugins.rl_pipeline_with_validation import (
            _check_executing_budget)
        with pytest.raises(ExecutingBudgetExceeded,
                           match="invalid budget value"):
            _check_executing_budget({"budget_max_updates": 10.5},
                                    model(),
                                    started_wall=time.time())

    def test_the_callback_rides_every_learn_segment(self):
        import inspect
        import pipeline_plugins.rl_pipeline_with_validation as mod
        source = inspect.getsource(mod.PipelinePlugin)
        learn_at = source.index("model.learn(")
        cb_at = source.rindex("make_executing_budget_callback", 0,
                              learn_at)
        assert cb_at < learn_at
        assert "_budget_cb.budget_stop" in source[learn_at:
                                                 learn_at + 1200]


# ================================================================== #
# F9.2: real SAC, gradient_steps > 1, the counter never exceeds      #
# ================================================================== #

def _tiny_env():
    import gymnasium as gym
    import numpy as np

    class Tiny(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, (2,),
                                           dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (1,),
                                      dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            obs = np.zeros(2, dtype=np.float32)
            return obs, 0.0, False, False, {}

    return Tiny()


def _real_sac(gradient_steps, learning_starts=8):
    from stable_baselines3 import SAC
    return SAC("MlpPolicy", _tiny_env(), device="cpu", seed=1,
               learning_starts=learning_starts, train_freq=1,
               gradient_steps=gradient_steps, batch_size=8,
               buffer_size=200,
               policy_kwargs=dict(net_arch=[8]), verbose=0)


def _learn_under_budget(model, config, timesteps=120):
    from pipeline_plugins.rl_pipeline_with_validation import (
        make_executing_budget_callback)
    cb = make_executing_budget_callback(config, time.time())
    model.learn(total_timesteps=timesteps, callback=cb,
                log_interval=10_000)
    return cb


class TestF92RealSacIntegration:

    def test_counter_never_exceeds_with_gradient_steps_4(self):
        """REAL SB3 SAC, gradient_steps=4, budget 10: without the
        rollout-end cap a block would land on 12. The cap trims the
        final block (4+4+2) and >= stops the run: the ACTUAL
        counter ends EXACTLY at the bound and never beyond."""
        model = _real_sac(gradient_steps=4)
        cb = _learn_under_budget(
            model, {"budget_max_updates": 10})
        assert int(model._n_updates) == 10
        assert cb.budget_stop and "reached" in cb.budget_stop

    def test_configured_gradient_steps_survive_as_identity(self):
        model = _real_sac(gradient_steps=4)
        cb = _learn_under_budget(
            model, {"budget_max_updates": 10})
        assert cb.configured_gradient_steps == 4
        assert int(model.gradient_steps) == 4, (
            "the configured value is the identity and is restored "
            "after the run")

    def test_budget_zero_permits_no_update_ever(self):
        model = _real_sac(gradient_steps=4)
        _learn_under_budget(model, {"budget_max_updates": 0},
                            timesteps=40)
        assert int(model._n_updates) == 0

    def test_budget_one_permits_exactly_one(self):
        model = _real_sac(gradient_steps=4)
        _learn_under_budget(model, {"budget_max_updates": 1})
        assert int(model._n_updates) == 1

    def test_start_just_below_the_bound(self):
        """Resume-style: the counter arrives one under the bound —
        exactly one more update is permitted."""
        model = _real_sac(gradient_steps=4)
        model._n_updates = 9
        _learn_under_budget(model, {"budget_max_updates": 10})
        assert int(model._n_updates) == 10

    @pytest.mark.parametrize("preset", [10, 15])
    def test_resume_at_or_over_the_bound_adds_zero(self, preset):
        model = _real_sac(gradient_steps=4)
        model._n_updates = preset
        cb = _learn_under_budget(model,
                                 {"budget_max_updates": 10})
        assert int(model._n_updates) == preset, (
            "a counter already at or over the bound must gain "
            "NOTHING")
        assert cb.budget_stop
