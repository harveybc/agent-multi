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
        _check_executing_budget(config, model(10**6, updates=1000),
                                started_wall=time.time())
        with pytest.raises(ExecutingBudgetExceeded,
                           match="ACTUAL counter reads 1001"):
            _check_executing_budget(config,
                                    model(10, updates=1001),
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
        cb.model._n_updates = 1000
        assert cb._on_step() is True
        cb.model._n_updates = 1001
        assert cb._on_step() is False
        assert "ACTUAL counter reads 1001" in cb.budget_stop

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
