from __future__ import annotations

import json

from agent_plugins._progress_callback import SCHEMA_VERSION, make_progress_callback
from agent_plugins.dqn_agent import Plugin as DQNPlugin
from agent_plugins.ppo_agent import Plugin as PPOPlugin
from agent_plugins.sac_agent import Plugin as SACPlugin


def test_progress_callback_writes_exact_step_percent(tmp_path):
    progress_file = tmp_path / "progress.json"
    callback = make_progress_callback(
        {
            "training_progress_file": str(progress_file),
            "run_id": "unit_run",
            "agent_plugin": "ppo_agent",
            "asset": "ETHUSDT_4h",
            "timeframe": "4h",
            "features_preset": "tech_stat",
            "train_seed": 2,
            "progress_update_interval_steps": 10,
        },
        total_timesteps=1000,
    )
    assert callback is not None

    callback.num_timesteps = 250
    callback._on_training_start()
    payload = json.loads(progress_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["run_id"] == "unit_run"
    assert payload["progress_percent"] == 25.0
    assert payload["progress_pct"] == 25.0
    assert payload["current_step"] == 250
    assert payload["num_timesteps"] == 250
    assert payload["elapsed_seconds"] >= 0.0
    assert payload["progress_detail"] == "250/1000 SB3 timesteps"

    callback.num_timesteps = 1000
    callback._on_training_end()
    payload = json.loads(progress_file.read_text(encoding="utf-8"))
    assert payload["status"] == "training_complete"
    assert payload["progress_percent"] == 100.0


def test_progress_callback_writes_live_trade_action_metrics(tmp_path):
    class FakeBridge:
        trade_count = 0
        equity = 10000.0
        execution_diagnostics = {
            "entry_actions_seen": 0,
            "entry_orders_submitted": 0,
            "blocked_atr_warmup": 0,
            "event_context_no_trade_active_steps": 3,
            "event_context_action_overrides": 2,
            "event_context_blocked_entries": 1,
            "event_context_forced_flat_actions": 1,
            "event_context_forced_flat_orders": 1,
        }

    class FakeVecEnv:
        def env_method(self, name):
            assert name == "summary"
            return [
                {
                    "trades_total": 0,
                    "total_return": 0.0,
                    "final_equity": 10000.0,
                    "action_diagnostics": {
                        "steps": 50,
                        "hold_actions": 50,
                        "long_actions": 0,
                        "short_actions": 0,
                        "non_hold_actions": 0,
                        "continuous_deadband_actions": 50,
                        "raw_abs_sum": 0.0,
                        "raw_min": 0.0,
                        "raw_max": 0.0,
                    },
                    "execution_diagnostics": FakeBridge.execution_diagnostics,
                }
            ]

        def get_attr(self, name):
            if name == "bridge":
                return [FakeBridge()]
            if name == "_action_diagnostics":
                return [
                    {
                        "steps": 50,
                        "hold_actions": 50,
                        "long_actions": 0,
                        "short_actions": 0,
                        "non_hold_actions": 0,
                        "continuous_deadband_actions": 50,
                        "raw_abs_sum": 0.0,
                    }
                ]
            return []

    class FakeModel:
        def get_env(self):
            return FakeVecEnv()

    progress_file = tmp_path / "progress.json"
    callback = make_progress_callback(
        {
            "training_progress_file": str(progress_file),
            "run_id": "unit_run",
            "agent_plugin": "sac_agent",
            "asset": "ETHUSDT_4h",
            "timeframe": "4h",
            "features_preset": "tech_stat",
            "train_seed": 2,
            "progress_update_interval_steps": 10,
        },
        total_timesteps=1000,
    )
    callback.model = FakeModel()
    callback.num_timesteps = 500
    callback._on_step()
    payload = json.loads(progress_file.read_text(encoding="utf-8"))

    assert payload["progress_percent"] == 50.0
    assert payload["progress_pct"] == 50.0
    assert payload["current_step"] == 500
    assert payload["trades_total"] == 0
    assert payload["profit_percent"] == 0.0
    assert payload["equity"] == 10000.0
    assert payload["no_trade_anomaly"] is True
    assert payload["action_non_hold_rate"] == 0.0
    assert payload["action_deadband_rate"] == 1.0
    assert payload["execution_entry_actions_seen"] == 0
    assert payload["execution_event_context_no_trade_active_steps"] == 3
    assert payload["execution_event_context_action_overrides"] == 2
    assert payload["execution_event_context_blocked_entries"] == 1
    assert payload["execution_event_context_forced_flat_actions"] == 1
    assert payload["execution_event_context_forced_flat_orders"] == 1
    assert payload["no_trade_diagnosis"] == "training_policy_hold_collapse"


def test_progress_config_keys_are_accepted_by_sb3_plugins(tmp_path):
    cfg = {
        "training_progress_file": str(tmp_path / "progress.json"),
        "progress_update_interval_steps": 123,
    }
    for plugin_cls in (PPOPlugin, SACPlugin, DQNPlugin):
        plugin = plugin_cls(cfg)
        plugin.set_params(**cfg)
        assert plugin.params["training_progress_file"] == cfg["training_progress_file"]
        assert plugin.params["progress_update_interval_steps"] == 123
