"""
env_plugins/gym_fx_env.py — env.plugins adapter for agent-multi.

Builds a GymFxEnv (from the installed gym-fx package) using the plugin
names specified in the config. All gym-fx configuration keys are
forwarded through transparently.

Requires `gym-fx` to be installed (editable install recommended):
    pip install -e ../gym-fx
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Dict


class Plugin:
    plugin_params: Dict[str, Any] = {
        "env_mode": "inference",
        "input_data_file": None,
        "price_column": "CLOSE",
        "date_column": "DATE_TIME",
        "headers": True,
        "max_rows": None,
        "window_size": 32,
        "initial_cash": 10000.0,
        "position_size": 1.0,
        "commission": 0.0,
        "slippage": 0.0,
        "data_feed_plugin": "default_data_feed",
        "broker_plugin": "default_broker",
        "strategy_plugin": "default_strategy",
        "preprocessor_plugin": "default_preprocessor",
        "reward_plugin": "pnl_reward",
        "metrics_plugin": "default_metrics",
    }

    plugin_debug_vars = [
        "env_mode", "input_data_file", "price_column", "window_size",
        "initial_cash", "commission", "slippage",
        "data_feed_plugin", "broker_plugin", "strategy_plugin",
        "preprocessor_plugin", "reward_plugin", "metrics_plugin",
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        self._env = None
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]) -> None:
        debug_info.update(self.get_debug_info())

    def _build_env_config(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.params)
        merged.update(agent_config)
        # gym-fx uses key "mode"; we relabel from env_mode to avoid
        # stomping on agent-multi's own "mode".
        merged["mode"] = merged.get("env_mode", "inference")
        return merged

    def _load_bundle_plugin(self, group: str, name: str, cfg: Dict[str, Any]):
        eps = entry_points().select(group=group)
        ep = next((e for e in eps if e.name == name), None)
        if ep is None:
            raise ImportError(f"Plugin '{name}' not found in group '{group}'")
        klass = ep.load()
        inst = klass(cfg)
        inst.set_params(**cfg)
        return inst

    def make_env(self, agent_config: Dict[str, Any]):
        try:
            from gym_fx import GymFxEnv
        except ImportError as exc:
            raise ImportError(
                "gym-fx is not installed. Run: pip install -e ../gym-fx"
            ) from exc

        env_config = self._build_env_config(agent_config)
        data_feed = self._load_bundle_plugin("data_feed.plugins", env_config["data_feed_plugin"], env_config)
        broker = self._load_bundle_plugin("broker.plugins", env_config["broker_plugin"], env_config)
        strategy = self._load_bundle_plugin("strategy.plugins", env_config["strategy_plugin"], env_config)
        preprocessor = self._load_bundle_plugin("preprocessor.plugins", env_config["preprocessor_plugin"], env_config)
        reward = self._load_bundle_plugin("reward.plugins", env_config["reward_plugin"], env_config)
        metrics = self._load_bundle_plugin("metrics.plugins", env_config["metrics_plugin"], env_config)

        self._env = GymFxEnv(
            config=env_config,
            data_feed_plugin=data_feed,
            broker_plugin=broker,
            strategy_plugin=strategy,
            preprocessor_plugin=preprocessor,
            reward_plugin=reward,
            metrics_plugin=metrics,
        )
        return self._env

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
