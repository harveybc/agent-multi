from __future__ import annotations

"""
gym_fx_env.py — env.plugins adapter for agent-multi.

This is the only env plugin agent-multi needs to know about.
It instantiates GymFxEnv from the gym-fx package and exposes the
standard Gymnasium-compatible interface (reset / step / summary)
that every agent in agent-multi can rely on.

To use, agent-multi config must contain:
    "env_plugin": "gym_fx_env"

and any gym-fx config keys (input_data_file, window_size, etc.)
which are forwarded transparently to GymFxEnv.

Dependencies:
    pip install -e ../gym-fx   (or add gym-fx to install_requires)
"""

from typing import Any, Dict, Tuple


class Plugin:
    """env.plugins entry that wraps gym_fx.GymFxEnv."""

    plugin_params = {
        # forwarded to gym-fx – only the most common keys listed here;
        # the full list lives in gym-fx/app/config.py DEFAULT_VALUES
        "env_mode": "inference",          # gym-fx mode (training/optimization/inference)
        "input_data_file": None,          # required: path to OHLCV CSV
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
        "preprocessor_plugin": "default_preprocessor",
        "reward_plugin": "pnl_reward",
        "metrics_plugin": "default_metrics",
        # strategy_plugin is left as default because the agent decides actions
        "strategy_plugin": "default_strategy",
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        self._env = None
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def _build_env_config(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge agent_config over plugin defaults to produce the full gym-fx config."""
        merged = dict(self.params)
        merged.update(agent_config)
        merged["mode"] = merged.pop("env_mode", "inference")
        return merged

    def make_env(self, agent_config: Dict[str, Any]):
        """Instantiate and return a GymFxEnv, caching it in self._env."""
        try:
            from gym_fx import GymFxEnv  # gym-fx must be installed / on sys.path
        except ImportError as exc:
            raise ImportError(
                "gym-fx is not installed. Run: pip install -e path/to/gym-fx"
            ) from exc

        from importlib.metadata import entry_points

        env_config = self._build_env_config(agent_config)

        def _load(group, name, cfg):
            eps = entry_points().select(group=group)
            ep = next((e for e in eps if e.name == name), None)
            if ep is None:
                raise ImportError(f"Plugin '{name}' not found in group '{group}'")
            klass = ep.load()
            inst = klass(cfg)
            inst.set_params(**cfg)
            return inst

        data_feed   = _load("data_feed.plugins",   env_config["data_feed_plugin"],   env_config)
        broker      = _load("broker.plugins",       env_config["broker_plugin"],      env_config)
        strategy    = _load("strategy.plugins",     env_config["strategy_plugin"],    env_config)
        preprocessor= _load("preprocessor.plugins", env_config["preprocessor_plugin"],env_config)
        reward      = _load("reward.plugins",       env_config["reward_plugin"],      env_config)
        metrics     = _load("metrics.plugins",      env_config["metrics_plugin"],     env_config)

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

    # ------------------------------------------------------------------
    # Convenience wrappers – agents can also call env directly
    # ------------------------------------------------------------------
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("Call make_env(config) before reset()")
        return self._env.reset()

    def step(self, action) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("Call make_env(config) before step()")
        return self._env.step(action)

    def summary(self) -> Dict[str, Any]:
        if self._env is None:
            return {}
        return self._env.summary()
