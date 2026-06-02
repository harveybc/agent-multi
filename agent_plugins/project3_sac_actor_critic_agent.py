"""
project3_sac_actor_critic_agent.py — Project 3 ETHUSDT 4h SAC defaults.

Thin subclass of `sac_agent.Plugin` that bakes in the canonical Project 3
hyperparameters and adds a small validation layer for the reproduction
profile (continuous action space, canonical strategy/reward in strict
mode). All RL lifecycle behavior (build/train/predict/save/load,
seeding, FlattenObservation wrapping) is inherited unchanged.

See docs/PROJECT3_ETHUSDT_4H_SAC_ACTOR_CRITIC_PLUGIN_SPEC.md for context.

This is an external screening experiment. Do not derive Stage C / held-out
performance claims from runs of this plugin.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent_plugins.sac_agent import Plugin as SacPlugin


class Plugin(SacPlugin):
    # Project 3 canonical defaults (Stage A first-wave run).
    plugin_params: Dict[str, Any] = {
        # SAC core
        "total_timesteps": 25_000,
        "learning_rate": 1e-4,
        "buffer_size": 200_000,
        "learning_starts": 5_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto",
        "use_sde": True,
        "net_arch": (256, 256),
        "device": "cuda",
        "agent_verbose": 0,
        "train_seed": 0,
        "ga_fitness_dd_lambda": 1.0,

        # Project 3 environment / strategy expectations
        "action_space_mode": "continuous",
        "continuous_action_threshold": 0.1,
        "window_size": 32,
        "price_column": "CLOSE",
        "strategy_plugin": "direct_atr_sltp",
        "reward_plugin": "pnl_reward",
        "atr_period": 14,
        "k_sl": 2.0,
        "k_tp": 3.0,
        "rel_volume": 0.05,
        "size_mode": "notional",
        "leverage": 1.0,
        "min_order_volume": 0.0,
        "max_order_volume": 100.0,
        "commission": 0.0002,
        "slippage": 0.0,
        "initial_cash": 10000.0,

        # Strict mode hard-fails on canonical reproduction violations
        # (action_space_mode). Non-canonical tunables (window_size,
        # price_column, strategy_plugin, reward_plugin) are warning-only
        # so DEAP/optimizer sweeps remain possible.
        "project3_strict": True,
    }

    _PARAM_KEYS = tuple(plugin_params.keys())

    plugin_debug_vars: List[str] = [
        "total_timesteps",
        "learning_rate",
        "batch_size",
        "buffer_size",
        "use_sde",
        "net_arch",
        "device",
        "train_seed",
        "action_space_mode",
        "continuous_action_threshold",
        "window_size",
        "strategy_plugin",
        "reward_plugin",
    ]

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]) -> None:
        debug_info.update(self.get_debug_info())

    def _validate_project3(self, config: Dict[str, Any]) -> None:
        action_mode = str(
            config.get("action_space_mode", self.params["action_space_mode"])
        ).lower()
        if action_mode != "continuous":
            raise ValueError(
                "project3_sac_actor_critic_agent requires "
                "action_space_mode='continuous'; got "
                f"'{action_mode}'."
            )

        strict = bool(config.get("project3_strict", self.params["project3_strict"]))
        if not strict:
            return

        warnings: List[str] = []
        canonical = {
            "window_size": 32,
            "price_column": "CLOSE",
            "strategy_plugin": "direct_atr_sltp",
            "reward_plugin": "pnl_reward",
        }
        for k, expected in canonical.items():
            actual = config.get(k, self.params.get(k))
            if actual != expected:
                warnings.append(f"{k}={actual!r} (canonical: {expected!r})")
        if warnings and not config.get("quiet_mode", False):
            print(
                "[project3_sac_actor_critic_agent] non-canonical reproduction "
                "settings detected: " + ", ".join(warnings)
            )

    # SAC plugin's build() already calls _require_continuous; we add the
    # Project 3 validation layer on top.
    def build(self, env, config: Dict[str, Any]):
        self._validate_project3(config)
        return super().build(env, config)

    def load(self, path: str, env):
        return super().load(path, env)

    def hparam_schema(self) -> List[Tuple[str, float, float, str]]:
        # Compact initial schema; expand only after smoke stability.
        return [
            ("learning_rate", 1e-5, 5e-4, "float"),
            ("batch_size", 128, 512, "int"),
            ("gamma", 0.95, 0.999, "float"),
            ("tau", 1e-3, 2e-2, "float"),
            ("train_freq", 1, 8, "int"),
            ("gradient_steps", 1, 8, "int"),
            ("continuous_action_threshold", 0.05, 0.35, "float"),
        ]
