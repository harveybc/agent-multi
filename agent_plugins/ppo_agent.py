"""
ppo_agent.py — stable-baselines3 PPO wrapper as an agent plugin.

Uses the MultiInputPolicy because gym-fx's observation space is a Dict.
All hyperparameters are config-driven; defaults live in plugin_params.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ._progress_callback import make_progress_callback


class Plugin:
    plugin_params: Dict[str, Any] = {
        "total_timesteps": 10_000,
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "net_arch_pi": (64, 64),
        "net_arch_vf": (64, 64),
        "device": "auto",
        "agent_verbose": 0,
        "train_seed": 0,
        "training_progress_file": None,
        "progress_file": None,
        "progress_update_interval_steps": 1000,
        "ga_fitness_dd_lambda": 1.0,
    }

    plugin_debug_vars: List[str] = [
        "total_timesteps", "learning_rate", "n_steps", "batch_size",
        "n_epochs", "gamma", "clip_range", "device", "train_seed",
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]) -> None:
        debug_info.update(self.get_debug_info())

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.params or k in (
                "total_timesteps", "learning_rate", "n_steps", "batch_size", "n_epochs",
                "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
                "max_grad_norm", "device", "agent_verbose", "train_seed",
                "training_progress_file", "progress_file", "progress_update_interval_steps",
                "net_arch_pi", "net_arch_vf", "ga_fitness_dd_lambda",
            ):
                self.params[k] = v

    # --- lifecycle -----------------------------------------------------------
    def build(self, env, config: Dict[str, Any]):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover
            raise ImportError("stable-baselines3 is required for ppo_agent") from exc

        p = self._resolve(config)
        policy_kwargs = {
            "net_arch": dict(pi=list(p["net_arch_pi"]), vf=list(p["net_arch_vf"])),
        }
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=float(p["learning_rate"]),
            n_steps=int(p["n_steps"]),
            batch_size=int(p["batch_size"]),
            n_epochs=int(p["n_epochs"]),
            gamma=float(p["gamma"]),
            gae_lambda=float(p["gae_lambda"]),
            clip_range=float(p["clip_range"]),
            ent_coef=float(p["ent_coef"]),
            vf_coef=float(p["vf_coef"]),
            max_grad_norm=float(p["max_grad_norm"]),
            policy_kwargs=policy_kwargs,
            verbose=int(p["agent_verbose"]),
            seed=int(p["train_seed"]),
            device=str(p["device"]),
        )
        return model

    def train(self, model, config: Dict[str, Any]):
        p = self._resolve(config)
        total_timesteps = int(p["total_timesteps"])
        callback = make_progress_callback({**config, **p}, total_timesteps)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        return model

    def predict(self, model, obs, deterministic: bool = True):
        action, _ = model.predict(obs, deterministic=deterministic)
        try:
            return int(action)
        except (TypeError, ValueError):
            return action

    def save(self, model, path: str) -> None:
        model.save(path)

    def load(self, path: str, env):
        from stable_baselines3 import PPO
        return PPO.load(path, env=env)

    def fitness(self, summary: Dict[str, Any], config: Dict[str, Any]) -> float:
        total_return = float(summary.get("total_return") or 0.0)
        max_dd = summary.get("max_drawdown_pct") or 0.0
        try:
            max_dd = float(max_dd)
        except (TypeError, ValueError):
            max_dd = 0.0
        lam = float(config.get("ga_fitness_dd_lambda", self.params["ga_fitness_dd_lambda"]))
        # max_drawdown_pct from backtrader is in percentage points
        return total_return - lam * (max_dd / 100.0)

    # --- GA schema -----------------------------------------------------------
    def hparam_schema(self) -> List[Tuple[str, float, float, str]]:
        """(name, low, high, type) used by the DEAP optimizer."""
        return [
            ("learning_rate", 1e-5, 1e-3, "float"),
            ("n_steps", 64, 1024, "int"),
            ("batch_size", 32, 256, "int"),
            ("n_epochs", 3, 15, "int"),
            ("gamma", 0.9, 0.999, "float"),
            ("gae_lambda", 0.8, 0.99, "float"),
            ("clip_range", 0.1, 0.4, "float"),
            ("ent_coef", 0.0, 0.05, "float"),
        ]

    # --- internal ------------------------------------------------------------
    def _resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.params)
        for k in self.params:
            if k in config and config[k] is not None:
                merged[k] = config[k]
        return merged
