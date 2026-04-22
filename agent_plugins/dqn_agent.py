"""
dqn_agent.py — stable-baselines3 DQN wrapper as an agent plugin.

DQN requires a flat Box observation, but gym-fx emits a Dict obs. We wrap
the env with gymnasium.wrappers.FlattenObservation inside `build()` so the
wider pipeline (which passes us a DummyVecEnv or raw env) can be used
unchanged by other agents.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


class Plugin:
    plugin_params: Dict[str, Any] = {
        "total_timesteps": 10_000,
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "max_grad_norm": 10.0,
        "net_arch": (128, 128),
        "device": "auto",
        "agent_verbose": 0,
        "train_seed": 0,
        "ga_fitness_dd_lambda": 1.0,
    }

    _PARAM_KEYS = tuple(plugin_params.keys())

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self._PARAM_KEYS:
                self.params[k] = v

    # --- lifecycle -----------------------------------------------------------
    def build(self, env, config: Dict[str, Any]):
        try:
            from stable_baselines3 import DQN
        except ImportError as exc:
            raise ImportError("stable-baselines3 is required for dqn_agent") from exc

        p = self._resolve(config)
        policy_kwargs = {"net_arch": list(p["net_arch"])}
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=float(p["learning_rate"]),
            buffer_size=int(p["buffer_size"]),
            learning_starts=int(p["learning_starts"]),
            batch_size=int(p["batch_size"]),
            tau=float(p["tau"]),
            gamma=float(p["gamma"]),
            train_freq=int(p["train_freq"]),
            gradient_steps=int(p["gradient_steps"]),
            target_update_interval=int(p["target_update_interval"]),
            exploration_fraction=float(p["exploration_fraction"]),
            exploration_initial_eps=float(p["exploration_initial_eps"]),
            exploration_final_eps=float(p["exploration_final_eps"]),
            max_grad_norm=float(p["max_grad_norm"]),
            policy_kwargs=policy_kwargs,
            verbose=int(p["agent_verbose"]),
            seed=int(p["train_seed"]),
            device=str(p["device"]),
        )
        return model

    def train(self, model, config: Dict[str, Any]):
        p = self._resolve(config)
        model.learn(total_timesteps=int(p["total_timesteps"]))
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
        from stable_baselines3 import DQN
        return DQN.load(path, env=env)

    def wrap_env(self, env, config: Dict[str, Any] | None = None):
        """Called by the pipeline before build/load — DQN needs flat obs."""
        return self._flatten_if_dict(env)

    def fitness(self, summary: Dict[str, Any], config: Dict[str, Any]) -> float:
        total_return = float(summary.get("total_return") or 0.0)
        max_dd = summary.get("max_drawdown_pct") or 0.0
        try:
            max_dd = float(max_dd)
        except (TypeError, ValueError):
            max_dd = 0.0
        lam = float(config.get("ga_fitness_dd_lambda", self.params["ga_fitness_dd_lambda"]))
        return total_return - lam * (max_dd / 100.0)

    # --- GA schema -----------------------------------------------------------
    def hparam_schema(self) -> List[Tuple[str, float, float, str]]:
        return [
            ("learning_rate", 1e-5, 1e-3, "float"),
            ("batch_size", 32, 256, "int"),
            ("gamma", 0.9, 0.999, "float"),
            ("exploration_fraction", 0.05, 0.5, "float"),
            ("exploration_final_eps", 0.01, 0.2, "float"),
            ("target_update_interval", 250, 5000, "int"),
            ("train_freq", 1, 8, "int"),
        ]

    # --- internal ------------------------------------------------------------
    def _resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.params)
        for k in self.params:
            if k in config and config[k] is not None:
                merged[k] = config[k]
        return merged

    @staticmethod
    def _flatten_if_dict(env):
        """Wrap the env with FlattenObservation if its obs space is a Dict."""
        try:
            from gymnasium import spaces
            from gymnasium.wrappers import FlattenObservation
        except ImportError:
            return env

        obs_space = getattr(env, "observation_space", None)
        # Unwrap vec envs transparently: SB3 accepts either wrapped env or vec env
        if isinstance(obs_space, spaces.Dict):
            return FlattenObservation(env)
        return env
