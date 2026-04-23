"""
sac_agent.py — stable-baselines3 SAC wrapper as an agent plugin.

SAC uses a continuous action space. gym-fx exposes `action_space_mode` —
when set to "continuous", the env returns Box(-1, +1, shape=(1,)) and
BTBridgeStrategy thresholds at ±0.33 to {-1, 0, +1} (per plan decision,
apples-to-apples with discrete baselines).

SAC also needs a flat obs; we wrap the env with FlattenObservation in build.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


class Plugin:
    plugin_params: Dict[str, Any] = {
        "total_timesteps": 10_000,
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 1_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto",
        "use_sde": False,
        "net_arch": (256, 256),
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

    def build(self, env, config: Dict[str, Any]):
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise ImportError("stable-baselines3 is required for sac_agent") from exc

        self._require_continuous(env)
        p = self._resolve(config)
        policy_kwargs = {"net_arch": list(p["net_arch"])}
        seed = int(p["train_seed"])
        model = SAC(
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
            ent_coef=p["ent_coef"],
            target_update_interval=int(p["target_update_interval"]),
            target_entropy=p["target_entropy"],
            use_sde=bool(p["use_sde"]),
            policy_kwargs=policy_kwargs,
            verbose=int(p["agent_verbose"]),
            seed=seed,
            device=str(p["device"]),
        )
        # Explicitly re-seed after construction so gSDE exploration noise,
        # torch, numpy, env, and action_space all share this seed (SB3's
        # constructor-time seeding does not always reseed gSDE on SAC).
        model.set_random_seed(seed)
        if bool(p["use_sde"]):
            try:
                model.policy.reset_noise()
            except Exception:
                pass
        return model

    def train(self, model, config: Dict[str, Any]):
        p = self._resolve(config)
        model.learn(total_timesteps=int(p["total_timesteps"]))
        return model

    def predict(self, model, obs, deterministic: bool = True):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    def save(self, model, path: str) -> None:
        model.save(path)

    def load(self, path: str, env):
        from stable_baselines3 import SAC
        self._require_continuous(env)
        return SAC.load(path, env=env)

    def wrap_env(self, env, config: Dict[str, Any] | None = None):
        """Called by the pipeline before build/load — SAC needs flat obs."""
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

    def hparam_schema(self) -> List[Tuple[str, float, float, str]]:
        return [
            ("learning_rate", 1e-5, 1e-3, "float"),
            ("batch_size", 64, 512, "int"),
            ("gamma", 0.9, 0.999, "float"),
            ("tau", 1e-3, 5e-2, "float"),
            ("train_freq", 1, 8, "int"),
            ("gradient_steps", 1, 8, "int"),
        ]

    def _resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.params)
        for k in self.params:
            if k in config and config[k] is not None:
                merged[k] = config[k]
        return merged

    @staticmethod
    def _flatten_if_dict(env):
        try:
            from gymnasium import spaces
            from gymnasium.wrappers import FlattenObservation
        except ImportError:
            return env
        obs_space = getattr(env, "observation_space", None)
        if isinstance(obs_space, spaces.Dict):
            return FlattenObservation(env)
        return env

    @staticmethod
    def _require_continuous(env) -> None:
        try:
            from gymnasium import spaces
        except ImportError:
            return
        act = getattr(env, "action_space", None)
        if act is not None and not isinstance(act, spaces.Box):
            raise ValueError(
                "sac_agent requires a continuous Box action_space. "
                "Set 'action_space_mode': 'continuous' in the env config."
            )
