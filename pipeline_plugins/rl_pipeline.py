"""
rl_pipeline.py — orchestrates an RL run on gym-fx.

Mode semantics:
  - train:        build env → build agent → train → save model → evaluate → summary.
  - inference:    build env → load agent → evaluate → summary.
  - optimization: no-op here; main.py dispatches to the optimizer plugin first.

Evaluation is a single deterministic rollout (config['eval_episodes']
reserved for future use). The summary comes from env.summary() which
reads backtrader analyzers, plus a few extra keys (episode_reward,
episode_length, eval_seed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class PipelinePlugin:
    plugin_params: Dict[str, Any] = {
        "eval_seed": 0,
        "train_seed": 0,
        "total_timesteps": 10_000,
        "save_model": "./agent_model.zip",
        "load_model": None,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        *,
        config: Dict[str, Any],
        env_plugin,
        agent_plugin,
        mode: str = "train",
    ) -> Dict[str, Any]:
        mode = str(mode).lower()
        base_env = env_plugin.make_env(config)
        # Agents may need a wrapped env (e.g. FlattenObservation for DQN/SAC).
        wrap_fn = getattr(agent_plugin, "wrap_env", None)
        env = wrap_fn(base_env, config) if callable(wrap_fn) else base_env
        try:
            if mode == "train":
                model = agent_plugin.build(env, config)
                model = agent_plugin.train(model, config)
                save_path = config.get("save_model")
                if save_path:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    agent_plugin.save(model, save_path)
            elif mode == "inference":
                load_path = config.get("load_model")
                if not load_path:
                    raise ValueError("inference mode requires config['load_model']")
                model = agent_plugin.load(load_path, env)
            else:
                raise ValueError(f"unsupported pipeline mode: {mode}")

            summary = self._evaluate(env, agent_plugin, model, config)
            summary["mode"] = mode
            return summary
        finally:
            # make_env owns env creation; env_plugin.close() tears down cleanly.
            try:
                env_plugin.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _evaluate(self, env, agent_plugin, model, config: Dict[str, Any]) -> Dict[str, Any]:
        seed = int(config.get("eval_seed", self.params["eval_seed"]))
        obs, _info = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = agent_plugin.predict(model, obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            if steps > 1_000_000:  # hard safety
                break

        # Unwrap any gymnasium wrappers so we get the base GymFxEnv with summary().
        base_env = env
        while hasattr(base_env, "env") and not hasattr(base_env, "summary"):
            base_env = base_env.env
        summary = base_env.summary() if hasattr(base_env, "summary") else {}
        summary.update(
            episode_reward=total_reward,
            episode_length=steps,
            eval_seed=seed,
        )
        return summary
