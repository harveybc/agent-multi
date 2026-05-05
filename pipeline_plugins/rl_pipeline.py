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

When ``return_trace_file`` is set in the config the per-step trace and
its Stage B metadata sidecar are emitted via
:mod:`pipeline_plugins._return_trace`. The trace is fail-closed against
Stage C rows (timestamps >= 2025-01-01) unless the config explicitly
authorizes a final Stage C evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from . import _return_trace as _trace_mod


class PipelinePlugin:
    plugin_params: Dict[str, Any] = {
        "eval_seed": 0,
        "train_seed": 0,
        "total_timesteps": 10_000,
        "save_model": "./agent_model.zip",
        "load_model": None,
        "return_trace_file": None,
    }

    plugin_debug_vars = [
        "eval_seed", "train_seed", "total_timesteps", "save_model", "load_model",
        "return_trace_file",
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]) -> None:
        debug_info.update(self.get_debug_info())

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
        deterministic = bool(config.get("eval_deterministic", True))
        run_id = _trace_mod.make_run_id(config)
        episode_id = f"{run_id}::eval0"
        asset = str(config.get("asset", "unknown_asset"))
        timeframe = str(config.get("timeframe", config.get("timeframe_label", "")))
        split_label = str(config.get("eval_split", "evaluation"))

        obs, _info = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        done = False
        trace_rows = []
        prev_equity = _safe_float(_info.get("equity"))
        while not done:
            action = agent_plugin.predict(model, obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _info = env.step(action)
            equity = _safe_float(_info.get("equity"))
            trace_rows.append(
                _trace_mod.build_trace_row(
                    env=env,
                    step=steps + 1,
                    action=action,
                    reward=reward,
                    info=_info,
                    prev_equity=prev_equity,
                    asset=asset,
                    timeframe=timeframe,
                    split=split_label,
                    seed=seed,
                    run_id=run_id,
                    episode_id=episode_id,
                )
            )
            prev_equity = equity
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
        trace_file = config.get("return_trace_file") or self.params.get("return_trace_file")
        if trace_file:
            metadata = _trace_mod.write_return_trace(
                str(trace_file),
                trace_rows,
                config=config,
                split=split_label,
                seed=seed,
                asset=asset,
                timeframe=timeframe,
                run_id=run_id,
                episode_id=episode_id,
                feature_list=config.get("feature_list"),
            )
            summary["return_trace_file"] = metadata["trace_file"]
            summary["return_trace_metadata_file"] = metadata["metadata_file"]
            evidence = _trace_mod.build_return_trace_evidence(
                [metadata],
                config=config,
                run_id=run_id,
                pipeline_plugin="rl_pipeline",
            )
            evidence_path = _trace_mod.derive_evidence_path(trace_file=str(trace_file))
            evidence["evidence_file"] = _trace_mod.write_return_trace_evidence(
                evidence, evidence_path,
            )
            summary["return_trace_evidence"] = evidence
            summary["return_trace_evidence_file"] = evidence["evidence_file"]
        return summary


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
