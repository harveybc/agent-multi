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

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from . import _return_trace as _trace_mod
from ._observation_contract import validate_observation_contract


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
        validate_observation_contract(config)
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
        eval_progress_file = config.get("training_progress_file") or config.get("progress_file")
        action_stats = _new_action_stats(
            continuous_threshold=_safe_float(config.get("continuous_action_threshold")),
        )
        while not done:
            action = agent_plugin.predict(model, obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _info = env.step(action)
            equity = _safe_float(_info.get("equity"))
            _update_action_stats(action_stats, action, _info)
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
            _write_eval_progress_if_needed(
                eval_progress_file,
                config,
                status="evaluation",
                step=steps,
                info=_info,
                action_stats=action_stats,
                force=steps == 1 or steps % int(config.get("progress_update_interval_steps") or 1000) == 0,
            )
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
        summary.update(_action_summary_fields(action_stats, summary))
        min_trades = int(config.get("no_trade_min_trades") or 0)
        trades_total = int(float(summary.get("trades_total") or 0))
        summary["no_trade_min_trades"] = min_trades
        summary["no_trade_gate_passed"] = bool(trades_total >= min_trades) if min_trades else True
        if min_trades and trades_total < min_trades:
            summary["no_trade_gate_reason"] = (
                f"trades_total={trades_total} below required minimum {min_trades}; "
                "hard kill for promotion, diagnostic rerun required"
            )
        _write_eval_progress_if_needed(
            eval_progress_file,
            config,
            status="evaluation_complete",
            step=steps,
            info=_info,
            action_stats=action_stats,
            force=True,
            summary=summary,
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
                env=env,
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_action_stats(*, continuous_threshold: float | None = None) -> Dict[str, Any]:
    return {
        "steps": 0,
        "hold_actions": 0,
        "long_actions": 0,
        "short_actions": 0,
        "non_hold_actions": 0,
        "raw_abs_sum": 0.0,
        "raw_sum": 0.0,
        "raw_square_sum": 0.0,
        "raw_min": None,
        "raw_max": None,
        "continuous_deadband_actions": 0,
        "continuous_action_threshold": continuous_threshold,
        "entry_actions_seen": 0,
        "entry_orders_submitted": 0,
        "blocked_atr_warmup": 0,
        "blocked_session_filter": 0,
        "blocked_non_positive_atr": 0,
        "blocked_non_positive_size": 0,
        "blocked_non_positive_price": 0,
    }


def _raw_action_value(action: Any) -> float:
    try:
        import numpy as np

        arr = np.asarray(action).reshape(-1)
        if len(arr):
            return float(arr[0])
    except Exception:
        pass
    try:
        return float(action)
    except Exception:
        return 0.0


def _update_action_stats(stats: Dict[str, Any], action: Any, info: Dict[str, Any]) -> None:
    raw = _raw_action_value(action)
    coerced = info.get("coerced_action")
    try:
        coerced_i = int(coerced)
    except Exception:
        coerced_i = None
    stats["steps"] += 1
    stats["raw_abs_sum"] += abs(raw)
    stats["raw_sum"] += raw
    stats["raw_square_sum"] += raw * raw
    stats["raw_min"] = raw if stats["raw_min"] is None else min(float(stats["raw_min"]), raw)
    stats["raw_max"] = raw if stats["raw_max"] is None else max(float(stats["raw_max"]), raw)
    if coerced_i == 1:
        stats["long_actions"] += 1
        stats["non_hold_actions"] += 1
    elif coerced_i == 2:
        stats["short_actions"] += 1
        stats["non_hold_actions"] += 1
    elif coerced_i == 0:
        stats["hold_actions"] += 1
    threshold = stats.get("continuous_action_threshold")
    if threshold is not None and abs(raw) < float(threshold):
        stats["continuous_deadband_actions"] += 1
    exec_diag = info.get("execution_diagnostics") if isinstance(info, dict) else {}
    if isinstance(exec_diag, dict):
        for key in (
            "entry_actions_seen",
            "entry_orders_submitted",
            "blocked_atr_warmup",
            "blocked_session_filter",
            "blocked_non_positive_atr",
            "blocked_non_positive_size",
            "blocked_non_positive_price",
        ):
            try:
                stats[key] = max(int(stats.get(key, 0)), int(exec_diag.get(key, 0)))
            except Exception:
                pass


def _action_summary_fields(stats: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    steps = max(1, int(stats.get("steps") or 0))
    non_hold = int(stats.get("non_hold_actions") or 0)
    trades = int(float(summary.get("trades_total") or 0))
    deadband = int(stats.get("continuous_deadband_actions") or 0)
    entry_actions = int(stats.get("entry_actions_seen") or 0)
    orders = int(stats.get("entry_orders_submitted") or 0)
    raw_mean = float(stats.get("raw_sum") or 0.0) / steps
    raw_second_moment = float(stats.get("raw_square_sum") or 0.0) / steps
    raw_std = max(0.0, raw_second_moment - raw_mean * raw_mean) ** 0.5
    action_counts = {
        "hold": int(stats.get("hold_actions") or 0),
        "long": int(stats.get("long_actions") or 0),
        "short": int(stats.get("short_actions") or 0),
    }
    dominant_side, dominant_count = max(action_counts.items(), key=lambda item: item[1])
    raw_min = stats.get("raw_min")
    raw_max = stats.get("raw_max")
    raw_range = (
        float(raw_max) - float(raw_min)
        if raw_min is not None and raw_max is not None
        else None
    )
    if trades > 0:
        diagnosis = "traded"
    elif non_hold <= 0:
        diagnosis = "policy_hold_collapse"
    elif entry_actions <= 0:
        diagnosis = "action_mapping_or_env_blocked_before_strategy"
    elif orders <= 0:
        diagnosis = "execution_guard_blocked_all_entries"
    else:
        diagnosis = "orders_submitted_but_no_closed_trades"
    return {
        "action_steps": int(stats.get("steps") or 0),
        "action_hold_count": int(stats.get("hold_actions") or 0),
        "action_long_count": int(stats.get("long_actions") or 0),
        "action_short_count": int(stats.get("short_actions") or 0),
        "action_non_hold_count": non_hold,
        "action_non_hold_rate": non_hold / steps,
        "action_abs_mean": float(stats.get("raw_abs_sum") or 0.0) / steps,
        "action_raw_mean": raw_mean,
        "action_raw_std": raw_std,
        "action_raw_min": raw_min,
        "action_raw_max": raw_max,
        "action_raw_range": raw_range,
        "action_dominant_side": dominant_side,
        "action_dominant_count": dominant_count,
        "action_dominant_rate": dominant_count / steps,
        "action_deadband_count": deadband,
        "action_deadband_rate": deadband / steps,
        "execution_entry_actions_seen": entry_actions,
        "execution_entry_orders_submitted": orders,
        "execution_blocked_atr_warmup": int(stats.get("blocked_atr_warmup") or 0),
        "execution_blocked_session_filter": int(stats.get("blocked_session_filter") or 0),
        "execution_blocked_non_positive_atr": int(stats.get("blocked_non_positive_atr") or 0),
        "execution_blocked_non_positive_size": int(stats.get("blocked_non_positive_size") or 0),
        "execution_blocked_non_positive_price": int(stats.get("blocked_non_positive_price") or 0),
        "no_trade_diagnosis": diagnosis if trades == 0 else "",
    }


def _write_eval_progress_if_needed(
    progress_file: str | None,
    config: Dict[str, Any],
    *,
    status: str,
    step: int,
    info: Dict[str, Any],
    action_stats: Dict[str, Any],
    force: bool = False,
    summary: Dict[str, Any] | None = None,
) -> None:
    if not progress_file or not force:
        return
    equity = _safe_float(info.get("equity")) if isinstance(info, dict) else None
    initial_cash = _safe_float(config.get("initial_cash")) or 10000.0
    total_return = None
    if equity is not None and initial_cash:
        total_return = (equity - initial_cash) / initial_cash
    trades = None
    if summary is not None:
        trades = summary.get("trades_total")
        total_return = summary.get("total_return", total_return)
        equity = summary.get("final_equity", equity)
    if trades is None and isinstance(info, dict):
        trades = info.get("trades")
    payload = {
        "schema_version": "project3_training_progress_v1",
        "status": status,
        "source": "rl_pipeline_evaluation",
        "run_id": config.get("run_id") or Path(str(config.get("save_model") or "")).parent.name,
        "agent_plugin": config.get("agent_plugin"),
        "asset": config.get("asset"),
        "timeframe": config.get("timeframe"),
        "features_preset": config.get("features_preset"),
        "seed": config.get("eval_seed", config.get("train_seed")),
        "pid": None,
        "updated_at_utc": _utc_now(),
        "elapsed_seconds": None,
        "current_step": step,
        "num_timesteps": config.get("total_timesteps"),
        "total_timesteps": config.get("total_timesteps"),
        "progress_pct": 100.0 if status == "evaluation_complete" else 99.0,
        "progress_percent": 100.0 if status == "evaluation_complete" else 99.0,
        "progress_detail": f"evaluation step {step}; trades={trades}; return={total_return}",
        "eval_step": step,
        "trades_total": trades,
        "total_return": total_return,
        "profit_percent": None if total_return is None else float(total_return) * 100.0,
        "equity": equity,
        "final_equity": equity,
        **_action_summary_fields(action_stats, summary or {"trades_total": trades or 0}),
    }
    path = Path(str(progress_file))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
