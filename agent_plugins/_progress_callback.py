"""JSON training-progress callback for Stable-Baselines3 agents.

The callback is intentionally dependency-light and opt-in. When a config
contains ``training_progress_file`` (or ``progress_file``), PPO/SAC/DQN write
an atomic JSON heartbeat during ``model.learn()`` so external supervisors can
report exact step progress instead of inferring it from wall-clock time.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


SCHEMA_VERSION = "project3_training_progress_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def make_progress_callback(config: Dict[str, Any], total_timesteps: int):
    """Return an SB3 callback or ``None`` when progress reporting is disabled."""
    progress_file = config.get("training_progress_file") or config.get("progress_file")
    if not progress_file:
        return None
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:  # pragma: no cover - SB3 agents already require SB3.
        return None

    class JsonTrainingProgressCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.path = Path(str(progress_file))
            self.total_timesteps = max(1, int(total_timesteps))
            self.update_interval_steps = max(1, int(config.get("progress_update_interval_steps") or 1000))
            self.started_at = utc_now()
            self.started_monotonic = time.monotonic()
            self.last_written_step = -1
            self.last_episode_trades: int | None = None
            self.cumulative_trades = 0
            self.trade_counter_reset_count = 0
            try:
                prior = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(prior, dict):
                    self.cumulative_trades = _safe_int(
                        prior.get("trades_total_cumulative"),
                        _safe_int(prior.get("trades_total"), 0),
                    )
                    self.trade_counter_reset_count = _safe_int(
                        prior.get("trade_counter_reset_count"), 0,
                    )
            except Exception:
                pass

        def _on_training_start(self) -> None:
            self._write("training_started", force=True)

        def _on_step(self) -> bool:
            step = int(getattr(self, "num_timesteps", 0) or 0)
            if step - self.last_written_step >= self.update_interval_steps:
                self._write("training")
            return True

        def _on_training_end(self) -> None:
            self._write("training_complete", force=True)

        def _write(self, status: str, force: bool = False) -> None:
            step = int(getattr(self, "num_timesteps", 0) or 0)
            if not force and step == self.last_written_step:
                return
            self.last_written_step = step
            percent = min(100.0, max(0.0, (step / self.total_timesteps) * 100.0))
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "source": "stable_baselines3_callback",
                "run_id": config.get("run_id") or Path(str(config.get("save_model") or "")).parent.name,
                "agent_plugin": config.get("agent_plugin"),
                "asset": config.get("asset"),
                "timeframe": config.get("timeframe"),
                "features_preset": config.get("features_preset"),
                "seed": config.get("train_seed"),
                "pid": os.getpid(),
                "started_at_utc": self.started_at,
                "updated_at_utc": utc_now(),
                "elapsed_seconds": round(max(0.0, time.monotonic() - self.started_monotonic), 4),
                "current_step": step,
                "num_timesteps": step,
                "total_timesteps": self.total_timesteps,
                "progress_pct": round(percent, 4),
                "progress_percent": round(percent, 4),
                "progress_detail": f"{step}/{self.total_timesteps} SB3 timesteps",
            }
            payload.update(self._training_env_metrics())
            self._update_cumulative_trade_metrics(payload)
            trades_total = _safe_int(payload.get("trades_total_cumulative"), _safe_int(payload.get("trades_total"), -1))
            payload["no_trade_anomaly"] = bool(percent >= 20.0 and trades_total <= 0)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            tmp.replace(self.path)

        def _update_cumulative_trade_metrics(self, payload: Dict[str, Any]) -> None:
            """Normalize env-local trade counters into a monotonic counter.

            Some trading envs expose ``trades_total`` as the current episode's
            trade count. During long SB3 learning runs the env resets many
            times, so the raw value can decrease. Supervisors need a monotonic
            "have we traded at all during training?" field; keep the raw value
            as ``trades_total_current_episode`` and expose the cumulative value
            as both ``trades_total_cumulative`` and ``trades_total``.
            """
            if "trades_total" not in payload:
                return
            current = max(0, _safe_int(payload.get("trades_total")))
            payload["trades_total_current_episode"] = current
            if self.last_episode_trades is None:
                self.cumulative_trades += current
            elif current >= self.last_episode_trades:
                self.cumulative_trades += current - self.last_episode_trades
            else:
                self.trade_counter_reset_count += 1
                self.cumulative_trades += current
            self.last_episode_trades = current
            payload["trades_total_cumulative"] = self.cumulative_trades
            payload["trade_counter_reset_count"] = self.trade_counter_reset_count
            payload["trades_total"] = self.cumulative_trades

        def _training_env_metrics(self) -> Dict[str, Any]:
            """Best-effort live trade/action telemetry from the training env.

            This must never break model training. Different SB3 wrappers expose
            different pieces of the underlying env, so the callback tries the
            rich summary first and then falls back to bridge/action attributes.
            """
            env = getattr(self, "training_env", None)
            if env is None:
                return {}
            payload: Dict[str, Any] = {}
            summary: Dict[str, Any] = {}
            try:
                summaries = env.env_method("summary")
                if summaries and isinstance(summaries[0], dict):
                    summary = summaries[0]
            except Exception:
                summary = {}
            if summary:
                if "trades_total" in summary:
                    payload["trades_total"] = _safe_int(summary.get("trades_total"))
                elif "trades" in summary:
                    payload["trades_total"] = _safe_int(summary.get("trades"))
                if "total_return" in summary:
                    total_return = _safe_float(summary.get("total_return"))
                    payload["total_return"] = total_return
                    payload["profit_percent"] = total_return * 100.0
                if "final_equity" in summary:
                    payload["final_equity"] = _safe_float(summary.get("final_equity"))
                    payload["equity"] = payload["final_equity"]
                action_diag = summary.get("action_diagnostics")
                if isinstance(action_diag, dict):
                    payload.update(_action_payload(action_diag))
                exec_diag = summary.get("execution_diagnostics")
                if isinstance(exec_diag, dict):
                    payload.update(_execution_payload(exec_diag))
            try:
                bridges = env.get_attr("bridge")
            except Exception:
                bridges = []
            if bridges:
                bridge = bridges[0]
                if "trades_total" not in payload:
                    payload["trades_total"] = _safe_int(getattr(bridge, "trade_count", 0))
                if "final_equity" not in payload:
                    payload["final_equity"] = _safe_float(getattr(bridge, "equity", 0.0))
                if "equity" not in payload:
                    payload["equity"] = payload.get("final_equity")
                exec_diag = getattr(bridge, "execution_diagnostics", None)
                if isinstance(exec_diag, dict):
                    payload.update(_execution_payload(exec_diag))
            try:
                action_diags = env.get_attr("_action_diagnostics")
            except Exception:
                action_diags = []
            if action_diags and isinstance(action_diags[0], dict):
                payload.update(_action_payload(action_diags[0]))
            if _safe_int(payload.get("trades_total"), -1) == 0:
                non_hold = _safe_float(payload.get("action_non_hold_rate"), -1.0)
                deadband = _safe_float(payload.get("action_deadband_rate"), -1.0)
                if non_hold == 0.0:
                    payload["no_trade_diagnosis"] = "training_policy_hold_collapse"
                elif deadband >= 0.999:
                    payload["no_trade_diagnosis"] = "training_policy_deadband_collapse"
                elif payload:
                    payload["no_trade_diagnosis"] = "training_no_trades_observed"
            return payload

    return JsonTrainingProgressCallback()


def _action_payload(action_diag: Dict[str, Any]) -> Dict[str, Any]:
    steps = max(1, _safe_int(action_diag.get("steps")))
    non_hold = _safe_int(action_diag.get("non_hold_actions"))
    deadband = _safe_int(action_diag.get("continuous_deadband_actions"))
    raw_abs_sum = _safe_float(action_diag.get("raw_abs_sum"))
    return {
        "action_steps": _safe_int(action_diag.get("steps")),
        "action_hold_count": _safe_int(action_diag.get("hold_actions")),
        "action_long_count": _safe_int(action_diag.get("long_actions")),
        "action_short_count": _safe_int(action_diag.get("short_actions")),
        "action_non_hold_count": non_hold,
        "action_non_hold_rate": non_hold / steps,
        "action_deadband_count": deadband,
        "action_deadband_rate": deadband / steps,
        "action_abs_mean": raw_abs_sum / steps,
        "action_raw_min": action_diag.get("raw_min"),
        "action_raw_max": action_diag.get("raw_max"),
    }


def _execution_payload(exec_diag: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "execution_entry_actions_seen": _safe_int(exec_diag.get("entry_actions_seen")),
        "execution_entry_orders_submitted": _safe_int(exec_diag.get("entry_orders_submitted")),
        "execution_blocked_atr_warmup": _safe_int(exec_diag.get("blocked_atr_warmup")),
        "execution_blocked_session_filter": _safe_int(exec_diag.get("blocked_session_filter")),
        "execution_blocked_non_positive_atr": _safe_int(exec_diag.get("blocked_non_positive_atr")),
        "execution_blocked_non_positive_size": _safe_int(exec_diag.get("blocked_non_positive_size")),
        "execution_blocked_non_positive_price": _safe_int(exec_diag.get("blocked_non_positive_price")),
        "execution_event_context_no_trade_active_steps": _safe_int(
            exec_diag.get("event_context_no_trade_active_steps")
        ),
        "execution_event_context_action_overrides": _safe_int(
            exec_diag.get("event_context_action_overrides")
        ),
        "execution_event_context_blocked_entries": _safe_int(
            exec_diag.get("event_context_blocked_entries")
        ),
        "execution_event_context_forced_flat_actions": _safe_int(
            exec_diag.get("event_context_forced_flat_actions")
        ),
        "execution_event_context_forced_flat_orders": _safe_int(
            exec_diag.get("event_context_forced_flat_orders")
        ),
    }
