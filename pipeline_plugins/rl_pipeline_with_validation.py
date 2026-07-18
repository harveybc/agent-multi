"""
rl_pipeline_with_validation.py — train/val/test pipeline with L1 early stopping.

This pipeline mirrors predictor's three-mode pattern (train / inference /
optimization upstream) and adds per-epoch validation evaluation with
level-1 early stopping based on a composite watch metric:

    selection_mean = 0.5 * (train_tail_score + val_score)
    composite = selection_mean - beta * abs(train_tail_score - val_score)

When `selection_metric=risk_adjusted_return`, train_tail_score and val_score
are RAP = total_return - lambda * max_drawdown_fraction. Patience resets when
the L1 composite improves over the best so far. Training stops when patience
>= configured `l1_patience` or `max_epochs` is hit.
The train-side watch window is the last week of the training period, not
the full multi-year training slice, so a large historical train return cannot
hide no-trade or bad validation behavior.

Per-epoch logs include:
    epoch | L1 patience X/N | L2 patience Y/M | trades | win% | sharpe |
    profit | balance      (validation rollout)

Splits are time-ordered:
    train: first `train_years` (chronological)
    val:   next  `val_years`
    test:  next  `test_years` (typically the last block)

Final output: ASCII table over train/val/test plus results.json with the
same content next to the saved model.
"""
from __future__ import annotations

import json
import math
import tempfile
from copy import deepcopy
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import _return_trace as _trace_mod
from agent_plugins._progress_callback import make_progress_callback


_METRIC_KEYS = ("trades_total", "win_pct", "sharpe_ratio", "total_return", "final_equity")


def _load_env_plugin(name: str, config: Dict[str, Any]):
    eps = entry_points().select(group="env.plugins")
    ep = next((e for e in eps if e.name == name), None)
    if ep is None:
        raise ImportError(f"env plugin '{name}' not found")
    klass = ep.load()
    inst = klass(config)
    inst.set_params(**config)
    return inst


def _win_pct(summary: Dict[str, Any]) -> float:
    won = summary.get("trades_won")
    total = summary.get("trades_total")
    try:
        won = float(won) if won is not None else 0.0
        total = float(total) if total is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return (won / total * 100.0) if total > 0 else 0.0


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trade_count(summary: Dict[str, Any]) -> int:
    return int(_safe_float(summary.get("trades_total")) or 0)


def _drawdown_fraction(summary: Dict[str, Any]) -> float:
    """Return max drawdown as a positive fraction of equity.

    Backtrader's DrawDown analyzer reports ``max.drawdown`` as a percentage
    value (for example 2.5 means 2.5%). Some older/imported summaries may
    already carry fractional values under ``max_drawdown``; keep that fallback
    deliberately conservative.
    """
    raw_pct = _safe_float(summary.get("max_drawdown_pct"))
    if not math.isnan(raw_pct):
        return max(0.0, raw_pct / 100.0)
    raw_fraction = _safe_float(summary.get("max_drawdown"))
    if not math.isnan(raw_fraction):
        return abs(raw_fraction)
    return 0.0


def _risk_adjusted_return(summary: Dict[str, Any], risk_lambda: float) -> float:
    ret = _safe_float(summary.get("total_return"))
    if math.isnan(ret):
        ret = 0.0
    return ret - float(risk_lambda) * _drawdown_fraction(summary)


def _annotate_risk_adjusted(summary: Dict[str, Any], risk_lambda: float) -> None:
    drawdown = _drawdown_fraction(summary)
    ret = _safe_float(summary.get("total_return"))
    if math.isnan(ret):
        ret = 0.0
    summary["max_drawdown_fraction"] = drawdown
    summary["risk_penalty_lambda"] = float(risk_lambda)
    summary["risk_adjusted_total_return"] = ret - float(risk_lambda) * drawdown


def _resolve_l1_min_checkpoint_timesteps(
    config: Dict[str, Any],
    default: int | None = None,
) -> int:
    """Return the earliest timestep at which an L1 checkpoint is trainable.

    Off-policy agents do not update their networks before ``learning_starts``.
    Letting an earlier rollout compete for the best checkpoint makes every
    hyperparameter candidate collapse to the same seeded, untrained policy.
    """
    configured = config.get("l1_min_checkpoint_timesteps", default)
    if configured is None:
        learning_starts = max(0, int(config.get("learning_starts", 0)))
        return learning_starts + 1 if learning_starts else 0
    return max(0, int(configured))


def _update_l1_checkpoint_state(
    *,
    composite: float,
    best_composite: float,
    no_improve: int,
    min_delta: float,
    eligible: bool,
) -> tuple[float, int, bool]:
    """Update checkpoint/patience state without charging warm-up epochs."""
    if not eligible:
        return best_composite, no_improve, False
    improved = composite > (best_composite + min_delta)
    if improved:
        return composite, 0, True
    return best_composite, no_improve + 1, False


def _selection_value(summary: Dict[str, Any], *, selection_metric: str, risk_lambda: float) -> float:
    metric = str(selection_metric or "total_return").strip().lower()
    if metric in {"risk_adjusted_return", "risk_adjusted_total_return", "rap"}:
        return _risk_adjusted_return(summary, risk_lambda)
    ret = _safe_float(summary.get("total_return"))
    return 0.0 if math.isnan(ret) else ret


def _selection_pair_details(
    train_tail_summary: Dict[str, Any],
    val_summary: Dict[str, Any],
    *,
    selection_metric: str,
    risk_lambda: float,
    gap_penalty_beta: float,
) -> Dict[str, float]:
    train_tail_score = _selection_value(
        train_tail_summary,
        selection_metric=selection_metric,
        risk_lambda=risk_lambda,
    )
    val_score = _selection_value(
        val_summary,
        selection_metric=selection_metric,
        risk_lambda=risk_lambda,
    )
    mean_score = 0.5 * (train_tail_score + val_score)
    gap = abs(train_tail_score - val_score)
    gap_penalty = float(gap_penalty_beta) * gap
    return {
        "train_tail_selection_score": train_tail_score,
        "validation_selection_score": val_score,
        "train_validation_selection_mean_score": mean_score,
        "train_validation_selection_gap": gap,
        "train_validation_selection_gap_penalty": gap_penalty,
        "train_validation_selection_score": mean_score - gap_penalty,
    }


def _early_stop_composite(
    train_tail_summary: Dict[str, Any],
    val_summary: Dict[str, Any],
    *,
    min_trades: int,
    no_trade_penalty: float,
    selection_metric: str = "total_return",
    risk_lambda: float = 1.0,
    gap_penalty_beta: float = 0.25,
) -> Tuple[float, float, bool, float, float, int, int]:
    train_tail_ret = _safe_float(train_tail_summary.get("total_return"))
    val_ret = _safe_float(val_summary.get("total_return"))
    if math.isnan(train_tail_ret):
        train_tail_ret = 0.0
    if math.isnan(val_ret):
        val_ret = 0.0
    details = _selection_pair_details(
        train_tail_summary,
        val_summary,
        selection_metric=selection_metric,
        risk_lambda=risk_lambda,
        gap_penalty_beta=gap_penalty_beta,
    )
    raw = details["train_validation_selection_score"]
    train_tail_trades = _trade_count(train_tail_summary)
    val_trades = _trade_count(val_summary)
    trade_gate_passed = train_tail_trades >= min_trades and val_trades >= min_trades
    composite = raw if trade_gate_passed else raw - no_trade_penalty
    return composite, raw, trade_gate_passed, train_tail_ret, val_ret, train_tail_trades, val_trades


def _normalize_split_label(name: str) -> str:
    n = str(name).strip().lower()
    if n in _trace_mod.ALLOWED_SPLITS:
        return n
    if n in ("val", "valid"):
        return "validation"
    if n.endswith("_epoch") and n[: -len("_epoch")] in ("train", "validation", "val"):
        return n if n in _trace_mod.ALLOWED_SPLITS else "evaluation"
    return "evaluation"


def _format_table(rows: List[Tuple[str, Dict[str, Any]]]) -> str:
    headers = ["Split", "Trades", "Win %", "Sharpe", "Profit", "Balance"]
    fmt_rows = []
    for name, s in rows:
        fmt_rows.append([
            name,
            str(int(_safe_float(s.get("trades_total")) or 0)),
            f"{_win_pct(s):.2f}",
            f"{_safe_float(s.get('sharpe_ratio')):.4f}",
            f"{_safe_float(s.get('total_return')) * 100:.2f}%",
            f"{_safe_float(s.get('final_equity')):.2f}",
        ])
    widths = [max(len(h), max(len(r[i]) for r in fmt_rows)) for i, h in enumerate(headers)]
    sep = "+".join("-" * (w + 2) for w in widths)
    sep = "+" + sep + "+"
    def fmt_row(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    out = [sep, fmt_row(headers), sep]
    for r in fmt_rows:
        out.append(fmt_row(r))
    out.append(sep)
    return "\n".join(out)


class PipelinePlugin:
    plugin_params: Dict[str, Any] = {
        # split widths (years)
        "train_years": 4,
        "val_years": 1,
        "test_years": 1,
        "train_days": None,
        "val_days": None,
        "test_days": None,
        "train_start": None,
        "train_end": None,
        "validation_start": None,
        "validation_end": None,
        "val_start": None,
        "val_end": None,
        "test_start": None,
        "test_end": None,
        "min_split_rows": 100,
        "split_anchor": "start",  # "start" or "end" of dataset

        # epoch loop
        "epoch_timesteps": 2_000,
        "max_epochs": 500,
        "l1_patience": 20,
        "l1_min_delta": 1e-4,
        "l1_min_checkpoint_timesteps": None,
        "early_stop_train_tail_days": 7,
        "early_stop_min_trades": 1,
        "early_stop_no_trade_penalty": 1_000_000.0,
        "selection_metric": "total_return",
        "risk_penalty_lambda": 1.0,
        "l1_generalization_gap_penalty_beta": 0.25,

        # eval
        "eval_seed": 0,
        "train_seed": 0,
        "save_model": "./agent_model.zip",
        "load_model": None,
        "warm_start_model": None,
        "return_trace_dir": None,
        "evaluate_test_split": True,
        "write_results_sidecar": True,
    }

    plugin_debug_vars = [
        "train_years", "val_years", "test_years",
        "train_days", "val_days", "test_days",
        "train_start", "train_end", "validation_start", "validation_end",
        "val_start", "val_end", "test_start", "test_end",
        "min_split_rows",
        "epoch_timesteps", "max_epochs", "l1_patience", "l1_min_delta",
        "l1_min_checkpoint_timesteps",
        "early_stop_train_tail_days", "early_stop_min_trades", "early_stop_no_trade_penalty",
        "selection_metric", "risk_penalty_lambda", "l1_generalization_gap_penalty_beta",
        "warm_start_model", "return_trace_dir", "evaluate_test_split",
        "write_results_sidecar",
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        self._tempdir: Optional[tempfile.TemporaryDirectory] = None
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
    def _split_csv(self, config: Dict[str, Any]) -> Dict[str, str]:
        src = config["input_data_file"]
        date_col = config.get("date_column", "DATE_TIME")
        df = pd.read_csv(src)
        if date_col not in df.columns:
            raise ValueError(f"date_column '{date_col}' missing from {src}")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        train_y = float(config.get("train_years", self.params["train_years"]))
        val_y = float(config.get("val_years", self.params["val_years"]))
        test_y = float(config.get("test_years", self.params["test_years"]))
        train_d = _safe_float_or_none(config.get("train_days", self.params["train_days"]))
        val_d = _safe_float_or_none(config.get("val_days", self.params["val_days"]))
        test_d = _safe_float_or_none(config.get("test_days", self.params["test_days"]))
        min_split_rows = int(config.get("min_split_rows", self.params["min_split_rows"]))
        anchor = str(config.get("split_anchor", self.params["split_anchor"])).lower()
        use_day_splits = train_d is not None and val_d is not None and test_d is not None
        explicit_train_start = config.get("train_start", self.params["train_start"])
        explicit_train_end = config.get("train_end", self.params["train_end"])
        explicit_val_start = (
            config.get("validation_start", self.params["validation_start"])
            or config.get("val_start", self.params["val_start"])
        )
        explicit_val_end = (
            config.get("validation_end", self.params["validation_end"])
            or config.get("val_end", self.params["val_end"])
        )
        explicit_test_start = config.get("test_start", self.params["test_start"])
        explicit_test_end = config.get("test_end", self.params["test_end"])
        explicit_ranges = [
            explicit_train_start,
            explicit_train_end,
            explicit_val_start,
            explicit_val_end,
            explicit_test_start,
            explicit_test_end,
        ]
        use_explicit_splits = all(v not in (None, "") for v in explicit_ranges)
        if any(v not in (None, "") for v in explicit_ranges) and not use_explicit_splits:
            raise ValueError(
                "Explicit weekly split windows require train_start, train_end, "
                "validation_start/val_start, validation_end/val_end, test_start, and test_end."
            )

        first = df[date_col].iloc[0]
        last = df[date_col].iloc[-1]
        if use_explicit_splits:
            train_start = pd.Timestamp(explicit_train_start)
            train_end = pd.Timestamp(explicit_train_end)
            val_start = pd.Timestamp(explicit_val_start)
            val_end = pd.Timestamp(explicit_val_end)
            test_start = pd.Timestamp(explicit_test_start)
            test_end = pd.Timestamp(explicit_test_end)
            if not train_start < train_end <= val_start < val_end <= test_start < test_end:
                raise ValueError(
                    "Explicit weekly split windows must be ordered as "
                    "train_start < train_end <= validation_start < validation_end <= test_start < test_end."
                )
        elif anchor == "end":
            test_end = last
            if use_day_splits:
                test_start = test_end - pd.DateOffset(days=int(test_d))
                val_end = test_start
                val_start = val_end - pd.DateOffset(days=int(val_d))
                train_end = val_start
                train_start = train_end - pd.DateOffset(days=int(train_d))
            else:
                test_start = test_end - pd.DateOffset(years=int(test_y))
                val_end = test_start
                val_start = val_end - pd.DateOffset(years=int(val_y))
                train_end = val_start
                train_start = train_end - pd.DateOffset(years=int(train_y))
        else:
            train_start = first
            if use_day_splits:
                train_end = train_start + pd.DateOffset(days=int(train_d))
                val_start = train_end
                val_end = val_start + pd.DateOffset(days=int(val_d))
                test_start = val_end
                test_end = test_start + pd.DateOffset(days=int(test_d))
            else:
                train_end = train_start + pd.DateOffset(years=int(train_y))
                val_start = train_end
                val_end = val_start + pd.DateOffset(years=int(val_y))
                test_start = val_end
                test_end = test_start + pd.DateOffset(years=int(test_y))

        train_df = df[(df[date_col] >= train_start) & (df[date_col] < train_end)]
        val_df = df[(df[date_col] >= val_start) & (df[date_col] < val_end)]
        test_df = df[(df[date_col] >= test_start) & (df[date_col] < test_end)]
        train_tail_days = _safe_float_or_none(
            config.get("early_stop_train_tail_days", self.params["early_stop_train_tail_days"])
        )
        train_tail_df = train_df
        if train_tail_days is not None and train_tail_days > 0:
            train_tail_start = train_end - pd.DateOffset(days=int(train_tail_days))
            train_tail_df = df[(df[date_col] >= train_tail_start) & (df[date_col] < train_end)]

        for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
            if len(part) < min_split_rows:
                raise ValueError(
                    f"{name} split has only {len(part)} rows; minimum is {min_split_rows} (range "
                    f"{train_start if name=='train' else val_start if name=='val' else test_start} "
                    f"-> {train_end if name=='train' else val_end if name=='val' else test_end}). "
                    f"Adjust split_anchor, *_years, *_days, or min_split_rows."
                )

        self._tempdir = tempfile.TemporaryDirectory(prefix="agent_multi_split_")
        out_dir = Path(self._tempdir.name)
        paths = {}
        for name, part in (
            ("train", train_df),
            ("train_tail", train_tail_df),
            ("val", val_df),
            ("test", test_df),
        ):
            p = out_dir / f"{name}.csv"
            part.to_csv(p, index=False)
            paths[name] = str(p)
        if not config.get("quiet_mode"):
            print(
                f"[split] train={len(train_df):>6} rows ({train_df[date_col].iloc[0].date()} -> {train_df[date_col].iloc[-1].date()})  "
                f"val={len(val_df):>5} rows ({val_df[date_col].iloc[0].date()} -> {val_df[date_col].iloc[-1].date()})  "
                f"test={len(test_df):>5} rows ({test_df[date_col].iloc[0].date()} -> {test_df[date_col].iloc[-1].date()})"
            )
        return paths

    def _make_split_env(self, env_plugin_name: str, base_config: Dict[str, Any], csv_path: str, agent_plugin):
        cfg = deepcopy(base_config)
        cfg["input_data_file"] = csv_path
        env_plugin = _load_env_plugin(env_plugin_name, cfg)
        env = env_plugin.make_env(cfg)
        wrap = getattr(agent_plugin, "wrap_env", None)
        if callable(wrap):
            env = wrap(env, cfg)
        return env_plugin, env

    def _eval_on_split(
        self,
        env_plugin_name: str,
        config: Dict[str, Any],
        csv_path: str,
        agent_plugin,
        model,
        seed: int,
        split_name: str,
    ) -> Dict[str, Any]:
        """Build a fresh env just for evaluation, roll out once, close it.

        Critical: never reuse the training env for evaluation — that pollutes
        SAC's replay buffer with terminal/post-reset transitions and freezes
        the actor weights from epoch 2 onward.
        """
        plug, env = self._make_split_env(env_plugin_name, config, csv_path, agent_plugin)
        try:
            split_label = _normalize_split_label(split_name)
            run_id = _trace_mod.make_run_id(config)
            episode_id = f"{run_id}::{split_label}"
            asset = str(config.get("asset", "unknown_asset"))
            timeframe = str(config.get("timeframe", config.get("timeframe_label", "")))

            summary = self._rollout(
                env, agent_plugin, model, seed,
                asset=asset, timeframe=timeframe, split=split_label,
                run_id=run_id, episode_id=episode_id,
            )
            trace_rows = summary.pop("_return_trace_rows", None)
            trace_dir = config.get("return_trace_dir")
            if trace_dir and trace_rows is not None:
                trace_path = _trace_mod.derive_split_trace_path(str(trace_dir), split_label)
                # Per-split config view so the metadata sidecar's data_file
                # hash matches the slice that was actually evaluated.
                split_config = dict(config)
                split_config["_run_config_hash"] = config.get("_run_config_hash") or _trace_mod._hash_config(config)
                split_config["input_data_file"] = csv_path
                split_config["_split"] = split_label
                metadata = _trace_mod.write_return_trace(
                    trace_path,
                    trace_rows,
                    config=split_config,
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
                # Stash the full sidecar so _final_eval can roll the per-split
                # metadata items into the run-level evidence index.
                summary["_return_trace_metadata"] = metadata
            return summary
        finally:
            try:
                plug.close()
            except Exception:
                pass

    @staticmethod
    def _rollout(
        env, agent_plugin, model, seed: int,
        *,
        asset: str = "unknown_asset",
        timeframe: str = "",
        split: str = "evaluation",
        run_id: str = "run",
        episode_id: str = "run::ep0",
    ) -> Dict[str, Any]:
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        done = False
        trace_rows: List[Dict[str, Any]] = []
        prev_equity = _safe_float_or_none(info.get("equity"))
        while not done:
            action = agent_plugin.predict(model, obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            equity = _safe_float_or_none(info.get("equity"))
            trace_rows.append(
                _trace_mod.build_trace_row(
                    env=env,
                    step=steps + 1,
                    action=action,
                    reward=reward,
                    info=info,
                    prev_equity=prev_equity,
                    asset=asset,
                    timeframe=timeframe,
                    split=split,
                    seed=int(seed),
                    run_id=run_id,
                    episode_id=episode_id,
                )
            )
            prev_equity = equity
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            if steps > 1_000_000:
                break
        base = env
        while hasattr(base, "env") and not hasattr(base, "summary"):
            base = base.env
        summary = base.summary() if hasattr(base, "summary") else {}
        summary["episode_reward"] = total_reward
        summary["episode_length"] = steps
        summary["_return_trace_rows"] = trace_rows
        return summary

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
        env_plugin_name = config.get("env_plugin", "gym_fx_env")
        try:
            paths = self._split_csv(config)

            train_env_plugin, train_env = self._make_split_env(
                env_plugin_name, config, paths["train"], agent_plugin
            )

            try:
                if mode == "inference":
                    load_path = config.get("load_model")
                    if not load_path:
                        raise ValueError("inference mode requires config['load_model']")
                    model = agent_plugin.load(load_path, train_env)
                    final = self._final_eval(
                        agent_plugin, model, train_env,
                        env_plugin_name, paths, config, agent_plugin
                    )
                    return final

                # training mode. Optional warm-start continues from a previous
                # weekly checkpoint but evaluates/saves under the current
                # split windows and run id.
                warm_start_model = config.get("warm_start_model", self.params["warm_start_model"])
                if warm_start_model:
                    warm_start_path = Path(str(warm_start_model))
                    if not warm_start_path.exists():
                        raise FileNotFoundError(f"warm_start_model not found: {warm_start_path}")
                    if not config.get("quiet_mode"):
                        print(f"[train] warm-start loading {warm_start_path}", flush=True)
                    model = agent_plugin.load(str(warm_start_path), train_env)
                    try:
                        model.set_env(train_env)
                    except Exception:
                        pass
                else:
                    model = agent_plugin.build(train_env, config)
                pretrain_summary = None
                pretrain_behavior = getattr(agent_plugin, "pretrain_behavior", None)
                if callable(pretrain_behavior) and bool(config.get("oracle_behavior_pretrain_enabled", False)):
                    pretrain_summary = pretrain_behavior(model, train_env, config)
                if not hasattr(model, "learn"):
                    best_model_path = config.get("save_model") or "./agent_model.zip"
                    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
                    agent_plugin.save(model, best_model_path)
                    final = self._final_eval(
                        agent_plugin, model, train_env,
                        env_plugin_name, paths, config, agent_plugin,
                    )
                    final["mode"] = "deterministic_baseline"
                    final["history"] = []
                    final["best_composite"] = None
                    final["best_model_path"] = str(Path(best_model_path).resolve())
                    final["oracle_behavior_pretrain"] = pretrain_summary
                    return final

                epoch_ts = int(config.get("epoch_timesteps", self.params["epoch_timesteps"]))
                max_epochs = int(config.get("max_epochs", self.params["max_epochs"]))
                total_progress_timesteps = int(config.get("total_timesteps") or epoch_ts * max_epochs)
                l1_patience = int(config.get("l1_patience", self.params["l1_patience"]))
                l1_min_delta = float(config.get("l1_min_delta", self.params["l1_min_delta"]))
                l1_min_checkpoint_timesteps = _resolve_l1_min_checkpoint_timesteps(
                    config,
                    self.params["l1_min_checkpoint_timesteps"],
                )
                seed = int(config.get("eval_seed", self.params["eval_seed"]))

                # L2 patience info shown in logs (driven externally by optimizer if any)
                l2_patience = config.get("optimization_patience", "-")
                l2_counter = config.get("_l2_counter", "-")

                best_composite = -math.inf
                no_improve = 0
                best_checkpoint_saved = False
                best_model_path = config.get("save_model") or "./agent_model.zip"
                Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)

                history: List[Dict[str, Any]] = []

                if not config.get("quiet_mode"):
                    print(
                        f"[train] starting: epoch_timesteps={epoch_ts} max_epochs={max_epochs} "
                        f"l1_patience={l1_patience} "
                        f"(L1=mean(train_tail_score,val_score)-beta*gap, no-trade penalized)"
                    )

                def _policy_checksum(m) -> Tuple[float, float, float]:
                    try:
                        actor = sum(float(p.detach().abs().sum().item())
                                    for p in m.policy.actor.parameters())
                    except Exception:
                        actor = float("nan")
                    try:
                        critic = sum(float(p.detach().abs().sum().item())
                                     for p in m.policy.critic.parameters())
                    except Exception:
                        critic = float("nan")
                    try:
                        ent = float(m.log_ent_coef.detach().exp().item()) if hasattr(m, "log_ent_coef") else float("nan")
                    except Exception:
                        ent = float("nan")
                    return actor, critic, ent

                for epoch in range(1, max_epochs + 1):
                    a_b, c_b, e_b = _policy_checksum(model)
                    nts_before = int(getattr(model, "num_timesteps", 0))
                    rb_before = int(getattr(getattr(model, "replay_buffer", None), "size", lambda: 0)()) if hasattr(model, "replay_buffer") else 0
                    # On epoch 1 we set up cleanly; on subsequent epochs use
                    # reset_num_timesteps=False to *continue* training on the
                    # same SAC instance without re-initializing the schedule.
                    model.learn(
                        total_timesteps=epoch_ts,
                        reset_num_timesteps=(epoch == 1),
                        log_interval=max(1, epoch_ts // 1000),
                        callback=make_progress_callback(config, total_progress_timesteps),
                    )
                    a_a, c_a, e_a = _policy_checksum(model)
                    nts_after = int(getattr(model, "num_timesteps", 0))
                    rb_after = int(getattr(getattr(model, "replay_buffer", None), "size", lambda: 0)()) if hasattr(model, "replay_buffer") else 0

                    train_summary = self._eval_on_split(
                        env_plugin_name, config, paths["train"], agent_plugin, model, seed, "train_epoch"
                    )
                    train_tail_summary = self._eval_on_split(
                        env_plugin_name, config, paths.get("train_tail", paths["train"]),
                        agent_plugin, model, seed, "train_tail_epoch"
                    )
                    val_summary = self._eval_on_split(
                        env_plugin_name, config, paths["val"], agent_plugin, model, seed, "validation_epoch"
                    )
                    selection_metric = str(
                        config.get("selection_metric", self.params["selection_metric"])
                    )
                    risk_lambda = float(
                        config.get("risk_penalty_lambda", self.params["risk_penalty_lambda"])
                    )
                    l1_gap_beta = float(
                        config.get(
                            "l1_generalization_gap_penalty_beta",
                            self.params["l1_generalization_gap_penalty_beta"],
                        )
                    )
                    for split_summary in (train_summary, train_tail_summary, val_summary):
                        _annotate_risk_adjusted(split_summary, risk_lambda)

                    train_ret = _safe_float(train_summary.get("total_return"))
                    if math.isnan(train_ret):
                        train_ret = 0.0
                    early_stop_min_trades = int(
                        config.get("early_stop_min_trades", self.params["early_stop_min_trades"])
                    )
                    no_trade_penalty = float(
                        config.get(
                            "early_stop_no_trade_penalty",
                            self.params["early_stop_no_trade_penalty"],
                        )
                    )
                    (
                        composite,
                        composite_raw,
                        trade_gate_passed,
                        train_tail_ret,
                        val_ret,
                        train_tail_trades,
                        val_trades,
                    ) = _early_stop_composite(
                        train_tail_summary,
                        val_summary,
                        min_trades=early_stop_min_trades,
                        no_trade_penalty=no_trade_penalty,
                        selection_metric=selection_metric,
                        risk_lambda=risk_lambda,
                        gap_penalty_beta=l1_gap_beta,
                    )
                    selection_details = _selection_pair_details(
                        train_tail_summary,
                        val_summary,
                        selection_metric=selection_metric,
                        risk_lambda=risk_lambda,
                        gap_penalty_beta=l1_gap_beta,
                    )

                    checkpoint_eligible = nts_after >= l1_min_checkpoint_timesteps
                    best_composite, no_improve, improved = _update_l1_checkpoint_state(
                        composite=composite,
                        best_composite=best_composite,
                        no_improve=no_improve,
                        min_delta=l1_min_delta,
                        eligible=checkpoint_eligible,
                    )
                    if improved:
                        agent_plugin.save(model, best_model_path)
                        best_checkpoint_saved = True

                    history.append({
                        "epoch": epoch,
                        "train_total_return": train_ret,
                        "train_tail_total_return": train_tail_ret,
                        "val_total_return": val_ret,
                        "selection_metric": selection_metric,
                        "risk_penalty_lambda": risk_lambda,
                        "l1_generalization_gap_penalty_beta": l1_gap_beta,
                        "train_tail_risk_adjusted_total_return": train_tail_summary.get(
                            "risk_adjusted_total_return"
                        ),
                        "val_risk_adjusted_total_return": val_summary.get(
                            "risk_adjusted_total_return"
                        ),
                        "train_tail_max_drawdown_fraction": train_tail_summary.get(
                            "max_drawdown_fraction"
                        ),
                        "val_max_drawdown_fraction": val_summary.get(
                            "max_drawdown_fraction"
                        ),
                        **selection_details,
                        "composite_raw": composite_raw,
                        "composite": composite,
                        "best_composite": best_composite if best_checkpoint_saved else None,
                        "l1_checkpoint_eligible": checkpoint_eligible,
                        "l1_min_checkpoint_timesteps": l1_min_checkpoint_timesteps,
                        "early_stop_trade_gate_passed": trade_gate_passed,
                        "early_stop_min_trades": early_stop_min_trades,
                        "l1_patience_used": no_improve,
                        "l1_patience_max": l1_patience,
                        "policy_actor_l1_before": a_b,
                        "policy_actor_l1_after": a_a,
                        "policy_actor_delta": a_a - a_b,
                        "policy_critic_l1_before": c_b,
                        "policy_critic_l1_after": c_a,
                        "policy_critic_delta": c_a - c_b,
                        "ent_coef": e_a,
                        "train_trades": int(_safe_float(train_summary.get("trades_total")) or 0),
                        "train_win_pct": _win_pct(train_summary),
                        "train_sharpe": _safe_float(train_summary.get("sharpe_ratio")),
                        "train_profit_pct": train_ret * 100.0,
                        "train_balance": _safe_float(train_summary.get("final_equity")),
                        "train_tail_trades": train_tail_trades,
                        "train_tail_win_pct": _win_pct(train_tail_summary),
                        "train_tail_sharpe": _safe_float(train_tail_summary.get("sharpe_ratio")),
                        "train_tail_profit_pct": train_tail_ret * 100.0,
                        "train_tail_balance": _safe_float(train_tail_summary.get("final_equity")),
                        "val_trades": val_trades,
                        "val_win_pct": _win_pct(val_summary),
                        "val_sharpe": _safe_float(val_summary.get("sharpe_ratio")),
                        "val_profit_pct": val_ret * 100.0,
                        "val_balance": _safe_float(val_summary.get("final_equity")),
                    })

                    l1_status = (
                        f"{no_improve}/{l1_patience}"
                        if checkpoint_eligible
                        else f"warmup<{l1_min_checkpoint_timesteps}"
                    )
                    checkpoint_status = (
                        "(IMPROVED, model saved)"
                        if improved
                        else "(checkpoint ineligible)" if not checkpoint_eligible else ""
                    )
                    print(
                        f"[epoch {epoch:>3}/{max_epochs}] "
                        f"L1 {l1_status}  "
                        f"L2 {l2_counter}/{l2_patience}  "
                        f"{selection_metric} composite={composite:+.4f} raw={composite_raw:+.4f} "
                        f"trade_gate={'PASS' if trade_gate_passed else 'FAIL'} "
                        f"best={best_composite:+.4f} "
                        f"{checkpoint_status} "
                        f"actor|w|={a_a:.2f} Δa={a_a-a_b:+.4f} "
                        f"critic|w|={c_a:.2f} Δc={c_a-c_b:+.4f} ent={e_a:.4f} "
                        f"steps={nts_before}->{nts_after} buf={rb_before}->{rb_after}",
                        flush=True,
                    )
                    print(
                        f"            TRAIN trades={int(_safe_float(train_summary.get('trades_total')) or 0):>4} "
                        f"win%={_win_pct(train_summary):>5.2f} "
                        f"sharpe={_safe_float(train_summary.get('sharpe_ratio')):+.4f} "
                        f"profit={train_ret*100:+.2f}% "
                        f"bal={_safe_float(train_summary.get('final_equity')):.2f}",
                        flush=True,
                    )
                    print(
                        f"            TRAIN_TAIL trades={train_tail_trades:>4} "
                        f"win%={_win_pct(train_tail_summary):>5.2f} "
                        f"sharpe={_safe_float(train_tail_summary.get('sharpe_ratio')):+.4f} "
                        f"profit={train_tail_ret*100:+.2f}% "
                        f"bal={_safe_float(train_tail_summary.get('final_equity')):.2f}",
                        flush=True,
                    )
                    print(
                        f"            VAL   trades={int(_safe_float(val_summary.get('trades_total')) or 0):>4} "
                        f"win%={_win_pct(val_summary):>5.2f} "
                        f"sharpe={_safe_float(val_summary.get('sharpe_ratio')):+.4f} "
                        f"profit={val_ret*100:+.2f}% "
                        f"bal={_safe_float(val_summary.get('final_equity')):.2f}",
                        flush=True,
                    )

                    if checkpoint_eligible and no_improve >= l1_patience:
                        print(
                            f"[train] L1 EARLY STOP at epoch {epoch} "
                            f"(no improvement for {no_improve} epochs, patience={l1_patience})",
                            flush=True,
                        )
                        break

                if not best_checkpoint_saved:
                    raise RuntimeError(
                        "training ended before an L1 checkpoint became eligible: "
                        f"num_timesteps={int(getattr(model, 'num_timesteps', 0))}, "
                        f"l1_min_checkpoint_timesteps={l1_min_checkpoint_timesteps}"
                    )

                # Reload best model for final evaluation.
                model = agent_plugin.load(best_model_path, train_env)

                final = self._final_eval(
                    agent_plugin, model, train_env,
                    env_plugin_name, paths, config, agent_plugin,
                )
                final["mode"] = mode
                final["history"] = history
                final["best_composite"] = best_composite
                final["best_model_path"] = str(Path(best_model_path).resolve())
                final["oracle_behavior_pretrain"] = pretrain_summary
                return final
            finally:
                try:
                    train_env_plugin.close()
                except Exception:
                    pass
        finally:
            if self._tempdir is not None:
                try:
                    self._tempdir.cleanup()
                except Exception:
                    pass
                self._tempdir = None

    # ------------------------------------------------------------------
    def _final_eval(
        self,
        agent_plugin,
        model,
        train_env,
        env_plugin_name: str,
        paths: Dict[str, str],
        config: Dict[str, Any],
        agent_plugin_for_wrap,
    ) -> Dict[str, Any]:
        seed = int(config.get("eval_seed", self.params["eval_seed"]))
        train_summary = self._eval_on_split(
            env_plugin_name, config, paths["train"], agent_plugin_for_wrap, model, seed, "train"
        )
        train_tail_summary = self._eval_on_split(
            env_plugin_name, config, paths.get("train_tail", paths["train"]),
            agent_plugin_for_wrap, model, seed, "train_tail"
        )
        val_summary = self._eval_on_split(
            env_plugin_name, config, paths["val"], agent_plugin_for_wrap, model, seed, "validation"
        )
        evaluate_test = bool(
            config.get("evaluate_test_split", self.params["evaluate_test_split"])
        )
        if evaluate_test:
            test_summary = self._eval_on_split(
                env_plugin_name,
                config,
                paths["test"],
                agent_plugin_for_wrap,
                model,
                seed,
                "test",
            )
        else:
            test_summary = {
                "evaluation_skipped": True,
                "skip_reason": "protected_test_disabled_for_optimization",
            }
        selection_metric = str(config.get("selection_metric", self.params["selection_metric"]))
        risk_lambda = float(config.get("risk_penalty_lambda", self.params["risk_penalty_lambda"]))
        l1_gap_beta = float(
            config.get(
                "l1_generalization_gap_penalty_beta",
                self.params["l1_generalization_gap_penalty_beta"],
            )
        )
        metric_summaries = [train_summary, train_tail_summary, val_summary]
        if evaluate_test:
            metric_summaries.append(test_summary)
        for split_summary in metric_summaries:
            _annotate_risk_adjusted(split_summary, risk_lambda)
        selection_details = _selection_pair_details(
            train_tail_summary,
            val_summary,
            selection_metric=selection_metric,
            risk_lambda=risk_lambda,
            gap_penalty_beta=l1_gap_beta,
        )
        risk_adjusted_mean = 0.5 * (
            float(train_tail_summary.get("risk_adjusted_total_return") or 0.0)
            + float(val_summary.get("risk_adjusted_total_return") or 0.0)
        )

        rows = [
            ("Train", train_summary),
            ("TrainTail", train_tail_summary),
            ("Validation", val_summary),
        ]
        if evaluate_test:
            rows.append(("Test", test_summary))
        table = _format_table(rows)
        print("\n=== Final results (best-composite checkpoint) ===")
        print(table, flush=True)

        # Pop transient evidence-bearing fields out of each split summary
        # before exporting, then build the run-level evidence index.
        metadata_items: List[Dict[str, Any]] = []
        for s in (train_summary, train_tail_summary, val_summary, test_summary):
            meta = s.pop("_return_trace_metadata", None)
            if meta is not None:
                metadata_items.append(meta)

        # Build the export payload
        out = {
            "splits": {
                "train": train_summary,
                "train_tail": train_tail_summary,
                "validation": val_summary,
                "test": test_summary,
            },
            "summary_table": table,
            "selection_metric": selection_metric,
            "risk_penalty_lambda": risk_lambda,
            "l1_generalization_gap_penalty_beta": l1_gap_beta,
            "train_validation_risk_adjusted_composite_score": risk_adjusted_mean,
            "train_validation_risk_adjusted_mean_score": risk_adjusted_mean,
            "train_validation_l1_score": selection_details["train_validation_selection_score"],
            **selection_details,
        }
        # also surface top-level metrics from validation for compatibility
        out.update({
            "total_return": val_summary.get("total_return"),
            "risk_adjusted_total_return": val_summary.get("risk_adjusted_total_return"),
            "max_drawdown_fraction": val_summary.get("max_drawdown_fraction"),
            "final_equity": val_summary.get("final_equity"),
            "max_drawdown_pct": val_summary.get("max_drawdown_pct"),
            "sharpe_ratio": val_summary.get("sharpe_ratio"),
            "trades_total": val_summary.get("trades_total"),
            "episode_reward": val_summary.get("episode_reward"),
            "episode_length": val_summary.get("episode_length"),
            "eval_seed": seed,
        })

        if metadata_items:
            evidence = _trace_mod.build_return_trace_evidence(
                metadata_items,
                config=config,
                run_id=_trace_mod.make_run_id(config),
                pipeline_plugin="rl_pipeline_with_validation",
            )
            trace_dir = config.get("return_trace_dir")
            evidence_path = _trace_mod.derive_evidence_path(
                trace_dir=str(trace_dir) if trace_dir else None,
                trace_file=metadata_items[0].get("trace_file"),
            )
            evidence["evidence_file"] = _trace_mod.write_return_trace_evidence(
                evidence, evidence_path,
            )
            out["return_trace_evidence"] = evidence
            out["return_trace_evidence_file"] = evidence["evidence_file"]

        # Save results.json next to the model file
        model_path = config.get("save_model")
        if model_path and bool(
            config.get("write_results_sidecar", self.params["write_results_sidecar"])
        ):
            results_path = Path(model_path).with_name("results.json")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with results_path.open("w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, default=str)
            print(f"[results] wrote {results_path}", flush=True)
        return out
