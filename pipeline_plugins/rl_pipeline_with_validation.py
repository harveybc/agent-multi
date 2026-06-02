"""
rl_pipeline_with_validation.py — train/val/test pipeline with L1 early stopping.

This pipeline mirrors predictor's three-mode pattern (train / inference /
optimization upstream) and adds per-epoch validation evaluation with
level-1 early stopping based on a composite watch metric:

    composite = 0.5 * (train_total_return + val_total_return)

Patience resets when composite improves over the best so far. Training
stops when patience >= configured `l1_patience` or `max_epochs` is hit.

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
        "min_split_rows": 100,
        "split_anchor": "start",  # "start" or "end" of dataset

        # epoch loop
        "epoch_timesteps": 2_000,
        "max_epochs": 30,
        "l1_patience": 5,
        "l1_min_delta": 1e-4,

        # eval
        "eval_seed": 0,
        "train_seed": 0,
        "save_model": "./agent_model.zip",
        "load_model": None,
        "return_trace_dir": None,
    }

    plugin_debug_vars = [
        "train_years", "val_years", "test_years",
        "train_days", "val_days", "test_days", "min_split_rows",
        "epoch_timesteps", "max_epochs", "l1_patience", "l1_min_delta",
        "return_trace_dir",
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

        first = df[date_col].iloc[0]
        last = df[date_col].iloc[-1]
        if anchor == "end":
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
        for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
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

                # training mode
                model = agent_plugin.build(train_env, config)
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
                    return final

                epoch_ts = int(config.get("epoch_timesteps", self.params["epoch_timesteps"]))
                max_epochs = int(config.get("max_epochs", self.params["max_epochs"]))
                total_progress_timesteps = int(config.get("total_timesteps") or epoch_ts * max_epochs)
                l1_patience = int(config.get("l1_patience", self.params["l1_patience"]))
                l1_min_delta = float(config.get("l1_min_delta", self.params["l1_min_delta"]))
                seed = int(config.get("eval_seed", self.params["eval_seed"]))

                # L2 patience info shown in logs (driven externally by optimizer if any)
                l2_patience = config.get("optimization_patience", "-")
                l2_counter = config.get("_l2_counter", "-")

                best_composite = -math.inf
                no_improve = 0
                best_model_path = config.get("save_model") or "./agent_model.zip"
                Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)

                history: List[Dict[str, Any]] = []

                if not config.get("quiet_mode"):
                    print(
                        f"[train] starting: epoch_timesteps={epoch_ts} max_epochs={max_epochs} "
                        f"l1_patience={l1_patience} (composite=mean(train_return,val_return))"
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
                    val_summary = self._eval_on_split(
                        env_plugin_name, config, paths["val"], agent_plugin, model, seed, "validation_epoch"
                    )

                    train_ret = _safe_float(train_summary.get("total_return"))
                    val_ret = _safe_float(val_summary.get("total_return"))
                    if math.isnan(train_ret):
                        train_ret = 0.0
                    if math.isnan(val_ret):
                        val_ret = 0.0
                    composite = 0.5 * (train_ret + val_ret)

                    improved = composite > (best_composite + l1_min_delta)
                    if improved:
                        best_composite = composite
                        no_improve = 0
                        agent_plugin.save(model, best_model_path)
                    else:
                        no_improve += 1

                    history.append({
                        "epoch": epoch,
                        "train_total_return": train_ret,
                        "val_total_return": val_ret,
                        "composite": composite,
                        "best_composite": best_composite,
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
                        "val_trades": int(_safe_float(val_summary.get("trades_total")) or 0),
                        "val_win_pct": _win_pct(val_summary),
                        "val_sharpe": _safe_float(val_summary.get("sharpe_ratio")),
                        "val_profit_pct": val_ret * 100.0,
                        "val_balance": _safe_float(val_summary.get("final_equity")),
                    })

                    print(
                        f"[epoch {epoch:>3}/{max_epochs}] "
                        f"L1 {no_improve}/{l1_patience}  "
                        f"L2 {l2_counter}/{l2_patience}  "
                        f"composite={composite:+.4f} best={best_composite:+.4f} "
                        f"{'(IMPROVED, model saved)' if improved else ''} "
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
                        f"            VAL   trades={int(_safe_float(val_summary.get('trades_total')) or 0):>4} "
                        f"win%={_win_pct(val_summary):>5.2f} "
                        f"sharpe={_safe_float(val_summary.get('sharpe_ratio')):+.4f} "
                        f"profit={val_ret*100:+.2f}% "
                        f"bal={_safe_float(val_summary.get('final_equity')):.2f}",
                        flush=True,
                    )

                    if no_improve >= l1_patience:
                        print(
                            f"[train] L1 EARLY STOP at epoch {epoch} "
                            f"(no improvement for {no_improve} epochs, patience={l1_patience})",
                            flush=True,
                        )
                        break

                # Reload best model for final evaluation
                if Path(best_model_path).exists():
                    model = agent_plugin.load(best_model_path, train_env)

                final = self._final_eval(
                    agent_plugin, model, train_env,
                    env_plugin_name, paths, config, agent_plugin,
                )
                final["mode"] = mode
                final["history"] = history
                final["best_composite"] = best_composite
                final["best_model_path"] = str(Path(best_model_path).resolve())
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
        val_summary = self._eval_on_split(
            env_plugin_name, config, paths["val"], agent_plugin_for_wrap, model, seed, "validation"
        )
        test_summary = self._eval_on_split(
            env_plugin_name, config, paths["test"], agent_plugin_for_wrap, model, seed, "test"
        )

        rows = [
            ("Train", train_summary),
            ("Validation", val_summary),
            ("Test", test_summary),
        ]
        table = _format_table(rows)
        print("\n=== Final results (best-composite checkpoint) ===")
        print(table, flush=True)

        # Pop transient evidence-bearing fields out of each split summary
        # before exporting, then build the run-level evidence index.
        metadata_items: List[Dict[str, Any]] = []
        for s in (train_summary, val_summary, test_summary):
            meta = s.pop("_return_trace_metadata", None)
            if meta is not None:
                metadata_items.append(meta)

        # Build the export payload
        out = {
            "splits": {
                "train": train_summary,
                "validation": val_summary,
                "test": test_summary,
            },
            "summary_table": table,
        }
        # also surface top-level metrics from validation for compatibility
        out.update({
            "total_return": val_summary.get("total_return"),
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
        if model_path:
            results_path = Path(model_path).with_name("results.json")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with results_path.open("w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, default=str)
            print(f"[results] wrote {results_path}", flush=True)
        return out
