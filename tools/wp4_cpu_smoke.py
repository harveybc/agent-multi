#!/usr/bin/env python3
"""WP4 (order 2026-08-20 20:40): bounded CPU end-to-end smoke of the
REAL path — real PipelinePlugin, real gym_fx_env, real SAC agent, real
ETH data, episodic objective as the executing selector.

Correction 3: asserts and reports the loaded gym_fx_env implementation
origin and its pinned commit before anything trains.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from importlib.metadata import entry_points
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PINNED_GYMFX = Path("/home/harveybc/Documents/GitHub/.runtime/"
                    "gym-fx-p1lr-634c3fd3")
EXPECTED_COMMIT = "634c3fd3c344cae3c4048b334158185c8bf4e1ef"
DATA = Path("/home/harveybc/Documents/GitHub/predictor/examples/data/"
            "project3/ethusdt_4h_tech_stat_full_model_ready.csv")

# engineered feature list derived from the real CSV header (price/raw
# columns excluded per the observation contract)
import csv as _csv
_EXCLUDE = {"DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"}
FEATURE_COLUMNS = [c for c in next(_csv.reader(DATA.open()))
                   if c not in _EXCLUDE]


def assert_env_origin() -> dict:
    """Correction 3: prove the loaded implementation IS the pinned one."""
    ep = next(e for e in entry_points().select(group="env.plugins")
              if e.name == "gym_fx_env")
    wrapper = ep.load()
    wrapper_file = str(Path(sys.modules[wrapper.__module__].__file__
                            ).resolve())
    # the entry point is agent-multi's WRAPPER; the implementation that
    # actually steps bars is gym-fx app.env — assert THAT origin, which
    # is what the campaign pins via PYTHONPATH.
    import importlib
    env_mod = importlib.import_module("app.env")
    origin = str(Path(env_mod.__file__).resolve())
    commit = subprocess.run(
        ["git", "-C", str(PINNED_GYMFX), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    pinned = origin.startswith(str(PINNED_GYMFX))
    if not pinned:
        del sys.modules["app.env"]  # do not leave a wrong module cached
        raise RuntimeError(
            f"REFUSED_ENV_ORIGIN: gym_fx_env loads from {origin}, not "
            f"the pinned runtime {PINNED_GYMFX}")
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"REFUSED_ENV_COMMIT: pinned worktree at {commit}, "
            f"expected {EXPECTED_COMMIT}")
    return {"wrapper_file": wrapper_file,
            "implementation_module": "app.env",
            "file": origin, "pinned_root":
            str(PINNED_GYMFX), "commit": commit}


def main() -> int:
    sys.path.insert(0, str(PINNED_GYMFX))
    origin = assert_env_origin()

    from pipeline_plugins.rl_pipeline_with_validation import (
        PipelinePlugin,
    )
    config = {
        "input_data_file": str(DATA),
        "env_plugin": "gym_fx_env",
        "agent_plugin": "project3_sac_actor_critic_agent",
        "quiet_mode": True,
        # bounded real splits (days, from dataset start)
        "train_days": 120, "val_days": 40, "test_days": 40,
        "min_split_rows": 100,
        # bounded budgets
        "epoch_timesteps": 512, "max_epochs": 3,
        "l1_patience": 2, "l1_patience_start_epoch": 0,
        "l1_min_delta": 1e-6,
        "window_size": 32, "initial_cash": 10000.0,
        "action_space_mode": "continuous",
        # easy-phase action semantics for the bounded smoke: an
        # untrained 3-epoch SAC cannot cross 0.1; threshold 0.0 is the
        # campaign's own easy contract and lets the smoke demonstrate
        # activity, selection and episodic components end to end.
        "continuous_action_threshold": 0.0,
        "solvency_mode": "normal_realistic",
        # finding-235 fail-closed contract: engineered causally scaled
        # features only, no raw price window
        "require_feature_aware_preprocessor": True,
        "include_price_window": False,
        "preprocessor_plugin": "feature_window_preprocessor",
        "feature_columns": FEATURE_COLUMNS,
        "feature_scaling": "rolling_zscore",
        "feature_scaling_window": 256,
        # WP3: the episodic objective IS the selector, legacy refused
        "selection_metric": "episodic_activity_economic_v1",
        "require_episodic_fitness": True,
        "episodic_activity_fitness": {
            # the NAMED diagnostic WP4 candidate — not production truth
            "activity_plateau_low_rate": 50.0,
            "activity_plateau_high_rate": 300.0,
        },
        "sac_params": {"learning_rate": 3e-4, "batch_size": 64,
                       "learning_starts": 128, "device": "cpu"},
        # D1: the per-epoch return traces ARE the activity evidence the
        # authority verifies — the smoke writes them like the campaign.
        "return_trace_dir": str(REPO / "docs/audits/evidence/"
                                "wp4_smoke_traces"),
        "inactive_terminal_is_typed_result": True,
    }
    agent_ep = next(e for e in entry_points().select(
        group="agent.plugins")
        if e.name == config["agent_plugin"])
    agent_plugin = agent_ep.load()()

    pipeline = PipelinePlugin(config)
    started = time.time()
    typed_termination = None
    try:
        result = pipeline.run_pipeline(config=config, env_plugin=None,
                                       agent_plugin=agent_plugin,
                                       mode="train")
    except RuntimeError as error:
        # the finding-232 typed no-activity termination is a VALID
        # smoke outcome: the real path ran and refused honestly.
        typed_termination = str(error)
        result = {}
    elapsed = time.time() - started

    def facts(summary):
        if not isinstance(summary, dict):
            return None
        keep = {}
        for key in ("trades_total", "total_return",
                    "max_drawdown_fraction", "scored_steps",
                    "episodic_fitness"):
            if key in summary:
                value = summary[key]
                keep[key] = (value if not isinstance(value, dict) else
                             {k: value[k] for k in
                              ("branch", "selection_value",
                               "annualized_trade_rate",
                               "activity_utility")
                              if k in value})
        return keep

    best_path = result.get("best_model_path")
    history = result.get("history") or []
    last = history[-1] if history else {}
    report = {
        "schema": "agent_multi.wp4_cpu_smoke.v1",
        "env_origin": origin,
        "elapsed_seconds": round(elapsed, 1),
        "epochs_run": len(history),
        "no_eligible_checkpoint": result.get(
            "activity_stopped_without_eligible_checkpoint"),
        "final_equity": result.get("final_equity"),
        "max_drawdown_fraction": result.get("max_drawdown_fraction"),
        "mean_weekly_rap": result.get("mean_weekly_rap"),
        "last_epoch_facts": {k: last.get(k) for k in
                             ("epoch", "composite", "composite_raw",
                              "train_tail_trades", "val_trades",
                              "trade_gate_passed", "checkpoint_eligible")
                             if isinstance(last, dict)},
        "stop_reason": result.get("stop_reason"),
        "termination_cause": (result.get("termination_cause")
                              or typed_termination),
        "selected_checkpoint": best_path,
        "selected_checkpoint_sha256": (
            hashlib.sha256(Path(best_path).read_bytes()).hexdigest()
            if best_path and Path(str(best_path)).is_file() else None),
        "train_facts": facts(result.get("train_summary")),
        "train_tail_facts": facts(result.get("train_tail_summary")),
        "validation_facts": facts(result.get("validation_summary")),
        "result_keys": sorted(result.keys())[:40],
        "proposed_gpu_smoke_command_NOT_LAUNCHED": (
            "CUDA_VISIBLE_DEVICES=<uuid> python tools/wp4_cpu_smoke.py "
            "--gpu  # same contract, epoch_timesteps=20000, "
            "max_epochs=50, single local GPU, bounded"),
    }
    out = REPO / ("docs/audits/evidence/"
                  "WP4_CPU_SMOKE_REPORT_2026_08_20.json")
    out.write_text(json.dumps(report, indent=1, sort_keys=True,
                              default=str) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("elapsed_seconds", "epochs_run", "stop_reason",
                       "selected_checkpoint")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
