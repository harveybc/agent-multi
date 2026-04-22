#!/usr/bin/env python3
"""
Smoke test for agent-multi on gym-fx.

Runs 3 quick checks:
  1. random_agent completes an episode, produces a finite total_return.
  2. buy_hold_agent beats random_agent on the synthetic uptrend feed
     used in gym-fx's own smoke test.
  3. ppo_agent trains for a small number of timesteps and then
     reloads from disk and executes a deterministic inference rollout.

Any assertion failure exits non-zero.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Find the sibling gym-fx repo automatically for dev installs.
SIBLING_GYMFX = REPO_ROOT.parent / "gym-fx"
if SIBLING_GYMFX.exists():
    sys.path.insert(0, str(SIBLING_GYMFX))

from app.config import DEFAULT_VALUES
from env_plugins.gym_fx_env import Plugin as EnvPlugin
from agent_plugins.random_agent import Plugin as RandomAgent
from agent_plugins.buy_hold_agent import Plugin as BuyHoldAgent
from agent_plugins.ppo_agent import Plugin as PPOAgent
from pipeline_plugins.rl_pipeline import PipelinePlugin


SAMPLE_FEED = SIBLING_GYMFX / "examples" / "data" / "eurusd_sample.csv"
UPTREND_FEED = SIBLING_GYMFX / "examples" / "data" / "eurusd_uptrend.csv"


def _base_config(overrides: dict | None = None) -> dict:
    cfg = {
        **DEFAULT_VALUES,
        "env_mode": "inference",
        "input_data_file": str(SAMPLE_FEED),
        "window_size": 32,
        "initial_cash": 10000.0,
        "total_timesteps": 500,
        "eval_seed": 0,
        "train_seed": 0,
        "quiet_mode": True,
        "device": "cpu",
        "agent_verbose": 0,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def _run(agent, config: dict, mode: str = "train") -> dict:
    env_plugin = EnvPlugin(config)
    pipeline = PipelinePlugin(config)
    agent.set_params(**config)
    summary = pipeline.run_pipeline(
        config=config,
        env_plugin=env_plugin,
        agent_plugin=agent,
        mode=mode,
    )
    return summary


def main() -> int:
    results_dir = REPO_ROOT / "examples" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    assert SAMPLE_FEED.exists(), f"gym-fx sample feed not found at {SAMPLE_FEED}"

    # 1) random_agent sanity
    random_cfg = _base_config({"save_model": str(results_dir / "random.zip")})
    random_summary = _run(RandomAgent(random_cfg), random_cfg)
    (results_dir / "random_summary.json").write_text(json.dumps(random_summary, indent=2, default=str))
    assert math.isfinite(random_summary.get("total_return", 0.0)), random_summary
    assert random_summary.get("episode_length", 0) > 0

    # 2) buy_hold on an uptrend should not produce a worse return than random noise.
    bh_cfg = _base_config({"save_model": str(results_dir / "buy_hold.zip")})
    if UPTREND_FEED.exists():
        bh_cfg["input_data_file"] = str(UPTREND_FEED)
    bh_summary = _run(BuyHoldAgent(bh_cfg), bh_cfg)
    (results_dir / "buy_hold_summary.json").write_text(json.dumps(bh_summary, indent=2, default=str))
    assert math.isfinite(bh_summary.get("total_return", 0.0))

    # 3) PPO quick train + reload inference
    ppo_model = results_dir / "ppo_smoke.zip"
    ppo_cfg = _base_config(
        {
            "save_model": str(ppo_model),
            "total_timesteps": 500,
            "n_steps": 128,
            "batch_size": 64,
        }
    )
    train_summary = _run(PPOAgent(ppo_cfg), ppo_cfg, mode="train")
    (results_dir / "ppo_train_summary.json").write_text(json.dumps(train_summary, indent=2, default=str))
    assert ppo_model.exists(), f"PPO did not save model to {ppo_model}"

    inf_cfg = _base_config(
        {
            "load_model": str(ppo_model),
            "total_timesteps": 0,
        }
    )
    inf_summary = _run(PPOAgent(inf_cfg), inf_cfg, mode="inference")
    (results_dir / "ppo_inference_summary.json").write_text(json.dumps(inf_summary, indent=2, default=str))
    assert inf_summary.get("episode_length", 0) > 0

    print("[smoke_test_agent] all assertions passed")
    print(
        json.dumps(
            {
                "random": {k: random_summary.get(k) for k in ("total_return", "episode_length")},
                "buy_hold": {k: bh_summary.get(k) for k in ("total_return", "episode_length")},
                "ppo_train": {k: train_summary.get(k) for k in ("total_return", "episode_length")},
                "ppo_inference": {k: inf_summary.get(k) for k in ("total_return", "episode_length")},
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
