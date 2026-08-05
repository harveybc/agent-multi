#!/usr/bin/env python3
"""Fixed-genome, paired-seed N / E / EN acceptance fixture (order WP-E).

One frozen candidate contract (the proven ETH SAC v2 run config), one
seed, one small declared budget, three arms:

1. ``normal``: normal-only training;
2. ``easy``: easy-only training, then evaluated under NORMAL conditions;
3. ``easy_normal``: easy -> normal warm continuation (WP-D pipeline).

Raw same-scale metrics per split are written to one JSON report. This
fixture validates the MECHANISM; it does not replace the full DOIN
comparison and its numbers must never be reported as champion quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import entry_points
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins.rl_pipeline_with_validation import (  # noqa: E402
    PipelinePlugin as ValidationPipeline,
)
from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (  # noqa: E402
    PipelinePlugin as CurriculumPipeline,
)

ETH_BASE = (REPO / "examples/results/"
            "project3_ethusdt_4h_sac_train_val_test_v2/config_out.json")
DATA_FILE = ("/home/harveybc/Documents/GitHub/predictor/examples/data/"
             "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
SPLITS = {
    "train_start": "2017-09-28T04:00:00",
    "train_end": "2023-12-31T23:59:59",
    "validation_start": "2024-01-01T00:00:00",
    "validation_end": "2024-12-31T23:59:59",
    "test_start": "2025-01-01T00:00:00",
    "test_end": "2025-12-31T23:59:59",
}
RAW_METRICS = (
    "initial_cash", "final_equity", "total_return", "mean_weekly_return",
    "annualized_return", "max_drawdown_fraction", "max_drawdown_pct",
    "trades_total", "trades_won", "trades_lost", "episode_length",
)


def _agent_plugin(name: str):
    eps = entry_points().select(group="agent.plugins")
    ep = next((e for e in eps if e.name == name), None)
    if ep is None:
        raise SystemExit(f"agent plugin {name!r} not found")
    return ep.load()()


def _base_config(out_dir: Path, arm: str, *, epoch_timesteps: int,
                 max_epochs: int, seed: int) -> dict:
    config = json.loads(ETH_BASE.read_text())
    config.update(SPLITS)
    config["input_data_file"] = DATA_FILE
    config["env_mode"] = "training"
    config["eval_seed"] = seed
    config["train_seed"] = seed
    config["epoch_timesteps"] = epoch_timesteps
    config["max_epochs"] = max_epochs
    config["l1_patience"] = max_epochs          # budget-bound, no early cut
    config["execution_cost_curriculum_epochs"] = max(2, max_epochs)
    config["l1_min_checkpoint_timesteps"] = 1
    config["easy_max_epochs"] = max_epochs
    config["easy_patience"] = max_epochs
    config["evaluate_test_split"] = True
    config["selection_metric"] = "lexicographic_weekly_v1"
    config["selection_min_trades"] = 0          # mechanism fixture only
    config["save_model"] = str(out_dir / arm / "model.zip")
    config["quiet_mode"] = True
    return config


def _raw(summary: dict) -> dict:
    return {key: summary.get(key) for key in RAW_METRICS
            if summary.get(key) is not None}


def _splits_raw(result: dict) -> dict:
    splits = result.get("splits") or {}
    return {name: _raw(summary) for name, summary in splits.items()
            if isinstance(summary, dict)}


def run_arm(arm: str, out_dir: Path, *, epoch_timesteps: int,
            max_epochs: int, seed: int, agent_name: str) -> dict:
    config = _base_config(out_dir, arm, epoch_timesteps=epoch_timesteps,
                          max_epochs=max_epochs, seed=seed)
    agent = _agent_plugin(agent_name)
    if arm == "normal":
        config["solvency_mode"] = "normal_realistic"
        pipeline = ValidationPipeline(config)
        result = pipeline.run_pipeline(
            config=config, env_plugin=None, agent_plugin=agent,
            mode="train")
    elif arm == "easy":
        # Easy-ONLY training, evaluated under normal conditions.
        pipeline = CurriculumPipeline(config)
        post_easy = pipeline._train_easy_phase(
            config=config, env_plugin=None, agent_plugin=agent)
        eval_config = dict(config)
        eval_config["solvency_mode"] = "normal_realistic"
        eval_config["load_model"] = post_easy["path"]
        result = pipeline.run_pipeline(
            config=eval_config, env_plugin=None, agent_plugin=agent,
            mode="inference")
        result["post_easy"] = post_easy["meta"]
    elif arm == "easy_normal":
        pipeline = CurriculumPipeline(config)
        result = pipeline.run_pipeline(
            config=config, env_plugin=None, agent_plugin=agent,
            mode="train")
    else:
        raise SystemExit(f"unknown arm {arm!r}")
    return {
        "arm": arm,
        "splits_raw": _splits_raw(result),
        "selection_contract": (
            (result.get("splits") or {}).get("validation", {})
            .get("selection_contract")),
        "curriculum": result.get("curriculum"),
        "best_model_path": result.get("best_model_path"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch-timesteps", type=int, default=4000)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2703)
    parser.add_argument("--agent", default="project3_sac_actor_critic_agent")
    parser.add_argument("--arms", default="normal,easy,easy_normal")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "agent_multi.eth_curriculum_fixture.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixed_genome_source": str(ETH_BASE),
        "data_file": DATA_FILE,
        "data_sha256": hashlib.sha256(
            Path(DATA_FILE).read_bytes()).hexdigest(),
        "seed": args.seed,
        "budget": {"epoch_timesteps": args.epoch_timesteps,
                   "max_epochs": args.max_epochs},
        "note": ("mechanism fixture only — never a champion claim;"
                 " raw same-scale metrics per split"),
        "arms": {},
    }
    for arm in args.arms.split(","):
        arm = arm.strip()
        print(f"[fixture] running arm {arm}", flush=True)
        report["arms"][arm] = run_arm(
            arm, args.output_dir, epoch_timesteps=args.epoch_timesteps,
            max_epochs=args.max_epochs, seed=args.seed,
            agent_name=args.agent)
        report_path = args.output_dir / "fixture_report.json"
        report_path.write_text(
            json.dumps(report, indent=1, sort_keys=True, default=str),
            encoding="utf-8")
        print(f"[fixture] arm {arm} done -> {report_path}", flush=True)
    print(json.dumps({"report": str(args.output_dir
                                    / "fixture_report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
