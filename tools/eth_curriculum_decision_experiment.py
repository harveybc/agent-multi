#!/usr/bin/env python3
"""Paired ETH/SAC curriculum decision experiment (decision order WP-C).

LOCAL-ONLY: writes no DOIN blocks, populations, champion archives or
succession artifacts. One seed per invocation (one per GPU); each host
runs ALL arms for its seed so host/GPU is a blocking factor, never an
arm confound.

Arms (equal-compute primary comparison):
  N14    — 14 normal epochs;
  EN4_10 — 4 easy epochs then 10 normal epochs (fresh replay buffer at
           the dynamics boundary via the WP-D pipeline);
  E4     — 4 easy epochs, inference-only normal evaluation (diagnostic).

Fixed contracts: frozen ETHUSD 4h dataset (sha asserted), 83-feature
causal observation (window 32, rolling 256), one anchor initialization
shared by every arm of the seed, 20,000 timesteps/epoch, NO early
stopping, 2024-validation-only selection metrics, disclosed 2025
disabled and absent from every payload.

Finding 114 evidence per arm: fully resolved config + sha256, anchor
and artifact hashes, return traces (train/train_tail/validation) with
sha256, per-epoch learning/activity trace, lineage, raw same-scale
metrics. Artifacts are replicated to a second host by the wrapper
script; this tool records the primary paths and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from importlib.metadata import entry_points  # noqa: E402

DATA_FILE = ("/home/harveybc/Documents/GitHub/predictor/examples/data/"
             "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
DATA_SHA256 = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435"
               "ebe440f")
ETH_BASE = (REPO / "examples/results/"
            "project3_ethusdt_4h_sac_train_val_test_v2/config_out.json")
SPLITS = {
    "train_start": "2017-09-28T04:00:00",
    "train_end": "2023-12-31T23:59:59",
    "validation_start": "2024-01-01T00:00:00",
    "validation_end": "2024-12-31T23:59:59",
    "test_start": "2025-01-01T00:00:00",
    "test_end": "2025-12-31T23:59:59",
}
EPOCH_TIMESTEPS = 20_000
ARMS = ("N14", "EN4_10", "E4")
ALLOWED_SPLITS = ("train", "train_tail", "validation")
RAW_METRICS = (
    "initial_cash", "final_equity", "total_return", "mean_weekly_return",
    "annualized_return", "max_drawdown_fraction", "max_drawdown_pct",
    "trades_total", "trades_won", "trades_lost", "episode_length",
    "would_margin_call_count", "termination_cause",
    "recapitalization_count", "recapitalization_debt",
    "action_counts", "sharpe_ratio",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _agent_plugin(name: str):
    ep = next(e for e in entry_points().select(group="agent.plugins")
              if e.name == name)
    return ep.load()()


def _git_rev(repo: str) -> str:
    return subprocess.run(
        ["git", "-C", f"/home/harveybc/Documents/GitHub/{repo}",
         "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()


def _base_config(out_dir: Path, arm: str, seed: int, *,
                 epoch_timesteps: int) -> dict:
    config = json.loads(ETH_BASE.read_text())
    config.update(SPLITS)
    config["input_data_file"] = DATA_FILE
    config["env_mode"] = "training"
    config["eval_seed"] = seed
    config["train_seed"] = seed
    config["ga_seed"] = seed
    config["epoch_timesteps"] = epoch_timesteps
    config["l1_min_checkpoint_timesteps"] = 1
    config["evaluate_test_split"] = False        # disclosed 2025 disabled
    config["selection_metric"] = "lexicographic_weekly_v1"
    config["selection_min_trades"] = 0           # mechanism packet
    config["quiet_mode"] = True
    config["save_model"] = str(out_dir / "model.zip")
    config["return_trace_dir"] = str(out_dir / "return_traces")
    return config


def _raw(summary: dict) -> dict:
    return {key: summary.get(key) for key in RAW_METRICS
            if summary.get(key) is not None}


def _splits_raw(result: dict) -> dict:
    splits = result.get("splits") or {}
    for name in splits:
        assert name in ALLOWED_SPLITS or name == "train_tail", (
            f"forbidden split {name!r} in result")
    return {name: _raw(summary) for name, summary in splits.items()
            if isinstance(summary, dict) and name in ALLOWED_SPLITS}


def _hash_traces(trace_dir: Path) -> dict:
    out = {}
    if trace_dir.exists():
        for path in sorted(trace_dir.rglob("*")):
            if path.is_file():
                out[str(path.relative_to(trace_dir))] = _sha(path)
    return out


def _make_anchor(out_dir: Path, seed: int, agent_name: str) -> Path:
    """One anchor initialization shared by every arm of this seed."""
    from stable_baselines3 import SAC
    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)

    anchor = out_dir / f"anchor_seed{seed}.zip"
    if anchor.exists():
        return anchor
    config = _base_config(out_dir, "anchor", seed,
                          epoch_timesteps=EPOCH_TIMESTEPS)
    config["solvency_mode"] = "normal_realistic"
    env_plugin = _load_env_plugin(
        str(config.get("env_plugin", "gym_fx_env")), config)
    env = env_plugin.make_env(config)
    agent = _agent_plugin(agent_name)
    wrap = getattr(agent, "wrap_env", None)
    if callable(wrap):
        env = wrap(env, config)
    model = SAC("MlpPolicy", env, seed=seed, device="cuda",
                policy_kwargs={"net_arch": [256, 256]})
    model.save(str(anchor))
    env.close()
    return anchor


def run_arm(arm: str, seed: int, out_root: Path, *, agent_name: str,
            epoch_timesteps: int, anchor: Path) -> dict:
    from pipeline_plugins.rl_pipeline_with_validation import (
        PipelinePlugin as ValidationPipeline)
    from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
        PipelinePlugin as CurriculumPipeline)

    out_dir = out_root / f"seed{seed}" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _base_config(out_dir, arm, seed,
                          epoch_timesteps=epoch_timesteps)
    config["warm_start_model"] = str(anchor)
    config["warm_start_model_sha256"] = _sha(anchor)
    agent = _agent_plugin(agent_name)
    started = datetime.now(timezone.utc)

    if arm == "N14":
        config["max_epochs"] = 14
        config["l1_patience"] = 10_000       # no early stopping
        config["execution_cost_curriculum_epochs"] = 14
        config["solvency_mode"] = "normal_realistic"
        pipeline = ValidationPipeline(config)
        result = pipeline.run_pipeline(
            config=config, env_plugin=None, agent_plugin=agent,
            mode="train")
    elif arm == "EN4_10":
        config["max_epochs"] = 10
        config["l1_patience"] = 10_000
        config["execution_cost_curriculum_epochs"] = 10
        config["easy_max_epochs"] = 4
        config["easy_patience"] = 10_000     # budget-only easy phase
        pipeline = CurriculumPipeline(config)
        result = pipeline.run_pipeline(
            config=config, env_plugin=None, agent_plugin=agent,
            mode="train")
    elif arm == "E4":
        config["easy_max_epochs"] = 4
        config["easy_patience"] = 10_000
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
    else:
        raise SystemExit(f"unknown arm {arm!r}")

    finished = datetime.now(timezone.utc)
    resolved_path = out_dir / "resolved_config.json"
    resolved_text = json.dumps(config, indent=1, sort_keys=True,
                               default=str)
    resolved_path.write_text(resolved_text, encoding="utf-8")
    model_path = Path(config["save_model"])
    weights = {}
    for label, path in (("final", model_path),
                        ("post_easy", out_dir / "model.post_easy.zip")):
        if path.exists():
            weights[label] = {"path": str(path), "sha256": _sha(path)}
    record = {
        "arm": arm,
        "seed": seed,
        "wall_time_seconds": (finished - started).total_seconds(),
        "resolved_config_sha256": hashlib.sha256(
            resolved_text.encode()).hexdigest(),
        "anchor_sha256": config["warm_start_model_sha256"],
        "artifacts": weights,
        "splits_raw": _splits_raw(result),
        "selection_contract": (result.get("splits") or {}).get(
            "validation", {}).get("selection_contract"),
        "curriculum": result.get("curriculum"),
        "epoch_history": result.get("history") or result.get(
            "epoch_history"),
        "return_trace_sha256": _hash_traces(
            Path(config["return_trace_dir"])),
    }
    (out_dir / "arm_record.json").write_text(
        json.dumps(record, indent=1, sort_keys=True, default=str),
        encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epoch-timesteps", type=int,
                        default=EPOCH_TIMESTEPS)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--agent",
                        default="project3_sac_actor_critic_agent")
    args = parser.parse_args()

    assert hashlib.sha256(
        Path(DATA_FILE).read_bytes()).hexdigest() == DATA_SHA256, (
        "dataset sha mismatch — frozen contract violated")

    out_root = args.output_root
    seed_dir = out_root / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    anchor = _make_anchor(seed_dir, args.seed, args.agent)
    print(json.dumps({"anchor": str(anchor), "sha256": _sha(anchor)}),
          flush=True)

    packet = {
        "schema": "agent_multi.eth_curriculum_decision.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "epoch_timesteps": args.epoch_timesteps,
        "data_sha256": DATA_SHA256,
        "lineage": {repo: _git_rev(repo) for repo in
                    ("agent-multi", "gym-fx", "doin-node",
                     "doin-plugins", "trading-contracts")},
        "note": ("LOCAL-ONLY paired decision packet; order key is"
                 " transport evidence only; 2025 disclosed period"
                 " disabled and absent"),
        "arms": {},
    }
    for arm in args.arms.split(","):
        arm = arm.strip()
        print(f"[decision] seed={args.seed} arm={arm} starting",
              flush=True)
        packet["arms"][arm] = run_arm(
            arm, args.seed, out_root, agent_name=args.agent,
            epoch_timesteps=args.epoch_timesteps, anchor=anchor)
        (seed_dir / "seed_packet.json").write_text(
            json.dumps(packet, indent=1, sort_keys=True, default=str),
            encoding="utf-8")
        print(f"[decision] seed={args.seed} arm={arm} done", flush=True)
    print(json.dumps({"seed_packet": str(seed_dir / "seed_packet.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
