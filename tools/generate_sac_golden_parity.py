#!/usr/bin/env python3
"""Generate the SAC live-adapter golden parity fixture (ETH order WP-F).

Training-side half of the parity contract: build the gym-fx environment
from the frozen ETH SAC v2 run config, roll the frozen policy.zip
deterministically for K steps, and record every (observation, raw
action, mapped action) triple plus the identity hashes. The lts-side
test replays the SAME observations through the live adapter and must
reproduce the SAME raw actions bit-exactly and the SAME long/short/hold
mapping. Any drift in SB3 loading, device placement or threshold
semantics fails the gate BEFORE any venue authority switch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "examples/results/project3_ethusdt_4h_sac_train_val_test_v2"
DATA_FILE = ("/home/harveybc/Documents/GitHub/predictor/examples/data/"
             "project3/ethusdt_4h_tech_stat_full_model_ready.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path,
                        default=RUN_DIR / "policy.zip")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2703)
    args = parser.parse_args()

    from stable_baselines3 import SAC
    from pipeline_plugins.rl_pipeline_with_validation import _load_env_plugin

    config = json.loads((RUN_DIR / "config_out.json").read_text())
    config["input_data_file"] = DATA_FILE
    config["env_mode"] = "training"
    config["solvency_mode"] = "normal_realistic"
    config["quiet_mode"] = True

    env_plugin = _load_env_plugin(
        str(config.get("env_plugin", "gym_fx_env")), config)
    env = env_plugin.make_env(config)
    from importlib.metadata import entry_points
    agent_name = str(config.get(
        "agent_plugin", "project3_sac_actor_critic_agent"))
    agent_ep = next(e for e in entry_points().select(group="agent.plugins")
                    if e.name == agent_name)
    agent_plugin = agent_ep.load()()
    wrap = getattr(agent_plugin, "wrap_env", None)
    if callable(wrap):
        env = wrap(env, config)
    artifact = args.artifact
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    model = SAC.load(str(artifact), device="cpu")

    threshold = float(config.get("continuous_action_threshold", 0.33))
    def _serialize(observation):
        if isinstance(observation, dict):
            parts = {key: np.asarray(value, dtype=np.float32)
                     for key, value in sorted(observation.items())}
            digest = hashlib.sha256()
            for key, value in parts.items():
                digest.update(key.encode())
                digest.update(value.tobytes())
            return (
                {key: {"shape": list(value.shape),
                       "values": value.reshape(-1).tolist()}
                 for key, value in parts.items()},
                digest.hexdigest(),
            )
        array = np.asarray(observation, dtype=np.float32)
        return (
            {"__array__": {"shape": list(array.shape),
                           "values": array.reshape(-1).tolist()}},
            hashlib.sha256(array.tobytes()).hexdigest(),
        )

    obs, _info = env.reset(seed=args.seed)
    records = []
    for _step in range(args.steps):
        serialized, obs_sha = _serialize(obs)
        raw, _state = model.predict(obs, deterministic=True)
        value = float(np.asarray(raw).reshape(-1)[0])
        mapped = ("long" if value >= threshold
                  else "short" if value <= -threshold else "hold")
        records.append({
            "observation": serialized,
            "observation_sha256": obs_sha,
            "raw_action": value,
            "mapped_action": mapped,
        })
        obs, _reward, terminated, truncated, _info = env.step(raw)
        if terminated or truncated:
            obs, _info = env.reset(seed=args.seed)
    env.close()

    fixture = {
        "schema": "lts.sac_golden_parity.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_file": str(artifact),
        "artifact_sha256": artifact_sha,
        "observation_keys": sorted(records[0]["observation"]),
        "continuous_action_threshold": threshold,
        "seed": args.seed,
        "steps": args.steps,
        "env_identity": {
            "env_plugin": config.get("env_plugin"),
            "preprocessor_plugin": config.get("preprocessor_plugin"),
            "feature_scaling": config.get("feature_scaling"),
            "feature_scaling_window": config.get("feature_scaling_window"),
            "window_size": config.get("window_size"),
            "feature_columns_sha256": hashlib.sha256(json.dumps(
                config.get("feature_columns"), sort_keys=True
            ).encode()).hexdigest(),
            "data_file": DATA_FILE,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
    actions = [r["mapped_action"] for r in records]
    print(json.dumps({
        "fixture": str(args.output),
        "records": len(records),
        "artifact_sha256": artifact_sha,
        "action_counts": {a: actions.count(a) for a in set(actions)},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
