#!/usr/bin/env python
"""Count parameters of BUILT SAC/DQN models for each materialized cell (no training).

The env is a synthetic CSV with the cell's feature count and window (parameter
counts depend on observation geometry and net_arch only, not on data values).
Writes <matrix_dir>/PARAMETERS_MEASURED.json and stamps each cell's
accounting.measured block.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _count(module):
    return int(sum(p.numel() for p in module.parameters()))


def measure(cell: dict, work: Path) -> dict:
    import numpy as np
    import pandas as pd
    from gymnasium.wrappers import FlattenObservation

    from env_plugins.gym_fx_env import Plugin as EnvPlugin

    feats = list(cell["feature_columns"])
    rows = int(cell["window_size"]) + int(cell["feature_scaling_window"]) + 40
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, rows))
    frame = pd.DataFrame({"DATE_TIME": pd.date_range("2020-01-01", periods=rows, freq="4h").strftime("%Y-%m-%d %H:%M:%S"),
                          "OPEN": close, "HIGH": close, "LOW": close, "CLOSE": close, "VOLUME": 1.0})
    for f in feats:
        frame[f] = rng.normal(size=rows)
    csv = work / f"synthetic_{len(feats)}f.csv"
    frame.to_csv(csv, index=False)
    env_cfg = {k: v for k, v in cell.items() if not isinstance(v, (dict, list)) or k in ("feature_columns", "net_arch")}
    env_cfg.update(input_data_file=str(csv), max_rows=None, env_mode="training", device="cpu")
    env = FlattenObservation(EnvPlugin().make_env(dict(env_cfg)))
    try:
        cfg = dict(env_cfg)
        cfg["representation"] = cell["representation"]
        cfg["learning_starts"] = 1
        cfg["buffer_size"] = 16
        if cell["algorithm"] == "SAC":
            from agent_plugins.sac_agent import Plugin
        else:
            from agent_plugins.dqn_agent import Plugin
        model = Plugin().build(env, cfg)
        out = {"observation_dim": int(np.prod(env.observation_space.shape)), "algorithm": cell["algorithm"],
               "representation": cell["representation"]["kind"], "networks": {}}
        nets = ({"actor": model.actor, "critic": model.critic, "critic_target": model.critic_target}
                if cell["algorithm"] == "SAC" else {"q_net": model.q_net, "q_net_target": model.q_net_target})
        seen = set()
        unique = 0
        for name, net in nets.items():
            ext = net.features_extractor
            out["networks"][name] = {"total": _count(net), "features_extractor": _count(ext),
                                     "head_excluding_extractor": _count(net) - _count(ext),
                                     "features_dim": int(ext.features_dim)}
            for p in net.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    unique += p.numel()
        out["unique_parameters_all_networks"] = int(unique)
        out["policy_total_parameters"] = _count(model.policy)
        if cell["algorithm"] == "SAC":
            out["shared_features_extractor"] = model.critic.features_extractor is model.actor.features_extractor
        return out
    finally:
        env.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args(argv)
    matrix = Path(args.matrix)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in sorted(matrix.glob("RL-*_seed*.json")):
        cell = json.loads(path.read_text())
        key = f"{cell['arm']}"
        if key not in results:  # geometry is identical across seeds
            results[key] = measure(cell, work)
        cell["accounting"]["measured"] = {**results[key], "label": "MEASURED_FROM_BUILT_MODEL_NO_TRAINING",
                                          "formula_values_superseded": True}
        path.write_text(json.dumps(cell, indent=2, sort_keys=True, default=str) + "\n")
    (matrix / "PARAMETERS_MEASURED.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: {"obs": v["observation_dim"], "unique": v["unique_parameters_all_networks"],
                          "nets": {n: d["head_excluding_extractor"] for n, d in v["networks"].items()},
                          "extractor": next(iter(v["networks"].values()))["features_extractor"]}
                      for k, v in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
