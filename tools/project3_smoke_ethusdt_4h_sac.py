#!/usr/bin/env python3
"""
project3_smoke_ethusdt_4h_sac.py — short smoke runner for the
Project 3 ETHUSDT 4h SAC actor-critic plugin.

Loads the canonical config, applies short-run overrides
(total_timesteps=100, isolated output paths) and executes the
agent-multi pipeline in-process. Prints the resulting summary.

Usage:
    python tools/project3_smoke_ethusdt_4h_sac.py
    python tools/project3_smoke_ethusdt_4h_sac.py --feature_aware
    python tools/project3_smoke_ethusdt_4h_sac.py --total_timesteps 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config import DEFAULT_VALUES  # noqa: E402
from app.config_handler import load_config, save_config  # noqa: E402
from app.config_merger import merge_config  # noqa: E402
from app.main import _run  # noqa: E402


REPRODUCTION_CONFIG = (
    REPO_ROOT / "examples/config/project3_ethusdt_4h_sac_actor_critic.json"
)
FEATURE_AWARE_CONFIG = (
    REPO_ROOT
    / "examples/config/project3_ethusdt_4h_sac_actor_critic_feature_aware.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature_aware",
        action="store_true",
        help="Use the feature-aware preprocessor variant.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100,
        help="Smoke run timestep budget (default: 100).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. 'cpu' to avoid GPU during smoke).",
    )
    args = parser.parse_args()

    cfg_path = FEATURE_AWARE_CONFIG if args.feature_aware else REPRODUCTION_CONFIG
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2

    file_config = load_config(str(cfg_path))
    overrides = {"total_timesteps": int(args.total_timesteps)}
    if args.device:
        overrides["device"] = args.device

    suffix = "feature_aware_smoke" if args.feature_aware else "smoke"
    out_dir = REPO_ROOT / f"examples/results/project3_ethusdt_4h_sac_actor_critic_{suffix}"
    overrides.update(
        save_model=str(out_dir / "policy.zip"),
        results_file=str(out_dir / "summary.json"),
        save_config=str(out_dir / "config_out.json"),
        quiet_mode=True,
    )

    config = merge_config(DEFAULT_VALUES.copy(), {}, {}, file_config, overrides, {})
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _run(config)

    results_file = Path(config["results_file"])
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    if config.get("save_config"):
        save_config(config, config["save_config"])

    required = (
        "total_return",
        "final_equity",
        "max_drawdown_pct",
        "sharpe_ratio",
        "trades_total",
        "episode_reward",
        "episode_length",
    )
    missing = [k for k in required if k not in summary]
    print(json.dumps(summary, indent=2, default=str))
    if missing:
        print(f"WARNING: summary missing keys: {missing}", file=sys.stderr)
        return 1
    print(f"\nSmoke OK — summary at {results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
