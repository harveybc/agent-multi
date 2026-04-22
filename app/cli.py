"""
cli.py — argparse entrypoint for agent-multi.

Keeps a small stable surface (mode, use_optimizer, load/save model,
load/save config, plugin selection, quiet). All other config keys are
passed through as `--key value` and parsed as unknown args.
"""
from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="agent-multi — plugin-based RL trainer/optimizer for gym-fx.",
    )
    parser.add_argument("--mode", type=str, choices=["train", "inference", "optimization"],
                        help="Execution mode.")
    parser.add_argument("--use_optimizer", action="store_true",
                        help="Run hyperparameter optimization before the pipeline.")
    parser.add_argument("--load_model", type=str,
                        help="Path to a pre-trained agent checkpoint (forces inference).")
    parser.add_argument("--save_model", type=str,
                        help="Where to save the trained agent checkpoint.")
    parser.add_argument("--load_config", type=str, help="Path to a JSON config file to load.")
    parser.add_argument("--save_config", type=str, help="Path to write the resolved config.")
    parser.add_argument("--results_file", type=str, help="Path to write the run summary JSON.")

    parser.add_argument("--env_plugin", type=str, help="env.plugins name.")
    parser.add_argument("--agent_plugin", type=str, help="agent.plugins name.")
    parser.add_argument("--pipeline_plugin", type=str, help="pipeline.plugins name.")
    parser.add_argument("--optimizer_plugin", type=str, help="optimizer.plugins name.")

    parser.add_argument("--input_data_file", type=str, help="Path to OHLCV CSV (forwarded to gym-fx).")
    parser.add_argument("--total_timesteps", type=int, help="Training timesteps.")
    parser.add_argument("--quiet_mode", action="store_true", help="Reduce console output.")
    return parser.parse_known_args()
