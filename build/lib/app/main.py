#!/usr/bin/env python3
"""
main.py — entrypoint for agent-multi.

Responsibilities:
  1. Merge config from CLI, --load_config JSON and optional remote source.
  2. Load env / agent / pipeline / optimizer plugins by name.
  3. Dispatch to train / inference / optimization mode.
  4. Persist results + resolved config.

Mode dispatch mirrors the predictor pattern:
  - use_optimizer=True and load_model is None → run optimizer.optimize()
    then re-run the pipeline with the best hyperparameters.
  - load_model set → inference.
  - otherwise → train (or whatever the pipeline decides from `mode`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.config_handler import load_config, remote_load_config, save_config
from app.config_merger import merge_config, process_unknown_args
from app.plugin_loader import load_plugin


def _load_plugin_instance(group: str, name: str, config: Dict[str, Any]):
    klass, _ = load_plugin(group, name)
    instance = klass(config)
    instance.set_params(**config)
    return instance


def _resolve_mode(config: Dict[str, Any]) -> str:
    if config.get("load_model"):
        return "inference"
    return str(config.get("mode", "train")).lower()


def _run(config: Dict[str, Any]) -> Dict[str, Any]:
    env_plugin = _load_plugin_instance("env.plugins", config["env_plugin"], config)
    agent_plugin = _load_plugin_instance("agent.plugins", config["agent_plugin"], config)
    pipeline_plugin = _load_plugin_instance("pipeline.plugins", config["pipeline_plugin"], config)

    plugin_defaults: Dict[str, Any] = {}
    for inst in (env_plugin, agent_plugin, pipeline_plugin):
        plugin_defaults.update(getattr(inst, "plugin_params", {}))
    for k, v in plugin_defaults.items():
        config.setdefault(k, v)

    if config.get("use_optimizer", False) and not config.get("load_model"):
        if not config.get("quiet_mode"):
            print("Running hyperparameter optimization...")
        optimizer_plugin = _load_plugin_instance(
            "optimizer.plugins", config["optimizer_plugin"], config
        )
        optimal = optimizer_plugin.optimize(
            env_plugin=env_plugin,
            agent_plugin=agent_plugin,
            pipeline_plugin=pipeline_plugin,
            config=config,
        )
        out = Path(config.get("optimizer_output_file", "optimizer_output.json"))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(optimal, fh, indent=2, default=str)
        config.update({k: v for k, v in optimal.items() if not k.startswith("_")})
        agent_plugin.set_params(**config)
        if not config.get("quiet_mode"):
            print(f"Optimizer wrote best params to {out}")

    mode = _resolve_mode(config)
    if not config.get("quiet_mode"):
        print(f"Running pipeline in mode={mode}")
    summary = pipeline_plugin.run_pipeline(
        config=config,
        env_plugin=env_plugin,
        agent_plugin=agent_plugin,
        mode=mode,
    )
    return summary


def main():
    args, unknown_args = parse_args()
    cli_args = {k: v for k, v in vars(args).items() if v is not None and v is not False}
    unknown_args_dict = process_unknown_args(unknown_args)

    file_config: Dict[str, Any] = {}
    if getattr(args, "load_config", None):
        file_config = load_config(args.load_config)

    config = merge_config(DEFAULT_VALUES.copy(), {}, {}, file_config, cli_args, unknown_args_dict)

    summary = _run(config)

    results_file = Path(config.get("results_file", "results.json"))
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    if config.get("save_config"):
        save_config(config, config["save_config"])

    if not config.get("quiet_mode", False):
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
