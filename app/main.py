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
from app.canonical_config import load_json_object, resolve_config, write_json_file
from app.config import DEFAULT_VALUES
from app.config_handler import save_config
from app.config_merger import process_unknown_args
from app.plugin_loader import load_plugin
from app.runtime_overlay import resolve_runtime_overlay


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
    cli_args = {
        k: v
        for k, v in vars(args).items()
        if v is not None and v is not False
        and k not in {"load_config", "base_config", "candidate_patch", "runtime_overlay"}
    }
    unknown_args_dict = process_unknown_args(unknown_args)
    cli_args.update(unknown_args_dict)

    source_descriptors: Dict[str, str] = {}
    base_config: Dict[str, Any] | None = None
    if getattr(args, "base_config", None):
        base_config = load_json_object(args.base_config)
        source_descriptors["base_profile"] = str(Path(args.base_config).resolve())

    file_config: Dict[str, Any] | None = None
    if getattr(args, "load_config", None):
        file_config = load_json_object(args.load_config)
        source_descriptors["file_config"] = str(Path(args.load_config).resolve())

    candidate_patch: Dict[str, Any] | None = None
    if getattr(args, "candidate_patch", None):
        candidate_patch = load_json_object(args.candidate_patch)
        source_descriptors["candidate_patch"] = str(Path(args.candidate_patch).resolve())

    runtime_overlay: Dict[str, Any] | None = None
    runtime_overlay_path: Path | None = None
    if getattr(args, "runtime_overlay", None):
        runtime_overlay_path = Path(args.runtime_overlay).resolve()
        runtime_overlay = load_json_object(runtime_overlay_path)
        source_descriptors["runtime_overlay"] = str(runtime_overlay_path)

    resolution = resolve_config(
        DEFAULT_VALUES,
        base_profile=base_config,
        file_config=file_config,
        cli_overrides=cli_args,
        candidate_patch=candidate_patch,
        source_descriptors=source_descriptors,
    )
    expected_repositories = resolution.canonical.code.get("repositories", {})
    runtime_resolution = resolve_runtime_overlay(
        resolution.runtime,
        overlay_payload=runtime_overlay,
        overlay_base_dir=(runtime_overlay_path.parent if runtime_overlay_path else Path.cwd()),
        expected_repositories=(
            expected_repositories if isinstance(expected_repositories, dict) else {}
        ),
    )
    config = runtime_resolution.runtime
    manifest = dict(resolution.manifest)
    manifest["runtime"] = runtime_resolution.manifest

    write_json_file(
        config.get("resolved_config_file", "resolved_config.json"),
        resolution.canonical.model_dump(mode="json", exclude_none=True),
    )
    write_json_file(
        config.get("config_manifest_file", "config_manifest.json"),
        manifest,
    )

    summary = _run(config)
    if isinstance(summary, dict):
        summary.setdefault("canonical_config_hash", resolution.canonical.canonical_hash)
        summary.setdefault("canonical_schema_version", resolution.canonical.schema_version)

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
