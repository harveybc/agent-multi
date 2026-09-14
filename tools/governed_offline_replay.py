#!/usr/bin/env python3
"""Run one small offline unit of agent-multi work from a governed, flat configuration.

P5 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_CAUSAL_PIPELINE_AND_OFFLINE_DOIN_2026_09_14.md`.

The governed wrapper (`tools/governed_run.py`) speaks a flat configuration: data-gov
substitutes the delivered input path into `input_data_file` and rewrites the output keys.
This runner merges that flat configuration over the nested experiment template this
repository actually uses, runs the pipeline **offline** (no DOIN, no venue, no network, CPU
only), and writes the metrics the wrapper reports.

It adds no remote call inside a training step and no new schema: the experiment contract,
the runtime overlay and the observation contract are the repository's own.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def merged_config(flat: dict, template: Path, output_dir: Path) -> dict:
    config = json.loads(template.read_text(encoding="utf-8"))
    data = config.setdefault("data", {})
    data["input_data_file"] = flat["input_data_file"]
    experiment = config.setdefault("experiment", {})
    experiment["name"] = flat.get("experiment_name", experiment.get("name", "governed-offline"))
    # bounded by construction: this is a mechanical replay, not a training campaign
    training = config.setdefault("training", {})
    training["total_timesteps"] = int(flat.get("total_timesteps", training.get("total_timesteps", 64)))
    environment = config.setdefault("environment", {})
    environment["max_rows"] = int(flat.get("max_rows", environment.get("max_rows", 384)))
    return config


def metrics_of(summary: dict, wall_seconds: float) -> dict:
    """Numbers the wrapper reports; absent ones are absent, never invented."""
    out = {"wall_seconds": wall_seconds}
    for key in ("best_fitness", "champion_fitness", "final_reward", "total_timesteps",
                "candidates_evaluated", "generations"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load_config", required=True, help="the flat governed configuration")
    args = parser.parse_args(argv)
    flat = json.loads(Path(args.load_config).read_text(encoding="utf-8"))
    template = Path(flat["experiment_config"])
    overlay = Path(flat["runtime_overlay"])
    output_dir = Path(flat.get("save_log", "./replay_log.json")).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = merged_config(flat, template, output_dir)
    merged_path = output_dir / "experiment_resolved.json"
    merged_path.write_text(json.dumps(merged, indent=1), encoding="utf-8")

    started = time.monotonic()
    command = [sys.executable, str(REPO / "app" / "main.py"),
               "--load_config", str(merged_path),
               "--runtime_overlay", str(overlay), "--mode", "train"]
    log_path = output_dir / "replay_stdout.log"
    with open(log_path, "wb") as handle:
        code = subprocess.run(command, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                       MKL_NUM_THREADS="1")).returncode
    wall = time.monotonic() - started

    summary = {}
    for candidate in (output_dir / "summary.json", REPO / "config_out.json"):
        if candidate.is_file():
            try:
                summary = json.loads(candidate.read_text(encoding="utf-8"))
                break
            except ValueError:
                continue
    body = {"schema": "agent_multi_offline_replay.v1", "exit_code": code,
            "experiment_config_sha256": None, "metrics": metrics_of(summary, wall),
            "offline": True, "network": "none", "device": "cpu",
            "stdout_tail": log_path.read_text(errors="replace")[-1500:] if code else ""}
    Path(flat.get("save_log", output_dir / "replay_log.json")).write_text(
        json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"exit_code": code, "wall_seconds": round(wall, 2),
                      "metrics": body["metrics"]}, indent=1))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
