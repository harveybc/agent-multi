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
import hashlib
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
    # the application reads the flat key (app/config.py), so the delivered path has to land
    # there too: writing it only under "data" left the run opening the sample file instead
    config["input_data_file"] = flat["input_data_file"]
    # the application writes its summary to `results_file`; pointing that at the run's own
    # directory is what lets the runner read THIS run's observed counters instead of guessing
    config["results_file"] = str(output_dir / "summary.json")
    experiment = config.setdefault("experiment", {})
    experiment["name"] = flat.get("experiment_name", experiment.get("name", "governed-offline"))
    # bounded by construction: this is a mechanical replay, not a training campaign
    training = config.setdefault("training", {})
    budget = int(flat.get("total_timesteps", training.get("total_timesteps", 64)))
    training["total_timesteps"] = budget
    # the agent plugin resolves its parameters from the TOP LEVEL of the configuration
    # (`_resolve` reads config[k] for each of its own params), so a budget written only under
    # "training" never reached it and the agent used its own default. Measured on
    # doin-offline-replay-prod-12: 64 requested, 10,240 observed steps and 400 updates.
    config["total_timesteps"] = budget
    environment = config.setdefault("environment", {})
    environment["max_rows"] = int(flat.get("max_rows", environment.get("max_rows", 384)))
    return config


#: What the run ASKED for. Never reported as work done: R3 of Musashi's order —
#: "a configured step budget is not automatically a measured step count".
REQUESTED_KEYS = {"total_timesteps": "requested_timesteps"}
#: What the runtime SAYS it did. Absent means absent.
OBSERVED_KEYS = ("observed_timesteps", "observed_updates", "best_fitness",
                 "champion_fitness", "final_reward", "candidates_evaluated", "generations")


def digest_of(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None


def metrics_of(summary: dict, wall_seconds: float, requested: dict) -> dict:
    """Numbers the wrapper reports.

    Two rules, both from the audit: the requested budget travels under its own name and is
    never renamed into work performed, and an observation that the runtime did not make is
    simply absent — the budget is not substituted for it.
    """
    out = {"wall_seconds": wall_seconds}
    out.update(requested)
    for key in OBSERVED_KEYS:
        value = summary.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
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
    started_wall = time.time()
    # the entry point is declared, so a custody test can stand a small application in its
    # place without pretending to have trained anything
    entry_point = Path(flat.get("entry_point") or (REPO / "app" / "main.py"))
    command = [sys.executable, str(entry_point),
               "--load_config", str(merged_path),
               "--runtime_overlay", str(overlay), "--mode", "train"]
    log_path = output_dir / "replay_stdout.log"
    with open(log_path, "wb") as handle:
        code = subprocess.run(command, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                       MKL_NUM_THREADS="1")).returncode
    wall = time.monotonic() - started

    # Custody (R3): only THIS run's own summary counts. The repository-global config_out.json
    # was read as a source of metrics, so a file left by any earlier run could be reported as
    # this one's work; and a summary that predates the run is refused for the same reason.
    summary, summary_origin = {}, "absent"
    own_summary = output_dir / "summary.json"
    if own_summary.is_file():
        if own_summary.stat().st_mtime + 1e-6 < started_wall:
            summary_origin = "stale_ignored"
        else:
            try:
                summary = json.loads(own_summary.read_text(encoding="utf-8"))
                summary_origin = "this_run"
            except ValueError:
                summary_origin = "unreadable"
    requested = {name: float(merged.get("training", {}).get(key, 0) or 0)
                 for key, name in REQUESTED_KEYS.items()}
    measured = metrics_of(summary, wall, requested)
    # the collector reads flattened names: keeping the numbers at the top level is what makes
    # the measured cost land in the terminal instead of being reported as "no metric"
    body = {"schema": "agent_multi_offline_replay.v2", "exit_code": code,
            # every input of the run bound to it, so a receipt names what produced it
            "custody": {
                "experiment_template_sha256": digest_of(template),
                "runtime_overlay_sha256": digest_of(overlay),
                "resolved_config_sha256": digest_of(merged_path),
                "runner_source_sha256": digest_of(Path(__file__).resolve()),
                "entry_point_sha256": digest_of(entry_point),
                "input_data_sha256": digest_of(Path(flat["input_data_file"])),
                "summary_origin": summary_origin,
            },
            "experiment_config_sha256": hashlib.sha256(
                merged_path.read_bytes()).hexdigest(), **measured,
            "offline": True, "network": "none", "device": "cpu",
            "stdout_tail": log_path.read_text(errors="replace")[-1500:] if code else ""}
    Path(flat.get("save_log", output_dir / "replay_log.json")).write_text(
        json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"exit_code": code, "wall_seconds": round(wall, 2),
                      "metrics": measured}, indent=1))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
