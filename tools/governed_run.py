#!/usr/bin/env python3
"""Governed offline replay of one small agent-multi unit (data-gov Flow v3).

P5 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_CAUSAL_PIPELINE_AND_OFFLINE_DOIN_2026_09_14.md`.

This repository already has its contracts: the experiment schema, the runtime overlay and the
observation contract that refuses a run which does not declare what reaches the policy. None
of them is replaced here. What this adds is the governed envelope the other consumers use —
campaign before data, a delivery with its hash and availability contract, a terminal with
metrics and cost, reconciliation — around a **bounded, offline** unit of real work.

Offline means offline: no DOIN, no venue, no broker, no network call inside a step, CPU only.
The protocol itself lives in data-gov (`tools/governed_exec.py`); the flat configuration it
substitutes into is merged over the nested experiment template by
`tools/governed_offline_replay.py`.

usage:
  governed_run.py --load_config flat.json --experiment-key k --gov-url URL
      --api-key-file FILE --lake synthetic_fixtures --lake-root DIR --out-dir DIR
      [--classification NON_GOVERNING]
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Numbers the replay writes; a run whose log lacks them reports no metric rather than
#: inventing one.
METRIC_KEYS = ["wall_seconds", "best_fitness", "champion_fitness", "final_reward",
               "total_timesteps", "candidates_evaluated", "generations"]


def _governed_exec():
    checkout = Path(os.environ.get("DATA_GOV_CHECKOUT") or REPO_ROOT.parent / "data-gov").expanduser()
    path = checkout / "tools" / "governed_exec.py"
    if not path.is_file():
        raise SystemExit(f"governed_run: data-gov checkout not found at {checkout} "
                         "(set DATA_GOV_CHECKOUT)")
    spec = importlib.util.spec_from_file_location("data_gov_governed_exec", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_gov_governed_exec"] = module
    spec.loader.exec_module(module)
    return module


def _metrics(config: dict) -> dict:
    log = Path(str(config.get("save_log") or "./replay_log.json")).name
    return {"kind": "json_numbers", "path": log, "keys": METRIC_KEYS}


PROFILE = {
    "project": "agent-multi",
    "input_keys": ["input_data_file"],
    "output_keys": ["save_log", "save_config"],
    "command": [sys.executable, "{repo_root}/tools/governed_offline_replay.py",
                "--load_config", "{config}"],
    "cwd": "{out_dir}",
    "metrics": _metrics,
    "artifacts": {"replay_log": "save_log", "effective_config": "save_config"},
    "tags": {"offline": "true", "network": "none", "device": "cpu"},
}


def main(argv=None) -> int:
    return _governed_exec().consumer_main(PROFILE, argv, repo_root=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
