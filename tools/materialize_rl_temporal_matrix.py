#!/usr/bin/env python
"""Materialize the four-arm x paired-seed RL config matrix from a task declaration.

    python tools/materialize_rl_temporal_matrix.py --task examples/config/rl_temporal/eth_4h_task.json \
        --out examples/config/rl_temporal/eth_4h_draft

The task declaration names the selected-feature manifest (path + expected sha256),
the selected variant, window/sample_hours, the data file and the seeds. Configs are
bound to the manifest status: a DRAFT_NOT_FROZEN manifest yields configs whose
pilot_gate.real_data_fit_allowed is false; nothing here fits anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_temporal.arms import build_matrix, check_pairing  # noqa: E402
from rl_temporal.lake_binding import SelectedFeatureBinding  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    task = json.loads(Path(args.task).read_text())
    manifest = task["manifest"]
    binding = SelectedFeatureBinding.from_manifest(manifest["path"], selected_variant=task["selected_variant"],
                                                   expected_sha256=manifest.get("sha256"))
    resources = binding.verify_resources()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = build_matrix(binding, out_root=task.get("out_root", str(out / "runs")), seeds=task["seeds"],
                         window=task["window"], sample_hours=task["sample_hours"],
                         input_data_file=task.get("input_data_file"),
                         observation_extras_dim=task.get("observation_extras_dim", 4))
    index = []
    for cell in cells:
        name = f"{cell['arm']}_seed{cell['train_seed']}.json"
        (out / name).write_text(json.dumps(cell, indent=2, sort_keys=True, default=str) + "\n")
        index.append({"file": name, "arm": cell["arm"], "seed": cell["train_seed"],
                      "representation_status": cell["representation"]["status"],
                      "real_data_fit_allowed": cell["pilot_gate"]["real_data_fit_allowed"],
                      "config_sha256": cell["config_sha256"]})
    summary = {"task": task, "manifest_sha256": binding.manifest_sha256, "manifest_status": binding.status,
               "selected_variant": binding.selected_variant, "feature_count": len(binding.feature_order),
               "resources": resources, "pairing": check_pairing(cells), "cells": index}
    (out / "MATRIX.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"cells": len(index), "manifest_status": binding.status,
                      "all_paired": all(v["identical_outside_representation"] for v in summary["pairing"].values()),
                      "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
