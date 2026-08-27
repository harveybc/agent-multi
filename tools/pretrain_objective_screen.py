#!/usr/bin/env python3
"""WP2 multi-objective mechanics screen (post-transfer order
2026-08-27). CPU-only, bounded:

1. each objective ALONE (derived single-objective contracts) — every
   applicable family must show finite losses, nonzero encoder
   gradients, non-constant targets and no representation collapse;
2. ALL objectives together under the predeclared balancing — plus a
   gradient norm/cosine conflict table;
3. resume interruption/replay parity — bitwise artifact equality;
4. (--real) one bounded real-data o2022 smoke, mechanics only.

Typed REJECTIONS: constant targets, zero encoder gradient,
representation collapse, non-finite values, purge-boundary violation
(leakage), and materially unresolved gradient conflict (any objective
pair with cosine < -0.8 in EVERY recorded epoch). Weights are NEVER
chosen from monitor performance — balancing stays the predeclared
inverse-initial-loss-on-calibration rule.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    sha256_file, sha256_obj)

SINGLE_OBJECTIVE_KEYS = ("masked_patch_reconstruction",
                         "multi_horizon_quantile",
                         "hierarchical_contrastive", "volatility",
                         "barrier_hit")
COLLAPSE_FLOOR = 1e-4
CONFLICT_COSINE = -0.8


def run_runner(contract_file: Path, out_dir: Path, epochs: int,
               max_windows: int, *extra) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools/pretrain_branches.py"),
         "--contract", str(contract_file), "--output-dir", str(out_dir),
         "--epochs", str(epochs), "--max-windows", str(max_windows),
         *extra],
        capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0:
        raise SystemExit(f"runner failed for {contract_file.name}:\n"
                         f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def target_degeneracy(contract: dict, data_csv: Path) -> dict:
    """Reject constant targets BEFORE training: compute target spread
    per forward-looking objective on the eligible steps."""
    import numpy as np

    from agent_plugins.branch_pretraining import (
        barrier_hit_labels, build_step_index, forward_log_return_targets,
        load_fit_slice, realized_volatility_targets, validate_contract)
    parsed = validate_contract(contract)
    df, _cols, close_col = load_fit_slice(data_csv, contract)
    steps = build_step_index(len(df), parsed["warmup_bars"],
                             parsed["window_stride"],
                             parsed["max_horizon_all_objectives"],
                             contract.get("max_windows"))
    close = df[close_col].to_numpy()
    report = {}
    objectives = contract["objectives"]
    if "multi_horizon_quantile" in objectives:
        t = forward_log_return_targets(
            close, steps, objectives["multi_horizon_quantile"][
                "horizons"])
        report["quantile_target_std"] = round(float(np.std(t)), 8)
    if "volatility" in objectives:
        spec = objectives["volatility"]
        annualization = spec["annualization"]
        t = realized_volatility_targets(
            close, steps, spec["horizons"], float(spec["epsilon"]),
            None if annualization == "none"
            else annualization["periods_per_year"])
        report["volatility_target_std"] = round(float(np.std(t)), 8)
    if "barrier_hit" in objectives:
        spec = objectives["barrier_hit"]
        labels = barrier_hit_labels(
            close, steps, spec["horizons"],
            int(spec["barrier_scale"]["lookback"]),
            float(spec["upper_mult"]), float(spec["lower_mult"]),
            float(spec["barrier_scale"]["epsilon"]))
        distribution = {int(k): int(v) for k, v in
                        zip(*__import__("numpy").unique(
                            labels, return_counts=True))}
        report["barrier_label_distribution"] = distribution
        report["barrier_classes_present"] = len(distribution)
    return report


def evaluate_manifest(manifest: dict) -> dict:
    """Post-run rejection screen over every family."""
    rejections = []
    conflict_table = {}
    for family, progress in manifest["progress"].items():
        for record in progress["losses"]:
            for name, value in record["train"].items():
                if name != "weighted_total" and not (
                        value == value and abs(value) != float("inf")):
                    rejections.append(f"{family}: non-finite {name}")
            monitor = record["monitor_fit_tail"]
            if monitor.get("representation_std", 1.0) < COLLAPSE_FLOOR:
                rejections.append(
                    f"{family}: representation collapse "
                    f"({monitor.get('representation_std')})")
            contrastive = monitor.get("contrastive_diagnostics")
            if contrastive and contrastive.get(
                    "projection_std", 1.0) < COLLAPSE_FLOOR:
                rejections.append(f"{family}: projection collapse")
            norms = record["gradient_diagnostics"]["norms"]
            for objective, norm in norms.items():
                if norm == 0:
                    rejections.append(
                        f"{family}: ZERO encoder gradient from "
                        f"{objective} (epoch {record['epoch']})")
        # gradient conflict: persistent across ALL epochs
        pair_history: dict[str, list] = {}
        for record in progress["losses"]:
            for key, value in record["gradient_diagnostics"].items():
                if key.startswith("cosine:"):
                    pair_history.setdefault(key, []).append(value)
        conflict_table[family] = {
            k: {"min": min(v), "mean": round(sum(v) / len(v), 4)}
            for k, v in pair_history.items()}
        for pair, values in pair_history.items():
            if values and all(v < CONFLICT_COSINE for v in values):
                rejections.append(
                    f"{family}: materially unresolved gradient "
                    f"conflict {pair} (all epochs < {CONFLICT_COSINE})")
    # leakage: purged boundaries must hold in the manifest itself
    parts = manifest["partitions"]
    if not (parts["train"]["target_range"]["last_target_row"]
            < parts["calibration"]["first_step"] - 1
            and parts["calibration"]["target_range"]["last_target_row"]
            < parts["monitor"]["first_step"] - 1):
        rejections.append("purge-boundary violation (target leakage)")
    return {"rejections": rejections, "conflict_table": conflict_table}


def artifacts_bitwise_equal(a: Path, b: Path) -> bool:
    import torch

    for file_a in sorted(a.glob("branch_*.pt")):
        file_b = b / file_a.name
        state_a = torch.load(file_a, weights_only=True)
        state_b = torch.load(file_b, weights_only=True)
        if state_a.keys() != state_b.keys():
            return False
        for key in state_a:
            if not torch.equal(state_a[key], state_b[key]):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True,
                        help="five-objective contract")
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=900)
    parser.add_argument("--real", action="store_true",
                        help="stage 4: bounded real-data smoke label")
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text())
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "agent_multi.pretrain_objective_screen.v1",
              "order": "post-transfer objectives order 2026-08-27 WP2",
              "mode": "real_bounded_o2022" if args.real
              else "synthetic_fixtures",
              "contract": Path(args.contract).name,
              "contract_sha256": sha256_file(Path(args.contract)),
              "bounded": {"epochs": args.epochs,
                          "max_windows": args.max_windows},
              "target_degeneracy": target_degeneracy(
                  contract, Path(args.data_csv)),
              "stages": {}, "rejections": []}
    if report["target_degeneracy"].get("barrier_classes_present",
                                       3) < 2:
        report["rejections"].append("barrier targets degenerate "
                                    "(<2 classes present)")
    for key, value in report["target_degeneracy"].items():
        if key.endswith("_std") and value == 0:
            report["rejections"].append(f"constant targets: {key}")

    # stage 1: each objective alone (reconstruction stays paired with
    # nothing — truly single) on EVERY family via the full extractor run
    for objective in SINGLE_OBJECTIVE_KEYS:
        if objective not in contract["objectives"]:
            continue
        solo = json.loads(json.dumps(contract))
        solo["objectives"] = {objective:
                              contract["objectives"][objective]}
        solo_file = workdir / f"contract_solo_{objective}.json"
        solo_file.write_text(json.dumps(solo, indent=1))
        out = workdir / f"solo_{objective}"
        run_runner(solo_file, out, args.epochs, args.max_windows)
        manifest = json.loads((out / "pretrain_manifest.json"
                               ).read_text())
        evaluation = evaluate_manifest(manifest)
        report["stages"][f"solo_{objective}"] = {
            "families": {f: p["losses"][-1]["train"]
                         for f, p in manifest["progress"].items()},
            "rejections": evaluation["rejections"]}
        report["rejections"] += [f"solo_{objective}: {r}"
                                 for r in evaluation["rejections"]]

    # stage 2: all objectives together, predeclared balancing
    joint_out = workdir / "joint"
    run_runner(Path(args.contract), joint_out, args.epochs,
               args.max_windows)
    joint_manifest = json.loads(
        (joint_out / "pretrain_manifest.json").read_text())
    joint_eval = evaluate_manifest(joint_manifest)
    report["stages"]["joint"] = {
        "effective_weights": {
            f: p["effective_weights"]["effective"]
            for f, p in joint_manifest["progress"].items()},
        "gradient_conflict_table": joint_eval["conflict_table"],
        "monitor_final": {f: p["losses"][-1]["monitor_fit_tail"]
                          for f, p in joint_manifest[
                              "progress"].items()},
        "rejections": joint_eval["rejections"]}
    report["rejections"] += [f"joint: {r}"
                             for r in joint_eval["rejections"]]

    # stage 3: resume interruption/replay parity
    split_out = workdir / "joint_split"
    run_runner(Path(args.contract), split_out, args.epochs,
               args.max_windows, "--stop-after-epochs", "3")
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools/pretrain_branches.py"),
         "--contract", str(Path(args.contract)),
         "--output-dir", str(split_out), "--resume"],
        capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0:
        raise SystemExit(f"resume failed:\n{proc.stderr[-1500:]}")
    parity = artifacts_bitwise_equal(joint_out, split_out)
    report["stages"]["resume_parity"] = {
        "bitwise_artifact_equality": parity}
    if not parity:
        report["rejections"].append("resume replay parity FAILED")

    report["verdict"] = ("REJECTED: " + "; ".join(report["rejections"])
                         if report["rejections"] else
                         "MECHANICS_PASS (no economic claim)")
    Path(args.output).write_text(json.dumps(report, indent=1))
    print(json.dumps({"verdict": report["verdict"],
                      "target_degeneracy": report["target_degeneracy"]},
                     indent=1))
    return 0 if not report["rejections"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
