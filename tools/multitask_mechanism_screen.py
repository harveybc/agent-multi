#!/usr/bin/env python3
"""M2 predeclared CPU 2x2 mechanism screen (multitask gradient order
2026-08-27). Executes EXACTLY the committed predeclaration
(`MULTITASK_2X2_PREDECLARATION_2026_08_27.json`): solo references once
under control settings, then the four balancing x combiner cells at the
identical budget/seed/probe, and the LEXICOGRAPHIC winner rule with the
declared tie tolerance. Monitor and economic returns are never
consulted; nothing is tuned post hoc.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import sha256_file  # noqa: E402

OBJECTIVES = ("masked_patch_reconstruction", "multi_horizon_quantile",
              "hierarchical_contrastive", "volatility", "barrier_hit")
SHORT = {"masked_patch_reconstruction": "reconstruction",
         "multi_horizon_quantile": "quantile",
         "hierarchical_contrastive": "contrastive",
         "volatility": "volatility", "barrier_hit": "barrier"}


def run_runner(contract_file: Path, out_dir: Path, epochs: int,
               max_windows: int) -> tuple[dict, float]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools/pretrain_branches.py"),
         "--contract", str(contract_file), "--output-dir", str(out_dir),
         "--epochs", str(epochs), "--max-windows", str(max_windows)],
        capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0:
        raise SystemExit(f"runner failed for {contract_file.name}:\n"
                         f"{proc.stderr[-1500:]}")
    manifest = json.loads((out_dir / "pretrain_manifest.json"
                           ).read_text())
    return manifest, round(time.perf_counter() - started, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predeclaration = json.loads(
        (REPO / "docs/audits/evidence/"
         "MULTITASK_2X2_PREDECLARATION_2026_08_27.json").read_text())
    epochs = predeclaration["shared"]["budget"]["epochs"]
    max_windows = predeclaration["shared"]["budget"]["max_windows"]
    cells = predeclaration["cells"]
    contract = json.loads(Path(args.contract).read_text())
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # solo references ONCE, control settings
    solo_final: dict[str, dict[str, float]] = {}
    for objective in OBJECTIVES:
        if objective not in contract["objectives"]:
            continue
        solo = json.loads(json.dumps(contract))
        solo["objectives"] = {objective:
                              contract["objectives"][objective]}
        solo["objective_balancing"] = {"method": "inverse_initial_loss",
                                       "floor": 1e-6}
        solo["gradient_combiner"] = {"plugin": "ordinary_sum"}
        solo_file = workdir / f"solo_{objective}.json"
        solo_file.write_text(json.dumps(solo, indent=1))
        manifest, _ = run_runner(solo_file,
                                 workdir / f"solo_{objective}",
                                 epochs, max_windows)
        name = SHORT[objective]
        solo_final[name] = {
            family: progress["losses"][-1]["mechanics_probe"][
                "probe_losses"][name]
            for family, progress in manifest["progress"].items()}

    report = {"schema": "agent_multi.multitask_2x2_screen.v1",
              "order": "M2", "predeclaration_sha256": sha256_file(
                  REPO / "docs/audits/evidence/"
                  "MULTITASK_2X2_PREDECLARATION_2026_08_27.json"),
              "contract_sha256": sha256_file(Path(args.contract)),
              "solo_references": solo_final, "cells": {}}
    summary = {}
    for cell in cells:
        variant = json.loads(json.dumps(contract))
        variant["objective_balancing"] = {
            "method": cell["balancing"],
            "floor": 1e-6 if cell["balancing"] == "inverse_initial_loss"
            else 1e-8}
        variant["gradient_combiner"] = {"plugin": cell["combiner"]}
        cell_file = workdir / f"cell_{cell['id']}.json"
        cell_file.write_text(json.dumps(variant, indent=1))
        manifest, runtime = run_runner(cell_file,
                                       workdir / f"cell_{cell['id']}",
                                       epochs, max_windows)
        ratios = {}
        degraded_pairs = []
        weighted_shares = {}
        combiner_facts = {}
        representation = {}
        negatives = {}
        for family, progress in manifest["progress"].items():
            records = progress["losses"]
            final_probe = records[-1]["mechanics_probe"]["probe_losses"]
            for objective, solo_by_family in solo_final.items():
                joint = final_probe.get(objective)
                solo = solo_by_family.get(family)
                if joint is None or not solo:
                    continue
                ratio = round(joint / solo, 4)
                ratios[f"{family}:{objective}"] = ratio
                if ratio > 1.2:
                    degraded_pairs.append(f"{family}:{objective}")
            weights = progress["effective_weights"]["effective"]
            weighted_shares[family] = [
                {objective: round(weights[objective] * norm, 6)
                 for objective, norm in record["mechanics_probe"][
                     "gradient_diagnostics"]["norms"].items()}
                for record in records]
            combiner_facts[family] = [record["gradient_combination"]
                                      for record in records]
            representation[family] = [
                record["mechanics_probe"]["representation_std"]
                for record in records]
            negatives[family] = [
                record["mechanics_probe"].get(
                    "contrastive_diagnostics", {}).get(
                    "effective_negatives_mean")
                for record in records]
        ratio_values = sorted(ratios.values())
        summary[cell["id"]] = {
            "degraded_count": len(degraded_pairs),
            "worst_ratio": ratio_values[-1] if ratio_values else None,
            "median_ratio": round(statistics.median(ratio_values), 4)
            if ratio_values else None}
        report["cells"][cell["id"]] = {
            "balancing": cell["balancing"],
            "combiner": cell["combiner"],
            "joint_over_solo_ratios": ratios,
            "materially_degraded_pairs": degraded_pairs,
            "weighted_gradient_shares_by_epoch": weighted_shares,
            "gradient_combination_by_epoch": combiner_facts,
            "representation_std_by_epoch": representation,
            "effective_negatives_by_epoch": negatives,
            "runtime_seconds_descriptive": runtime}

    # predeclared lexicographic winner with declared tolerance
    ordering = sorted(summary.items(),
                      key=lambda kv: (kv[1]["degraded_count"],
                                      kv[1]["worst_ratio"],
                                      kv[1]["median_ratio"]))
    best_id, best = ordering[0]
    ties = [cid for cid, facts in ordering[1:]
            if facts["degraded_count"] == best["degraded_count"]
            and abs(facts["worst_ratio"] - best["worst_ratio"]) <= 0.02
            and abs(facts["median_ratio"] - best["median_ratio"])
            <= 0.02]
    if ties:
        verdict = (f"INCONCLUSIVE: {best_id} ties with "
                   f"{ties} within the declared tolerance")
        winner = None
    elif best["degraded_count"] > 0:
        verdict = (f"NO_CELL_REMOVES_MATERIAL_DEGRADATION: best is "
                   f"{best_id} with {best['degraded_count']} degraded "
                   f"pairs -> M3 routing ablation")
        winner = None
    else:
        verdict = f"WINNER: {best_id} (zero materially degraded pairs)"
        winner = best_id
    report["summary"] = summary
    report["mechanical_selection"] = {"rule": predeclaration[
        "winner_rule_lexicographic"], "verdict": verdict,
        "winner": winner}
    Path(args.output).write_text(json.dumps(report, indent=1))
    print(json.dumps({"summary": summary, "verdict": verdict},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
