#!/usr/bin/env python3
"""C4 bounded conflict calibration (DATA-SOTA-368). CPU only.

Executes the PREDECLARED design
(`CONFLICT_CALIBRATION_PREDECLARATION_2026_08_27.json`, committed
BEFORE this run): 5 solo runs + 1 joint run at the identical budget on
the identical frozen train-tail probe, then reports per epoch and per
family: cosine sign frequency / median / lower quartile / persistence,
weighted gradient norms, solo-vs-joint PROBE loss trajectories,
representation variance and effective negatives — and applies the
FIXED disposition rule. The monitor is never consulted; nothing is
auto-removed (removal authority stays with the auditor).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
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
               max_windows: int) -> dict:
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools/pretrain_branches.py"),
         "--contract", str(contract_file), "--output-dir", str(out_dir),
         "--epochs", str(epochs), "--max-windows", str(max_windows)],
        capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0:
        raise SystemExit(f"runner failed for {contract_file.name}:\n"
                         f"{proc.stderr[-1500:]}")
    return json.loads((out_dir / "pretrain_manifest.json").read_text())


def pair_statistics(values: list[float]) -> dict:
    negatives = [v < 0 for v in values]
    longest = run = 0
    for negative in negatives:
        run = run + 1 if negative else 0
        longest = max(longest, run)
    ordered = sorted(values)
    return {"sign_negative_frequency": round(
                sum(negatives) / len(values), 3),
            "median": round(statistics.median(values), 4),
            "p25": round(ordered[max(0, len(ordered) // 4)], 4),
            "max_consecutive_negative": longest,
            "epochs": len(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predeclaration = json.loads(
        (REPO / "docs/audits/evidence/"
         "CONFLICT_CALIBRATION_PREDECLARATION_2026_08_27.json"
         ).read_text())
    epochs = predeclaration["budget"]["epochs"]
    max_windows = predeclaration["budget"]["max_windows"]
    contract = json.loads(Path(args.contract).read_text())
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    solo_manifests = {}
    for objective in OBJECTIVES:
        if objective not in contract["objectives"]:
            continue
        solo = json.loads(json.dumps(contract))
        solo["objectives"] = {objective:
                              contract["objectives"][objective]}
        solo_file = workdir / f"solo_{objective}.json"
        solo_file.write_text(json.dumps(solo, indent=1))
        solo_manifests[SHORT[objective]] = run_runner(
            solo_file, workdir / f"solo_{objective}", epochs,
            max_windows)
    joint = run_runner(Path(args.contract), workdir / "joint", epochs,
                       max_windows)

    report = {"schema": "agent_multi.conflict_calibration.v1",
              "order": "C4 (DATA-SOTA-368)",
              "predeclaration_sha256": sha256_file(
                  REPO / "docs/audits/evidence/"
                  "CONFLICT_CALIBRATION_PREDECLARATION_2026_08_27.json"),
              "contract_sha256": sha256_file(Path(args.contract)),
              "budget": predeclaration["budget"],
              "families": {}, "dispositions": {}}
    rule = predeclaration["disposition_rule_fixed_before_the_run"]
    for family, progress in joint["progress"].items():
        records = progress["losses"]
        weights = progress["effective_weights"]["effective"]
        pair_series: dict[str, list] = {}
        weighted_norms_by_epoch = []
        probe_losses_by_epoch = []
        representation = []
        negatives = []
        for record in records:
            probe = record["mechanics_probe"]
            for key, value in probe["gradient_diagnostics"].items():
                if key.startswith("cosine:"):
                    pair_series.setdefault(
                        key.replace("cosine:", ""), []).append(value)
            weighted_norms_by_epoch.append(
                {objective: round(weights[objective] * norm, 6)
                 for objective, norm in probe["gradient_diagnostics"][
                     "norms"].items()})
            probe_losses_by_epoch.append(probe.get("probe_losses", {}))
            representation.append(probe["representation_std"])
            contrastive = probe.get("contrastive_diagnostics")
            if contrastive:
                negatives.append(
                    contrastive["effective_negatives_mean"])
        solo_vs_joint = {}
        for objective, manifest in solo_manifests.items():
            solo_records = manifest["progress"][family]["losses"]
            solo_final = solo_records[-1]["mechanics_probe"].get(
                "probe_losses", {}).get(objective)
            joint_final = probe_losses_by_epoch[-1].get(objective)
            solo_first = solo_records[0]["mechanics_probe"].get(
                "probe_losses", {}).get(objective)
            joint_first = probe_losses_by_epoch[0].get(objective)
            degradation = (round(joint_final / solo_final, 4)
                           if solo_final else None)
            solo_vs_joint[objective] = {
                "solo_probe_first_to_final": [solo_first, solo_final],
                "joint_probe_first_to_final": [joint_first,
                                               joint_final],
                "joint_over_solo_final_ratio": degradation}
        pair_stats = {pair: pair_statistics(values)
                      for pair, values in pair_series.items()}
        # FIXED disposition per pair (predeclared; names outcomes only)
        dispositions = {}
        for pair, stats in pair_stats.items():
            a, b = pair.split("|")
            degraded = []
            for objective in (a, b):
                ratio = solo_vs_joint.get(objective, {}).get(
                    "joint_over_solo_final_ratio")
                if ratio is not None and ratio > 1.2:
                    degraded.append(objective)
            if degraded and (stats["median"] < -0.5
                             and stats["sign_negative_frequency"]
                             >= 0.75):
                dispositions[pair] = ("MATERIAL_CONFLICT_DEGRADATION: "
                                      + ",".join(degraded))
            elif degraded:
                dominance_check = []
                for epoch_norms in weighted_norms_by_epoch:
                    for objective in degraded:
                        partner = b if objective == a else a
                        if epoch_norms.get(partner, 0) > 0 and \
                                epoch_norms.get(objective, 0) \
                                < epoch_norms[partner] / 5:
                            dominance_check.append(True)
                        else:
                            dominance_check.append(False)
                if dominance_check and (sum(dominance_check)
                                        >= len(dominance_check) / 2):
                    dispositions[pair] = (
                        "MATERIAL_CONFLICT_DEGRADATION: "
                        + ",".join(degraded) + " (dominated)")
                else:
                    dispositions[pair] = ("PERSISTENT_TENSION_NO_HARM"
                                          if stats[
                                              "sign_negative_frequency"]
                                          >= 0.6 and stats["median"]
                                          < -0.3 else
                                          "TRANSIENT_DISAGREEMENT")
            elif stats["sign_negative_frequency"] >= 0.6 and \
                    stats["median"] < -0.3:
                dispositions[pair] = "PERSISTENT_TENSION_NO_HARM"
            else:
                dispositions[pair] = "TRANSIENT_DISAGREEMENT"
        # DOMINANCE across every other objective
        dominance_epochs = 0
        for epoch_norms in weighted_norms_by_epoch:
            for objective, norm in epoch_norms.items():
                others = [v for k, v in epoch_norms.items()
                          if k != objective]
                if others and all(norm > 5 * v for v in others):
                    dominance_epochs += 1
                    break
        if dominance_epochs >= 6:
            dispositions["GLOBAL"] = (
                f"DOMINANCE in {dominance_epochs}/{len(records)} epochs")
        report["families"][family] = {
            "pair_statistics": pair_stats,
            "weighted_gradient_norms_by_epoch": weighted_norms_by_epoch,
            "probe_losses_by_epoch": probe_losses_by_epoch,
            "solo_vs_joint": solo_vs_joint,
            "representation_std_by_epoch": representation,
            "effective_negatives_by_epoch": negatives}
        report["dispositions"][family] = dispositions
    report["rule_applied"] = rule
    Path(args.output).write_text(json.dumps(report, indent=1))
    summary = {family: {pair: d for pair, d in dispositions.items()
                        if "TRANSIENT" not in d}
               for family, dispositions in report["dispositions"].items()}
    print(json.dumps({"non_transient_dispositions": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
