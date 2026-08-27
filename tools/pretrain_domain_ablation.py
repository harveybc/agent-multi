#!/usr/bin/env python3
"""Bounded CPU objective-domain ablation (DATA-SOTA-350, order WP2).

Runs the executing pretraining runner twice from ONE base contract:

* arm ``adapters`` — ``runtime_domain_with_target_adapters`` with the
  declared families' reconstruction TARGETS transformed by
  ``window_zscore_visible`` (the preferred design: the encoder always
  consumes the runtime tensor; adapters are objective-only);
* arm ``single`` — ``single_domain_raw_targets``: every objective uses
  the raw runtime tensor end to end.

It then writes a side-by-side comparison of losses, gradient
norms/cosines and representation scale per branch. DIAGNOSTIC ONLY —
this tool selects no winner, promotes nothing and never touches SAC.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import sha256_file  # noqa: E402


def derive_contract(base: dict, arm: str,
                    zscore_families: list[str]) -> dict:
    contract = json.loads(json.dumps(base))
    if arm == "adapters":
        contract["objective_domain"] = \
            "runtime_domain_with_target_adapters"
        for family in zscore_families:
            if family not in contract["normalization_policies"]:
                raise SystemExit(f"unknown family {family}")
            contract["normalization_policies"][family] = {
                "policy": "window_zscore_visible", "eps": 1e-5}
    elif arm == "single":
        contract["objective_domain"] = "single_domain_raw_targets"
        for family in contract["normalization_policies"]:
            contract["normalization_policies"][family] = {
                "policy": "identity_preprocessed"}
    else:
        raise SystemExit(f"unknown arm {arm}")
    contract["ablation_arm"] = arm
    return contract


def summarize(manifest: dict) -> dict:
    summary = {}
    for branch, progress in manifest["progress"].items():
        last = progress["losses"][-1]
        summary[branch] = {
            "train_final": last["train"],
            "monitor_final": last["monitor_fit_tail"],
            "gradient_final": last["gradient_diagnostics"],
            "effective_weights":
                progress["effective_weights"]["effective"],
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=1200)
    parser.add_argument("--zscore-families",
                        default="volume_flow,returns_momentum")
    args = parser.parse_args()

    base = json.loads(Path(args.contract).read_text())
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    zscore_families = [f.strip() for f in
                      args.zscore_families.split(",") if f.strip()]
    report = {
        "schema": "agent_multi.pretrain_domain_ablation.v1",
        "order": "DATA-SOTA-350 WP2 (order @e9af87a3)",
        "authority": ("DIAGNOSTIC_ONLY — no winner selection, no "
                      "promotion, no SAC transfer"),
        "base_contract": args.contract,
        "base_contract_sha256": sha256_file(Path(args.contract)),
        "bounded": {"epochs": args.epochs,
                    "max_windows": args.max_windows},
        "zscore_families_in_adapters_arm": zscore_families,
        "arms": {},
    }
    for arm in ("adapters", "single"):
        contract = derive_contract(base, arm, zscore_families)
        contract_path = out_root / f"contract_{arm}.json"
        contract_path.write_text(json.dumps(contract, indent=1))
        arm_dir = out_root / arm
        proc = subprocess.run(
            [sys.executable, str(REPO / "tools/pretrain_branches.py"),
             "--contract", str(contract_path),
             "--output-dir", str(arm_dir),
             "--epochs", str(args.epochs),
             "--max-windows", str(args.max_windows)],
            capture_output=True, text=True, cwd=str(REPO))
        if proc.returncode != 0:
            raise SystemExit(f"arm {arm} failed:\n{proc.stdout}"
                             f"\n{proc.stderr}")
        manifest = json.loads(
            (arm_dir / "pretrain_manifest.json").read_text())
        report["arms"][arm] = {
            "contract_sha256": sha256_file(contract_path),
            "objective_domain": contract["objective_domain"],
            "summary": summarize(manifest),
        }
    report_path = out_root / "domain_ablation_report.json"
    report_path.write_text(json.dumps(report, indent=1))
    print(json.dumps({arm: {b: v["monitor_final"]
                            for b, v in data["summary"].items()}
                      for arm, data in report["arms"].items()},
                     indent=1))
    print(f"ABLATION COMPLETE (diagnostic only) -> "
          f"{report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
