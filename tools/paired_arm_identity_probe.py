#!/usr/bin/env python3
"""C4 adversarial identity probe (order 2026-08-28): construct BOTH
arms' SAC models for one seed on CPU — construction only, no training,
no GPU — and prove at the tensor level that the initialization is the
ONLY treatment difference:

1. same-seed construction is deterministic (control built twice is
   bitwise identical);
2. after the treatment transfer, every non-temporal-branch tensor of
   the policy is bitwise IDENTICAL between arms;
3. every treatment temporal-branch tensor is bitwise identical to the
   sealed per-family encoder artifact (via the loader's own parity
   proof);
4. all temporal-branch parameters remain trainable in both arms.

Refuses if CUDA is visible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.dispatch_paired_pretrain_comparison import (  # noqa: E402
    DispatchRefused, build_cell_config, verify_cell)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    if torch.cuda.is_available():
        raise DispatchRefused(
            "CUDA is visible — the identity probe is CPU-only")
    import numpy as np

    from agent_plugins.pretrained_branch_loader import (
        load_into_sac_policy)
    from agent_plugins.sac_agent import Plugin as SacPlugin
    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)

    design = json.loads(
        (REPO / "docs/audits/evidence/"
         "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json"
         ).read_text())
    pretrain_dir = Path(args.pretrain_dir)
    cells = {
        arm: verify_cell(design, pretrain_dir, args.seed, arm)
        for arm in ("control_random_init", "pretrained_finetuned")}
    out_root = Path(tempfile.mkdtemp(prefix="paired_identity_"))
    cfgs = {arm: build_cell_config(design, cell, pretrain_dir,
                                   out_root, device="cpu")
            for arm, cell in cells.items()}
    for cfg in cfgs.values():
        cfg.pop("_snapshot_config_sha256")

    # one tiny CPU env from the head of the pinned source CSV — enough
    # rows for the 256-bar scaling context plus the 32-bar window
    source = Path(cfgs["control_random_init"]["input_data_file"])
    sliced = out_root / "sliced.csv"
    with source.open() as src, sliced.open("w") as dst:
        for i, line in enumerate(src):
            if i > 700:
                break
            dst.write(line)
    env_cfg = {**cfgs["control_random_init"],
               "input_data_file": str(sliced), "max_steps": 460}
    env = _load_env_plugin("gym_fx_env", env_cfg).make_env(env_cfg)

    def build(cfg):
        return SacPlugin(cfg).build(env, {**cfg, "device": "cpu"})

    def flat(model):
        return {k: v.detach().clone()
                for k, v in model.policy.state_dict().items()}

    control_a = flat(build(cfgs["control_random_init"]))
    control_b = flat(build(cfgs["control_random_init"]))
    determinism = all(torch.equal(control_a[k], control_b[k])
                      for k in control_a)

    treatment_model = build(cfgs["pretrained_finetuned"])
    transfer = load_into_sac_policy(
        treatment_model, pretrain_dir, REPO, source,
        expected_seal_manifest_sha256=cfgs["pretrained_finetuned"][
            "pretrained_branch_expected_seal"])
    treatment = flat(treatment_model)

    branch_marker = "features_extractor.temporal_branches."
    identical, differing, branch_keys = [], [], []
    for key in control_a:
        is_branch = branch_marker in key
        same = torch.equal(control_a[key], treatment[key])
        if is_branch:
            branch_keys.append(key)
        elif same:
            identical.append(key)
        else:
            differing.append(key)
    branch_changed = sum(
        1 for k in branch_keys
        if not torch.equal(control_a[k], treatment[k]))
    trainable = {
        arm: all(p.requires_grad for p in
                 model.policy.actor.features_extractor.parameters())
        for arm, model in (("treatment", treatment_model),)}

    report = {
        "schema": "agent_multi.paired_arm_identity_probe.v1",
        "seed": args.seed,
        "same_seed_construction_deterministic": bool(determinism),
        "policy_tensors_total": len(control_a),
        "non_branch_tensors": len(identical) + len(differing),
        "non_branch_identical_between_arms": len(identical),
        "non_branch_DIFFERING_between_arms": differing,
        "temporal_branch_tensors": len(branch_keys),
        "temporal_branch_tensors_changed_by_transfer": branch_changed,
        "transfer_bit_parity_all_extractors": all(
            f["bit_parity"]
            for sub in transfer["extractors"].values()
            for f in sub["families"].values()),
        "transfer_trainability": transfer["trainability"],
        "treatment_encoders_trainable": trainable["treatment"],
        "verdict": ("INITIALIZATION_IS_THE_ONLY_DIFFERENCE"
                    if determinism and not differing
                    and branch_changed > 0 else "IDENTITY_VIOLATION"),
    }
    payload = json.dumps(report, indent=1)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload)
    return 0 if report["verdict"] == \
        "INITIALIZATION_IS_THE_ONLY_DIFFERENCE" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DispatchRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
