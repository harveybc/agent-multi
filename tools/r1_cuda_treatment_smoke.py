#!/usr/bin/env python3
"""R1 CUDA treatment smoke (GPU runtime correction order 2026-08-28,
DATA-SOTA-381). Bounded, single-GPU, Musashi-ordered: build the REAL
strong-route SAC model on CUDA, load the five sealed encoders through
the corrected loader (actor + critic + critic_target, bit parity in
the common device domain), then run REAL environment interaction with
at least one forward and one genuine gradient update — not just
construction. No long training: the budget is a few dozen steps.

Refuses if zero or more than one CUDA device is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            f"REFUSED: {torch.cuda.device_count()} CUDA devices "
            "visible — the bounded smoke uses exactly one")

    from agent_plugins.dispatch_authorization import (
        cudnn_micro_preflight)
    from agent_plugins.pretrained_branch_loader import (
        load_into_sac_policy)
    from agent_plugins.sac_agent import Plugin as SacPlugin
    from tools.dispatch_paired_pretrain_comparison import (
        build_cell_config, verify_cell)
    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)

    micro = cudnn_micro_preflight("cuda")
    design = json.loads(
        (REPO / "docs/audits/evidence/"
         "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json"
         ).read_text())
    pretrain_dir = Path(args.pretrain_dir)
    cell = verify_cell(design, pretrain_dir, args.seed,
                       "pretrained_finetuned")
    out_root = Path(tempfile.mkdtemp(prefix="r1_cuda_smoke_"))
    cfg = build_cell_config(design, cell, pretrain_dir, out_root,
                            device="cuda")
    cfg.pop("_snapshot_config_sha256")

    source = Path(cfg["input_data_file"])
    sliced = out_root / "sliced.csv"
    with source.open() as src, sliced.open("w") as dst:
        for i, line in enumerate(src):
            if i > 700:
                break
            dst.write(line)
    env_cfg = {**cfg, "input_data_file": str(sliced),
               "max_steps": 460}
    env = _load_env_plugin("gym_fx_env", env_cfg).make_env(env_cfg)

    # bounded SAC budget: one rollout burst, real updates
    smoke_cfg = {**cfg, "total_timesteps": 64, "learning_starts": 32,
                 "batch_size": 32, "buffer_size": 1000,
                 "agent_verbose": 0}
    model = SacPlugin(smoke_cfg).build(env, smoke_cfg)
    transfer = load_into_sac_policy(
        model, pretrain_dir, REPO, source,
        expected_seal_manifest_sha256=cfg[
            "pretrained_branch_expected_seal"])
    parity = all(f["bit_parity"]
                 for sub in transfer["extractors"].values()
                 for f in sub["families"].values())
    devices = sorted({str(p.device) for p in
                      model.policy.actor.features_extractor
                      .parameters()})

    updates_before = int(getattr(model, "_n_updates", 0))
    model.learn(total_timesteps=64)
    updates_after = int(getattr(model, "_n_updates", 0))
    obs, _ = env.reset(seed=7)
    action, _ = model.predict(obs, deterministic=True)
    torch.cuda.synchronize()

    report = {
        "schema": "agent_multi.r1_cuda_treatment_smoke.v1",
        "finding": "DATA-SOTA-381",
        "device_class_sanitized": torch.cuda.get_device_name(0),
        "cudnn_micro_preflight": micro,
        "treatment_seal": transfer["seal_manifest_sha256"],
        "transfer_bit_parity_all_extractors": parity,
        "transfer_trainability": transfer["trainability"],
        "extractor_parameter_devices": devices,
        "timesteps_run": int(model.num_timesteps),
        "gradient_updates_real": updates_after - updates_before,
        "post_update_predict_finite": bool(
            torch.isfinite(torch.as_tensor(action)).all()),
        "verdict": ("CUDA_TREATMENT_PATH_EXECUTES"
                    if parity and updates_after > updates_before
                    and devices == ["cuda:0"]
                    else "CUDA_TREATMENT_PATH_BROKEN"),
    }
    payload = json.dumps(report, indent=1)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload)
    return 0 if report["verdict"] == "CUDA_TREATMENT_PATH_EXECUTES" \
        else 3


if __name__ == "__main__":
    raise SystemExit(main())
