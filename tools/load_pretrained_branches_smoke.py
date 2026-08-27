#!/usr/bin/env python3
"""ONE bounded CPU transfer-loader smoke (Musashi dispatch 2026-08-27,
automatic consequence of accepted DATA-SOTA-353..356).

Loads the sealed o2022 v4 pretrained branch ENCODERS into the declared
grouped extractor through `agent_plugins.pretrained_branch_loader`
(full identity chain, encoder-only strict load by named family, bit
parity by re-serialization), then runs ONE finite CPU forward over an
observation produced by the real ETH H4 GymFxEnv executing
preprocessor. Publishes runtime, peak host memory, per-family digests,
loaded/rejected key counts and forward-shape evidence.

The result is MECHANICS_ONLY_NOT_ECONOMICALLY_ELIGIBLE: no GPU, no
economic comparison, no promotion, no collector activation.

Public-evidence discipline: logical paths only; no host/operator
identity.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.pretrained_branch_loader import (  # noqa: E402
    TRANSFER_STATUS, TransferLoadError, check_finite_forward,
    load_family_encoders, verify_source)
from tools.pretrain_branches import (  # noqa: E402
    logical_interpreter, repo_relative, resolve_data_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--arch-config", required=True,
                        help="grouped-extractor experiment config "
                             "(state branch + fusion declaration)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true", default=True)
    args = parser.parse_args()

    import numpy as np
    import torch

    import agent_plugins.grouped_features_extractor as gfe
    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)

    t0 = time.perf_counter()
    data_path, data_logical = resolve_data_path()
    pretrain_dir = Path(args.pretrain_dir)
    source = verify_source(pretrain_dir, REPO, data_path)
    contract = source["contract"]
    manifest = source["manifest"]

    # real executing env over the fit-slice head (same fixture shape as
    # the accepted Tier-A parity tests)
    cfg = json.loads(Path(args.arch_config).read_text())
    sliced = Path(os.environ.get("TMPDIR", "/tmp")) / "loader_eth_700.csv"
    with data_path.open() as src, sliced.open("w") as dst:
        for i, line in enumerate(src):
            if i > 700:
                break
            dst.write(line)
    cfg["input_data_file"] = str(sliced)
    cfg["max_steps"] = 460
    env = _load_env_plugin("gym_fx_env", cfg).make_env(cfg)

    # declared topology ONLY — branches come from the sealed contract,
    # never inferred from checkpoint shapes
    arch = {
        "schema": "agent_multi.grouped_features.v1",
        "feature_columns": list(contract["feature_columns"]),
        "branches": [{"name": b["name"], "plugin": b["plugin"],
                      "features": list(b["features"]),
                      "params": b.get("params") or {}}
                     for b in contract["branches"]],
        "state_keys": [k for k in env.observation_space.spaces
                       if k != "features"],
        "state_branch": {"plugin": "mlp_branch",
                         "params": {"hidden_dims": [32],
                                    "output_dim": 16}},
        "fusion": {"plugin": "cross_family_attention",
                   "params": {"d_model": 32, "n_heads": 4,
                              "output_dim": 96}},
    }
    Extractor = gfe.build_grouped_extractor_class()
    torch.manual_seed(0)
    extractor = Extractor(env.observation_space, arch)

    families = load_family_encoders(pretrain_dir, manifest, contract,
                                    extractor)

    obs, _ = env.reset(seed=7)
    batch = {k: torch.tensor(
        np.repeat(np.asarray(v, dtype=np.float32)[None, ...], 3,
                  axis=0)) for k, v in obs.items()}
    extractor.eval()
    with torch.no_grad():
        out = check_finite_forward(extractor(batch))
        out_repeat = check_finite_forward(extractor(batch))
    deterministic_repeat = bool(torch.equal(out, out_repeat))
    wall = time.perf_counter() - t0
    peak_host_mb = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024.0

    packet = {
        "schema": "agent_multi.transfer_loader_smoke.v1",
        "dispatch": ("MUSASHI_TO_GENERAL_SATOSHI_TRANSFER_LOADER_CPU_"
                     "SMOKE_DISPATCH_2026_08_27 (exactly one bounded "
                     "CPU smoke)"),
        "status": TRANSFER_STATUS,
        "pretrain_dir": f"external:{pretrain_dir.name}",
        "arch_config": repo_relative(Path(args.arch_config)),
        "data_source": data_logical,
        "interpreter": logical_interpreter(),
        "device": "cpu (CUDA_VISIBLE_DEVICES empty)",
        "code_identity": source["code_identity_report"],
        "sealed_identity_verified": [
            "generation_seal", "contract_v4_digest",
            "source_data_digest", "ordered_83_feature_partition",
            "family_ordered_digests", "branch_topology_digest",
            "origin_plan_digest", "normalization_policies_digest",
            "preprocessor_module_sha", "preprocessing_config_digest",
            "training_code_file_shas"],
        "families": families,
        "tensors_loaded_total": int(sum(
            f["tensors_loaded"] for f in families.values())),
        "rejected_keys_total": 0,
        "state_branch_and_fusion": "random-init, DECLARED untransferred",
        "ordered_families": extractor.ordered_families,
        "family_digest": extractor.fusion.family_digest,
        "observation_shapes": {k: list(v.shape)
                               for k, v in batch.items()},
        "forward_output_shape": list(out.shape),
        "forward_output_finite": True,
        "deterministic_repeat_forward_equal": deterministic_repeat,
        "wall_seconds": round(wall, 3),
        "peak_host_memory_mb": round(peak_host_mb, 1),
        "gpu": "NONE", "economics": "NONE", "promotion": "NONE",
        "collector_activation": "NONE",
    }
    out_path = Path(args.output) if args.output else (
        REPO / "docs/audits/evidence/"
        "TRANSFER_LOADER_CPU_SMOKE_2026_08_27.json")
    out_path.write_text(json.dumps(packet, indent=1))
    print(json.dumps(packet, indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TransferLoadError as exc:
        print(f"TRANSFER REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
