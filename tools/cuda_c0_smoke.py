#!/usr/bin/env python3
"""ONE bounded CUDA C0 mechanics smoke (Musashi dispatch 2026-08-26).

Strong grouped route on the REAL GymFxEnv observation, moved to CUDA:
forward/backward mechanics only — NO economic comparison, NO checkpoint
promotion, NO B4. Publishes device (redacted UUID), CUDA/torch
versions, wall time, peak memory, parameter count, output finiteness,
per-named-branch/fusion/actor-facing gradients, save/load parity; binds
command, config digest, data digest, code commit and family digest.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True,
                                     default=str).encode()).hexdigest()


def public_gpu_identity(device_index: int) -> dict:
    """DATA-SOTA-339: PUBLIC evidence carries the model class and the
    run-local ordinal ONLY — no UUID fragment, no UUID hash (a hash of
    a stable private identifier is a pseudonym, not a redaction)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                          "--format=csv,noheader", "-i",
                          str(device_index)],
                         capture_output=True, text=True).stdout.strip()
    return {"gpu_model": out, "gpu_run_local_ordinal": device_index}


DECLARED_OUTPUTS = None  # set in main() after C0_OUTPUT resolves


def _tree_status() -> list:
    out = subprocess.run(["git", "-C", str(REPO), "status",
                          "--porcelain"],
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def _clean_with_declared_outputs() -> bool:
    allowed = {str(p) for p in (DECLARED_OUTPUTS or [])}
    for line in _tree_status():
        path = line[3:].strip()
        full = str((REPO / path).resolve())
        if full not in allowed:
            return False
    return True


def main() -> int:
    import numpy as np
    import torch

    global DECLARED_OUTPUTS
    out_declared = Path(os.environ.get(
        "C0_OUTPUT",
        str(REPO / "docs/audits/evidence/"
            "CUDA_C0_SMOKE_V2_2026_08_26.json")))
    DECLARED_OUTPUTS = [out_declared.resolve()]
    # preflight: the tree must already be clean (incl. untracked)
    # BEFORE the run creates its declared output
    pre = _tree_status()
    if pre:
        raise SystemExit(f"REFUSED: tree not clean at preflight "
                         f"(incl. untracked): {pre[:5]}")
    if not torch.cuda.is_available():
        raise SystemExit("REFUSED: CUDA unavailable")
    eth = os.environ.get("AGENT_MULTI_ETH_CSV") or str(
        REPO.parent.parent / "predictor/examples/data/project3/"
        "ethusdt_4h_tech_stat_full_model_ready.csv")
    eth = Path(eth)
    if not eth.is_file():
        raise SystemExit("REFUSED: Tier-A fixture absent")

    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)
    import agent_plugins.grouped_features_extractor as gfe
    from agent_plugins.feature_families import semantic_feature_families

    cfg = json.loads((REPO / "examples/config/"
                      "project3_ethusdt_4h_sac_grouped_features_v1.json"
                      ).read_text())
    sliced = Path(os.environ.get("TMPDIR", "/tmp")) / "c0_eth_700.csv"
    with eth.open() as src, sliced.open("w") as dst:
        for i, line in enumerate(src):
            if i > 700:
                break
            dst.write(line)
    cfg["input_data_file"] = str(sliced)
    cfg["max_steps"] = 460
    env = _load_env_plugin("gym_fx_env", cfg).make_env(cfg)
    cols = list(cfg["feature_columns"])
    fams = semantic_feature_families(cols)
    arch = {
        "schema": "agent_multi.grouped_features.v1",
        "feature_columns": cols,
        "branches": [
            {"name": "returns_momentum", "plugin": "patchtst_branch",
             "features": fams["returns_momentum"],
             "params": {"d_model": 32, "n_heads": 4, "n_layers": 1}},
            {"name": "trend_level", "plugin": "tft_branch",
             "features": fams["trend_level"],
             "params": {"hidden": 32, "n_heads": 4}},
            {"name": "volatility_distribution",
             "plugin": "timesnet_branch",
             "features": fams["volatility_distribution"],
             "params": {"d_model": 32, "top_k": 2}},
            {"name": "oscillators", "plugin": "tcn_branch",
             "features": fams["oscillators"],
             "params": {"channels": [32, 32]}},
            {"name": "volume_flow", "plugin": "gru_branch",
             "features": fams["volume_flow"]},
        ],
        "state_keys": [k for k in env.observation_space.spaces
                       if k != "features"],
        "state_branch": {"plugin": "mlp_branch",
                         "params": {"hidden_dims": [32],
                                    "output_dim": 16}},
        "fusion": {"plugin": "cross_family_attention",
                   "params": {"d_model": 32, "n_heads": 4,
                              "output_dim": 96}},
    }

    device = torch.device("cuda:0")
    # DATA-SOTA-338: measure THROUGH construction — synchronize, clear
    # cache, reset peaks BEFORE the model exists; baseline reported
    # separately.
    torch.cuda.init()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_before, total = torch.cuda.mem_get_info(device)
    baseline_device_used_mb = round((total - free_before) / 1e6, 1)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    Extractor = gfe.build_grouped_extractor_class()
    torch.manual_seed(0)
    model = Extractor(env.observation_space, arch).to(device)
    actor_probe = torch.nn.Linear(96, 1).to(device)  # actor-facing path

    obs, _ = env.reset(seed=7)
    batch = {k: torch.tensor(
        np.repeat(np.asarray(v, dtype=np.float32)[None, ...], 4,
                  axis=0)).to(device) for k, v in obs.items()}
    out = model(batch)
    actor_out = actor_probe(out)
    finite = bool(torch.isfinite(out).all()
                  and torch.isfinite(actor_out).all())
    actor_out.sum().backward()

    grads = {}
    for i, branch in enumerate(model.temporal_branches):
        name = arch["branches"][i]["name"]
        grads[name] = float(sum(
            p.grad.abs().sum() for p in branch.parameters()
            if p.grad is not None))
    grads["account_state"] = float(sum(
        p.grad.abs().sum() for p in model.state_branch.parameters()
        if p.grad is not None))
    grads["fusion"] = float(sum(
        p.grad.abs().sum() for p in model.fusion.parameters()
        if p.grad is not None))
    grads["actor_facing_probe"] = float(sum(
        p.grad.abs().sum() for p in actor_probe.parameters()
        if p.grad is not None))

    # save/load output parity on-device
    model.eval()
    with torch.no_grad():
        ref = model(batch)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "c0_strong.pt"
    torch.save(model.state_dict(), tmp)
    model2 = Extractor(env.observation_space, arch).to(device)
    model2.load_state_dict(torch.load(tmp, weights_only=True,
                                      map_location=device))
    model2.eval()
    with torch.no_grad():
        out2 = model2(batch)
    parity = bool(torch.equal(ref, out2))
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0

    packet = {
        "schema": "agent_multi.cuda_c0_smoke.v2",
        "replaces_rejected_packet": (
            "docs/audits/evidence/CUDA_C0_SMOKE_2026_08_26.json "
            "(REJECTED by AUDIT_SATOSHI_CUDA_C0_RETURN_2026_08_26; "
            "preserved unmodified as historical evidence)"),
        "dispatch": ("MUSASHI_TO_GENERAL_SATOSHI_CUDA_C0_DISPATCH_"
                     "2026_08_26 (single bounded mechanics smoke)"),
        "code_commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        # DATA-SOTA-337: cleanliness INCLUDES untracked files; the only
        # tolerated entries are the declared output artifacts of THIS
        # run, verified against the recorded preflight.
        "tree_status_including_untracked": _tree_status(),
        "clean_tree_including_untracked": _clean_with_declared_outputs(),
        "runner_sha256": sha_file(Path(__file__)),
        "argv_full": list(sys.argv),
        "interpreter": {"path": sys.executable,
                        "python": sys.version.split()[0]},
        "environment_identity": {
            "CUDA_VISIBLE_DEVICES": os.environ.get(
                "CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "TMPDIR": os.environ.get("TMPDIR")},
        "command": " ".join(sys.argv),
        "data_digest": sha_file(eth),
        "config_sha256": sha_obj(arch),
        "family_digest": model.fusion.family_digest,
        "family_ids_ordered": model.fusion.family_ids,
        **public_gpu_identity(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "device": str(device),
        "wall_seconds": round(wall, 3),
        "baseline_device_used_mb": baseline_device_used_mb,
        "peak_allocated_mb_through_construction": round(
            torch.cuda.max_memory_allocated(device) / 1e6, 1),
        "peak_reserved_mb_through_construction": round(
            torch.cuda.max_memory_reserved(device) / 1e6, 1),
        "parameter_count": int(sum(p.numel()
                                   for p in model.parameters())),
        "output_finite": finite,
        "nonzero_gradients": {k: v > 0 for k, v in grads.items()},
        "gradient_l1": {k: round(v, 4) for k, v in grads.items()},
        "save_load_output_parity": parity,
        "economic_comparison": "NONE (mechanics only)",
        "checkpoint_promotion": "NONE",
        "b4_dispatch": "NONE",
    }
    out_path = Path(os.environ.get(
        "C0_OUTPUT",
        str(REPO / "docs/audits/evidence/"
            "CUDA_C0_SMOKE_V2_2026_08_26.json")))
    out_path.write_text(json.dumps(packet, indent=1))
    print(json.dumps(packet, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
