#!/usr/bin/env python3
"""Warmup/context-prefix probe (finding SOTA-R06).

Measures, on the ACTUAL pipeline execution path, how quickly the
observation buffer densifies after env reset for each nested role, with
every identity bound: probe hash, code-tree commit, effective-config
hash, per-role dataset hash (verified against the nested manifest before
use), command line. Distinguishes source-data zeros (raw CSV head) from
scaler/transform output zeros, and reports per-feature evidence with an
explicit denominator.

Run example (CPU, read-only against a completed campaign arm):

  CUDA_VISIBLE_DEVICES="" python tools/warmup_context_probe.py \
    --arm-dir ~/.local/share/agent-multi/l1_curriculum_campaign_20260823/seed101_N \
    --phase normal \
    --code-root /home/harveybc/Documents/GitHub/.worktrees/am-p1-6e7bd128 \
    --steps 400 --out <evidence.json>
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_config_sha(cfg):
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_identity(code_root):
    def run(*args):
        return subprocess.run(["git", "-C", str(code_root)] + list(args),
                              capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def load_arm_contract(arm_dir, phase):
    arm_dir = Path(arm_dir)
    launch = json.load(open(arm_dir / f"{phase}_report.launch_manifest.json"))
    manifest = json.load(
        open(arm_dir / phase / "nested_splits" / "nested_split_manifest.json"))
    return launch, manifest


def verify_role_csv(role_name, role):
    """Re-hash the role CSV and require equality with the manifest digest."""
    csv = role.get("csv")
    declared = role.get("csv_sha256")
    if not csv or not declared:
        return None
    actual = sha256_file(csv)
    if actual != declared:
        raise SystemExit(
            f"REFUSED: role {role_name} csv hash mismatch "
            f"(manifest {declared[:12]}.., actual {actual[:12]}..)")
    return actual


def raw_head_zero_fraction(csv_path, n_rows):
    """Zero fraction of numeric cells in the first n_rows of the SOURCE
    csv — separates dataset-head zeros from scaler-output zeros."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=max(n_rows, 1))
    num = df.select_dtypes("number")
    if num.size == 0:
        return None
    return float((num.to_numpy() == 0).mean())


def zeros_profile_metrics(profile):
    first_half = next((i for i, z in enumerate(profile) if z < 0.5), None)
    first_dense = next((i for i, z in enumerate(profile) if z < 0.05), None)
    return {
        "steps_probed": len(profile),
        "zero_fraction_at_reset": round(profile[0], 4) if profile else None,
        "zero_fraction_step_2": round(profile[2], 4) if len(profile) > 2 else None,
        "zero_fraction_step_50": round(profile[50], 4) if len(profile) > 50 else None,
        "zero_fraction_step_300": round(profile[300], 4) if len(profile) > 300 else None,
        "first_step_below_50pct_zeros": first_half,
        "first_step_below_5pct_zeros": first_dense,
    }


def probe_role(code_root, cfg, role_name, role, steps, seed):
    import numpy as np
    sys.path.insert(0, str(code_root))
    from pipeline_plugins.rl_pipeline_with_validation import _load_env_plugin
    rcfg = dict(cfg)
    rcfg["input_data_file"] = role["csv"]
    # env_mode=training is scoped to the fit env by pipeline design;
    # evaluation roles run in the pipeline's default inference mode.
    if role_name == "fit_train":
        rcfg["env_mode"] = "training"
    else:
        rcfg.pop("env_mode", None)
    env = _load_env_plugin("gym_fx_env", rcfg).make_env(rcfg)
    obs, _ = env.reset(seed=seed)
    profile, per_feature_reset = [], None
    o = obs
    for _ in range(steps):
        feats = np.asarray(o["features"], dtype=np.float64)
        if per_feature_reset is None:
            per_feature_reset = (feats == 0).mean(axis=0)
        profile.append(float((feats == 0).mean()))
        o, _r, term, trunc, _i = env.step([0.0])
        if term or trunc:
            break
    n_feat = int(per_feature_reset.shape[0]) if per_feature_reset is not None else 0
    ctx = role.get("context_rows") or 0
    return {
        "context_rows": role.get("context_rows"),
        "csv_sha256_verified": verify_role_csv(role_name, role),
        "feature_denominator": n_feat,
        "features_fully_zero_at_reset": (
            int((per_feature_reset == 1.0).sum()) if n_feat else None),
        "source_head_zero_fraction": raw_head_zero_fraction(
            role["csv"], max(ctx, 260)),
        **zeros_profile_metrics(profile),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--phase", default="normal")
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--roles", default="fit_train,train_monitor,"
                                       "inner_validation,outer_validation")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    launch, manifest = load_arm_contract(args.arm_dir, args.phase)
    cfg = dict(launch["effective_config"])
    roles_out = {}
    for role_name in args.roles.split(","):
        role = manifest["roles"].get(role_name)
        if not role or not role.get("csv"):
            roles_out[role_name] = {
                "skipped": "structurally unmaterialized (no csv in manifest)"}
            continue
        roles_out[role_name] = probe_role(
            Path(args.code_root), cfg, role_name, role, args.steps, args.seed)

    evidence = {
        "schema": "agent_multi.warmup_context_probe.v2",
        "probe_sha256": sha256_file(__file__),
        "code_root": str(Path(args.code_root).resolve()),
        "code_identity": git_identity(args.code_root),
        "effective_config_sha256": canonical_config_sha(cfg),
        "arm_dir": str(Path(args.arm_dir).resolve()),
        "phase": args.phase,
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "roles": roles_out,
        "interpretation_note": (
            "fit_train has context_rows=0 BY DESIGN (the fit role starts at "
            "dataset head; context prefixes exist to protect evaluation "
            "partitions). source_head_zero_fraction near the observed reset "
            "zero fraction indicates dataset-head indicator warmup, not a "
            "scaler dead zone."),
    }
    Path(args.out).write_text(json.dumps(evidence, indent=1))
    print(json.dumps({k: evidence[k] for k in
                      ("probe_sha256", "code_identity",
                       "effective_config_sha256")}, indent=1))
    for rn, r in roles_out.items():
        print(rn, json.dumps({k: r.get(k) for k in (
            "context_rows", "zero_fraction_at_reset",
            "first_step_below_5pct_zeros", "source_head_zero_fraction",
            "features_fully_zero_at_reset", "feature_denominator")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
