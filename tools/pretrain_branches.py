#!/usr/bin/env python3
"""WP-PRETRAIN executing runner (Data-First order @7886de39): CPU
branch-wise self-supervised pretraining of the strong grouped route.

First two ordered objectives are wired: masked-patch reconstruction and
multi-horizon quantile. The artifact carries 316/317-grade identity
(contract/data/feature/code digests, seed, fit boundary) and resume
REFUSES on any identity drift. Rows after ``fit_end`` are never loaded:
development_outer 2024 and sealed 2025 stay structurally absent.

Public-evidence discipline: the manifest records LOGICAL identities
only — no absolute paths, no host names, no operator name.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    PretrainContractError, build_window_index, canonical_feature_digest,
    forward_log_return_targets, instance_normalize, load_fit_slice,
    masked_reconstruction_loss, pinball_loss, refuse_on_identity_drift,
    resume_identity, sample_span_mask, sha256_file, sha256_obj,
    validate_contract)


def resolve_data_path() -> tuple[Path, str]:
    override = os.environ.get("AGENT_MULTI_ETH_CSV")
    if override:
        return Path(override), "env:AGENT_MULTI_ETH_CSV"
    conventional = (REPO.parent.parent / "predictor/examples/data/project3/"
                    "ethusdt_4h_tech_stat_full_model_ready.csv")
    return conventional, ("dataset:ethusdt_4h_tech_stat_full_model_ready"
                          ".csv@conventional-sibling-checkout")


def logical_interpreter() -> str:
    env_name = Path(sys.prefix).name
    return f"python:{sys.version.split()[0]}@env:{env_name}"


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return f"external:{path.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="continue from checkpoint.pt in output-dir; "
                             "refuses on any identity drift")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override contract epochs (recorded)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="override contract max_windows (recorded)")
    parser.add_argument("--stop-after-epochs", type=int, default=None,
                        help="bounded interruption: exit cleanly after N "
                             "epoch checkpoints (resume continues the "
                             "exact trajectory)")
    args = parser.parse_args()

    import numpy as np
    import torch

    from app.plugin_loader import load_plugin

    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text())
    parsed = validate_contract(contract)
    epochs = args.epochs if args.epochs is not None else parsed["epochs"]
    if epochs < 1:
        raise PretrainContractError(f"epochs must be >= 1, got {epochs}")
    max_windows = (args.max_windows if args.max_windows is not None
                   else contract.get("max_windows"))

    data_path, data_logical = resolve_data_path()
    if not data_path.is_file():
        raise SystemExit(f"REFUSED: dataset absent ({data_logical}); "
                         f"set AGENT_MULTI_ETH_CSV")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, columns, close_col = load_fit_slice(data_path, contract)
    branches = contract["branches"]
    assignment = [{"name": b["name"], "plugin": b["plugin"],
                   "params": b.get("params") or {},
                   "features": list(b["features"])} for b in branches]
    claimed: set[str] = set()
    for spec in assignment:
        unknown = [f for f in spec["features"] if f not in columns]
        if unknown:
            raise PretrainContractError(
                f"branch {spec['name']} features not in contract "
                f"columns: {unknown}")
        overlap = claimed.intersection(spec["features"])
        if overlap:
            raise PretrainContractError(
                f"features assigned to multiple branches: "
                f"{sorted(overlap)}")
        claimed.update(spec["features"])

    identity = {
        "contract_path": repo_relative(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "data_source": data_logical,
        "data_sha256": sha256_file(data_path),
        "feature_columns_sha256": canonical_feature_digest(columns),
        "branch_assignment_sha256": sha256_obj(assignment),
        "library_sha256": sha256_file(
            REPO / "agent_plugins/branch_pretraining.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "code_commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "torch_version": torch.__version__,
        "interpreter": logical_interpreter(),
        "seed": parsed["seed"],
        "window_size": parsed["window_size"],
        "fit_end": parsed["fit_end"],
    }

    objectives = contract["objectives"]
    rec_spec = objectives.get("masked_patch_reconstruction")
    quant_spec = objectives.get("multi_horizon_quantile")
    horizons = list(quant_spec["horizons"]) if quant_spec else [1]
    max_horizon = max(horizons)

    window = parsed["window_size"]
    ends = build_window_index(len(df), window, parsed["window_stride"],
                              max_horizon, max_windows)
    features = df[columns].to_numpy(dtype=np.float32)
    windows_view = np.lib.stride_tricks.sliding_window_view(
        features, window, axis=0)  # (N-w+1, F, w)
    targets = None
    if quant_spec:
        targets = torch.tensor(forward_log_return_targets(
            df[close_col].to_numpy(), ends, horizons))

    manifest_path = out_dir / "pretrain_manifest.json"
    ckpt_path = out_dir / "checkpoint.pt"

    if args.resume:
        if not ckpt_path.is_file() or not manifest_path.is_file():
            raise PretrainContractError(
                "resume REFUSED: no checkpoint/manifest in output dir")
        manifest = json.loads(manifest_path.read_text())
        refuse_on_identity_drift(resume_identity(manifest),
                                 resume_identity({"identity": identity}))
        ckpt = torch.load(ckpt_path, weights_only=False)
        refuse_on_identity_drift(ckpt["identity"],
                                 resume_identity({"identity": identity}))
        start_branch = ckpt["branch_index"]
        start_epoch = ckpt["epochs_done_in_branch"]
    else:
        if manifest_path.exists():
            raise PretrainContractError(
                "output dir already holds a manifest; pass --resume or "
                "choose a fresh directory (existing artifacts are never "
                "silently overwritten)")
        manifest = {
            "schema": "agent_multi.branch_pretrain.v1",
            "order": ("Data-First SOTA Multibranch @7886de39 — "
                      "WP-PRETRAIN"),
            "identity": {**identity},
            "dataset": {
                "rows_in_fit_slice": int(len(df)),
                "eligible_windows": len(ends),
                "window_stride": parsed["window_stride"],
                "max_windows_applied": max_windows,
                "max_horizon": max_horizon,
                "boundary_note": ("rows after fit_end never loaded; "
                                  "windows with forward targets crossing "
                                  "fit_end dropped"),
            },
            "objectives": objectives,
            "cli": {"argv": [Path(a).name if os.sep in str(a) else str(a)
                             for a in sys.argv],
                    "epochs_effective": epochs},
            "progress": {b["name"]: {"status": "pending", "losses": []}
                         for b in assignment},
            "artifacts": {},
        }
        ckpt = None
        start_branch = 0
        start_epoch = 0
    # resume-supplied manifests keep their recorded epochs_effective
    epochs = manifest["cli"]["epochs_effective"]

    torch.manual_seed(parsed["seed"])
    gen = torch.Generator().manual_seed(parsed["seed"] + 1)
    epoch_checkpoints_this_invocation = 0
    batch_size = parsed["batch_size"]
    n = len(ends)
    start_idx = np.asarray(ends) - window + 1

    for bi in range(start_branch, len(assignment)):
        spec = assignment[bi]
        ch_idx = [columns.index(f) for f in spec["features"]]
        plugin_class, _ = load_plugin("feature_branch.plugins",
                                      spec["plugin"])
        from agent_plugins.component_config import deep_merge_strict
        params = deep_merge_strict(plugin_class.plugin_params,
                                   spec["params"],
                                   path=f"branches[{bi}].params")
        encoder, dim = plugin_class.build(len(ch_idx), window, params)
        heads = torch.nn.ModuleDict()
        if rec_spec:
            heads["reconstruction"] = torch.nn.Linear(
                dim, window * len(ch_idx))
        if quant_spec:
            heads["quantile"] = torch.nn.Linear(
                dim, len(horizons) * len(quant_spec["quantiles"]))
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(heads.parameters()),
            lr=parsed["lr"])
        epoch0 = 0
        if ckpt is not None and bi == start_branch:
            encoder.load_state_dict(ckpt["encoder_state"])
            heads.load_state_dict(ckpt["heads_state"])
            opt.load_state_dict(ckpt["optimizer_state"])
            gen.set_state(ckpt["generator_state"])
            torch.set_rng_state(ckpt["torch_rng_state"])
            epoch0 = start_epoch
            ckpt = None

        manifest["progress"][spec["name"]]["status"] = "training"
        for epoch in range(epoch0, epochs):
            t0 = time.perf_counter()
            perm = torch.randperm(n, generator=gen)
            sums = {"reconstruction": 0.0, "quantile": 0.0,
                    "total": 0.0}
            batches = 0
            for lo in range(0, n, batch_size):
                idx = perm[lo:lo + batch_size]
                batch_windows = instance_normalize(torch.tensor(
                    windows_view[start_idx[idx.numpy()]][:, ch_idx, :]
                ).permute(0, 2, 1).contiguous())  # (B, T, C) normalized
                loss = torch.zeros(())
                if rec_spec:
                    mask = sample_span_mask(
                        batch_windows.shape[0], window,
                        float(rec_spec["mask_ratio"]),
                        int(rec_spec["mask_span"]), gen)
                    rec = masked_reconstruction_loss(
                        encoder, heads["reconstruction"], batch_windows,
                        mask)
                    sums["reconstruction"] += float(rec)
                    loss = loss + float(rec_spec["weight"]) * rec
                if quant_spec:
                    pred = heads["quantile"](encoder(batch_windows)).view(
                        -1, len(horizons), len(quant_spec["quantiles"]))
                    ql = pinball_loss(pred, targets[idx],
                                      quant_spec["quantiles"])
                    sums["quantile"] += float(ql)
                    loss = loss + float(quant_spec["weight"]) * ql
                if not torch.isfinite(loss):
                    raise PretrainContractError(
                        f"non-finite loss in {spec['name']} epoch "
                        f"{epoch}: typed run failure")
                opt.zero_grad()
                loss.backward()
                opt.step()
                sums["total"] += float(loss)
                batches += 1
            record = {"epoch": epoch,
                      **{k: round(v / batches, 8)
                         for k, v in sums.items()},
                      "seconds": round(time.perf_counter() - t0, 3)}
            manifest["progress"][spec["name"]]["losses"].append(record)
            torch.save({"identity": resume_identity(
                            {"identity": identity}),
                        "branch_index": bi,
                        "epochs_done_in_branch": epoch + 1,
                        "encoder_state": encoder.state_dict(),
                        "heads_state": heads.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "generator_state": gen.get_state(),
                        "torch_rng_state": torch.get_rng_state()},
                       ckpt_path)
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(f"[{spec['name']}] epoch {epoch}: "
                  f"total {record['total']:.6f} "
                  f"(rec {record['reconstruction']:.6f} "
                  f"quant {record['quantile']:.6f}) "
                  f"{record['seconds']}s")
            epoch_checkpoints_this_invocation += 1
            if (args.stop_after_epochs is not None
                    and epoch_checkpoints_this_invocation
                    >= args.stop_after_epochs):
                print(f"INTERRUPTED after "
                      f"{epoch_checkpoints_this_invocation} epoch "
                      f"checkpoints; resume with --resume")
                return 0

        enc_file = out_dir / f"branch_{spec['name']}_encoder.pt"
        heads_file = out_dir / f"branch_{spec['name']}_heads.pt"
        torch.save(encoder.state_dict(), enc_file)
        torch.save(heads.state_dict(), heads_file)
        manifest["progress"][spec["name"]]["status"] = "complete"
        manifest["artifacts"][spec["name"]] = {
            "encoder_file": enc_file.name,
            "encoder_sha256": sha256_file(enc_file),
            "heads_file": heads_file.name,
            "heads_sha256": sha256_file(heads_file),
            "encoder_dim": int(dim),
            "parameters": int(sum(p.numel()
                                  for p in encoder.parameters())),
        }
        manifest_path.write_text(json.dumps(manifest, indent=1))

    manifest["completed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"COMPLETE: {len(assignment)} branches -> "
          f"{repo_relative(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
