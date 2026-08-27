#!/usr/bin/env python3
"""WP-PRETRAIN executing runner v3 (Data-First order @7886de39;
DATA-SOTA-341..346 and 347..352 corrected).

CPU branch-wise self-supervised pretraining of the strong grouped
route, with: verified causal origin chain (347), complete ordered
feature partition bound by digests (348), chronological
train/calibration/monitor partitions — weights calibrate on calibration
only, monitor only checkpoints/reports (349), ONE input domain — the
transferred encoder always consumes the exact runtime-preprocessed
tensor, normalization policies transform reconstruction TARGETS only
and heads are objective adapters excluded from transfer (350) — plus
the whole 341..346 discipline (executing-preprocessor windows,
mask-safe target statistics, monotone quantile head, gradient
diagnostics, complete resume identity over atomic digest-sealed
generations).

Artifacts are NOT_TRANSFER_ELIGIBLE until independently accepted; no
encoder from this runner loads into SAC without that acceptance.

Public-evidence discipline: the manifest records LOGICAL identities
only — no absolute paths, no host names, no operator name.
"""
from __future__ import annotations

import argparse
import inspect
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    PretrainContractError, assert_purged_boundaries,
    balance_objective_weights, barrier_hit_labels, barrier_loss,
    build_barrier_head, build_monotone_quantile_head, build_projection_head,
    build_step_index, build_volatility_head, canonical_feature_digest,
    collect_preprocessed_windows, forward_log_return_targets,
    frozen_class_weights, hierarchical_contrastive_loss, load_fit_slice,
    load_generation, masked_reconstruction_loss,
    objective_gradient_diagnostics, partition_evidence, pinball_loss,
    quantile_crossing_rate, realized_volatility_targets,
    reconstruction_target, refuse_on_identity_drift, resume_identity,
    sample_span_mask, sha256_file, sha256_obj, three_way_split,
    validate_contract, verify_earlier_origin_decision, write_generation,
    _fsync_write_bytes)

PREPROCESSING_IDENTITY_KEYS = (
    "window_size", "feature_scaling", "feature_scaling_window",
    "feature_binary_columns", "feature_clip")


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


def save_module_fsync(module, path: Path) -> str:
    import torch

    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    payload = buffer.getvalue()
    _fsync_write_bytes(path, payload)
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the sealed generation in "
                             "output-dir; refuses on identity drift or "
                             "a torn generation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override contract epochs (recorded)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="override contract max_windows (recorded)")
    parser.add_argument("--stop-after-epochs", type=int, default=None,
                        help="bounded interruption: exit cleanly after N "
                             "epoch generations (resume continues the "
                             "exact trajectory)")
    args = parser.parse_args()

    import numpy as np
    import torch

    from app.plugin_loader import load_plugin

    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text())
    parsed = validate_contract(contract)
    origin_decision = None
    if parsed["predecessor_origin_id"] is not None:
        # DATA-SOTA-347/356: loaded, digest-verified decision of the
        # EXACT immediate predecessor
        origin_decision = verify_earlier_origin_decision(contract, REPO)
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
    manifest_path = out_dir / "pretrain_manifest.json"
    if args.resume and manifest_path.exists():
        # the EFFECTIVE dataset bound is part of the run identity: a
        # resume must rebuild the exact same eligible-step set
        _peek_cli = json.loads(manifest_path.read_text())["cli"]
        epochs = _peek_cli["epochs_effective"]
        if args.max_windows is None:
            max_windows = _peek_cli.get("max_windows_effective",
                                        max_windows)

    df, columns, close_col = load_fit_slice(data_path, contract)

    # DATA-SOTA-342: the executing observation pipeline is the window
    # source. The source config must carry the same feature identity.
    source_config_path = Path(
        contract["observation_pipeline"]["source_config"])
    if not source_config_path.is_absolute():
        source_config_path = REPO / source_config_path
    env_config = json.loads(source_config_path.read_text())
    if canonical_feature_digest(env_config.get("feature_columns")) != \
            canonical_feature_digest(columns):
        raise PretrainContractError(
            "observation_pipeline.source_config feature_columns differ "
            "from the contract feature identity (DATA-SOTA-342)")
    plugin_name = contract["observation_pipeline"]["preprocessor_plugin"]
    plugin_class, _ = load_plugin("preprocessor.plugins", plugin_name)
    preprocessing_identity = {
        key: env_config.get(key, plugin_class.plugin_params.get(key))
        for key in PREPROCESSING_IDENTITY_KEYS}
    if int(preprocessing_identity["window_size"]) != \
            int(parsed["window_size"]):
        raise PretrainContractError(
            f"contract window_size={parsed['window_size']} differs from "
            f"the executing preprocessor window "
            f"{preprocessing_identity['window_size']} (DATA-SOTA-342)")

    # DATA-SOTA-348: complete ordered partition, digests bound
    partition = parsed["partition"]
    assignment = [{"name": b["name"], "plugin": b["plugin"],
                   "params": b.get("params") or {},
                   "features": list(b["features"])}
                  for b in contract["branches"]]

    identity = {
        "contract_path": repo_relative(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "data_source": data_logical,
        "data_sha256": sha256_file(data_path),
        "feature_columns_sha256": partition["global_ordered_digest"],
        "family_ordered_digests_sha256": sha256_obj(
            partition["family_ordered_digests"]),
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
        "fit_end": parsed["fit_end"].isoformat(),
        "score_origin": parsed["origin_id"],
        "warmup_bars": parsed["warmup_bars"],
        "calibration_fraction": parsed["calibration_fraction"],
        "monitor_fraction": parsed["monitor_fraction"],
        "objective_domain": parsed["objective_domain"],
        "origin_plan_sha256": sha256_obj(parsed["origin_plan"]),
        "earlier_origin_decision": origin_decision,
        "preprocessor_plugin": plugin_name,
        "preprocessor_module_sha256": sha256_file(
            Path(inspect.getfile(plugin_class))),
        "preprocessing_config_digest": sha256_obj(
            preprocessing_identity),
        "normalization_policies_digest": sha256_obj(
            parsed["normalization_policies"]),
        "objective_balancing_digest": sha256_obj(
            contract["objective_balancing"]),
    }

    objectives = contract["objectives"]
    rec_spec = objectives.get("masked_patch_reconstruction")
    quant_spec = objectives.get("multi_horizon_quantile")
    contrastive_spec = objectives.get("hierarchical_contrastive")
    vol_spec = objectives.get("volatility")
    barrier_spec = objectives.get("barrier_hit")
    horizons = list(quant_spec["horizons"]) if quant_spec else [1]
    quantiles = list(quant_spec["quantiles"]) if quant_spec else [0.5]
    # WP1: the purge horizon is the max forward horizon over ALL
    # objectives, derived mechanically by the validator
    max_horizon = parsed["max_horizon_all_objectives"]

    window = parsed["window_size"]
    steps = build_step_index(len(df), parsed["warmup_bars"],
                             parsed["window_stride"], max_horizon,
                             max_windows)
    # DATA-SOTA-353: purge derived MECHANICALLY from max(horizons);
    # it binds resume identity like every other executing fact
    purge_steps = max_horizon
    identity["purge_steps"] = purge_steps
    (train_steps, calibration_steps, monitor_steps,
     purged_steps) = three_way_split(
        steps, parsed["calibration_fraction"],
        parsed["monitor_fraction"], purge_steps)
    assert_purged_boundaries(train_steps, calibration_steps,
                             monitor_steps, max_horizon,
                             parsed["window_stride"])
    all_windows = collect_preprocessed_windows(df, contract, env_config,
                                               steps)
    step_pos = {t: i for i, t in enumerate(steps)}
    train_idx = np.array([step_pos[t] for t in train_steps])
    calibration_idx = np.array([step_pos[t] for t in calibration_steps])
    monitor_idx = np.array([step_pos[t] for t in monitor_steps])
    steps_array = np.asarray(steps)
    targets_all = None
    if quant_spec:
        targets_all = torch.tensor(forward_log_return_targets(
            df[close_col].to_numpy(), steps, horizons))
    vol_targets_all = None
    if vol_spec:
        annualization = vol_spec["annualization"]
        vol_targets_all = torch.tensor(realized_volatility_targets(
            df[close_col].to_numpy(), steps,
            list(vol_spec["horizons"]), float(vol_spec["epsilon"]),
            None if annualization == "none"
            else annualization["periods_per_year"]))
    barrier_labels_all = None
    barrier_class_weights = None
    if barrier_spec:
        barrier_labels_all = torch.tensor(barrier_hit_labels(
            df[close_col].to_numpy(), steps,
            list(barrier_spec["horizons"]),
            int(barrier_spec["barrier_scale"]["lookback"]),
            float(barrier_spec["upper_mult"]),
            float(barrier_spec["lower_mult"]),
            float(barrier_spec["barrier_scale"]["epsilon"])))

    stamps = df[str(contract.get("date_column") or "DATE_TIME")].tolist()
    context_rows = int(parsed["window_size"]) + int(
        preprocessing_identity.get("feature_scaling_window") or 0)
    partitions_evidence = {
        name: partition_evidence(name, part, stamps,
                                 parsed["window_size"], max_horizon,
                                 context_rows)
        for name, part in (("train", train_steps),
                           ("calibration", calibration_steps),
                           ("monitor", monitor_steps))}
    partitions_evidence["purge"] = {
        "purge_steps_each_boundary": purge_steps,
        "derived_from": "max(horizons) — mechanical, no free constant",
        "additional_embargo_steps": 0,
        "purged_windows_total": len(purged_steps),
        "purged_steps_sha256": sha256_obj(purged_steps)}

    if args.resume:
        ckpt, manifest, generation = load_generation(out_dir)
        refuse_on_identity_drift(resume_identity(manifest), identity)
        refuse_on_identity_drift(ckpt["identity"], identity)
        start_branch = ckpt["branch_index"]
        start_epoch = ckpt["epochs_done_in_branch"]
    else:
        if manifest_path.exists():
            raise PretrainContractError(
                "output dir already holds a manifest; pass --resume or "
                "choose a fresh directory (existing artifacts are never "
                "silently overwritten)")
        manifest = {
            "schema": "agent_multi.branch_pretrain.v3",
            "order": ("Data-First SOTA Multibranch @7886de39 — "
                      "WP-PRETRAIN, DATA-SOTA-341..352 corrected"),
            "transfer_eligibility": (
                "NOT_TRANSFER_ELIGIBLE_PENDING_INDEPENDENT_ACCEPTANCE"),
            "identity": {**identity},
            "dataset": {
                "rows_in_fit_slice": int(len(df)),
                "eligible_windows": len(steps),
                "window_stride": parsed["window_stride"],
                "max_windows_applied": max_windows,
                "max_horizon": max_horizon,
                "boundary_note": ("causal per-origin: rows after "
                                  "fit_end never loaded; targets "
                                  "crossing fit_end dropped"),
            },
            "partitions": partitions_evidence,
            "feature_partition": partition,
            "objectives": objectives,
            "objective_domain_note": (
                "DATA-SOTA-350: encoder consumes the runtime tensor "
                "for EVERY objective; policies transform "
                "reconstruction targets only; heads are objective "
                "adapters excluded from transferred encoder weights"),
            "normalization_policies": contract["normalization_policies"],
            "objective_balancing": contract["objective_balancing"],
            "cli": {"argv": [Path(a).name if os.sep in str(a) else str(a)
                             for a in sys.argv],
                    "epochs_effective": epochs,
                    "max_windows_effective": max_windows},
            "progress": {b["name"]: {"status": "pending",
                                     "effective_weights": None,
                                     "losses": []}
                         for b in assignment},
            "artifacts": {},
        }
        ckpt = None
        generation = 0
        start_branch = 0
        start_epoch = 0
    epochs = manifest["cli"]["epochs_effective"]

    torch.manual_seed(parsed["seed"])
    gen = torch.Generator().manual_seed(parsed["seed"] + 1)
    # fixed masks: monitor (reporting) and calibration (weight fitting)
    monitor_gen = torch.Generator().manual_seed(parsed["seed"] + 2)
    calibration_gen = torch.Generator().manual_seed(parsed["seed"] + 3)
    batch_size = parsed["batch_size"]
    n_train = len(train_idx)
    epoch_generations_this_invocation = 0
    declared_weights = {}
    if rec_spec:
        declared_weights["reconstruction"] = float(rec_spec["weight"])
    if quant_spec:
        declared_weights["quantile"] = float(quant_spec["weight"])
    if contrastive_spec:
        declared_weights["contrastive"] = float(
            contrastive_spec["weight"])
    if vol_spec:
        declared_weights["volatility"] = float(vol_spec["weight"])
    if barrier_spec:
        declared_weights["barrier"] = float(barrier_spec["weight"])
        # DATA-SOTA-349 discipline: class balance from CALIBRATION only
        barrier_class_weights = frozen_class_weights(
            barrier_labels_all[calibration_idx].numpy())
        manifest.setdefault("barrier_class_weights_calibration_only",
                            barrier_class_weights)

    for bi in range(start_branch, len(assignment)):
        spec = assignment[bi]
        policy = parsed["normalization_policies"][spec["name"]]
        ch_idx = [columns.index(f) for f in spec["features"]]
        plugin_class_b, _ = load_plugin("feature_branch.plugins",
                                        spec["plugin"])
        from agent_plugins.component_config import deep_merge_strict
        params = deep_merge_strict(plugin_class_b.plugin_params,
                                   spec["params"],
                                   path=f"branches[{bi}].params")
        encoder, dim = plugin_class_b.build(len(ch_idx), window, params)
        heads = torch.nn.ModuleDict()
        if rec_spec:
            heads["reconstruction"] = torch.nn.Linear(
                dim, window * len(ch_idx))
        if quant_spec:
            heads["quantile"] = build_monotone_quantile_head(
                dim, len(horizons), len(quantiles))
        if contrastive_spec:
            heads["projection"] = build_projection_head(
                dim, int(contrastive_spec["projection_dim"]))
        if vol_spec:
            heads["volatility"] = build_volatility_head(
                dim, len(vol_spec["horizons"]))
        if barrier_spec:
            heads["barrier"] = build_barrier_head(
                dim, len(barrier_spec["horizons"]))
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(heads.parameters()),
            lr=parsed["lr"])

        branch_windows = torch.tensor(
            all_windows[:, :, ch_idx].copy())  # (N, T, C)
        monitor_windows = branch_windows[monitor_idx]
        calibration_windows = branch_windows[calibration_idx]
        mask_ratio = float(rec_spec["mask_ratio"]) if rec_spec else 0.25
        mask_span = int(rec_spec["mask_span"]) if rec_spec else 4
        monitor_mask = sample_span_mask(
            len(monitor_idx), window, mask_ratio, mask_span,
            monitor_gen)
        calibration_mask = sample_span_mask(
            len(calibration_idx), window, mask_ratio, mask_span,
            calibration_gen)
        last_diagnostics = {}

        def objective_losses(values, mask, idx):
            losses = {}
            if rec_spec:
                target = reconstruction_target(values, mask, policy)
                losses["reconstruction"] = masked_reconstruction_loss(
                    encoder, heads["reconstruction"], values, target,
                    mask)
            if quant_spec:
                pred = heads["quantile"](encoder(values))
                losses["quantile"] = pinball_loss(
                    pred, targets_all[idx], quantiles)
            if contrastive_spec:
                loss, diag = hierarchical_contrastive_loss(
                    encoder, heads["projection"], values,
                    steps_array[idx].tolist(),
                    list(contrastive_spec["scales"]),
                    float(contrastive_spec["temperature"]),
                    int(contrastive_spec["negatives"][
                        "exclusion_steps"]))
                losses["contrastive"] = loss
                last_diagnostics["contrastive"] = diag
            if vol_spec:
                pred = heads["volatility"](encoder(values))
                losses["volatility"] = torch.nn.functional.mse_loss(
                    pred, vol_targets_all[idx])
            if barrier_spec:
                pred = heads["barrier"](encoder(values))
                losses["barrier"] = barrier_loss(
                    pred, barrier_labels_all[idx],
                    barrier_class_weights)
            return losses

        def monitor_report():
            with torch.no_grad():
                losses = objective_losses(monitor_windows, monitor_mask,
                                          monitor_idx)
                report = {k: round(float(v), 8)
                          for k, v in losses.items()}
                embedding = encoder(monitor_windows)
                report["representation_std"] = round(
                    float(embedding.std()), 6)
                if quant_spec:
                    pred = heads["quantile"](embedding)
                    report["quantile_crossing_rate"] = \
                        quantile_crossing_rate(pred)
                if contrastive_spec and "contrastive" in \
                        last_diagnostics:
                    report["contrastive_diagnostics"] = \
                        last_diagnostics["contrastive"]
            return report

        epoch0 = 0
        if ckpt is not None and bi == start_branch:
            encoder.load_state_dict(ckpt["encoder_state"])
            heads.load_state_dict(ckpt["heads_state"])
            opt.load_state_dict(ckpt["optimizer_state"])
            gen.set_state(ckpt["generator_state"])
            monitor_gen.set_state(ckpt["monitor_generator_state"])
            calibration_gen.set_state(
                ckpt["calibration_generator_state"])
            torch.set_rng_state(ckpt["torch_rng_state"])
            # the fixed masks above were drawn from FRESH generators;
            # the checkpointed masks are the branch's real ones
            monitor_mask = ckpt["monitor_mask"]
            calibration_mask = ckpt["calibration_mask"]
            effective = ckpt["effective_weights"]
            epoch0 = start_epoch
            ckpt = None
        else:
            # DATA-SOTA-349: weights calibrate ONCE on the CALIBRATION
            # partition; the monitor never calibrates anything.
            with torch.no_grad():
                initial = {k: float(v) for k, v in objective_losses(
                    calibration_windows, calibration_mask,
                    calibration_idx).items()}
            effective = balance_objective_weights(
                initial, declared_weights, parsed["balancing_floor"])
            manifest["progress"][spec["name"]]["effective_weights"] = {
                "initial_calibration_losses": {
                    k: round(v, 8) for k, v in initial.items()},
                "declared": declared_weights,
                "effective": {k: round(v, 8)
                              for k, v in effective.items()},
                "calibrated_on": "calibration partition only "
                                 "(DATA-SOTA-349)"}

        manifest["progress"][spec["name"]]["status"] = "training"
        for epoch in range(epoch0, epochs):
            t0 = time.perf_counter()
            perm = torch.randperm(n_train, generator=gen)
            sums = {k: 0.0 for k in declared_weights}
            sums["weighted_total"] = 0.0
            batches = 0
            for lo in range(0, n_train, batch_size):
                sel = train_idx[perm[lo:lo + batch_size].numpy()]
                values = branch_windows[sel]
                mask = sample_span_mask(values.shape[0], window,
                                        mask_ratio, mask_span, gen)
                losses = objective_losses(values, mask, sel)
                total = sum(effective[k] * v for k, v in losses.items())
                if not torch.isfinite(total):
                    raise PretrainContractError(
                        f"non-finite loss in {spec['name']} epoch "
                        f"{epoch}: typed run failure")
                opt.zero_grad()
                total.backward()
                opt.step()
                for k, v in losses.items():
                    sums[k] += float(v.detach())
                sums["weighted_total"] += float(total.detach())
                batches += 1
            # DATA-SOTA-345 diagnostics on the fixed monitor probe
            probe = monitor_windows[:batch_size]
            probe_mask = monitor_mask[:batch_size]
            grad_report = objective_gradient_diagnostics(
                encoder, objective_losses(probe, probe_mask,
                                          monitor_idx[:batch_size]))
            record = {"epoch": epoch,
                      "train": {k: round(v / batches, 8)
                                for k, v in sums.items()},
                      "monitor_fit_tail": monitor_report(),
                      "gradient_diagnostics": grad_report,
                      "seconds": round(time.perf_counter() - t0, 3)}
            manifest["progress"][spec["name"]]["losses"].append(record)
            generation += 1
            write_generation(
                out_dir,
                {"identity": identity, "branch_index": bi,
                 "epochs_done_in_branch": epoch + 1,
                 "encoder_state": encoder.state_dict(),
                 "heads_state": heads.state_dict(),
                 "optimizer_state": opt.state_dict(),
                 "generator_state": gen.get_state(),
                 "monitor_generator_state": monitor_gen.get_state(),
                 "calibration_generator_state":
                     calibration_gen.get_state(),
                 "monitor_mask": monitor_mask,
                 "calibration_mask": calibration_mask,
                 "torch_rng_state": torch.get_rng_state(),
                 "effective_weights": effective},
                manifest, generation)
            print(f"[{spec['name']}] epoch {epoch}: "
                  f"train {record['train']} | monitor "
                  f"{record['monitor_fit_tail']} "
                  f"{record['seconds']}s")
            epoch_generations_this_invocation += 1
            if (args.stop_after_epochs is not None
                    and epoch_generations_this_invocation
                    >= args.stop_after_epochs):
                print(f"INTERRUPTED after "
                      f"{epoch_generations_this_invocation} epoch "
                      f"generations; resume with --resume")
                return 0

        enc_file = out_dir / f"branch_{spec['name']}_encoder.pt"
        heads_file = out_dir / f"branch_{spec['name']}_heads.pt"
        enc_sha = save_module_fsync(encoder, enc_file)
        heads_sha = save_module_fsync(heads, heads_file)
        # DATA-SOTA-350: adapters are excluded from transfer — the
        # encoder artifact must share NO parameter key with the heads.
        overlap = set(encoder.state_dict()) & set(heads.state_dict())
        if overlap:
            raise PretrainContractError(
                f"objective adapters leak into the transferred "
                f"encoder: {sorted(overlap)}")
        manifest["progress"][spec["name"]]["status"] = "complete"
        manifest["artifacts"][spec["name"]] = {
            "encoder_file": enc_file.name,
            "encoder_sha256": enc_sha,
            "heads_file": heads_file.name,
            "heads_sha256": heads_sha,
            "adapters_excluded_from_transfer": True,
            "encoder_dim": int(dim),
            "parameters": int(sum(p.numel()
                                  for p in encoder.parameters())),
        }
        generation += 1
        write_generation(
            out_dir,
            {"identity": identity, "branch_index": bi,
             "epochs_done_in_branch": epochs,
             "encoder_state": encoder.state_dict(),
             "heads_state": heads.state_dict(),
             "optimizer_state": opt.state_dict(),
             "generator_state": gen.get_state(),
             "monitor_generator_state": monitor_gen.get_state(),
             "calibration_generator_state":
                 calibration_gen.get_state(),
             "monitor_mask": monitor_mask,
             "calibration_mask": calibration_mask,
             "torch_rng_state": torch.get_rng_state(),
             "effective_weights": effective},
            manifest, generation)

    manifest["completed"] = True
    generation += 1
    write_generation(out_dir,
                     {"identity": identity,
                      "branch_index": len(assignment),
                      "epochs_done_in_branch": 0,
                      "encoder_state": {}, "heads_state": {},
                      "optimizer_state": {},
                      "generator_state": gen.get_state(),
                      "monitor_generator_state":
                          monitor_gen.get_state(),
                      "calibration_generator_state":
                          calibration_gen.get_state(),
                      "monitor_mask": None, "calibration_mask": None,
                      "torch_rng_state": torch.get_rng_state(),
                      "effective_weights": {}},
                     manifest, generation)
    print(f"COMPLETE: {len(assignment)} branches -> "
          f"{repo_relative(out_dir)} "
          f"(transfer_eligibility: NOT_TRANSFER_ELIGIBLE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
