#!/usr/bin/env python3
"""WP4 bounded smoke of the REAL path (correction order 2026-08-20).

Strict CLI; requested vs effective device with CUDA assertion; learning
and checkpoint acceptance built in; complete hashed facts; typed
negative results. Never launches anything beyond its own bounded run.
"""
from __future__ import annotations

import argparse
import csv as _csv
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib.metadata import entry_points
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PINNED_GYMFX = Path("/home/harveybc/Documents/GitHub/.runtime/"
                    "gym-fx-p1lr-634c3fd3")
EXPECTED_COMMIT = "634c3fd3c344cae3c4048b334158185c8bf4e1ef"
DATA = Path("/home/harveybc/Documents/GitHub/predictor/examples/data/"
            "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
_EXCLUDE = {"DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_env_origin() -> dict:
    ep = next(e for e in entry_points().select(group="env.plugins")
              if e.name == "gym_fx_env")
    wrapper = ep.load()
    wrapper_file = str(Path(
        sys.modules[wrapper.__module__].__file__).resolve())
    import importlib
    env_mod = importlib.import_module("app.env")
    origin = str(Path(env_mod.__file__).resolve())
    commit = subprocess.run(
        ["git", "-C", str(PINNED_GYMFX), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    if not origin.startswith(str(PINNED_GYMFX)):
        raise RuntimeError(f"REFUSED_ENV_ORIGIN: {origin}")
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"REFUSED_ENV_COMMIT: {commit}")
    return {"wrapper_file": wrapper_file,
            "implementation_module": "app.env", "file": origin,
            "pinned_root": str(PINNED_GYMFX), "commit": commit}


def resolve_device(requested: str) -> dict:
    """Correction 2: requested vs EFFECTIVE device; CUDA must actually
    be selected when requested."""
    import torch
    facts = {"requested": requested,
             "cuda_visible_devices":
                 os.environ.get("CUDA_VISIBLE_DEVICES")}
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "REFUSED_CUDA_UNAVAILABLE: --device cuda was requested "
                "but torch reports no CUDA device; refusing a silent "
                "CPU run (WP4-A)")
        facts["effective"] = "cuda"
        facts["torch_device_name"] = torch.cuda.get_device_name(0)
        # Replication defect 2026-08-21 (seed 404): nvidia-smi
        # enumerates the HOST, not the CUDA_VISIBLE_DEVICES mask, so
        # index 0 here misattributed the 5070 Ti UUID to a run that
        # torch placed on the 5090. When the mask itself names a
        # GPU-UUID that IS the effective device; only an unmasked run
        # may fall back to host enumeration, and then only if it is
        # unambiguous (a single GPU).
        cvd = facts["cuda_visible_devices"]
        if cvd and cvd.startswith("GPU-") and "," not in cvd:
            facts["gpu_uuid"] = cvd
            facts["gpu_uuid_provenance"] = "cuda_visible_devices_mask"
        else:
            uuid = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid",
                 "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip().splitlines()
            facts["gpu_uuid"] = (
                uuid[0] if len(uuid) == 1 else None)
            facts["gpu_uuid_provenance"] = (
                "host_single_gpu" if len(uuid) == 1
                else "ambiguous_multi_gpu_unresolved")
    else:
        facts["effective"] = "cpu"
        facts["gpu_uuid"] = None
    return facts


def build_config(args, features) -> dict:
    return {
        "input_data_file": str(DATA), "env_plugin": "gym_fx_env",
        "agent_plugin": "project3_sac_actor_critic_agent",
        "quiet_mode": True,
        "train_days": 120, "val_days": 40, "test_days": 40,
        "min_split_rows": 100,
        "epoch_timesteps": args.epoch_timesteps,
        "max_epochs": args.max_epochs,
        # MUSASHI_CORRECTION_SMOKE_PATIENCE_WAS_UNAUTHORIZED_2026_08_21:
        # stopping semantics are NEVER derived from the runtime budget.
        # Both values are explicit CLI facts; argparse refuses their
        # absence. Requested == effective by construction and both are
        # persisted with provenance in the report.
        "l1_patience": args.l1_patience,
        "l1_patience_start_epoch": args.l1_patience_start_epoch,
        "l1_min_delta": 1e-6,
        "window_size": 32, "initial_cash": 10000.0,
        "action_space_mode": "continuous",
        "continuous_action_threshold": 0.0,
        "solvency_mode": "normal_realistic",
        "require_feature_aware_preprocessor": True,
        "include_price_window": False,
        "preprocessor_plugin": "feature_window_preprocessor",
        "feature_columns": features,
        "feature_scaling": "rolling_zscore",
        "feature_scaling_window": 256,
        # Order 2026-08-21 §3/§4: the smoke can run under either the
        # episodic contract (default) or the easy checkpoint monitor —
        # the matching fail-closed guard is asserted for whichever is
        # chosen, and the plateau-LR contract (optional) requires the
        # monitor metric.
        "selection_metric": args.selection_metric,
        "require_episodic_fitness": (
            args.selection_metric == "episodic_activity_economic_v1"),
        "require_easy_contracts": (
            args.selection_metric == "easy_checkpoint_monitor_v1"),
        **({"plateau_lr": json.loads(args.plateau_lr_json)}
           if args.plateau_lr_json else {}),
        "episodic_activity_fitness": {
            "activity_plateau_low_rate": 50.0,
            "activity_plateau_high_rate": 300.0},
        # the pipeline consumes these at TOP level (config.get), not
        # inside a sac_params dict — runtime finding follow-up
        "learning_rate": 3e-4, "batch_size": 64,
        "learning_starts": 128, "device": args.device,
        # correction 6: model artifacts land under --output-dir, never
        # the repo root
        "save_model": str(args.output_dir / "best_model.zip"),
        "seed": args.seed,
        "return_trace_dir": str(args.output_dir / "traces"),
        "output_dir": str(args.output_dir),
        "inactive_terminal_is_typed_result": True,
    }


def facts_from(history, trace_dir: Path) -> dict:
    """Correction: complete per-split facts, hash-bound to their traces."""
    last = history[-1] if history else {}
    traces = {}
    # AUD-F1-20260821-PLR-04: the 40-day holdout is reported as
    # "diagnostic_holdout", never bare "test" — it is NOT the sealed
    # 2025 test and repeated inspection under that name invites
    # adaptation. The source filename (written by the pipeline) is
    # retained verbatim in the descriptor's "file" field.
    for name in ("train_epoch", "train_tail_epoch", "validation_epoch",
                 "evaluation", "test"):
        report_key = "diagnostic_holdout" if name == "test" else name
        path = trace_dir / f"{name}_return_trace.csv"
        if path.is_file():
            rows = list(_csv.DictReader(path.open()))
            stamps = [r.get("timestamp", "") for r in rows]
            traces[report_key] = {
                "file": str(path), "sha256": _sha_file(path),
                "rows": len(rows),
                "closed_trades_cumulative": (
                    rows[-1].get("closed_trades_cumulative")
                    if rows else None),
                "split_label": rows[0].get("split") if rows else None,
                "first_timestamp": stamps[0] if stamps else None,
                "last_timestamp": stamps[-1] if stamps else None,
                "distinct_actions": len({r.get("action_raw")
                                         for r in rows}),
            }
    return {"last_epoch": {k: last.get(k) for k in (
                "epoch", "composite", "composite_raw",
                "l1_checkpoint_eligible", "early_stop_trade_gate_passed",
                "policy_actor_delta", "policy_critic_delta",
                "gradient_updates_total", "train_tail_trades",
                "val_trades") if isinstance(last, dict)},
            "traces": traces}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cpu")
    parser.add_argument("--epoch-timesteps", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "docs/audits/evidence/wp4_smoke")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--l1-patience", type=int, required=True,
        help="explicit early-stop patience; never derived from "
             "--max-epochs (correction 2026-08-21)")
    parser.add_argument(
        "--l1-patience-start-epoch", type=int, required=True,
        help="explicit epoch before which patience never counts; "
             "never derived from --max-epochs (correction 2026-08-21)")
    parser.add_argument(
        "--selection-metric",
        choices=["episodic_activity_economic_v1",
                 "easy_checkpoint_monitor_v1"],
        default="episodic_activity_economic_v1")
    parser.add_argument(
        "--plateau-lr-json", default=None,
        help="explicit plateau-LR contract as JSON (factor, lr_patience, "
             "min_lr, threshold, cooldown[, start_epoch]); requires "
             "--selection-metric easy_checkpoint_monitor_v1")
    parser.add_argument("--preflight", action="store_true",
                        help="non-mutating: assert env origin, device "
                             "and data hash, then exit")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(PINNED_GYMFX))
    origin = assert_env_origin()
    device = resolve_device(args.device)
    data_sha = _sha_file(DATA)
    if args.preflight:
        print(json.dumps({"outcome": "PREFLIGHT_OK",
                          "env_origin": origin, "device": device,
                          "data_sha256": data_sha}, indent=1))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = [c for c in next(_csv.reader(DATA.open()))
                if c not in _EXCLUDE]
    config = build_config(args, features)
    config_sha = hashlib.sha256(json.dumps(
        config, sort_keys=True, default=str).encode()).hexdigest()
    # Dispatch order 2026-08-22: config-minus-treatment identity hash,
    # computed AT MATERIALIZATION TIME — the pair-identity fact that
    # two arms are the same experiment except the scheduler treatment.
    pair_config_sha = hashlib.sha256(json.dumps(
        {k: v for k, v in config.items() if k != "plateau_lr"},
        sort_keys=True, default=str).encode()).hexdigest()
    pair_contract_doc = {
        "seed": args.seed, "data_sha256": data_sha,
        "epoch_timesteps": args.epoch_timesteps,
        "max_epochs": args.max_epochs,
        "l1_patience": args.l1_patience,
        "l1_patience_start_epoch": args.l1_patience_start_epoch,
        "selection_metric": args.selection_metric,
        "train_days": 120, "val_days": 40, "test_days": 40,
        "device_mask": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env_origin": origin, "learning_rate_initial": 3e-4,
        "pair_config_sha256": pair_config_sha,
    }
    arm_contract_doc = {
        "scheduler_policy": ("plateau" if args.plateau_lr_json
                             else "fixed"),
        "plateau_spec": (json.loads(args.plateau_lr_json)
                         if args.plateau_lr_json else None),
    }
    # Persist the launch manifest BEFORE training; an arm without it
    # refuses aggregation (durable pre-launch identity).
    if args.report is not None:
        launch_manifest = Path(str(args.report)).with_suffix(
            ".launch_manifest.json")
        launch_manifest.parent.mkdir(parents=True, exist_ok=True)
        launch_manifest.write_text(json.dumps({
            "schema": "agent_multi.wp4_smoke_launch.v1",
            "effective_config": {k: str(v) if not isinstance(
                v, (int, float, bool, str, list, dict, type(None)))
                else v for k, v in config.items()},
            "pair_contract": pair_contract_doc,
            "arm_contract": arm_contract_doc,
            "config_sha256": config_sha,
            "pair_config_sha256": pair_config_sha,
            "commit": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                capture_output=True, text=True).stdout.strip(),
            "data_sha256": data_sha,
            "argv": sys.argv,
        }, indent=1, default=str))

    from pipeline_plugins.rl_pipeline_with_validation import (
        PipelinePlugin,
    )
    agent_ep = next(e for e in entry_points().select(
        group="agent.plugins")
        if e.name == config["agent_plugin"])
    started = time.time()
    result = PipelinePlugin(config).run_pipeline(
        config=config, env_plugin=None,
        agent_plugin=agent_ep.load()(), mode="train")
    elapsed = time.time() - started

    history = result.get("history") or []
    trace_dir = Path(config["return_trace_dir"])
    detail = facts_from(history, trace_dir)
    actor_moved = sum(abs(float(row.get("policy_actor_delta") or 0))
                      for row in history)
    distinct = max((t.get("distinct_actions") or 0
                    for t in detail["traces"].values()), default=0)
    activity = any(
        float(t.get("closed_trades_cumulative") or 0) > 0
        for t in detail["traces"].values())
    episodic_path = any(row.get("composite_raw") is not None
                        for row in history)
    # TR-L2: the boundary proof requires the ACTUAL diagnostic test
    # trace — NO fallback to train/validation; absent -> acceptance
    # refuses.
    test_trace = detail["traces"].get("diagnostic_holdout") or {}
    best = result.get("best_model_path")
    eligible = bool(best) and Path(str(best)).is_file()

    sealed_ok = (
        (test_trace.get("last_timestamp") or "")[:4] < "2025"
        if test_trace.get("last_timestamp") else False)
    accepted = (actor_moved > 0 and distinct >= 10 and activity
                and episodic_path and eligible and sealed_ok)
    negative = None
    if not accepted:
        negative = {
            "type": "TYPED_NEGATIVE_SMOKE",
            "actor_parameters_moved": actor_moved > 0,
            "distinct_actions_ge_10": distinct >= 10,
            "real_split_activity": activity,
            "episodic_call_path": episodic_path,
            "eligible_checkpoint_selected": eligible,
            "sealed_proof_available": sealed_ok,
            "failed_evidence": {
                "last_epoch_gates": detail["last_epoch"],
                "trace_descriptors": {
                    k: {kk: v.get(kk) for kk in
                        ("file", "sha256", "closed_trades_cumulative")}
                    for k, v in detail["traces"].items()},
            },
            "promotion": "REFUSED",
        }


    sealed_proof = {
        "label": "diagnostic_holdout_120_40_40",
        "not_the_sealed_2025_test": True,
        "influences_selection": False,
        "note": ("selection consumes train_tail/validation only "
                 "(_early_stop_composite); the internal test table is "
                 "display-only in this pipeline"),
        "max_timestamp": test_trace.get("last_timestamp"),
        "sealed_2025_untouched": sealed_ok,
        "split_label": test_trace.get("split_label"),
        "first_timestamp": test_trace.get("first_timestamp"),
        "last_timestamp": test_trace.get("last_timestamp"),
        "sha256": test_trace.get("sha256"),
        "contains_heldout_rows": bool(test_trace.get("rows")),
        "proof_basis": ("actual test trace timestamps" if sealed_ok
                        else "REFUSED_NO_TEST_TRACE — acceptance "
                        "refused (TR-L2: no fallback to train or "
                        "validation)"),
        "selection_firewall": (
            "the executing stopping path (_early_stop_composite) "
            "consumes ONLY train_tail_summary and val_summary; the "
            "test split is written at final evaluation and never "
            "enters checkpoint or stopping state (pinned by "
            "test_wp3_episodic_wiring firewall test)"),
    }

    report = {
        "schema": "agent_multi.wp4_smoke.v2",
        "commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "config_sha256": config_sha,
        "data_sha256": data_sha,
        "env_origin": origin,
        "device": device,
        "budgets": {"epoch_timesteps": args.epoch_timesteps,
                    "max_epochs": args.max_epochs, "seed": args.seed},
        # Correction 2026-08-21: stopping semantics are explicit CLI
        # facts with provenance — never derived from max_epochs.
        "stopping_contract": {
            "l1_patience": {
                "requested": args.l1_patience,
                "effective": args.l1_patience,
                "provenance": "cli_explicit_required"},
            "l1_patience_start_epoch": {
                "requested": args.l1_patience_start_epoch,
                "effective": args.l1_patience_start_epoch,
                "provenance": "cli_explicit_required"},
            # AUD-F1-20260821-PLR-02: this tool's data contract is a
            # fixed 120/40/40-day window; no stopping contract makes it
            # long-horizon. The strongest truthful label is a bounded
            # scheduler mechanism screen.
            "classification": "MECHANICS_RANK_DIAGNOSTIC_ONLY"
            if args.l1_patience < 60 or args.max_epochs < 2000
            else "BOUNDED_120_40_40_DAY_SCHEDULER_SCREEN"},
        "data_horizon": {"train_days": 120, "val_days": 40,
                         "test_days": 40,
                         "note": ("bounded mechanism screen only; no "
                                  "claim about the multi-year easy "
                                  "curriculum (PLR-02)")},
        # AUD-F1-20260821-PLR-06: canonical pair/arm identity. The
        # aggregator requires exact equality of every pair_contract
        # field across a seed's two arms and permits the arm_contract
        # to differ ONLY as predeclared (fixed vs exact plateau spec).
        "pair_contract": pair_contract_doc,
        "arm_contract": arm_contract_doc,
        "pair_config_sha256": pair_config_sha,
        "elapsed_seconds": round(elapsed, 1),
        "epochs_run": len(history),
        "stop_reason": result.get("stop_reason"),
        "no_eligible_checkpoint": result.get(
            "activity_stopped_without_eligible_checkpoint"),
        "selected_checkpoint": best,
        "selected_checkpoint_sha256": (
            _sha_file(Path(str(best))) if eligible else None),
        "learning": {"actor_delta_sum": actor_moved,
                     "distinct_actions_max": distinct,
                     "gradient_updates": (history[-1].get(
                         "gradient_updates_total")
                         if history else None)},
        "split_facts": detail,
        "diagnostic_holdout": sealed_proof,
        "accepted": accepted,
        "typed_negative": negative,
        # WP1 2026-08-21: the FULL per-epoch history is durable
        # evidence — the 22-epoch GPU run could not be reconstructed
        # because this was missing; never again.
        "history": history,
        "model_artifacts_not_committed": {
            "policy": ("binaries stay OUT of git; hashes are the "
                       "evidence (audit 2026-08-20: duplicate 33.4MB "
                       "models removed)"),
            "best_sha256": (_sha_file(Path(str(best)))
                            if eligible else None),
        },
    }
    out = args.report or (REPO / "docs/audits/evidence/"
                          "WP4_CPU_SMOKE_REPORT_2026_08_20.json")
    out.write_text(json.dumps(report, indent=1, sort_keys=True,
                              default=str) + "\n")
    print(json.dumps({"accepted": accepted,
                      "epochs": len(history),
                      "elapsed": round(elapsed, 1),
                      "device": device["effective"],
                      "negative": bool(negative)}, default=str))
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
