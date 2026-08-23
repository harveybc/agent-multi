"""P1 fixed-LR L1 curriculum experiment: N / EN-W / EN-F arms.

Orders: fixed-LR order §3, continuity amendment, and the 307-312
correction order (2026-08-23). FLAT-MLP experiment identity — the
grouped extractor is NEVER enabled here.

Data contract (finding 311): the VERIFIED NESTED ROLE MANIFEST only —
fit_train (through 2022) for training, train_monitor (2022) + inner_
validation (2023) for checkpoint/stopping via the paired hierarchical
comparator, outer_validation (2024) for the ONE post-selection
treatment endpoint, sealed_test (2025) STRUCTURALLY unmaterialized.
Day-based splits refuse.

PREDECLARED DIRECTION RULE (fixed before any terminal arm exists):
primary endpoint = the selected checkpoint's ONE post-selection
outer-2024 economic/risk score (risk-adjusted total return =
return − lambda*max_drawdown), treatment arm minus N, per seed —
NEVER the selection composite (finding 310). Activity eligibility is
reported separately; raw return, drawdown, Sharpe, trades and action
facts stay visible. FOR needs >=3/4 seeds positive with positive
median; AGAINST mirrored; else INCONCLUSIVE. EN-W and EN-F are
interpreted SEPARATELY — never merged. Four seeds are directional,
not conclusive.

Arms — transition-state contracts (the ONLY declared differences):

- N    : cold start; no bundle.
- EN-W : selected easy CHECKPOINT BUNDLE (findings 307/308/309):
         model tensors + optimizers + entropy from the selected epoch,
         verified by EXACT named-state hash map equality after load;
         FRESH normal replay.
- EN-F : same bundle, PLUS the replay snapshot FROM THE SAME SELECTED
         EPOCH (never the terminal buffer — selected and terminal
         state never mix).

Handoff (finding 307) consumes ONLY the immutable selected-checkpoint
manifest: eligibility, epoch, tensor digests, and >=2 mapped normal
decision crossings counted from the manifest's OWN inner-validation
trace snapshot (sha-verified), never a mutable trace path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PHASE_RUNNER = REPO / "tools" / "wp4_cpu_smoke.py"
ARMS = ("N", "EN-W", "EN-F")
MIN_NORMAL_CROSSINGS = 2
PAIRED_METRIC = "paired_generalization_weekly_v1"
RISK_LAMBDA = 1.0

# Finding 312: fields that MAY differ across the three arms of one
# seed. Everything else in the effective config must be identical.
ARM_FACTOR_ALLOWLIST = frozenset({
    "solvency_mode", "warm_start_bundle", "warm_start_model",
    "warm_start_model_sha256", "warm_start_replay_from_bundle",
    "save_model", "return_trace_dir", "output_dir",
    "checkpoint_bundle_dir", "nested_split_dir", "max_epochs",
    "env_mode", "nested_split_manifest", "_source_data_sha256",
})


class CurriculumError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_normal_crossings(trace_csv: Path, threshold: float) -> int:
    rows = list(csv.DictReader(trace_csv.open()))
    state, crossings = 0, 0
    for row in rows:
        raw = row.get("action_raw")
        if raw in (None, ""):
            continue
        value = float(raw)
        zone = 1 if value > threshold else (-1 if value < -threshold
                                            else 0)
        if zone != 0 and zone != state:
            if state != 0 or crossings == 0:
                crossings += 1
            state = zone
        elif zone == 0:
            state = 0
    return crossings


def load_bundle_manifest(easy_dir: Path) -> dict:
    path = easy_dir / "selected_bundle" / (
        "selected_checkpoint_manifest.json")
    if not path.is_file():
        raise CurriculumError(
            "no selected-checkpoint bundle manifest; handoff cannot "
            "be authorized from mutable paths (finding 307)")
    doc = json.loads(path.read_text())
    if doc.get("schema") != "agent_multi.selected_checkpoint_bundle.v1":
        raise CurriculumError("foreign bundle schema")
    doc["_manifest_path"] = str(path)
    return doc


def verify_handoff(easy_report: dict, bundle: dict,
                   threshold: float) -> dict:
    if not easy_report.get("accepted"):
        raise CurriculumError(
            "easy phase not accepted; no handoff. NOTE: economic "
            "negativity alone never rejects easy — acceptance is the "
            "activity/learning gate")
    model = Path(bundle["model"]["path"])
    if not model.is_file() or _sha(model) != bundle["model"]["sha256"]:
        raise CurriculumError("bundle model artifact missing or drifted")
    trace_entry = (bundle.get("traces") or {}).get("inner_validation")
    if not trace_entry:
        raise CurriculumError("bundle lacks the inner-validation "
                              "trace snapshot")
    trace = Path(trace_entry["path"])
    if not trace.is_file() or _sha(trace) != trace_entry["sha256"]:
        raise CurriculumError(
            "bundle inner-validation trace missing or drifted; the "
            "crossings gate only trusts the SELECTED epoch's snapshot")
    crossings = count_normal_crossings(trace, threshold)
    if crossings < MIN_NORMAL_CROSSINGS:
        raise CurriculumError(
            f"selected checkpoint shows {crossings} mapped normal "
            f"decision crossings; handoff requires >= "
            f"{MIN_NORMAL_CROSSINGS}")
    return {"bundle_manifest": bundle["_manifest_path"],
            "bundle_epoch": bundle["epoch"],
            "artifact_sha256": bundle["model"]["sha256"],
            "named_state_tensors": len(bundle["named_state_sha256"]),
            "replay_snapshot_transitions": bundle["replay"].get("size"),
            "validation_crossings": crossings}


def verify_continuity(normal_report: dict, handoff: dict,
                      arm: str) -> dict:
    disposition = normal_report.get("replay_disposition") or {}
    expected = ("selected_epoch_full_continuity" if arm == "EN-F"
                else "fresh")
    if disposition.get("mode") != expected:
        raise CurriculumError(
            f"replay disposition {disposition.get('mode')!r} does not "
            f"match the declared {arm} semantics ({expected!r})")
    if disposition.get("bundle_epoch") != handoff["bundle_epoch"]:
        raise CurriculumError(
            "normal phase loaded a bundle from a different epoch than "
            "the handoff authorized — selected/terminal state mixing "
            "refused (finding 308)")
    verification = disposition.get("state_verification") or {}
    if verification.get("exact") is not True:
        raise CurriculumError(
            "loaded runtime state was not verified exact against the "
            "bundle's named-tensor map (finding 309)")
    if arm == "EN-F" and disposition.get("loaded_transitions", 0) <= 0:
        raise CurriculumError("EN-F loaded an empty replay buffer")
    return {"replay_disposition": disposition,
            "named_state_verified_exact": True,
            "tensors_verified": verification.get("tensors_verified")}


def _phase_cmd(*, device, seed, epoch_timesteps, max_epochs, patience,
               patience_start, nested_contract, out_dir, report,
               solvency, bundle=None, replay_from_bundle=False,
               buffer_size=None):
    cmd = [sys.executable, str(PHASE_RUNNER),
           "--device", device, "--seed", str(seed),
           "--epoch-timesteps", str(epoch_timesteps),
           "--max-epochs", str(max_epochs),
           "--l1-patience", str(patience),
           "--l1-patience-start-epoch", str(patience_start),
           "--nested-contract", str(nested_contract),
           "--solvency-mode", solvency,
           "--selection-metric", PAIRED_METRIC,
           "--output-dir", str(out_dir), "--report", str(report)]
    if buffer_size:
        cmd += ["--buffer-size", str(buffer_size)]
    if bundle:
        cmd += ["--warm-start-bundle", str(bundle)]
    if replay_from_bundle:
        cmd += ["--warm-start-replay-from-bundle"]
    return cmd


def outer_endpoint(normal_dir: Path, normal_report: dict) -> dict:
    """Finding 310: ONE post-selection outer-2024 evaluation of the
    selected checkpoint through the executing evaluation path. Runs
    strictly AFTER the phase terminated — outer facts cannot reach
    checkpoint, stopping or configuration retroactively."""
    manifest_path = Path(str(normal_dir / "launch_manifest_missing"))
    launch = normal_dir.parent / (normal_dir.name + "_report"
                                  ".launch_manifest.json")
    if not launch.is_file():
        # driver layout: report sits beside output dir
        candidates = list(normal_dir.parent.glob(
            "*_report.launch_manifest.json"))
        raise CurriculumError(
            f"normal phase launch manifest not found near "
            f"{normal_dir}; candidates={candidates}")
    effective = json.loads(launch.read_text())["effective_config"]
    nested_manifest = normal_dir / "nested_splits" / (
        "nested_split_manifest.json")
    if not nested_manifest.is_file():
        raise CurriculumError("nested split manifest missing; the "
                              "outer role cannot be verified")
    roles = json.loads(nested_manifest.read_text()).get("roles") or {}
    outer = roles.get("outer_validation") or {}
    if outer.get("status") != "MATERIALIZED":
        raise CurriculumError("outer_validation role not materialized")
    sealed = roles.get("sealed_test") or {}
    if sealed.get("status") == "MATERIALIZED":
        raise CurriculumError(
            "sealed_test is materialized in a P1 run — REFUSED "
            "(order §5: sealed 2025 is never read, traced or reported)")
    bundle = load_bundle_manifest(normal_dir)
    sys.path.insert(0, str(REPO))
    from importlib.metadata import entry_points
    from pipeline_plugins.rl_pipeline_with_validation import (
        PipelinePlugin,
    )
    config = {k: v for k, v in effective.items()}
    config["nested_split_manifest"] = str(nested_manifest)
    config["return_trace_dir"] = str(normal_dir / "outer_trace")
    plugin = PipelinePlugin(config)
    agent_ep = next(e for e in entry_points().select(
        group="agent.plugins")
        if e.name == config["agent_plugin"])
    agent = agent_ep.load()(config)
    from stable_baselines3 import SAC
    model = SAC.load(bundle["model"]["path"], device="auto")
    summary = plugin._eval_on_split(
        str(config["env_plugin"]), config, str(outer["csv"]), agent,
        model, int(config.get("eval_seed", 0)), "outer_validation")
    ret = float(summary.get("total_return") or 0.0)
    dd = float(summary.get("max_drawdown_fraction") or 0.0)
    return {"role": "outer_validation_2024",
            "role_rows": outer.get("rows"),
            "csv_sha256": outer.get("sha256"),
            "selected_bundle_epoch": bundle["epoch"],
            "primary_score_risk_adjusted_return": ret
            - RISK_LAMBDA * dd,
            "risk_lambda": RISK_LAMBDA,
            "raw": {"total_return": ret,
                    "max_drawdown_fraction": dd,
                    "sharpe_ratio": summary.get("sharpe_ratio"),
                    "trades_total": summary.get("trades_total"),
                    "exposure": summary.get("exposure_fraction"),
                    "action_non_hold_rate": summary.get(
                        "action_non_hold_rate")},
            "activity_eligibility_reported_separately": {
                "trades_total": summary.get("trades_total")},
            "evaluated_after_phase_terminal": True}


def arm_contracts(effective: dict, arm: str) -> dict:
    """Finding 312: canonical contracts. pair fields must be identical
    across arms; only the allowlisted factors may differ."""
    pair = {k: v for k, v in sorted(effective.items())
            if k not in ARM_FACTOR_ALLOWLIST}
    return {
        "pair_contract_sha256": hashlib.sha256(json.dumps(
            pair, sort_keys=True, default=str).encode()).hexdigest(),
        "pair_contract": pair,
        "arm_contract": {"arm": arm},
        "transition_state_contract": {
            "N": {"start": "cold", "replay": "empty"},
            "EN-W": {"start": "selected_bundle_model",
                     "replay": "fresh",
                     "verification": "exact_named_state_map"},
            "EN-F": {"start": "selected_bundle_model",
                     "replay": "selected_epoch_snapshot",
                     "verification": "exact_named_state_map"},
        }[arm],
    }


def verify_arm_identity(records: list) -> None:
    shas = {r["contracts"]["pair_contract_sha256"] for r in records}
    if len(shas) != 1:
        raise CurriculumError(
            "arm identity violated: pair contracts differ beyond the "
            f"declared factor allowlist ({len(shas)} distinct)")


def run_arm(args) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    common = dict(device=args.device, seed=args.seed,
                  epoch_timesteps=args.epoch_timesteps,
                  patience=args.l1_patience,
                  patience_start=args.l1_patience_start_epoch,
                  nested_contract=args.nested_contract,
                  buffer_size=args.buffer_size)
    record = {"schema": "agent_multi.l1_curriculum_arm.v2",
              "arm": args.arm, "seed": args.seed,
              "model_contract": "flat_mlp (feature_extractor_plugin "
                                "NEVER set)",
              "predeclared_rule": __doc__.split(
                  "PREDECLARED DIRECTION RULE")[1].split("Arms —")[0]}
    handoff = None
    if args.arm in ("EN-W", "EN-F"):
        easy_dir = out / "easy"
        easy_report_path = out / "easy_report.json"
        cmd = _phase_cmd(**common, max_epochs=args.easy_max_epochs,
                         solvency="easy_chronological_continuation",
                         out_dir=easy_dir, report=easy_report_path)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        record["easy_phase"] = {"cmd": cmd[1:], "exit": proc.returncode}
        if proc.returncode != 0:
            record["outcome"] = "EASY_PHASE_FAILED"
            record["stderr_tail"] = proc.stderr[-800:]
            return record
        easy_report = json.loads(easy_report_path.read_text())
        bundle = load_bundle_manifest(easy_dir)
        handoff = verify_handoff(easy_report, bundle,
                                 args.action_threshold)
        record["handoff"] = handoff
        record["easy_compute"] = {
            "epochs": easy_report.get("epochs_run"),
            "elapsed_seconds": easy_report.get("elapsed_seconds"),
            "note": "reported separately; never truncated into the "
                    "normal budget"}
        warm = dict(bundle=bundle["_manifest_path"],
                    replay_from_bundle=(args.arm == "EN-F"))
    else:
        warm = {}
    normal_dir = out / "normal"
    normal_report_path = out / "normal_report.json"
    cmd = _phase_cmd(**common, max_epochs=args.max_epochs,
                     solvency="normal_realistic", out_dir=normal_dir,
                     report=normal_report_path, **warm)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    record["normal_phase"] = {"cmd": cmd[1:], "exit": proc.returncode}
    if proc.returncode != 0:
        record["outcome"] = "NORMAL_PHASE_FAILED"
        record["stderr_tail"] = proc.stderr[-800:]
        return record
    normal_report = json.loads(normal_report_path.read_text())
    if handoff is not None:
        record["continuity"] = verify_continuity(
            normal_report, handoff, args.arm)
    else:
        record["continuity"] = {
            "replay_disposition": normal_report.get(
                "replay_disposition")}
    launch = json.loads(
        normal_report_path.with_suffix(".launch_manifest.json")
        .read_text())
    record["contracts"] = arm_contracts(
        launch["effective_config"], args.arm)
    record["outer_endpoint"] = outer_endpoint(normal_dir,
                                              normal_report)
    record["normal_accepted"] = normal_report.get("accepted")
    record["outcome"] = ("ARM_COMPLETE" if normal_report.get("accepted")
                         else "NORMAL_NOT_ACCEPTED")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cpu")
    parser.add_argument("--epoch-timesteps", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--easy-max-epochs", type=int, required=True)
    parser.add_argument("--l1-patience", type=int, required=True)
    parser.add_argument("--l1-patience-start-epoch", type=int,
                        required=True)
    parser.add_argument("--nested-contract", type=Path, required=True,
                        help="finding 311: the ONLY data contract; "
                             "day splits refuse structurally (this "
                             "driver has no day flags)")
    parser.add_argument("--action-threshold", type=float, default=0.0)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = run_arm(args)
    except CurriculumError as exc:
        record = {"schema": "agent_multi.l1_curriculum_arm.v2",
                  "arm": args.arm, "seed": args.seed,
                  "outcome": "REFUSED", "reason": str(exc)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps({"arm": args.arm, "seed": args.seed,
                      "outcome": record["outcome"]}))
    return 0 if record["outcome"] == "ARM_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
