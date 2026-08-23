"""P1 fixed-LR L1 curriculum experiment: N / EN-W / EN-F arms.

Orders: MUSASHI_TO_GENERAL_SATOSHI_EARLY_SCREEN_ACCEPTANCE_MT5_USDCAD_AND_FIXED_LR_ORDER_2026_08_23 §3
+ MT5_P0_CORRECTION_AND_P1_CONTINUITY_ORDER (three-arm state-factor
amendment). FLAT-MLP experiment identity — the grouped extractor is
NEVER enabled here.

PREDECLARED DIRECTION RULE (fixed before any terminal arm exists):
primary endpoint = paired NORMAL-PHASE best eligible monitor score,
treatment arm minus N, per seed. Direction FOR a treatment needs
>=3/4 seeds positive with positive median; AGAINST mirrored; else
INCONCLUSIVE. EN-W and EN-F are interpreted SEPARATELY — never merged
into one "easy pretraining" claim. Four seeds are a directional
screen, not statistical proof.

Arms (state factors are the ONLY declared differences):

- N    : normal-only, declared cold start.
- EN-W : easy phase, then normal warm-started from the SELECTED easy
         checkpoint artifact — actor, twin critics, target critics,
         entropy state and their optimizers restored by the artifact
         load; FRESH normal replay buffer (loaded_transitions == 0).
- EN-F : EN-W plus full replay continuity — the EASY TERMINAL replay
         buffer (declared: checkpoints do not snapshot replay) is
         sha-bound and loaded before the first normal update.

Both phases share seed, data, fixed LR 3e-4, stopping contract
(max 2000 epochs/phase, patience 60 inactive before epoch 40),
episodic activity/economic objective, identical action semantics and
evaluation roles. Sealed/outer data never influence stopping,
selection or configuration. The easy phase may be economically
negative while learning activity; it is never rejected for negative
profit alone. Handoff requires an ELIGIBLE easy checkpoint plus at
least two mapped normal decision crossings in its validation trace.
Easy-phase compute is reported separately — no equal-wall-clock
truncation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PHASE_RUNNER = REPO / "tools" / "wp4_cpu_smoke.py"
ARMS = ("N", "EN-W", "EN-F")
MIN_NORMAL_CROSSINGS = 2


class CurriculumError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member_hashes(artifact: Path) -> dict:
    """Component identity of an SB3 zip artifact: sha256 per member.

    policy.pth carries actor+critics+targets, *.optimizer.pth the
    optimizer states, pytorch_variables.pth the log entropy
    coefficient — byte identity of the members IS tensor identity."""
    out = {}
    with zipfile.ZipFile(artifact) as zf:
        for name in sorted(zf.namelist()):
            out[name] = hashlib.sha256(zf.read(name)).hexdigest()
    return out


def _phase_cmd(*, device, seed, epoch_timesteps, max_epochs, patience,
               patience_start, days, solvency, out_dir, report,
               warm_model=None, warm_model_sha=None, warm_replay=None,
               warm_replay_sha=None, save_replay=None):
    cmd = [sys.executable, str(PHASE_RUNNER),
           "--device", device, "--seed", str(seed),
           "--epoch-timesteps", str(epoch_timesteps),
           "--max-epochs", str(max_epochs),
           "--l1-patience", str(patience),
           "--l1-patience-start-epoch", str(patience_start),
           "--train-days", str(days[0]), "--val-days", str(days[1]),
           "--test-days", str(days[2]),
           "--solvency-mode", solvency,
           "--selection-metric", "episodic_activity_economic_v1",
           "--output-dir", str(out_dir), "--report", str(report)]
    if warm_model:
        cmd += ["--warm-start-model", str(warm_model),
                "--warm-start-model-sha256", warm_model_sha]
    if warm_replay:
        cmd += ["--warm-start-replay-buffer", str(warm_replay),
                "--warm-start-replay-buffer-sha256", warm_replay_sha]
    if save_replay:
        cmd += ["--save-replay-buffer", str(save_replay)]
    return cmd


def count_normal_crossings(trace_csv: Path, threshold: float) -> int:
    """Mapped normal decision crossings in the easy checkpoint's
    validation trace: transitions of the raw action across the normal
    threshold band (executing data, not a synthetic estimate)."""
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


def verify_handoff(easy_report: dict, easy_dir: Path,
                   threshold: float) -> dict:
    if not easy_report.get("accepted"):
        raise CurriculumError(
            "easy phase not accepted; no handoff. NOTE: economic "
            "negativity alone never rejects easy — acceptance is the "
            "activity/learning gate")
    best = easy_report.get("selected_checkpoint")
    if not best or not Path(str(best)).is_file():
        raise CurriculumError("no eligible easy checkpoint artifact")
    trace = easy_dir / "traces" / "validation_epoch_return_trace.csv"
    if not trace.is_file():
        raise CurriculumError("easy validation trace missing; "
                              "crossings unprovable")
    crossings = count_normal_crossings(trace, threshold)
    if crossings < MIN_NORMAL_CROSSINGS:
        raise CurriculumError(
            f"easy checkpoint shows {crossings} mapped normal decision "
            f"crossings; handoff requires >= {MIN_NORMAL_CROSSINGS}")
    return {"artifact": str(best), "artifact_sha256": _sha(Path(best)),
            "component_sha256": _member_hashes(Path(best)),
            "validation_crossings": crossings}


def verify_continuity(easy_report: dict, normal_report: dict,
                      handoff: dict, arm: str) -> dict:
    hist = normal_report.get("history") or []
    if not hist:
        raise CurriculumError("normal phase has no history")
    # A warm-started run prepends a baseline-evaluation row (the easy
    # checkpoint re-scored under normal semantics) that carries no
    # policy-L1 facts; continuity is proven on the first TRAINING
    # epoch's before-update checksum.
    first = next((r for r in hist
                  if r.get("policy_actor_l1_before") is not None),
                 None)
    if first is None:
        raise CurriculumError("continuity facts missing from history")
    sel_epoch = None
    for row in easy_report.get("history") or []:
        if row.get("checkpoint_improved"):
            sel_epoch = row
    easy_l1 = (sel_epoch or {}).get("policy_actor_l1_after")
    normal_l1 = first.get("policy_actor_l1_before")
    if easy_l1 is None or normal_l1 is None:
        raise CurriculumError("continuity facts missing from history")
    if abs(float(easy_l1) - float(normal_l1)) > 1e-6:
        raise CurriculumError(
            f"tensor continuity broken: easy selected actor L1 "
            f"{easy_l1} vs normal initial {normal_l1}")
    disposition = (normal_report.get("replay_disposition")
                   or {})
    expected = "full_continuity" if arm == "EN-F" else "fresh"
    if disposition.get("mode") != expected:
        raise CurriculumError(
            f"replay disposition {disposition!r} does not match the "
            f"declared {arm} semantics ({expected})")
    if arm == "EN-F" and disposition.get("loaded_transitions", 0) <= 0:
        raise CurriculumError("EN-F loaded an empty replay buffer")
    return {"actor_l1_continuity": {"easy": easy_l1,
                                    "normal_initial": normal_l1,
                                    "identical": True},
            "artifact_sha256": handoff["artifact_sha256"],
            "replay_disposition": disposition,
            "entropy_and_optimizers":
                "restored by SB3 artifact load (members hashed in "
                "handoff.component_sha256: policy.pth, "
                "*.optimizer.pth, pytorch_variables.pth)"}


def run_arm(args) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    days = (args.train_days, args.val_days, args.test_days)
    common = dict(device=args.device, seed=args.seed,
                  epoch_timesteps=args.epoch_timesteps,
                  patience=args.l1_patience,
                  patience_start=args.l1_patience_start_epoch,
                  days=days)
    record = {"schema": "agent_multi.l1_curriculum_arm.v1",
              "arm": args.arm, "seed": args.seed,
              "model_contract": "flat_mlp (feature_extractor_plugin "
                                "NEVER set; grouped extractor is a "
                                "separate experiment identity)",
              "predeclared_rule": __doc__.split(
                  "PREDECLARED DIRECTION RULE")[1].split("\n\n")[0]}
    if args.arm in ("EN-W", "EN-F"):
        easy_dir = out / "easy"
        easy_report_path = out / "easy_report.json"
        replay_path = (out / "easy_terminal_replay.pkl"
                       if args.arm == "EN-F" else None)
        cmd = _phase_cmd(**common, max_epochs=args.easy_max_epochs,
                         solvency="easy_chronological_continuation",
                         out_dir=easy_dir, report=easy_report_path,
                         save_replay=replay_path)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        record["easy_phase"] = {"cmd": cmd[1:], "exit": proc.returncode}
        if proc.returncode != 0:
            record["outcome"] = "EASY_PHASE_FAILED"
            record["stderr_tail"] = proc.stderr[-800:]
            return record
        easy_report = json.loads(easy_report_path.read_text())
        handoff = verify_handoff(
            easy_report, easy_dir, args.action_threshold)
        record["handoff"] = handoff
        record["easy_compute"] = {
            "epochs": easy_report.get("epochs_run"),
            "elapsed_seconds": easy_report.get("elapsed_seconds"),
            "note": "reported separately; never truncated into the "
                    "normal budget"}
        warm = dict(warm_model=handoff["artifact"],
                    warm_model_sha=handoff["artifact_sha256"])
        if args.arm == "EN-F":
            warm.update(warm_replay=replay_path,
                        warm_replay_sha=_sha(replay_path))
    else:
        easy_report, handoff, warm = None, None, {}
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
    if easy_report is not None:
        record["continuity"] = verify_continuity(
            easy_report, normal_report, handoff, args.arm)
    else:
        record["continuity"] = {"mode": "cold_start",
                                "replay_disposition":
                                    normal_report.get(
                                        "replay_disposition")}
    hist = normal_report.get("history") or []
    eligible = [r for r in hist if r.get("l1_checkpoint_eligible")
                and r.get("checkpoint_improved")]
    record["normal_best_monitor"] = (
        max(r["composite"] for r in eligible) if eligible else None)
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
    parser.add_argument("--train-days", type=int, required=True)
    parser.add_argument("--val-days", type=int, required=True)
    parser.add_argument("--test-days", type=int, required=True)
    parser.add_argument("--action-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = run_arm(args)
    except CurriculumError as exc:
        record = {"schema": "agent_multi.l1_curriculum_arm.v1",
                  "arm": args.arm, "seed": args.seed,
                  "outcome": "REFUSED", "reason": str(exc)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=1))
    print(json.dumps({"arm": args.arm, "seed": args.seed,
                      "outcome": record["outcome"]}))
    return 0 if record["outcome"] == "ARM_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
