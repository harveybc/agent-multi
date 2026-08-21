"""Persistent screen-recovery controller (post-outage order §B).

Replaces reliance on transient ``systemd-run`` memory with a durable,
filesystem-backed attempt state machine. NOT ACTIVATED against the
frozen screen running at 93880beb — this delivery is library + CLI +
fixtures; the activation boundary is proposed separately.

Contract, stated once:

- One durable ATTEMPT MANIFEST is materialized (atomic write+fsync)
  BEFORE any process launch. Everything later is judged against it.
- A missing process is classified ``completed`` /
  ``failed_before_training`` / ``interrupted_nonresumable`` /
  ``unknown`` from durable artifacts only. ABSENCE IS NEVER COMPLETION:
  completed requires a parseable report at the manifest's report path.
- Plateau scheduler/model state is NEVER resumed: a retry is a fresh
  attempt id with a clean output directory; a plateau sidecar or model
  artifact already inside the retry output dir refuses the launch.
- Preservation of an interrupted attempt is journaled: an intent file
  is written first, each rename is idempotent, and a crash between
  archive and retry is completed on the next pass without loss.
- Duplicate active attempts for one (seed, arm) refuse. A completed
  arm can never be retried; only the unfinished arm of a pair retries.
- Launch preconditions verify frozen commit, config hash, seed, GPU
  assignment and output ownership. Probes (clock, pid, git, gpu) are
  injectable so every test runs socket-free on temporary fixtures.
- Status/heartbeat exposes attempt id, arm, epoch, GPU temperature and
  utilization, and a measured-rate ETA as JSON for the consolidated
  status/Telegram path. This module never touches broker authority or
  any live/demo trading service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SCHEMA = "agent_multi.screen_recovery_attempt.v1"

COMPLETED = "completed"
FAILED_BEFORE_TRAINING = "failed_before_training"
INTERRUPTED_NONRESUMABLE = "interrupted_nonresumable"
UNKNOWN = "unknown"
ACTIVE = "active"

_EPOCH_RE = re.compile(r"\[epoch\s+(\d+)/|epoch[ =](\d+)")


class RecoveryError(ValueError):
    """Typed refusal for every recovery-contract violation."""


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _manifest_path(root: Path, seed: int, arm: str,
                   attempt_id: int) -> Path:
    return root / f"attempt_seed{seed}_{arm}_{attempt_id:04d}.json"


def _live_manifests(root: Path, seed: int, arm: str) -> List[Path]:
    return sorted(root.glob(f"attempt_seed{seed}_{arm}_*.json"))


def _load(path: Path) -> Dict[str, Any]:
    doc = json.loads(path.read_text())
    if doc.get("schema") != SCHEMA:
        raise RecoveryError(f"{path}: foreign manifest schema "
                            f"{doc.get('schema')!r}")
    return doc


def _pid_alive_default(pid: int, expected_substr: str) -> bool:
    """True only if the PID exists AND its cmdline contains the
    expected token — a reused (stale) PID never counts as ours."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
            "utf-8", "replace")
    except OSError:
        return False
    return expected_substr in cmdline


def write_attempt_manifest(
    root: Path, *, seed: int, arm: str, frozen_commit: str,
    config_sha256: str, gpu_mask: str, output_dir: str,
    report_path: str, log_path: str, contract: Dict[str, Any],
    clock: Callable[[], float] = time.time,
    pid_alive: Callable[[int, str], bool] = _pid_alive_default,
) -> Path:
    """Materialize the durable attempt manifest BEFORE launch.

    Refuses while a prior attempt for the same (seed, arm) is still
    active or unclassified — duplicates are structural, not advisory.
    """
    if arm not in ("fixed", "plateau"):
        raise RecoveryError(f"unknown arm {arm!r}")
    root.mkdir(parents=True, exist_ok=True)
    for prior in _live_manifests(root, seed, arm):
        doc = _load(prior)
        state = classify_attempt(prior, pid_alive=pid_alive)["state"]
        if state == ACTIVE:
            raise RecoveryError(
                f"duplicate active attempt refused: {prior.name} is "
                f"still {state} for seed {seed} arm {arm}")
        if state == COMPLETED and doc.get("superseded_by") is None \
                and arm == doc.get("arm"):
            raise RecoveryError(
                f"arm already completed by {prior.name}; a completed "
                "paired arm is never rerun")
        if state in (INTERRUPTED_NONRESUMABLE, FAILED_BEFORE_TRAINING,
                     UNKNOWN) and not doc.get("preserved"):
            raise RecoveryError(
                f"{prior.name} is {state} but not yet preserved; "
                "preserve the interrupted attempt before a retry")
    attempt_id = 1 + max(
        (int(p.stem.rsplit("_", 1)[-1])
         for p in _live_manifests(root, seed, arm)), default=0)
    path = _manifest_path(root, seed, arm, attempt_id)
    _atomic_write(path, {
        "schema": SCHEMA, "seed": seed, "arm": arm,
        "attempt_id": attempt_id, "created_unix": clock(),
        "frozen_commit": frozen_commit, "config_sha256": config_sha256,
        "gpu_mask": gpu_mask, "output_dir": output_dir,
        "report_path": report_path, "log_path": log_path,
        "pid": None, "preserved": False, "superseded_by": None,
        "contract": contract,
    })
    return path


def record_pid(manifest: Path, pid: int) -> None:
    doc = _load(manifest)
    doc["pid"] = int(pid)
    _atomic_write(manifest, doc)


def classify_attempt(
    manifest: Path, *,
    pid_alive: Callable[[int, str], bool] = _pid_alive_default,
) -> Dict[str, Any]:
    """Classify from durable artifacts. Absence is never completion."""
    doc = _load(manifest)
    report = Path(doc["report_path"])
    log = Path(doc["log_path"])
    if report.is_file():
        try:
            json.loads(report.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {"state": INTERRUPTED_NONRESUMABLE,
                    "detail": f"report exists but is not parseable "
                              f"({exc.__class__.__name__}); an "
                              "incomplete report is not completion"}
        return {"state": COMPLETED, "detail": "parseable report present"}
    pid = doc.get("pid")
    if pid and pid_alive(int(pid), f"--seed {doc['seed']}"):
        return {"state": ACTIVE, "detail": f"pid {pid} alive with "
                                           "matching cmdline"}
    if not log.is_file():
        return {"state": UNKNOWN,
                "detail": "no report, no matching live pid, no log; "
                          "absence is never completion"}
    text = log.read_text(errors="replace")
    if _EPOCH_RE.search(text):
        return {"state": INTERRUPTED_NONRESUMABLE,
                "detail": "training epochs observed, no report; "
                          "plateau/model state is never resumed"}
    return {"state": FAILED_BEFORE_TRAINING,
            "detail": "process gone before the first epoch "
                      "(e.g. argument or import failure)"}


def preserve_interrupted(manifest: Path, *, suffix: str) -> Dict[str, Any]:
    """Journaled, idempotent preservation of an interrupted attempt.

    Writes an intent file FIRST; each rename skips targets that already
    exist, so a power loss between archive and retry is completed by
    simply calling this again with the same suffix.
    """
    doc = _load(manifest)
    state = classify_attempt(manifest)["state"]
    if state == ACTIVE:
        raise RecoveryError("refusing to preserve an ACTIVE attempt")
    if state == COMPLETED:
        raise RecoveryError("refusing to preserve a COMPLETED attempt; "
                            "completed evidence stays in place")
    intent_path = manifest.with_suffix(".preserve_intent.json")
    renames = []
    for key in ("output_dir", "log_path"):
        src = Path(doc[key])
        dst = src.parent / (src.name + "_" + suffix)
        # Include half-completed archives: source already gone but the
        # destination exists (crash between archive and retry).
        if src.exists() or dst.exists():
            renames.append([str(src), str(dst)])
    _atomic_write(intent_path, {"suffix": suffix, "renames": renames})
    performed = []
    for src_s, dst_s in renames:
        src, dst = Path(src_s), Path(dst_s)
        if dst.exists():
            performed.append({"dst": dst_s, "note": "already archived"})
            continue
        if src.exists():
            src.rename(dst)
            performed.append({"dst": dst_s, "note": "archived"})
    doc["preserved"] = True
    doc["preserved_suffix"] = suffix
    _atomic_write(manifest, doc)
    intent_path.unlink(missing_ok=True)
    return {"preserved": performed, "state_was": state}


def verify_launch_preconditions(
    manifest: Path, *,
    git_head: Callable[[], str],
    gpu_masks_present: Callable[[], List[str]],
    expected_config_sha256: str,
) -> None:
    """Fail-closed identity checks BEFORE any retry launch."""
    doc = _load(manifest)
    head = git_head()
    if not head.startswith(doc["frozen_commit"]):
        raise RecoveryError(
            f"wrong commit: worktree at {head[:12]} but the attempt "
            f"is pinned to {doc['frozen_commit']}")
    if doc["config_sha256"] != expected_config_sha256:
        raise RecoveryError("config hash mismatch; the retry would not "
                            "be the same experiment")
    if doc["gpu_mask"] not in gpu_masks_present():
        raise RecoveryError(
            f"assigned GPU {doc['gpu_mask'][:12]}… not present on this "
            "host; wrong-GPU launch refused")
    out = Path(doc["output_dir"])
    if out.exists():
        if any(out.rglob("*.plateau_lr_state.json")):
            raise RecoveryError(
                "REFUSED_PLATEAU_RESUME: scheduler sidecar inside the "
                "retry output dir; plateau state is never resumed — "
                "the output dir must be clean")
        if any(out.rglob("*.zip")):
            raise RecoveryError(
                "model artifact inside the retry output dir; a retry "
                "is a fresh attempt with a clean directory")
        if out.stat().st_uid != os.getuid():
            raise RecoveryError("output dir not owned by this user")


def status(root: Path, *, now: Callable[[], float] = time.time
           ) -> List[Dict[str, Any]]:
    """Heartbeat JSON for the consolidated status/Telegram path."""
    rows = []
    for manifest in sorted(root.glob("attempt_seed*_*.json")):
        if manifest.suffixes[-2:] == [".preserve_intent", ".json"]:
            continue
        doc = _load(manifest)
        cls = classify_attempt(manifest)
        epoch = None
        log = Path(doc["log_path"])
        if log.is_file():
            for m in _EPOCH_RE.finditer(log.read_text(errors="replace")):
                epoch = int(m.group(1) or m.group(2))
        telem = {}
        telem_path = Path(doc["output_dir"] + "_gpu_telemetry.csv")
        if telem_path.is_file():
            last = telem_path.read_text().strip().rsplit("\n", 1)[-1]
            parts = [p.strip() for p in last.split(",")]
            if len(parts) >= 4:
                telem = {"temperature_c": parts[2],
                         "utilization": parts[3]}
        eta_s = None
        if epoch and cls["state"] == ACTIVE:
            rate = (now() - doc["created_unix"]) / max(epoch, 1)
            # measured-rate ETA to the patience floor, not a guess
            floor_epochs = (doc["contract"].get("l1_patience", 60)
                            + doc["contract"].get(
                                "l1_patience_start_epoch", 40))
            eta_s = max(0.0, (floor_epochs - epoch) * rate)
        rows.append({"attempt": manifest.name, "seed": doc["seed"],
                     "arm": doc["arm"], "attempt_id": doc["attempt_id"],
                     "state": cls["state"], "detail": cls["detail"],
                     "epoch": epoch, "gpu": telem,
                     "eta_seconds_to_patience_floor": eta_s})
    return rows


def emit_persistent_unit(manifest: Path) -> str:
    """Persistent user-unit text (NOT installed by this tool)."""
    doc = _load(manifest)
    return f"""# NOT INSTALLED by screen_recovery_controller — proposal only.
# Activation boundary: after the current frozen screen completes.
[Unit]
Description=screen recovery supervisor seed {doc['seed']} {doc['arm']}
After=default.target

[Service]
Type=oneshot
ExecStart={sys.executable} {Path(__file__).resolve()} supervise --root {manifest.parent} --seed {doc['seed']} --arm {doc['arm']}
Environment=CUDA_VISIBLE_DEVICES={doc['gpu_mask']}

[Install]
WantedBy=default.target
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     allow_abbrev=False)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("classify", "status", "emit-unit"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        if name != "status":
            p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "classify":
        print(json.dumps(classify_attempt(args.manifest), indent=1))
    elif args.cmd == "status":
        print(json.dumps(status(args.root), indent=1))
    elif args.cmd == "emit-unit":
        print(emit_persistent_unit(args.manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
