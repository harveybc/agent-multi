"""Persistent screen-recovery controller (post-outage order §B; REC-01..04).

Filesystem-backed attempt state machine with a REAL supervise
lifecycle. NOT activated against any live experiment until
independently verified.

Contract, stated once:

- One durable ATTEMPT MANIFEST is materialized before any launch, with
  the FULL 40-hex frozen commit and the sha256 of a materialized
  canonical launch artifact (exact argv + cwd + GPU mask). The
  supervisor launches EXACTLY that artifact — a mutation between check
  and launch fails the hash and refuses (REC-03).
- Durability is real: every manifest/intent write fsyncs the file AND
  its parent directory before the state is acknowledged (REC-04;
  injectable for failure tests — a failed fsync is a loud error).
- A missing process is classified from durable artifacts only:
  ``completed`` requires a SEMANTICALLY valid report — schema,
  ``accepted is True``, matching seed, arm, full commit, config hash,
  report ownership and a terminal stop reason. ``{}``, a typed
  negative, or a foreign/stale report is never completion (REC-02).
  Absence is never completion.
- Plateau scheduler/model state is never resumed; retries are fresh
  attempts with ABSENT-or-empty, non-symlink output directories.
- ``supervise`` (REC-01) is a real, CLI-exercised lifecycle: select
  manifest -> classify -> preserve/retry decision -> precondition
  verification -> launch from the artifact -> immediate PID record ->
  heartbeat -> exit reconciliation -> typed terminal state.
- Never touches broker authority or live/demo trading services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SCHEMA = "agent_multi.screen_recovery_attempt.v2"
REPORT_SCHEMA_PREFIX = "agent_multi.wp4_smoke"

COMPLETED = "completed"
FAILED_BEFORE_TRAINING = "failed_before_training"
INTERRUPTED_NONRESUMABLE = "interrupted_nonresumable"
UNKNOWN = "unknown"
ACTIVE = "active"

_EPOCH_RE = re.compile(r"\[epoch\s+(\d+)/|epoch[ =](\d+)")
_FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class RecoveryError(ValueError):
    """Typed refusal for every recovery-contract violation."""


def _fsync_dir_default(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, payload: Dict[str, Any], *,
                  fsync_dir: Callable[[Path], None] = _fsync_dir_default
                  ) -> None:
    """Durable write: fsync the file, rename, then fsync the parent
    directory (REC-04). A failing fsync raises — never silent."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    fsync_dir(path.parent)


def _canonical_launch_payload(argv: List[str], cwd: str,
                              gpu_mask: str) -> bytes:
    return json.dumps({"argv": list(argv), "cwd": cwd,
                       "gpu_mask": gpu_mask},
                      sort_keys=True).encode()


def _manifest_path(root: Path, seed: int, arm: str,
                   attempt_id: int) -> Path:
    return root / f"attempt_seed{seed}_{arm}_{attempt_id:04d}.json"


def _live_manifests(root: Path, seed: int, arm: str) -> List[Path]:
    return sorted(p for p in root.glob(
        f"attempt_seed{seed}_{arm}_*.json")
        if not p.name.endswith(".preserve_intent.json"))


def _load(path: Path) -> Dict[str, Any]:
    doc = json.loads(path.read_text())
    if doc.get("schema") != SCHEMA:
        raise RecoveryError(f"{path}: foreign manifest schema "
                            f"{doc.get('schema')!r}")
    return doc


def _pid_alive_default(pid: int, expected_substr: str) -> bool:
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
    argv: List[str], cwd: str,
    clock: Callable[[], float] = time.time,
    pid_alive: Callable[[int, str], bool] = _pid_alive_default,
    fsync_dir: Callable[[Path], None] = _fsync_dir_default,
) -> Path:
    """Materialize manifest + canonical launch artifact BEFORE launch."""
    if arm not in ("fixed", "plateau"):
        raise RecoveryError(f"unknown arm {arm!r}")
    if not _FULL_COMMIT_RE.match(frozen_commit or ""):
        raise RecoveryError(
            "frozen_commit must be the FULL 40-hex commit; short "
            f"prefixes collide (REC-03): got {frozen_commit!r}")
    if not argv:
        raise RecoveryError("argv is required; a manifest without the "
                            "exact command binds nothing (REC-03)")
    root.mkdir(parents=True, exist_ok=True)
    for prior in _live_manifests(root, seed, arm):
        doc = _load(prior)
        state = classify_attempt(prior, pid_alive=pid_alive)["state"]
        if state == ACTIVE:
            raise RecoveryError(
                f"duplicate active attempt refused: {prior.name}")
        if state == COMPLETED and doc.get("superseded_by") is None:
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
    payload = _canonical_launch_payload(argv, cwd, gpu_mask)
    artifact = root / f"launch_seed{seed}_{arm}_{attempt_id:04d}.json"
    tmp = artifact.with_suffix(".json.tmp")
    tmp.write_bytes(payload)
    tmp.replace(artifact)
    fsync_dir(artifact.parent)
    path = _manifest_path(root, seed, arm, attempt_id)
    _atomic_write(path, {
        "schema": SCHEMA, "seed": seed, "arm": arm,
        "attempt_id": attempt_id, "created_unix": clock(),
        "frozen_commit": frozen_commit, "config_sha256": config_sha256,
        "gpu_mask": gpu_mask, "output_dir": output_dir,
        "report_path": report_path, "log_path": log_path,
        "launch_artifact": str(artifact),
        "argv_sha256": hashlib.sha256(payload).hexdigest(),
        "pid": None, "preserved": False, "superseded_by": None,
        "heartbeat_unix": None, "contract": contract,
    }, fsync_dir=fsync_dir)
    return path


def record_pid(manifest: Path, pid: int, *,
               fsync_dir: Callable[[Path], None] = _fsync_dir_default
               ) -> None:
    doc = _load(manifest)
    doc["pid"] = int(pid)
    _atomic_write(manifest, doc, fsync_dir=fsync_dir)


def _semantic_completion(doc: Dict[str, Any], report: Path
                         ) -> Dict[str, Any]:
    """REC-02: completion is semantic, never 'parseable JSON exists'."""
    try:
        rep = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"state": INTERRUPTED_NONRESUMABLE,
                "detail": f"report unparseable ({exc.__class__.__name__});"
                          " an incomplete report is not completion"}
    problems = []
    if not str(rep.get("schema", "")).startswith(REPORT_SCHEMA_PREFIX):
        problems.append(f"schema {rep.get('schema')!r}")
    if rep.get("accepted") is not True:
        problems.append(f"accepted {rep.get('accepted')!r}")
    if (rep.get("budgets") or {}).get("seed") != doc["seed"]:
        problems.append("foreign seed")
    policy = (rep.get("arm_contract") or {}).get("scheduler_policy")
    if policy != doc["arm"]:
        problems.append(f"foreign arm {policy!r}")
    if rep.get("commit") != doc["frozen_commit"]:
        problems.append("foreign commit")
    if rep.get("config_sha256") != doc["config_sha256"]:
        problems.append("foreign config hash")
    if not isinstance(rep.get("stop_reason"), str) or not rep.get(
            "stop_reason"):
        problems.append("no terminal stop_reason")
    try:
        if report.stat().st_uid != os.getuid():
            problems.append("report not owned by this user")
    except OSError:
        problems.append("report unstatable")
    if problems:
        return {"state": INTERRUPTED_NONRESUMABLE,
                "detail": "report exists but is not a semantically "
                          "valid completion for this attempt: "
                          + "; ".join(problems)}
    return {"state": COMPLETED,
            "detail": "semantically valid accepted terminal report"}


def classify_attempt(
    manifest: Path, *,
    pid_alive: Callable[[int, str], bool] = _pid_alive_default,
) -> Dict[str, Any]:
    """Classify from durable artifacts. Absence is never completion."""
    doc = _load(manifest)
    report = Path(doc["report_path"])
    if report.is_file():
        return _semantic_completion(doc, report)
    pid = doc.get("pid")
    if pid and pid_alive(int(pid), f"--seed {doc['seed']}"):
        return {"state": ACTIVE, "detail": f"pid {pid} alive with "
                                           "matching cmdline"}
    log = Path(doc["log_path"])
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
            "detail": "process gone before the first epoch"}


def preserve_interrupted(
    manifest: Path, *, suffix: str,
    fsync_dir: Callable[[Path], None] = _fsync_dir_default,
) -> Dict[str, Any]:
    """Journaled, idempotent preservation (intent first, REC-04 fsync)."""
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
        if src.exists() or dst.exists():
            renames.append([str(src), str(dst)])
    _atomic_write(intent_path, {"suffix": suffix, "renames": renames},
                  fsync_dir=fsync_dir)
    performed = []
    for src_s, dst_s in renames:
        src, dst = Path(src_s), Path(dst_s)
        if dst.exists():
            performed.append({"dst": dst_s, "note": "already archived"})
            continue
        if src.exists():
            src.rename(dst)
            fsync_dir(src.parent)
            performed.append({"dst": dst_s, "note": "archived"})
    doc["preserved"] = True
    doc["preserved_suffix"] = suffix
    _atomic_write(manifest, doc, fsync_dir=fsync_dir)
    intent_path.unlink(missing_ok=True)
    fsync_dir(intent_path.parent)
    return {"preserved": performed, "state_was": state}


def verify_launch_preconditions(
    manifest: Path, *,
    git_head: Callable[[], str],
    git_dirty: Callable[[], bool],
    gpu_masks_present: Callable[[], List[str]],
    expected_config_sha256: str,
) -> None:
    """Fail-closed identity checks bound to what will execute (REC-03)."""
    doc = _load(manifest)
    head = git_head()
    if not _FULL_COMMIT_RE.match(head or ""):
        raise RecoveryError(
            f"git head {head!r} is not a full 40-hex commit; prefix "
            "comparison is a collision hazard (REC-03)")
    if head != doc["frozen_commit"]:
        raise RecoveryError(
            f"wrong commit: worktree at {head[:12]}… but the attempt "
            f"is pinned to {doc['frozen_commit'][:12]}…")
    if git_dirty():
        raise RecoveryError("worktree is dirty; a launch from an "
                            "unhashed tree binds nothing (REC-03)")
    if doc["config_sha256"] != expected_config_sha256:
        raise RecoveryError("config hash mismatch; the retry would not "
                            "be the same experiment")
    if doc["gpu_mask"] not in gpu_masks_present():
        raise RecoveryError(
            f"assigned GPU {doc['gpu_mask'][:12]}… not present; "
            "wrong-GPU launch refused")
    artifact = Path(doc["launch_artifact"])
    if not artifact.is_file():
        raise RecoveryError("launch artifact missing; nothing bound "
                            "to execute (REC-03)")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != doc["argv_sha256"]:
        raise RecoveryError(
            "launch artifact hash mismatch — the command changed after "
            "the check (REC-03 substitution refused)")
    out = Path(doc["output_dir"])
    if out.is_symlink():
        raise RecoveryError("output dir is a symlink; refused (REC-03)")
    if out.exists():
        if any(out.iterdir()):
            raise RecoveryError(
                "output dir is not empty; a retry requires an ABSENT "
                "or EMPTY directory (REC-03 — stale files of any kind "
                "refuse, not only model artifacts)")
        if out.stat().st_uid != os.getuid():
            raise RecoveryError("output dir not owned by this user")


def _spawn_default(manifest_doc: Dict[str, Any]) -> Any:
    artifact = Path(manifest_doc["launch_artifact"])
    payload = json.loads(artifact.read_text())
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = manifest_doc["gpu_mask"]
    log = open(manifest_doc["log_path"], "a")
    return subprocess.Popen(payload["argv"], cwd=payload["cwd"],
                            stdout=log, stderr=log, env=env)


def supervise(
    root: Path, seed: int, arm: str, *,
    git_head: Callable[[], str],
    git_dirty: Callable[[], bool],
    gpu_masks_present: Callable[[], List[str]],
    expected_config_sha256: str,
    spawn: Callable[[Dict[str, Any]], Any] = _spawn_default,
    poll_seconds: float = 30.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    fsync_dir: Callable[[Path], None] = _fsync_dir_default,
    preserve_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    """REC-01: the real lifecycle. Returns a typed terminal record."""
    manifests = _live_manifests(root, seed, arm)
    if not manifests:
        raise RecoveryError(
            f"no attempt manifest for seed {seed} arm {arm}; the "
            "supervisor never invents an experiment")
    current = manifests[-1]
    doc = _load(current)
    cls = classify_attempt(current)
    if cls["state"] == ACTIVE:
        raise RecoveryError(
            f"{current.name} already ACTIVE; duplicate supervision "
            "refused")
    if cls["state"] == COMPLETED:
        return {"terminal": COMPLETED, "attempt": current.name,
                "detail": cls["detail"], "action": "none"}
    if cls["state"] in (INTERRUPTED_NONRESUMABLE,
                        FAILED_BEFORE_TRAINING, UNKNOWN):
        if not doc.get("preserved"):
            preserve_interrupted(
                current,
                suffix=preserve_suffix or f"{cls['state']}_auto",
                fsync_dir=fsync_dir)
        payload = json.loads(
            Path(doc["launch_artifact"]).read_text())
        current = write_attempt_manifest(
            root, seed=seed, arm=arm,
            frozen_commit=doc["frozen_commit"],
            config_sha256=doc["config_sha256"],
            gpu_mask=doc["gpu_mask"], output_dir=doc["output_dir"],
            report_path=doc["report_path"], log_path=doc["log_path"],
            contract=doc["contract"], argv=payload["argv"],
            cwd=payload["cwd"], clock=clock, fsync_dir=fsync_dir)
        doc = _load(current)
    verify_launch_preconditions(
        current, git_head=git_head, git_dirty=git_dirty,
        gpu_masks_present=gpu_masks_present,
        expected_config_sha256=expected_config_sha256)
    proc = spawn(doc)
    record_pid(current, int(proc.pid), fsync_dir=fsync_dir)
    while proc.poll() is None:
        d = _load(current)
        d["heartbeat_unix"] = clock()
        _atomic_write(current, d, fsync_dir=fsync_dir)
        sleep(poll_seconds)
    final = classify_attempt(current)
    return {"terminal": final["state"], "attempt": current.name,
            "detail": final["detail"], "exit_code": proc.returncode,
            "action": "launched"}


def status(root: Path, *, now: Callable[[], float] = time.time
           ) -> List[Dict[str, Any]]:
    rows = []
    for manifest in sorted(root.glob("attempt_seed*_*.json")):
        if manifest.name.endswith(".preserve_intent.json"):
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
            floor_epochs = (doc["contract"].get("l1_patience", 60)
                            + doc["contract"].get(
                                "l1_patience_start_epoch", 40))
            eta_s = max(0.0, (floor_epochs - epoch) * rate)
        rows.append({"attempt": manifest.name, "seed": doc["seed"],
                     "arm": doc["arm"], "attempt_id": doc["attempt_id"],
                     "state": cls["state"], "detail": cls["detail"],
                     "heartbeat_unix": doc.get("heartbeat_unix"),
                     "epoch": epoch, "gpu": telem,
                     "eta_seconds_to_patience_floor": eta_s})
    return rows


def emit_persistent_unit(manifest: Path) -> str:
    """Persistent user-unit text (NOT installed by this tool). The
    ExecStart invokes the REAL, tested `supervise` subcommand."""
    doc = _load(manifest)
    return f"""# NOT INSTALLED by screen_recovery_controller — proposal only.
# Activation boundary: only after independent verification, never
# against a live experiment.
[Unit]
Description=screen recovery supervisor seed {doc['seed']} {doc['arm']}
After=default.target

[Service]
Type=oneshot
ExecStart={sys.executable} {Path(__file__).resolve()} supervise --root {manifest.parent} --seed {doc['seed']} --arm {doc['arm']} --expected-config-sha256 {doc['config_sha256']} --repo-dir {json.loads(Path(doc['launch_artifact']).read_text())['cwd']}

[Install]
WantedBy=default.target
"""


def _cli_git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True
                          ).stdout.strip()


def _cli_git_dirty(repo: Path) -> bool:
    out = subprocess.run(["git", "-C", str(repo), "status",
                          "--porcelain"], capture_output=True,
                         text=True).stdout.strip()
    return bool(out)


def _cli_gpu_masks() -> List[str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid",
                          "--format=csv,noheader"], capture_output=True,
                         text=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     allow_abbrev=False)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("classify", "status", "emit-unit", "supervise"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        if name in ("classify", "emit-unit"):
            p.add_argument("--manifest", type=Path, required=True)
        if name == "supervise":
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--arm", required=True,
                           choices=["fixed", "plateau"])
            p.add_argument("--expected-config-sha256", required=True)
            p.add_argument("--repo-dir", type=Path, required=True)
            p.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.cmd == "classify":
        print(json.dumps(classify_attempt(args.manifest), indent=1))
    elif args.cmd == "status":
        print(json.dumps(status(args.root), indent=1))
    elif args.cmd == "emit-unit":
        print(emit_persistent_unit(args.manifest))
    elif args.cmd == "supervise":
        result = supervise(
            args.root, args.seed, args.arm,
            git_head=lambda: _cli_git_head(args.repo_dir),
            git_dirty=lambda: _cli_git_dirty(args.repo_dir),
            gpu_masks_present=_cli_gpu_masks,
            expected_config_sha256=args.expected_config_sha256,
            poll_seconds=args.poll_seconds)
        print(json.dumps(result, indent=1))
        if result["terminal"] != COMPLETED:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
