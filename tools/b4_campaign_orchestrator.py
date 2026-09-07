#!/usr/bin/env python3
"""B4 campaign orchestrator (orders @0ce52740 C4, C9-C16).

C10 — ONE claimable object per logical cell and campaign generation:
the claim path is FIXED (`CLAIM_<generation>.json`), created with
O_EXCL|O_NOFOLLOW and validated descriptor-first; the random
attempt_id lives INSIDE the record. Two synchronized processes at
the vulnerable boundary produce exactly one winner. Uncertain
creation fails closed; success is never inferred from absence.

C11 — dry-run means ZERO writes: without --execute the orchestrator
verifies authority, materialization, ledger, health schema and
schedule, prints the plan, and provably alters nothing.

C12 — execution happens only under a verified EXECUTION LEASE bound
to campaign generation, cell, attempt, authorization and
materialization digests; the executor refuses without it.

C13 — the 96 GPU-hour ceiling is an INTRASEGMENT bound: each
dispatch derives the remaining campaign wall from every attempt
(failed and uncertain included, durable facts for replay, monotonic
for the live attempt) and the cell's effective wall budget is
min(per-cell limit, global remainder); a remainder smaller than one
segment fails closed.

C15 — resume ADJUDICATES terminals: verified-completed skips;
verified non-completed is retained and named; unsealed/malformed is
UNCERTAIN and blocks; an ambiguous claim blocks; absent claim and
terminal is pending. Exit status distinguishes COMPLETE /
COMPLETE_WITH_FAILED_CELLS / STOPPED_RESOURCE_CEILING / UNCERTAIN —
never success-by-file-existence."""
import argparse
import hashlib
import json
import os
import stat
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import b4_authority as b4a  # noqa: E402

SUBSTANTIAL_COMPUTE_MIB = 1024
MIN_SEGMENT_SECONDS = 600.0


class OrchestratorRefusal(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


CONTROL_DIR_MODE = 0o700
CONTROL_FILE_MODE = 0o600


def _secure_dir(path: Path, create: bool = True) -> None:
    """C24: control-plane directories are PRIVATE (0700) and
    validated descriptor-first. An existing permissive directory is
    REFUSED, never silently chmodded."""
    path = Path(path)
    if create and not path.exists():
        try:
            os.mkdir(str(path), CONTROL_DIR_MODE)
        except FileExistsError:
            pass
        except OSError as exc:
            raise OrchestratorRefusal(
                f"REFUSED: cannot create control directory "
                f"{path.name}: {exc}")
    try:
        dfd = os.open(str(path),
                      os.O_RDONLY | os.O_NOFOLLOW
                      | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise OrchestratorRefusal(
            f"REFUSED: control directory {path.name} unopenable "
            f"({exc}) — failing closed")
    try:
        st = os.fstat(dfd)
        if not stat.S_ISDIR(st.st_mode):
            raise OrchestratorRefusal(
                f"REFUSED: {path.name} is not a directory")
        if st.st_uid != os.getuid():
            raise OrchestratorRefusal(
                f"REFUSED: control directory {path.name} has a "
                "foreign owner")
        if stat.S_IMODE(st.st_mode) != CONTROL_DIR_MODE:
            raise OrchestratorRefusal(
                f"REFUSED: control directory {path.name} mode "
                f"{oct(stat.S_IMODE(st.st_mode))} is not the "
                f"private {oct(CONTROL_DIR_MODE)} — refused, not "
                "chmodded")
    finally:
        os.close(dfd)


def _excl_write(path: Path, payload: bytes,
                mode=CONTROL_FILE_MODE) -> None:
    """C24: O_EXCL|O_NOFOLLOW create of a PRIVATE (0600) control
    object under a verified 0700 directory; descriptor-first
    validation, explicit fchmod, fsync(file)+fsync(dir).
    Uncertainty fails closed."""
    path = Path(path)
    _secure_dir(path.parent)
    try:
        fd = os.open(str(path),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY
                     | os.O_NOFOLLOW, mode)
    except FileExistsError:
        raise OrchestratorRefusal(
            f"REFUSED: {path.name} already exists — exactly one "
            "winner per logical object")
    except OSError as exc:
        raise OrchestratorRefusal(
            f"REFUSED: uncertain exclusive create of {path.name}: "
            f"{exc} — failing closed")
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OrchestratorRefusal(
                f"REFUSED: {path.name} is not a regular file")
        os.fchmod(fd, mode)
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _secure_read(path: Path,
                 expected_mode=CONTROL_FILE_MODE) -> bytes:
    """C24: ONE descriptor per consumption — O_NOFOLLOW open, then
    regular-file/owner/exact-mode checks and the read all from that
    same descriptor. No check-by-path-then-reopen-by-path."""
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise OrchestratorRefusal(
            f"REFUSED: control object {Path(path).name} absent")
    except OSError as exc:
        raise OrchestratorRefusal(
            f"REFUSED: control object {Path(path).name} unopenable "
            f"({exc}) — symlinks and races fail closed")
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OrchestratorRefusal(
                f"REFUSED: {Path(path).name} is not a regular "
                "file")
        if st.st_uid != os.getuid():
            raise OrchestratorRefusal(
                f"REFUSED: {Path(path).name} has a foreign owner")
        if expected_mode is not None and \
                stat.S_IMODE(st.st_mode) != expected_mode:
            raise OrchestratorRefusal(
                f"REFUSED: {Path(path).name} mode "
                f"{oct(stat.S_IMODE(st.st_mode))} is not the "
                f"private {oct(expected_mode)} control mode — a "
                "permissive object is refused, not chmodded")
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _secure_json(path: Path, where: str,
                 expected_mode=CONTROL_FILE_MODE) -> dict:
    return b4a._strict_json_bytes(
        _secure_read(path, expected_mode), where)


def _self_sha(body: dict, exclude: str) -> str:
    return hashlib.sha256(json.dumps(
        {k: body[k] for k in sorted(body) if k != exclude},
        sort_keys=True).encode()).hexdigest()


# ---------------- C23: monotone epoch campaign lock ----------------
LOCK_SCHEMA_NAME = "agent_multi.b4_campaign_lock_epoch.v1"
LOCK_INTENT_SCHEMA = "agent_multi.b4_lock_release_intent.v1"
LOCK_COMPLETE_SCHEMA = "agent_multi.b4_lock_release_complete.v1"


def _epoch_paths(root: Path, epoch: int):
    root = Path(root)
    return (root / f"LOCK_EPOCH_{epoch}.json",
            root / f"LOCK_RELEASE_INTENT_{epoch}.json",
            root / f"LOCK_RELEASE_COMPLETE_{epoch}.json")


def _scan_epochs(root: Path) -> list:
    out = []
    for p in Path(root).glob("LOCK_EPOCH_*.json"):
        tail = p.name[len("LOCK_EPOCH_"):-len(".json")]
        if not tail.isdigit():
            raise OrchestratorRefusal(
                f"REFUSED: malformed lock epoch name {p.name}")
        out.append(int(tail))
    return sorted(out)


def lock_epoch_state(root: Path, epoch: int) -> dict:
    """PHYSICAL adjudication of one lock epoch:
    HELD -> lock only; RELEASING -> lock+intent; RELEASED ->
    lock+intent+completion all integral and mutually bound;
    anything partial/malformed/transplanted -> UNCERTAIN."""
    lock_p, intent_p, complete_p = _epoch_paths(root, epoch)
    out = {"epoch": epoch, "state": "UNCERTAIN", "record": None}
    try:
        rec = _secure_json(lock_p, f"lock epoch {epoch}")
    except SystemExit:
        return out
    if set(rec) != {"schema", "generation", "epoch", "holder_pid",
                    "acquire_id", "lock_sha256"} or \
            rec["schema"] != LOCK_SCHEMA_NAME or \
            rec["epoch"] != epoch or \
            rec["generation"] != b4a.CAMPAIGN_GENERATION or \
            type(rec["holder_pid"]) is not int or \
            type(rec["acquire_id"]) is not str or \
            _self_sha(rec, "lock_sha256") != rec["lock_sha256"]:
        return out
    out["record"] = rec
    intent_exists = intent_p.exists()
    complete_exists = complete_p.exists()
    if not intent_exists and not complete_exists:
        out["state"] = "HELD"
        return out
    if not intent_exists:
        return out              # completion without intent
    try:
        intent = _secure_json(intent_p,
                              f"lock release intent {epoch}")
    except SystemExit:
        return out
    if set(intent) != {"schema", "epoch", "acquire_id",
                       "holder_pid", "intent_sha256"} or \
            intent["schema"] != LOCK_INTENT_SCHEMA or \
            intent["epoch"] != epoch or \
            intent["acquire_id"] != rec["acquire_id"] or \
            intent["holder_pid"] != rec["holder_pid"] or \
            _self_sha(intent, "intent_sha256") != \
            intent["intent_sha256"]:
        return out
    if not complete_exists:
        out["state"] = "RELEASING"
        return out
    try:
        comp = _secure_json(complete_p,
                            f"lock release completion {epoch}")
    except SystemExit:
        return out
    intent_file_sha = hashlib.sha256(
        _secure_read(intent_p)).hexdigest()
    if set(comp) != {"schema", "epoch", "acquire_id",
                     "intent_file_sha256",
                     "completion_sha256"} or \
            comp["schema"] != LOCK_COMPLETE_SCHEMA or \
            comp["epoch"] != epoch or \
            comp["acquire_id"] != rec["acquire_id"] or \
            comp["intent_file_sha256"] != intent_file_sha or \
            _self_sha(comp, "completion_sha256") != \
            comp["completion_sha256"]:
        return out
    out["state"] = "RELEASED"
    return out


class GlobalLock:
    """C23: MONOTONE campaign lock — unlink is never an authorizing
    transition. Epoch n transitions in place through
    held -> releasing -> released by APPEND-ONLY intent/completion
    witnesses; reclaim creates epoch n+1 exclusively and only when
    epoch n is physically RELEASED. Every missing, partial,
    malformed, transplanted or uncertain state is operator
    disposition, never an absent lock."""

    def __init__(self, results_root: Path):
        self.root = Path(results_root)
        self.epoch = None
        self.acquire_id = None
        self.held = False

    def __enter__(self):
        import uuid as _uuid
        _secure_dir(self.root)
        epochs = _scan_epochs(self.root)
        if epochs:
            cur = epochs[-1]
            st = lock_epoch_state(self.root, cur)
            if st["state"] == "HELD":
                raise OrchestratorRefusal(
                    f"REFUSED: lock epoch {cur} is HELD — exactly "
                    "one winner; a crashed holder is operator "
                    "disposition, never auto-stolen")
            if st["state"] == "RELEASING":
                raise OrchestratorRefusal(
                    f"REFUSED: lock epoch {cur} release is "
                    "UNCERTAIN (intent without durable completion) "
                    "— operator disposition, no second holder")
            if st["state"] != "RELEASED":
                raise OrchestratorRefusal(
                    f"REFUSED: lock epoch {cur} state is "
                    "UNCERTAIN — operator disposition")
            nxt = cur + 1
        else:
            nxt = 1
        self.acquire_id = _uuid.uuid4().hex[:16]
        rec = {"schema": LOCK_SCHEMA_NAME,
               "generation": b4a.CAMPAIGN_GENERATION,
               "epoch": nxt,
               "holder_pid": os.getpid(),
               "acquire_id": self.acquire_id}
        rec["lock_sha256"] = _self_sha(rec, "lock_sha256")
        lock_p, _, _ = _epoch_paths(self.root, nxt)
        _excl_write(lock_p, json.dumps(rec, indent=1).encode())
        # C23: revalidate the COMPLETE tuple under the exclusive
        # choice — the predecessor must still be RELEASED and the
        # epoch we own must adjudicate HELD by us.
        if nxt > 1:
            prev = lock_epoch_state(self.root, nxt - 1)
            if prev["state"] != "RELEASED":
                # release OUR just-created epoch in order before
                # refusing, so a transient predecessor doubt never
                # strands an orphan HELD epoch; if even that release
                # is uncertain, the epoch stays for the operator.
                self.epoch = nxt
                self.held = True
                try:
                    self.__exit__()
                except SystemExit:
                    pass
                self.held = False
                raise OrchestratorRefusal(
                    f"REFUSED: predecessor lock epoch {nxt - 1} "
                    "is no longer RELEASED under the exclusive "
                    "choice — failing closed")
        mine = lock_epoch_state(self.root, nxt)
        if mine["state"] != "HELD" or \
                mine["record"]["holder_pid"] != os.getpid() or \
                mine["record"]["acquire_id"] != self.acquire_id:
            raise OrchestratorRefusal(
                f"REFUSED: acquired lock epoch {nxt} does not "
                "adjudicate as held by this process")
        self.epoch = nxt
        self.held = True
        return self

    def __exit__(self, *exc):
        if not self.held:
            return
        st = lock_epoch_state(self.root, self.epoch)
        if st["state"] != "HELD" or \
                st["record"] is None or \
                st["record"]["holder_pid"] != os.getpid() or \
                st["record"]["acquire_id"] != self.acquire_id:
            raise OrchestratorRefusal(
                "REFUSED: lock release by a non-holder or over an "
                "uncertain epoch — ownership is required")
        _, intent_p, complete_p = _epoch_paths(self.root,
                                               self.epoch)
        intent = {"schema": LOCK_INTENT_SCHEMA,
                  "epoch": self.epoch,
                  "acquire_id": self.acquire_id,
                  "holder_pid": os.getpid()}
        intent["intent_sha256"] = _self_sha(intent, "intent_sha256")
        try:
            _excl_write(intent_p,
                        json.dumps(intent, indent=1).encode())
        except OSError as exc2:
            raise OrchestratorRefusal(
                f"REFUSED: uncertain release intent ({exc2}) — "
                "the epoch stays for operator disposition")
        intent_file_sha = hashlib.sha256(
            _secure_read(intent_p)).hexdigest()
        comp = {"schema": LOCK_COMPLETE_SCHEMA,
                "epoch": self.epoch,
                "acquire_id": self.acquire_id,
                "intent_file_sha256": intent_file_sha}
        comp["completion_sha256"] = _self_sha(comp,
                                              "completion_sha256")
        try:
            _excl_write(complete_p,
                        json.dumps(comp, indent=1).encode())
        except OSError as exc2:
            raise OrchestratorRefusal(
                f"REFUSED: uncertain release completion ({exc2}) "
                "— the epoch stays RELEASING for operator "
                "disposition")
        self.held = False


def current_lock_epoch(root: Path) -> dict:
    """The newest epoch's physical state (NO_LOCK when none)."""
    epochs = _scan_epochs(root)
    if not epochs:
        return {"epoch": None, "state": "NO_LOCK", "record": None}
    return lock_epoch_state(root, epochs[-1])


def _claim_path(results_root: Path, cell_id: str) -> Path:
    return (Path(results_root) / cell_id /
            f"CLAIM_{b4a.CAMPAIGN_GENERATION}.json")


CLAIM_SCHEMA = {
    "schema": str, "campaign_generation": str, "attempt_id": str,
    "cell": str, "claimed_wall": float, "claimed_monotonic": float,
    "holder_pid": int, "terminal_sha256": type(None),
    "recovery_acta_sha256": str,
    "claim_sha256": str}


def claim_attempt(results_root: Path, cell_id: str) -> dict:
    """C10/C24/C36: the ONE claimable object per (cell,
    generation) — self-integral, private-mode, under a 0700 cell
    directory. NO claim can exist while the recovery gate is
    closed: the witness is RE-DERIVED here from the reviewed acta
    (a caller-supplied witness is never sufficient) and its digest
    enters the claim itself."""
    witness = b4a.require_v6_launch_open()
    _secure_dir(Path(results_root))
    cell_dir = Path(results_root) / cell_id
    _secure_dir(cell_dir)
    rec = {"schema": "agent_multi.b4_attempt_claim.v2",
           "campaign_generation": b4a.CAMPAIGN_GENERATION,
           "attempt_id": f"attempt_{uuid.uuid4().hex[:16]}",
           "cell": cell_id,
           "claimed_wall": time.time(),
           "claimed_monotonic": time.monotonic(),
           "holder_pid": os.getpid(),
           "terminal_sha256": None,
           "recovery_acta_sha256": witness["acta_sha256"]}
    rec["claim_sha256"] = _self_sha(rec, "claim_sha256")
    _excl_write(_claim_path(results_root, cell_id),
                json.dumps(rec, indent=1).encode())
    return rec


def load_claim(results_root: Path, cell_id: str) -> dict:
    """C24: exact typed self-integral schema, consumed from ONE
    descriptor."""
    p = _claim_path(results_root, cell_id)
    try:
        rec = _secure_json(p, f"claim {cell_id}")
    except SystemExit as exc:
        if "absent" in str(exc):
            raise OrchestratorRefusal(
                f"REFUSED: claim for {cell_id} absent or "
                "non-regular")
        raise
    if set(rec) != set(CLAIM_SCHEMA):
        raise OrchestratorRefusal(
            f"REFUSED: claim for {cell_id} keys are not the exact "
            "schema")
    for k, t in CLAIM_SCHEMA.items():
        if not (rec[k] is None if t is type(None)
                else type(rec[k]) is t):
            raise OrchestratorRefusal(
                f"REFUSED: claim field {k!r} has a foreign "
                "primitive type")
    if _self_sha(rec, "claim_sha256") != rec["claim_sha256"]:
        raise OrchestratorRefusal(
            f"REFUSED: claim for {cell_id} content digest does "
            "not re-derive — altered control object")
    if rec["cell"] != cell_id or \
            rec["campaign_generation"] != b4a.CAMPAIGN_GENERATION:
        raise OrchestratorRefusal(
            f"REFUSED: claim binding mismatch for {cell_id}")
    return rec


# ------------------- C12: execution lease -------------------------
LEASE_SCHEMA = {
    "schema": str, "campaign_generation": str, "cell": str,
    "attempt_id": str, "authorization_sha256": str,
    "materialization_sha256": str, "issued_monotonic": float,
    "holder_pid": int,
    "recovery_acta_sha256": str,
    "pinned_execution_commit": str,
    "lease_sha256": str}
LEASE_SCHEMA_NAME = "agent_multi.b4_execution_lease.v3"


def issue_lease(results_root: Path, cell_id: str, claim: dict,
                auth_sha: str, mat_root: Path) -> Path:
    # C36: the execution capability cannot be issued while the
    # recovery gate is closed — the witness is re-derived HERE.
    witness = b4a.require_v6_launch_open()
    if claim.get("recovery_acta_sha256") != \
            witness["acta_sha256"]:
        raise OrchestratorRefusal(
            "REFUSED: claim recovery-acta digest differs from "
            "the re-derived witness — transplanted authority")
    lease = {"schema": LEASE_SCHEMA_NAME,
             "campaign_generation": b4a.CAMPAIGN_GENERATION,
             "cell": cell_id,
             "attempt_id": claim["attempt_id"],
             "authorization_sha256": auth_sha,
             "materialization_sha256":
                 _sha_file(Path(mat_root) / "B4_MATERIALIZATION.json"),
             "issued_monotonic": float(time.monotonic()),
             "holder_pid": os.getpid(),
             "recovery_acta_sha256": witness["acta_sha256"],
             "pinned_execution_commit":
                 witness["pinned_commit"]}
    lease["lease_sha256"] = hashlib.sha256(json.dumps(
        {k: lease[k] for k in sorted(lease)},
        sort_keys=True).encode()).hexdigest()
    p = (Path(results_root) / cell_id /
         f"LEASE_{claim['attempt_id']}.json")
    _excl_write(p, json.dumps(lease, indent=1).encode())
    return p


def verify_lease(lease_path: Path, results_root: Path,
                 cell_id: str, mat_root: Path,
                 expected_auth_sha: str = None) -> dict:
    """C17: the lease is an EXECUTION CAPABILITY, not a file beside
    a claim — exact schema and primitive types, immutable content
    digest, the LIVE campaign authorization digest, and holder
    identity bound across lease == claim == global lock == the
    executing process. Revalidate under the same lock immediately
    before entering the pipeline; any foreign element produces zero
    compute."""
    p = Path(lease_path)
    lease = _secure_json(p, "execution lease")
    if set(lease) != set(LEASE_SCHEMA):
        raise OrchestratorRefusal(
            "REFUSED: lease keys are not the exact schema")
    for k, t in LEASE_SCHEMA.items():
        if type(lease[k]) is not t:
            raise OrchestratorRefusal(
                f"REFUSED: lease field {k!r} has a foreign "
                "primitive type")
    if lease["schema"] != LEASE_SCHEMA_NAME:
        raise OrchestratorRefusal(
            "REFUSED: foreign lease schema")
    body = {k: lease[k] for k in sorted(lease)
            if k != "lease_sha256"}
    if hashlib.sha256(json.dumps(
            body, sort_keys=True).encode()).hexdigest() != \
            lease["lease_sha256"]:
        raise OrchestratorRefusal(
            "REFUSED: lease content digest does not re-derive — "
            "the capability was altered")
    if lease["campaign_generation"] != b4a.CAMPAIGN_GENERATION \
            or lease["cell"] != cell_id:
        raise OrchestratorRefusal(
            "REFUSED: lease generation/cell binding mismatch")
    if expected_auth_sha is not None and \
            lease["authorization_sha256"] != expected_auth_sha:
        raise OrchestratorRefusal(
            "REFUSED: lease authorization digest differs from the "
            "reviewed campaign authorization")
    claim = load_claim(results_root, cell_id)
    if claim["attempt_id"] != lease["attempt_id"]:
        raise OrchestratorRefusal(
            "REFUSED: lease attempt differs from the unique claim")
    # C36: the capability is valid ONLY under the re-derived
    # recovery witness — caller-supplied values grant nothing.
    witness = b4a.require_v6_launch_open()
    if lease["recovery_acta_sha256"] != witness["acta_sha256"] \
            or lease["pinned_execution_commit"] != \
            witness["pinned_commit"] or \
            claim.get("recovery_acta_sha256") != \
            witness["acta_sha256"]:
        raise OrchestratorRefusal(
            "REFUSED: lease/claim recovery bindings differ from "
            "the re-derived witness — transplanted authority")
    if seal_state(results_root, cell_id) != "UNSEALED":
        raise OrchestratorRefusal(
            "REFUSED: the claimed attempt already reached a sealed "
            "or uncertain terminal")
    terminal = Path(results_root) / cell_id / "B4_CELL_TERMINAL.json"
    if terminal.exists():
        raise OrchestratorRefusal(
            "REFUSED: a terminal already exists for this cell")
    cur = current_lock_epoch(Path(results_root))
    if cur["state"] != "HELD":
        raise OrchestratorRefusal(
            "REFUSED: no live campaign lease/lock covers this "
            f"execution (lock state {cur['state']})")
    lockrec = cur["record"]
    if lockrec["generation"] != b4a.CAMPAIGN_GENERATION:
        raise OrchestratorRefusal(
            "REFUSED: campaign lock belongs to another generation")
    me = os.getpid()
    if not (lease["holder_pid"] == lockrec["holder_pid"]
            == claim.get("holder_pid") == me):
        raise OrchestratorRefusal(
            "REFUSED: lease/claim/lock holder identity does not "
            "bind to the executing process")
    mat_sha = _sha_file(Path(mat_root) / "B4_MATERIALIZATION.json")
    if lease["materialization_sha256"] != mat_sha:
        raise OrchestratorRefusal(
            "REFUSED: lease materialization digest is stale")
    return lease


# ---------------- C18: intent/completion durable seal -------------
def _seal_paths(results_root: Path, cell_id: str,
                attempt_id: str):
    d = Path(results_root) / cell_id
    return (d / f"SEAL_INTENT_{attempt_id}.json",
            d / f"SEAL_COMPLETE_{attempt_id}.json")


def seal_attempt(results_root: Path, cell_id: str,
                 attempt_id: str) -> None:
    """C18: append-only intent/completion — never an overwrite whose
    only proof is the final fsync. Recovery reads PHYSICAL data: a
    complete, self-integral completion that matches its intent seals
    the terminal; anything else is UNCERTAIN."""
    cell_dir = Path(results_root) / cell_id
    claim = load_claim(results_root, cell_id)
    if claim["attempt_id"] != attempt_id:
        raise OrchestratorRefusal(
            "REFUSED: sealing a foreign attempt")
    terminal = cell_dir / "B4_CELL_TERMINAL.json"
    if not terminal.is_file():
        raise OrchestratorRefusal(
            "REFUSED: no terminal exists to seal — the attempt "
            "stays UNCERTAIN for operator disposition")
    intent_p, complete_p = _seal_paths(results_root, cell_id,
                                       attempt_id)
    if complete_p.exists():
        raise OrchestratorRefusal("REFUSED: attempt already sealed")
    term_sha = hashlib.sha256(_secure_read(terminal)).hexdigest()
    intent = {"schema": "agent_multi.b4_seal_intent.v1",
              "campaign_generation": b4a.CAMPAIGN_GENERATION,
              "cell": cell_id, "attempt_id": attempt_id,
              "holder_pid": os.getpid(),
              "terminal_sha256": term_sha,
              # C37: the seal names the recovery authority via its
              # verified parent binding (the terminal carries the
              # full witness re-derived by the verifiers).
              "recovery_acta_sha256":
                  b4a.require_v6_launch_open()["acta_sha256"]}
    _excl_write(intent_p, json.dumps(intent, indent=1).encode())
    completion = {"schema": "agent_multi.b4_seal_completion.v1",
                  "intent_sha256": _sha_file(intent_p),
                  "terminal_sha256": term_sha,
                  "attempt_id": attempt_id}
    completion["completion_sha256"] = hashlib.sha256(json.dumps(
        {k: completion[k] for k in sorted(completion)},
        sort_keys=True).encode()).hexdigest()
    _excl_write(complete_p,
                json.dumps(completion, indent=1).encode())


def seal_state(results_root: Path, cell_id: str) -> str:
    """PHYSICAL adjudication of the seal: SEALED only when a
    complete, self-integral completion matches its intent AND the
    live terminal bytes; UNCERTAIN on any partial, malformed or
    transplanted witness; UNSEALED when neither exists."""
    try:
        claim = load_claim(results_root, cell_id)
    except SystemExit:
        return "NO_CLAIM"
    attempt_id = claim["attempt_id"]
    intent_p, complete_p = _seal_paths(results_root, cell_id,
                                       attempt_id)
    terminal = Path(results_root) / cell_id / "B4_CELL_TERMINAL.json"
    if not intent_p.exists() and not complete_p.exists():
        return "UNSEALED"
    if not complete_p.exists() or not intent_p.exists() or \
            not terminal.is_file():
        return "UNCERTAIN"
    try:
        completion = _secure_json(complete_p, "seal completion")
        body = {k: completion[k] for k in sorted(completion)
                if k != "completion_sha256"}
        if hashlib.sha256(json.dumps(
                body, sort_keys=True).encode()).hexdigest() != \
                completion.get("completion_sha256"):
            return "UNCERTAIN"
        if completion.get("intent_sha256") != hashlib.sha256(
                _secure_read(intent_p)).hexdigest():
            return "UNCERTAIN"
        if completion.get("attempt_id") != attempt_id:
            return "UNCERTAIN"
        if completion.get("terminal_sha256") != hashlib.sha256(
                _secure_read(terminal)).hexdigest():
            return "UNCERTAIN"
    except (SystemExit, OSError):
        return "UNCERTAIN"
    return "SEALED"


# ---------------- C13: global remaining wall ----------------------
def gpu_seconds_spent(results_root: Path) -> float:
    """Durable elapsed facts from EVERY attempt (failed and
    uncertain included). Malformed durations fail closed."""
    total = 0.0
    now = time.time()
    for claim_p in Path(results_root).glob("*/CLAIM_*.json"):
        rec = _secure_json(claim_p, claim_p.name)
        term = claim_p.parent / "B4_CELL_TERMINAL.json"
        if term.exists():
            t = _secure_json(term, term.name)
            w = t.get("wall_seconds")
            if type(w) not in (int, float) or w < 0 or \
                    not (w == w):
                raise OrchestratorRefusal(
                    f"REFUSED: malformed terminal duration in "
                    f"{claim_p.parent.name} — failing closed")
            total += float(w)
        else:
            start = rec.get("claimed_wall")
            if type(start) not in (int, float):
                raise OrchestratorRefusal(
                    "REFUSED: claim without a start fact — failing "
                    "closed")
            elapsed = now - float(start)
            if elapsed < 0:
                raise OrchestratorRefusal(
                    "REFUSED: clock rollback detected — failing "
                    "closed")
            total += elapsed
    return total


def remaining_global_seconds(results_root: Path,
                             limits: dict) -> float:
    """C32: the 96h budget NEVER restarts across generations — the
    superseded v5 generation's fixed charge from the incident acta
    (0.01 h) is deducted before this root's own spending."""
    ceiling = float(limits["global_gpu_hours_ceiling"]) * 3600.0
    return (ceiling - b4a.PRIOR_GENERATIONS_GPU_SECONDS
            - gpu_seconds_spent(results_root))


# ------------------- C15: resume adjudication ---------------------
def adjudicate_cell_state(results_root: Path, cell_id: str) -> str:
    cell_dir = Path(results_root) / cell_id
    claim_p = _claim_path(results_root, cell_id)
    terminal = cell_dir / "B4_CELL_TERMINAL.json"
    if not claim_p.exists() and not terminal.exists():
        return "PENDING"
    if claim_p.exists() and not terminal.exists():
        return "AMBIGUOUS_CLAIM"
    try:
        claim = load_claim(results_root, cell_id)
    except SystemExit:
        return "UNCERTAIN"
    try:
        term = _secure_json(terminal, f"terminal {cell_id}")
    except SystemExit:
        return "UNCERTAIN"
    seal = seal_state(results_root, cell_id)
    if seal != "SEALED":
        return "UNCERTAIN"          # unsealed/partial never accepted
    if term.get("cell") != cell_id or \
            term.get("attempt_id") != claim["attempt_id"]:
        return "UNCERTAIN"
    if term.get("terminal") == "COMPLETED":
        return "COMPLETED_VERIFIED"
    if term.get("terminal") in ("FAILED", "TIMED_OUT",
                                "THERMAL_STOP", "RESOURCE_STOP",
                                "EXTERNALLY_STOPPED",
                                # C31: typed post-claim failures
                                "FAILED_PLUGIN_ENVIRONMENT",
                                "FAILED_CONSTRUCTION",
                                "FAILED_PREFLIGHT_TYPED"):
        return f"TERMINAL_{term['terminal']}"
    return "UNCERTAIN"


def runtime_health(results_root: Path, device: str) -> dict:
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4run_orch", REPO / "tools/b4_run_cell.py")
    runner = ilu.module_from_spec(spec)
    spec.loader.exec_module(runner)
    health = {"device_available": False,
              "stop_file_present":
                  (Path(results_root) / "CAMPAIGN_STOP").exists(),
              "compute_apps_active": False}
    try:
        inv = runner.gpu_inventory(device)
        apps = runner.gpu_compute_apps(device)
    except SystemExit:
        return health
    if apps is None:
        return health
    health["device_available"] = True
    health["compute_apps_active"] = any(
        a["used_memory_mib"] > SUBSTANTIAL_COMPUTE_MIB for a in apps)
    return health


def _snapshot(root: Path) -> dict:
    if not Path(root).exists():
        return {}
    return {str(p.relative_to(root)): _sha_file(p)
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def run_campaign(mat_root: Path, ledger_path: Path,
                 results_root: Path, device: str,
                 execute: bool,
                 continue_after_failed: bool = False) -> int:
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4led_orch", REPO / "tools/b4_campaign_ledger.py")
    ledger_mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(ledger_mod)
    espec = ilu.spec_from_file_location(
        "b4exec_orch", REPO / "tools/b4_campaign_executor.py")
    executor = ilu.module_from_spec(espec)
    espec.loader.exec_module(executor)
    limits = b4a.load_resource_contract()
    results_root = Path(results_root)

    # C32: NO path — dry-run included — may interpret a
    # superseded-generation root: a v5 claim adjudicated by v6
    # code would look PENDING, which is a lie. Foreign objects
    # refuse before any state is reported.
    if Path(results_root).exists():
        for claim_p in Path(results_root).glob("*/CLAIM_*.json"):
            if claim_p.name != \
                    f"CLAIM_{b4a.CAMPAIGN_GENERATION}.json":
                raise OrchestratorRefusal(
                    f"REFUSED: results root holds a foreign-"
                    f"generation object {claim_p.name} — "
                    "superseded incident roots are immutable "
                    "history, never reused or re-adjudicated")
    # ---- C11: pure dry-run — ZERO writes anywhere ----
    if not execute:
        pre = _snapshot(results_root)
        # C30: the strong dry-run validates the EXECUTION
        # ENVIRONMENT — interpreter, versions, CUDA, entry points,
        # effective plugin imports from the frozen checkout,
        # dependencies and live authority — with zero writes. The
        # incident's plugin-blind dry-run is dead.
        env_facts = executor.preflight_environment(device)
        ledger = ledger_mod.verify_ledger(ledger_path, mat_root)
        b4a.verify_campaign_materialization(mat_root)
        states = {cid: adjudicate_cell_state(results_root, cid)
                  for cid in ledger_mod.EXPECTED_CELLS}
        plan = [cid for cid, st in states.items()
                if st == "PENDING"]
        spent_h = (gpu_seconds_spent(results_root) / 3600.0
                   if results_root.exists() else 0.0)
        prior_h = b4a.PRIOR_GENERATIONS_GPU_SECONDS / 3600.0
        print(json.dumps({
            "dry_run": True, "writes": 0,
            "generation": b4a.CAMPAIGN_GENERATION,
            "environment_preflight": env_facts,
            "cell_states": states,
            "dispatch_plan_in_order": plan,
            "gpu_hours_spent": round(spent_h, 2),
            "gpu_hours_charged_prior_generations":
                round(prior_h, 2),
            "gpu_hours_remaining": round(
                limits["global_gpu_hours_ceiling"] - prior_h
                - spent_h, 2),
        }, indent=1))
        post = _snapshot(results_root)
        if pre != post:
            raise OrchestratorRefusal(
                "REFUSED: dry-run altered the result root — "
                "impossible state, failing closed")
        return 0

    if executor.CAMPAIGN_AUTH_SHA is None:
        raise OrchestratorRefusal(
            "REFUSED: no Musashi campaign authorization record — "
            "the orchestrator dispatches nothing")
    # C33: the v6 launch is CLOSED until the external Musashi
    # recovery-audit acta exists — the candidate submission grants
    # nothing.
    b4a.require_v6_launch_open()
    # C30: the environment preflight runs with ZERO writes BEFORE
    # any claim, lease, binding or origin contract exists.
    executor.preflight_environment(device)
    ledger = ledger_mod.verify_ledger(ledger_path, mat_root)
    if not results_root.exists():
        os.makedirs(str(results_root), mode=0o700)
    _secure_dir(results_root)
    outcome = {"completed": [], "failed": [], "uncertain": [],
               "pending": []}
    with GlobalLock(results_root):
        for cid in ledger_mod.EXPECTED_CELLS:
            state = adjudicate_cell_state(results_root, cid)
            if state == "COMPLETED_VERIFIED":
                outcome["completed"].append(cid)
                continue
            if state.startswith("TERMINAL_"):
                outcome["failed"].append(f"{cid}:{state}")
                if not continue_after_failed:
                    raise OrchestratorRefusal(
                        f"REFUSED: {cid} holds verified terminal "
                        f"{state} and the reviewed policy does not "
                        "say to collect the remaining cells")
                continue
            if state in ("AMBIGUOUS_CLAIM", "UNCERTAIN"):
                raise OrchestratorRefusal(
                    f"REFUSED: {cid} is {state} — blocked for "
                    "operator disposition")
            remaining = remaining_global_seconds(results_root,
                                                 limits)
            if remaining < MIN_SEGMENT_SECONDS:
                raise OrchestratorRefusal(
                    "STOPPED_RESOURCE_CEILING: remaining global "
                    f"wall {remaining:.0f}s < one segment")
            gpu_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            health = (runtime_health(results_root, gpu_dev)
                      if device.startswith("cuda") else
                      {"device_available": True,
                       "stop_file_present":
                           (results_root / "CAMPAIGN_STOP"
                            ).exists(),
                       "compute_apps_active": False})
            nxt = ledger_mod.schedule_next(
                {"cells": {c: {"status":
                               ("PENDING" if c == cid else "DONE")}
                           for c in ledger_mod.EXPECTED_CELLS}},
                {k: health[k] for k in
                 ("device_available", "stop_file_present",
                  "compute_apps_active")})
            if nxt.startswith("HOLD"):
                raise OrchestratorRefusal(f"REFUSED: {nxt}")
            claim = claim_attempt(results_root, cid)
            lease = issue_lease(results_root, cid, claim,
                                executor.CAMPAIGN_AUTH_SHA,
                                mat_root)
            executor.execute_cell(
                cid, mat_root, results_root, device,
                lease_path=lease,
                global_wall_remaining_seconds=remaining)
            seal_attempt(results_root, cid, claim["attempt_id"])
            outcome["completed"].append(cid)
    # C20: scientific completion REQUIRES the strongest verifier —
    # comparator evidence derived from the reviewed materialization,
    # impossible to omit.
    if not outcome["failed"] and not outcome["uncertain"]:
        ledger_mod.verify_campaign_results(
            ledger_path, mat_root, results_root)
        status = "CAMPAIGN_COMPLETE"
        outcome["authorization_record_sha256"] = _sha_file(
            b4a.CAMPAIGN_AUTHORIZATION_RECORD_PATH)
        outcome["amendment_11_sha256"] = _sha_file(
            b4a.AMENDMENT_11_PATH)
        # C37: the final report binds the recovered authority,
        # re-derived at reporting time.
        _wit = b4a.require_v6_launch_open()
        outcome["campaign_generation"] = \
            _wit["campaign_generation"]
        outcome["recovery_acta_sha256"] = _wit["acta_sha256"]
        outcome["pinned_execution_commit"] = \
            _wit["pinned_commit"]
        outcome["latest_amendment_sha256"] = \
            _wit["latest_amendment_sha256"]
    else:
        status = "CAMPAIGN_COMPLETE_WITH_FAILED_CELLS"
    print(json.dumps({"status": status, **outcome}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialization-root", type=Path,
                    required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "cuda:0"])
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    return run_campaign(args.materialization_root, args.ledger,
                        args.results_root, args.device,
                        args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
