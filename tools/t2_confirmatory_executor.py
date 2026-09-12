#!/usr/bin/env python3
"""T2 confirmatory EXECUTOR (orders C42-C47 + C48-C56 + C57-C65)
— implemented, STRUCTURALLY CLOSED for science: `--execute`
consumes the SEALED v6 design only after ALL pure gates open,
including the external Musashi v2 EXECUTION record. That record
does not exist; no confirmatory score is computed.

C57: wall authority that a restart cannot renew — a hash-chained,
strictly parsed, DESCRIPTOR-BOUND append-only ledger consumed and
appended through ONE file descriptor; every interval is CHARGED
IN ADVANCE by a durable reservation (fsynced before the interval
may execute) and closed with its real elapsed; a reservation
without a close charges in full, so repeated sub-cadence crashes
can never recover time. Interior malformed/duplicated/reordered/
transplanted records fail closed; only a torn FINAL line (a crash
mid-append) is tolerated, because nothing executes before its
reservation's fsync returns. Boot identity is recorded; monotonic
clocks are never compared across sessions. Ledger replacement
between read and append is detected (inode identity at close).

C58: the lock release is an immutable RELEASE_INTENT plus a
separate RELEASE_DONE completion witness, both strict-schema,
self-integral and bound to the exact session record digest, UUID,
holder pid and campaign generation. An absent, empty, malformed,
permissive, symlinked, stale, transplanted or fsync-uncertain
completion never frees the lock; reclaim re-reads and revalidates
both AFTER winning the exclusive election.

C59: claims and terminals are v3 authority-bound objects carrying
the sealed design (physical+self), review/execution records,
manifest, census, executor code identity, pinned commit/tree,
campaign generation, the exact unit binding, and (terminals) the
digest of the claim they close. verify_unit_terminal() re-derives
ALL of it against current physical authority; a stale or
transplanted terminal adjudicates typed UNCERTAIN, never
TERMINAL_FAILED.

C60: operator disposition is EXTERNAL authority: the CLI first
passes the same current gates as execution, fully verifies the
uncertain claim, and then consumes a separate Musashi disposition
record at the private reviewer root pinning the claim digest,
unit, attempt, current execution record, decision and reason.
Candidate-authored text grants nothing.

C61: the results root is opened from a fixed trusted parent,
component by component with O_NOFOLLOW; the root and control
directories require exact uid/0700 and control/evidence objects
uid/0600; production writes go through directory descriptors, so
a symlink root or intermediate component, a permissive
preexisting directory or a path replacement fails before writes.
The heartbeat uses a random per-write temp name — no fixed shared
.tmp path.

C62: the fit supervisor receives the live wall authority, uses
min(remaining_global, per_fit_limit), refuses to start with no
positive budget, durably charges the non-interruptible interval
BEFORE dispatch, verifies the child exited (kill+reap otherwise)
after every harvest — OK, WALL_KILLED, RSS, CRASH and EOF — and
leaves no orphan. The RSS bound is enforced PER PROCESS (child
address-space ceiling at 2x the bound plus a post-fit peak check;
the parent's peak is checked at wall checkpoints); it is NOT a
bound on the simultaneous parent+child resident total.

C63: physical adjudication controls process success — any
persistence or verification failure after a claim halts the
campaign immediately as typed uncertainty (never counted and
continued), and before the lock is released with exit 0 EVERY
population unit is re-adjudicated deeply under current authority:
records fully re-verified against rebuilt physical series,
terminals fully re-verified, zero UNCERTAIN, counts equal to the
sealed population exactly."""
import argparse
import hashlib
import io
import json
import os
import resource
import stat as _stat
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import t2_confirmatory as conf  # noqa: E402
import t2_bank as bank  # noqa: E402
import t2_bank_census as census_mod  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"
SEALED_PATH = STATE / "t2_screen_design_SEALED_V6.json"
SUCCESSOR_PATH = (STATE
                  / "t2_screen_design_RESOURCE_SUCCESSOR_V1"
                  ".json")


def active_design_path() -> Path:
    """C76: the resource-only successor supersedes v6 for
    EXECUTION when present; v6 itself stays byte-immutable."""
    return SUCCESSOR_PATH if SUCCESSOR_PATH.exists() \
        else SEALED_PATH
MANIFEST_PATH = STATE / "t2_public_data_manifest_20260906.json"
CENSUS_PATH = STATE / "t2_bank_census_20260906.json"
DEV_UNITS = ("sm_co2", "sm_sunspots", "sm_nile")
MODES = ("confirmatory", "mechanical_rehearsal")
REHEARSAL_EXECUTION_SENTINEL = (
    "REHEARSAL_NO_EXECUTION_RECORD_MECHANICS_ONLY")
REHEARSAL_DIGEST_SENTINEL = "REHEARSAL_DEV_CENSUS_ONLY"
NO_DISPOSITION_SENTINEL = "NOT_AN_OPERATOR_DISPOSITION"
T2_CAMPAIGN_GENERATION = "t2_confirmatory_v6_generation_20260907"
PER_FIT_WALL_SECONDS = 120.0
RESERVE_QUANTUM_S = 30.0
# C66: the grammar bound on any single reservation — the idle
# quantum or one full supervised-fit charge, whichever is larger.
MAX_RESERVE_S = max(RESERVE_QUANTUM_S, PER_FIT_WALL_SECONDS)
BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
_TRUSTED_ROOT_PARENTS = (STATE, Path.home() / ".cache")

_WRAPPER_KEYS = {
    "schema", "mode", "unit_id", "attempt_id",
    "sealed_design_file_sha256", "sealed_design_self_sha256",
    "design_review_record_sha256", "execution_record_sha256",
    "manifest_sha256", "census_sha256", "unit_binding",
    "code_identity", "assay_record", "arrays_npz_sha256",
    "wall_seconds", "record_sha256"}
_REHEARSAL_BINDING_KEYS = {
    "dataset", "family", "seasonal_period", "horizon", "n_obs",
    "series_numeric_sha256", "origin_windows"}
_AUTHORITY_KEYS = (
    "sealed_design_file_sha256", "sealed_design_self_sha256",
    "design_review_record_sha256", "execution_record_sha256",
    "manifest_sha256", "census_sha256")
_CLAIM_KEYS = {
    "schema", "unit_id", "attempt_id", "mode",
    "campaign_generation", *_AUTHORITY_KEYS, "code_identity",
    "pinned_commit", "pinned_tree", "unit_binding", "pid",
    "claimed_wall", "claim_sha256"}
_TERMINAL_KEYS = {
    "schema", "unit_id", "attempt_id", "mode",
    "campaign_generation", *_AUTHORITY_KEYS, "code_identity",
    "pinned_commit", "pinned_tree", "unit_binding",
    "closed_claim_sha256", "terminal", "failure_class", "reason",
    "operator_disposition", "disposition_record_sha256",
    "wall_seconds", "terminal_sha256"}


class ExecutorRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


class T2BudgetStop(SystemExit):
    """C52.5: a typed budget stop naming its exact checkpoint. It
    deliberately does NOT write a unit terminal: an interrupted
    attempt adjudicates UNCERTAIN and awaits the EXTERNAL recorded
    operator disposition — it is never silently reused."""
    def __init__(self, msg, checkpoint):
        self.checkpoint = checkpoint
        super().__init__(f"T2_BUDGET_STOP at {checkpoint}: {msg}")


class T2AssayFailed(Exception):
    """C63: an assay-phase failure whose integral typed terminal
    WAS durably written — the only failure the campaign may count
    and continue past. Everything else halts."""
    def __init__(self, uid, reason):
        self.uid = uid
        super().__init__(f"{uid}: assay failed with terminal "
                         f"written ({reason})")


class SupervisedFitFailure(SystemExit):
    """Typed harvest of a supervised worker that did not return."""
    def __init__(self, msg):
        super().__init__(f"SUPERVISED_FIT_FAILURE: {msg}")


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _self_sha(body: dict, exclude: str) -> str:
    return hashlib.sha256(json.dumps(
        {k: body[k] for k in sorted(body) if k != exclude},
        sort_keys=True).encode()).hexdigest()


def _strict_parse(raw: bytes, what: str) -> dict:
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ExecutorRefusal(f"duplicate JSON key in {what}")
        return dict(pairs)
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_no_dupes,
            parse_constant=lambda c: (_ for _ in ()).throw(
                ExecutorRefusal(f"non-finite constant in {what}")))
    except json.JSONDecodeError as exc:
        raise ExecutorRefusal(
            f"{what} is not well-formed JSON ({exc.msg})")


def code_identity() -> dict:
    return conf.executor_code_identity(REPO)


def _git_head_tree() -> tuple:
    import subprocess
    h = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                        "HEAD"], capture_output=True,
                       text=True).stdout.strip()
    t = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                        "HEAD^{tree}"], capture_output=True,
                       text=True).stdout.strip()
    return h, t


def _boot_id() -> str:
    try:
        with open(BOOT_ID_PATH) as f:
            b = f.read().strip()
    except OSError:
        raise ExecutorRefusal(
            "boot identity unavailable — clock authority is "
            "ambiguous; stopping for review, never renewing the "
            "budget")
    if len(b) < 8:
        raise ExecutorRefusal("boot identity malformed")
    return b


# ---------------- C61: root and directory custody ----------------

class ResultsRoot:
    """C61: the results root opened from a fixed trusted parent,
    component by component with O_NOFOLLOW; the root and its
    control directories (units/, locks/) are exact uid/0700 and
    every production write goes through these directory
    descriptors. A symlink root or intermediate component, a
    permissive or foreign-owned directory, a non-normal path or a
    replaced path fails BEFORE any write."""

    def __init__(self, out_root: Path, create: bool = False):
        if ".." in Path(str(out_root)).parts:
            raise ExecutorRefusal(
                "results root path is not normal")
        rp = Path(os.path.abspath(str(out_root)))
        if not any(par == rp or par in rp.parents
                   for par in _TRUSTED_ROOT_PARENTS):
            raise ExecutorRefusal(
                "results root must live under the trusted state "
                "or cache parents — foreign roots are refused")
        self.path = rp
        parts = rp.parts
        fd = os.open("/", os.O_RDONLY
                     | getattr(os, "O_DIRECTORY", 0))
        try:
            for comp in parts[1:-1]:
                nfd = os.open(comp, os.O_RDONLY | os.O_NOFOLLOW
                              | getattr(os, "O_DIRECTORY", 0),
                              dir_fd=fd)
                os.close(fd)
                fd = nfd
            leaf = parts[-1]
            try:
                rfd = self._open_dir_at(fd, leaf)
            except FileNotFoundError:
                if not create:
                    os.close(fd)
                    raise ExecutorRefusal(
                        "results root does not exist")
                os.mkdir(leaf, mode=0o700, dir_fd=fd)
                rfd = self._open_dir_at(fd, leaf)
        except OSError as exc:
            try:
                os.close(fd)
            except OSError:
                pass
            raise ExecutorRefusal(
                "results-root path component refuses O_NOFOLLOW "
                f"custody (errno {exc.errno}: {exc.strerror}) — "
                "a symlink or non-directory component fails "
                "before writes")
        os.close(fd)
        self.root_fd = rfd
        self._require_private_dir(self.root_fd, "results root")
        self.units_fd = self._ensure_subdir("units")
        self.locks_fd = self._ensure_subdir("locks")
        self._pins = {}
        for name, dfd in (("root", self.root_fd),
                          ("units", self.units_fd),
                          ("locks", self.locks_fd)):
            st = os.fstat(dfd)
            self._pins[name] = (st.st_dev, st.st_ino)

    def revalidate(self):
        """C69: re-open the DECLARED path component by component
        with O_NOFOLLOW and prove it still names the held
        objects (device/inode pins for the root and every
        control directory). A rename-and-replace refuses before
        any side effect."""
        parts = self.path.parts
        fd = os.open("/", os.O_RDONLY
                     | getattr(os, "O_DIRECTORY", 0))
        try:
            for comp in parts[1:]:
                nfd = self._open_dir_at(fd, comp)
                os.close(fd)
                fd = nfd
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != self._pins["root"]:
                raise ExecutorRefusal(
                    "RESULTS_ROOT_IDENTITY_LOST: the declared "
                    "results-root path no longer names the held "
                    "root object (renamed or replaced) — "
                    "refusing before any side effect")
            for name in ("units", "locks"):
                sfd = self._open_dir_at(fd, name)
                try:
                    st2 = os.fstat(sfd)
                    if (st2.st_dev, st2.st_ino) != \
                            self._pins[name]:
                        raise ExecutorRefusal(
                            "RESULTS_ROOT_IDENTITY_LOST: control "
                            f"directory {name!r} at the declared "
                            "path is not the held object")
                finally:
                    os.close(sfd)
        except ExecutorRefusal:
            raise
        except OSError as exc:
            raise ExecutorRefusal(
                "RESULTS_ROOT_IDENTITY_LOST: the declared path "
                f"cannot be re-walked (errno {exc.errno}) — "
                "refusing before any side effect")
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    @staticmethod
    def _open_dir_at(dfd, name):
        return os.open(name, os.O_RDONLY | os.O_NOFOLLOW
                       | getattr(os, "O_DIRECTORY", 0),
                       dir_fd=dfd)

    @staticmethod
    def _require_private_dir(dfd, what):
        st = os.fstat(dfd)
        if not _stat.S_ISDIR(st.st_mode):
            raise ExecutorRefusal(f"{what} is not a directory")
        if st.st_uid != os.getuid():
            raise ExecutorRefusal(f"{what} has a foreign owner")
        if _stat.S_IMODE(st.st_mode) != 0o700:
            raise ExecutorRefusal(
                f"{what} mode "
                f"{oct(_stat.S_IMODE(st.st_mode))} is not the "
                "exact private 0700 — refused, never chmodded")

    def _ensure_subdir(self, name):
        try:
            fd = self._open_dir_at(self.root_fd, name)
        except FileNotFoundError:
            os.mkdir(name, mode=0o700, dir_fd=self.root_fd)
            fd = self._open_dir_at(self.root_fd, name)
        except OSError as exc:
            raise ExecutorRefusal(
                f"control directory {name!r} refuses custody "
                f"(errno {exc.errno})")
        self._require_private_dir(fd, f"control directory {name}")
        return fd

    def _post_identity(self, op: str):
        """C74: the truthful contract — no effect or read is
        accepted unless the DECLARED path names the held root
        both immediately before AND immediately after the
        operation. A failed post-check is a typed custody loss:
        the effect is NOT accepted, any in-flight unit stays
        UNCERTAIN, the lock is never released and continuation
        requires external disposition."""
        try:
            self.revalidate()
        except SystemExit:
            raise ExecutorRefusal(
                f"RESULTS_ROOT_CUSTODY_LOST_AFTER_{op}: the "
                "declared path no longer names the held root — "
                "the operation's effect is NOT accepted as "
                "successful; the attempt is UNCERTAIN; the lock "
                "stays held; external disposition is required "
                "before continuation")

    # -- descriptor-relative object IO --
    def excl_write(self, dfd, name, payload: bytes):
        self.revalidate()
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                     | os.O_NOFOLLOW, 0o600, dir_fd=dfd)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(dfd)
        self._post_identity("WRITE")

    def read_private(self, dfd, name, what) -> bytes:
        self.revalidate()
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW,
                         dir_fd=dfd)
        except FileNotFoundError:
            raise ExecutorRefusal(f"{what} does not exist")
        except OSError as exc:
            raise ExecutorRefusal(
                f"{what} unopenable without following links "
                f"(errno {exc.errno})")
        try:
            st = os.fstat(fd)
            if not _stat.S_ISREG(st.st_mode):
                raise ExecutorRefusal(
                    f"{what} is not a regular file")
            if st.st_uid != os.getuid():
                raise ExecutorRefusal(f"{what} has a foreign "
                                      "owner")
            if _stat.S_IMODE(st.st_mode) != 0o600:
                raise ExecutorRefusal(
                    f"{what} mode "
                    f"{oct(_stat.S_IMODE(st.st_mode))} is not "
                    "the exact private 0600")
            chunks = []
            while True:
                b = os.read(fd, 1 << 20)
                if not b:
                    break
                chunks.append(b)
        finally:
            os.close(fd)
        # C75: bytes from an obsolete root are never evidence
        self._post_identity("READ")
        return b"".join(chunks)

    def exists(self, dfd, name) -> bool:
        self.revalidate()
        try:
            os.stat(name, dir_fd=dfd, follow_symlinks=False)
            found = True
        except FileNotFoundError:
            found = False
        self._post_identity("STAT")
        return found

    def listdir(self, dfd):
        self.revalidate()
        fd2 = os.dup(dfd)
        try:
            with os.scandir(f"/proc/self/fd/{fd2}") as it:
                names = sorted(e.name for e in it)
        finally:
            os.close(fd2)
        self._post_identity("LIST")
        return names

    def close(self):
        for fd in (self.units_fd, self.locks_fd, self.root_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def _fsync_dir(d: Path) -> None:
    dfd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _excl_write(path: Path, payload: bytes) -> None:
    """Path-form exclusive create (0600, file+dir fsync) for
    objects outside a ResultsRoot's descriptors."""
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _read_evidence(src, what: str) -> bytes:
    """C69: production passes (ResultsRoot, name) so reads go
    through the HELD units descriptor; a bare Path stays for
    battery/inspection use."""
    if isinstance(src, tuple):
        rr, name = src
        return rr.read_private(rr.units_fd, name, what)
    return _read_private_bytes(Path(src), what)


def _src_name(src) -> str:
    return src[1] if isinstance(src, tuple) else Path(src).name


def _src_sibling(src, name: str):
    if isinstance(src, tuple):
        return (src[0], name)
    return Path(src).parent / name


def _src_exists(src) -> bool:
    if isinstance(src, tuple):
        rr, name = src
        return rr.exists(rr.units_fd, name)
    return Path(src).exists()


def _read_private_bytes(path: Path, what: str) -> bytes:
    try:
        fd = os.open(str(path), os.O_RDONLY
                     | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        raise ExecutorRefusal(f"{what} does not exist at "
                              f"{Path(path).name!r}")
    except OSError as exc:
        raise ExecutorRefusal(
            f"{what} unopenable without following links "
            f"(errno {exc.errno})")
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise ExecutorRefusal(f"{what} is not a regular file")
        if st.st_uid != os.getuid():
            raise ExecutorRefusal(f"{what} has a foreign owner")
        if _stat.S_IMODE(st.st_mode) != 0o600:
            raise ExecutorRefusal(
                f"{what} mode "
                f"{oct(_stat.S_IMODE(st.st_mode))} is not the "
                "exact private 0600")
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
    finally:
        os.close(fd)
    return b"".join(chunks)


def resolve_stop_file(design: dict) -> Path:
    decl = design["resource_contract"]["stop_file"]
    if decl != "<state_root>/T2_STOP":
        raise ExecutorRefusal(
            f"sealed stop_file declaration {decl!r} is not the "
            "known contract — refusing to guess a location")
    return STATE / "T2_STOP"


# ------------- C57: non-renewable wall authority -------------

_LEDGER_GENESIS = "T2_WALL_GENESIS"


class WallAuthority:
    """C57: charge-in-advance reservations on a hash-chained,
    strict, self-integral, DESCRIPTOR-BOUND append-only ledger.
    See the module docstring for the exact protocol."""

    def __init__(self, rr: ResultsRoot, limits: dict,
                 stop_file: Path, session_uuid: str,
                 ledger_name: str = "T2_WALL_LEDGER.jsonl"):
        self.rr = rr
        self.limits = limits
        self.stop_file = Path(stop_file)
        self.session_uuid = session_uuid
        self.ledger_name = ledger_name
        self.boot = _boot_id()
        try:
            self._fd = os.open(ledger_name,
                               os.O_RDWR | os.O_APPEND
                               | os.O_CREAT | os.O_NOFOLLOW,
                               0o600, dir_fd=rr.root_fd)
        except OSError as exc:
            raise ExecutorRefusal(
                f"wall ledger refuses custody (errno "
                f"{exc.errno})")
        st = os.fstat(self._fd)
        if not _stat.S_ISREG(st.st_mode) or \
                st.st_uid != os.getuid() or \
                _stat.S_IMODE(st.st_mode) != 0o600:
            os.close(self._fd)
            raise ExecutorRefusal(
                "wall ledger is not a private regular 0600 file")
        self._ino = (st.st_dev, st.st_ino)
        os.fsync(rr.root_fd)
        os.lseek(self._fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            b = os.read(self._fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        raw = b"".join(chunks)
        self.prior, self._last_sha, self._seq, _ = \
            self._replay(raw)
        self._closed_session = 0.0
        self._reserve_seq = None
        self._reserve_seconds = 0.0
        self._reserve_t0 = None
        self._append({"kind": "session_open",
                      "session": session_uuid,
                      "boot_id": self.boot,
                      "tail_torn_tolerated": False,
                      "pid": os.getpid()})
        # C57: the FIRST charge happens before any work interval
        self._rollover("session_start")

    # -- replay: C66 exact grammar + state machine --
    _LEDGER_KEYS = {
        "session_open": {"kind", "session", "boot_id",
                         "tail_torn_tolerated", "pid", "seq",
                         "prev_sha", "record_sha"},
        "reserve": {"kind", "session", "seconds", "seq",
                    "prev_sha", "record_sha"},
        "close": {"kind", "session", "reserve_seq", "elapsed",
                  "seq", "prev_sha", "record_sha"},
    }

    @staticmethod
    def _g_str(v):
        return type(v) is str and bool(v)

    @staticmethod
    def _g_posint(v):
        return type(v) is int and not isinstance(v, bool) and \
            v > 0

    @staticmethod
    def _g_finite(v):
        import math as _m
        return type(v) in (int, float) and \
            not isinstance(v, bool) and _m.isfinite(v)

    def _replay(self, raw: bytes):
        """C66: every record kind has an exact schema, exact
        primitive types and a finite positive domain BEFORE it
        affects state; C67: a torn final fragment is a REVIEW
        boundary — nothing may follow it; C68: a boot identity
        different from the current one is a REVIEW boundary."""
        charged = 0.0
        pending = {}
        opened_sessions = set()
        last_sha = _LEDGER_GENESIS
        seq = 0
        max_wall = float(self.limits["max_wall_seconds"])
        lines = raw.split(b"\n")
        body = lines[:-1] if lines and lines[-1] == b"" else lines
        if raw == b"":
            body = []
        for i, line in enumerate(body):
            is_last = (i == len(body) - 1)
            torn_candidate = is_last and lines[-1] != b""
            try:
                doc = _strict_parse(line, "wall ledger record")
            except SystemExit:
                if torn_candidate:
                    raise ExecutorRefusal(
                        "WALL_LEDGER_TORN_TAIL_REVIEW_REQUIRED: "
                        "the final ledger record is torn (a "
                        "crash mid-append); nothing may execute "
                        "behind it — external review decides "
                        "recovery, never truncation, overwrite "
                        "or continued appends")
                raise ExecutorRefusal(
                    "wall ledger interior record is malformed — "
                    "fail closed, stopping for review")
            kind = doc.get("kind")
            if kind not in self._LEDGER_KEYS:
                raise ExecutorRefusal(
                    f"wall ledger unknown record kind {kind!r} "
                    "— fail closed")
            if set(doc) != self._LEDGER_KEYS[kind]:
                raise ExecutorRefusal(
                    f"wall ledger {kind} keys are not the exact "
                    "schema — fail closed")
            if not (self._g_posint(doc["seq"])
                    and self._g_str(doc["prev_sha"])
                    and self._g_str(doc["record_sha"])
                    and self._g_str(doc["session"])):
                raise ExecutorRefusal(
                    "wall ledger chain fields are not canonical "
                    "positive/nonempty primitives — fail closed")
            if _self_sha(doc, "record_sha") != doc["record_sha"]:
                raise ExecutorRefusal(
                    "wall ledger record self-digest does not "
                    "re-derive — mutated or transplanted; fail "
                    "closed")
            if doc["prev_sha"] != last_sha or \
                    int(doc["seq"]) != seq + 1:
                raise ExecutorRefusal(
                    "wall ledger chain broken (reordered, "
                    "duplicated or replaced records) — fail "
                    "closed")
            last_sha = doc["record_sha"]
            seq = int(doc["seq"])
            if kind == "session_open":
                if not (self._g_str(doc["boot_id"])
                        and self._g_posint(doc["pid"])
                        and type(doc["tail_torn_tolerated"])
                        is bool):
                    raise ExecutorRefusal(
                        "wall ledger session_open fields are not "
                        "canonical — fail closed")
                if doc["boot_id"] != self.boot:
                    raise ExecutorRefusal(
                        "CLOCK_AUTHORITY_REVIEW_REQUIRED: the "
                        "ledger names a different boot identity "
                        f"({doc['boot_id'][:16]}…) than the "
                        "current boot — a reboot is a review "
                        "boundary; no reservation or work may "
                        "start without a separate external "
                        "record")
                opened_sessions.add(doc["session"])
            elif kind == "reserve":
                if doc["session"] not in opened_sessions:
                    raise ExecutorRefusal(
                        "wall ledger reserve precedes its "
                        "session_open — fail closed")
                sec = doc["seconds"]
                if not self._g_finite(sec) or float(sec) <= 0:
                    raise ExecutorRefusal(
                        "wall ledger reservation must be a "
                        "finite strictly positive duration — "
                        "fail closed")
                sec = float(sec)
                if sec > MAX_RESERVE_S + 0.25:
                    raise ExecutorRefusal(
                        "wall ledger reservation exceeds the "
                        "protocol quantum — fail closed")
                open_total = sum(x for x, _ in pending.values())
                if charged + open_total + sec > max_wall + 0.25:
                    raise ExecutorRefusal(
                        "wall ledger reservation exceeds the "
                        "sealed remaining budget at its point in "
                        "the chain — fail closed")
                pending[seq] = (sec, doc["session"])
            elif kind == "close":
                if not self._g_posint(doc["reserve_seq"]):
                    raise ExecutorRefusal(
                        "wall ledger close reserve_seq is not a "
                        "canonical positive integer — fail "
                        "closed")
                rs = int(doc["reserve_seq"])
                if rs not in pending:
                    raise ExecutorRefusal(
                        "wall ledger close without its OPEN "
                        "reservation (missing, duplicate or "
                        "already closed) — fail closed")
                res_s, res_sess = pending[rs]
                el = doc["elapsed"]
                if not self._g_finite(el):
                    raise ExecutorRefusal(
                        "wall ledger close elapsed is not a "
                        "finite number — fail closed")
                el = float(el)
                if doc["session"] != res_sess or el < 0 or \
                        el > res_s + 0.25:
                    raise ExecutorRefusal(
                        "wall ledger close does not bind its "
                        "reservation (cross-session or "
                        "out-of-range elapsed) — fail closed")
                pending.pop(rs)
                charged += el
        # C57: a reservation that was never closed charges IN FULL
        charged += sum(x for x, _ in pending.values())
        return charged, last_sha, seq, False

    def _append(self, doc: dict):
        self.rr.revalidate()
        doc = dict(doc)
        doc["seq"] = self._seq + 1
        doc["prev_sha"] = self._last_sha
        doc["record_sha"] = _self_sha(doc, "record_sha")
        os.write(self._fd, (json.dumps(doc, sort_keys=True)
                            + "\n").encode())
        os.fsync(self._fd)
        self._seq = doc["seq"]
        self._last_sha = doc["record_sha"]
        # C75: charges written into a replaced root are uncertain
        # evidence, never a successful effect
        self.rr._post_identity("WALL_APPEND")

    # -- accounting --
    def _elapsed_in_reserve(self) -> float:
        if self._reserve_t0 is None:
            return 0.0
        return time.monotonic() - self._reserve_t0

    def consumed(self) -> float:
        return (self.prior + self._closed_session
                + self._elapsed_in_reserve())

    def remaining(self) -> float:
        return float(self.limits["max_wall_seconds"]) \
            - self.consumed()

    def _close_reserve(self):
        if self._reserve_seq is None:
            return
        el = min(self._elapsed_in_reserve(),
                 self._reserve_seconds)
        self._append({"kind": "close",
                      "session": self.session_uuid,
                      "reserve_seq": self._reserve_seq,
                      "elapsed": round(el, 6)})
        self._closed_session += el
        self._reserve_seq = None
        self._reserve_t0 = None
        self._reserve_seconds = 0.0

    def _reserve(self, seconds: float):
        self._append({"kind": "reserve",
                      "session": self.session_uuid,
                      "seconds": round(float(seconds), 6)})
        self._reserve_seq = self._seq
        self._reserve_seconds = float(seconds)
        self._reserve_t0 = time.monotonic()

    def _rollover(self, label: str):
        self._close_reserve()
        rem = self.remaining()
        if rem <= 0:
            raise T2BudgetStop(
                f"accumulated wall {self.consumed():.3f}s "
                f"consumes the sealed "
                f"{self.limits['max_wall_seconds']}s (prior "
                f"sessions charged {self.prior:.3f}s — a crash "
                "or restart never renews the budget)", label)
        self._reserve(min(RESERVE_QUANTUM_S, rem))

    def ensure_reserved(self, seconds: float,
                        label: str) -> float:
        """C62: durably charge a non-interruptible interval BEFORE
        dispatch; returns the effective bound actually granted."""
        rem = self.remaining()
        if rem <= 0:
            self._close_reserve()
            raise T2BudgetStop(
                "no positive wall budget remains for a "
                "supervised interval", label)
        want = min(float(seconds), rem)
        left = self._reserve_seconds - self._elapsed_in_reserve()
        if left < want:
            self._close_reserve()
            rem = self.remaining()
            if rem <= 0:
                raise T2BudgetStop(
                    "no positive wall budget remains for a "
                    "supervised interval", label)
            want = min(float(seconds), rem)
            self._reserve(min(max(RESERVE_QUANTUM_S, want),
                              rem, MAX_RESERVE_S))
        return want

    def check(self, label: str) -> None:
        if self.stop_file.exists():
            self._close_reserve()
            raise T2BudgetStop(
                "external stop request present at the "
                f"design-declared state root "
                f"({self.stop_file.name})", label)
        rss = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss * 1024
        rss_c = resource.getrusage(
            resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        if max(rss, rss_c) > int(self.limits["max_rss_bytes"]):
            self._close_reserve()
            raise T2BudgetStop(
                f"peak RSS {max(rss, rss_c)} bytes exceeds the "
                "sealed per-process bound", label)
        if self.remaining() <= 0:
            self._close_reserve()
            raise T2BudgetStop(
                f"accumulated wall {self.consumed():.3f}s "
                f"exceeds the sealed "
                f"{self.limits['max_wall_seconds']}s (prior "
                f"{self.prior:.3f}s — resume never renews)",
                label)
        left = self._reserve_seconds - self._elapsed_in_reserve()
        if left <= 0.05:
            self._rollover(label)

    def close(self):
        self._close_reserve()
        try:
            st_now = os.stat(self.ledger_name,
                             dir_fd=self.rr.root_fd,
                             follow_symlinks=False)
            same = (st_now.st_dev, st_now.st_ino) == self._ino
        except OSError:
            same = False
        os.close(self._fd)
        self._fd = None
        if not same:
            raise ExecutorRefusal(
                "the wall ledger path was REPLACED while held — "
                "charges landed on the original inode; stopping "
                "for review, never renewing the budget")
        # C67: replay the complete active chain from a FRESH
        # descriptor and require equality with in-memory state
        fd2 = os.open(self.ledger_name,
                      os.O_RDONLY | os.O_NOFOLLOW,
                      dir_fd=self.rr.root_fd)
        try:
            chunks = []
            while True:
                b = os.read(fd2, 1 << 20)
                if not b:
                    break
                chunks.append(b)
        finally:
            os.close(fd2)
        charged2, last2, seq2, _ = self._replay(b"".join(chunks))
        want = self.prior + self._closed_session
        if last2 != self._last_sha or seq2 != self._seq or \
                abs(charged2 - want) > 0.3:
            raise ExecutorRefusal(
                "fresh-descriptor ledger replay does not equal "
                "the in-memory wall state — charges would be "
                "lost or forged; stopping for review")
        self.rr._post_identity("WALL_CLOSE")


# ------------- C62: wall-aware supervised fits -------------

def make_fit_supervisor(limits: dict, wall: WallAuthority):
    """C62: min(remaining_global, per_fit); durable charge before
    dispatch; typed harvests; child always reaped; no orphan."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    rss_cap = int(limits["max_rss_bytes"])

    def _reap(p):
        if p.is_alive():
            p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join(5)
        if p.exitcode is None:
            raise ExecutorRefusal(
                "supervised worker could not be reaped — "
                "stopping for review")

    def supervise(fn, label):
        eff = wall.ensure_reserved(
            min(PER_FIT_WALL_SECONDS, 10 ** 9), label)
        eff = min(eff, PER_FIT_WALL_SECONDS)
        if eff <= 0:
            raise T2BudgetStop(
                "no positive wall budget remains for a "
                "supervised fit", label)
        recv, send = ctx.Pipe(duplex=False)

        def _child():
            try:
                resource.setrlimit(resource.RLIMIT_AS,
                                   (2 * rss_cap, 2 * rss_cap))
                out = fn()
                rss = resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss * 1024
                if rss > rss_cap:
                    send.send(("RSS_EXCEEDED",
                               f"{rss} bytes > {rss_cap}"))
                else:
                    send.send(("OK", out))
            except MemoryError:
                send.send(("RSS_EXCEEDED",
                           "address-space ceiling"))
            except BaseException as exc:
                try:
                    send.send(("CRASH",
                               f"{type(exc).__name__}: "
                               f"{str(exc)[:120]}"))
                except BaseException:
                    pass

        p = ctx.Process(target=_child, daemon=True)
        p.start()
        send.close()
        try:
            if recv.poll(eff):
                try:
                    kind, payload = recv.recv()
                except EOFError:
                    _reap(p)
                    raise SupervisedFitFailure(
                        f"{label} harvested EOF (worker died "
                        f"before reporting; exitcode "
                        f"{p.exitcode})")
                _reap(p)
                if kind == "OK":
                    return payload
                raise SupervisedFitFailure(
                    f"{label} harvested {kind}: {payload}")
            _reap(p)
            raise SupervisedFitFailure(
                f"{label} exceeded its effective wall bound "
                f"({eff:.3f}s = min(remaining global, per-fit "
                f"{PER_FIT_WALL_SECONDS}s)) — WALL_KILLED")
        finally:
            recv.close()
            if p.is_alive():
                _reap(p)
    return supervise


# ------------- C58: complete lock protocol -------------

def _lock_doc_verify(raw: bytes, what: str, keys: set,
                     schema: str, self_key: str) -> dict:
    doc = _strict_parse(raw, what)
    if set(doc) != keys:
        raise ExecutorRefusal(
            f"{what} keys are not the exact schema")
    if _self_sha(doc, self_key) != doc.get(self_key):
        raise ExecutorRefusal(
            f"{what} self-digest does not re-derive")
    if doc.get("schema") != schema:
        raise ExecutorRefusal(f"{what} carries a foreign schema")
    return doc


_SESSION_KEYS = {"schema", "session", "session_uuid", "pid",
                 "boot_id", "campaign_generation", "started_wall",
                 "session_sha256"}
_TAKEOVER_KEYS = {"schema", "over_session", "dead_pid",
                  "by_session_uuid", "campaign_generation",
                  "at_wall", "takeover_sha256"}
_INTENT_KEYS = {"schema", "session", "session_sha256",
                "session_uuid", "pid", "campaign_generation",
                "released_wall", "intent_sha256"}
_DONE_KEYS = {"schema", "session", "intent_sha256",
              "done_sha256"}


def _scan_sessions(rr: ResultsRoot) -> list:
    out = []
    for name in rr.listdir(rr.locks_fd):
        if name.startswith("SESSION_") and name.endswith(".json"):
            try:
                n = int(name[len("SESSION_"):-len(".json")])
            except ValueError:
                raise ExecutorRefusal(
                    f"foreign object {name!r} in the lock "
                    "protocol directory")
            out.append(n)
    return sorted(out)


def _release_complete(rr: ResultsRoot, n: int,
                      sess_doc: dict) -> bool:
    """C58: True ONLY when a strict, self-integral RELEASE_INTENT
    and its RELEASE_DONE completion witness both exist and bind
    the exact session record digest, UUID, holder pid and
    campaign generation. Anything else never frees the lock."""
    try:
        intent = _lock_doc_verify(
            rr.read_private(rr.locks_fd,
                            f"RELEASE_INTENT_{n:06d}.json",
                            f"release intent {n}"),
            f"release intent {n}", _INTENT_KEYS,
            "agent_multi.t2_lock_release_intent.v1",
            "intent_sha256")
        done = _lock_doc_verify(
            rr.read_private(rr.locks_fd,
                            f"RELEASE_DONE_{n:06d}.json",
                            f"release completion {n}"),
            f"release completion {n}", _DONE_KEYS,
            "agent_multi.t2_lock_release_done.v1",
            "done_sha256")
    except SystemExit:
        return False
    if int(intent["session"]) != n or int(done["session"]) != n:
        return False
    if intent["session_sha256"] != sess_doc["session_sha256"]:
        return False
    if intent["session_uuid"] != sess_doc["session_uuid"] or \
            int(intent["pid"]) != int(sess_doc["pid"]):
        return False
    if intent["campaign_generation"] != \
            sess_doc["campaign_generation"] or \
            sess_doc["campaign_generation"] != \
            T2_CAMPAIGN_GENERATION:
        return False
    if done["intent_sha256"] != intent["intent_sha256"]:
        return False
    return True


def acquire_lock(root, session_uuid: str,
                 takeover_stale: bool = False) -> tuple:
    """C54/C58: monotonic sessions; a session is free ONLY when
    its release intent AND completion witness fully verify; both
    are re-read and revalidated AFTER winning the exclusive
    election. Recovery follows physical facts: a live pid
    refuses; a provably dead pid requires the explicit recorded
    takeover; an inconclusive probe refuses as UNCERTAIN."""
    rr = root if isinstance(root, ResultsRoot) else \
        ResultsRoot(root, create=True)
    sessions = _scan_sessions(rr)
    nxt = 1
    prev = None
    if sessions:
        n = sessions[-1]
        nxt = n + 1
        sess_doc = _lock_doc_verify(
            rr.read_private(rr.locks_fd, f"SESSION_{n:06d}.json",
                            f"lock session {n}"),
            f"lock session {n}", _SESSION_KEYS,
            "agent_multi.t2_lock_session.v2",
            "session_sha256")
        prev = (n, sess_doc)
        if not _release_complete(rr, n, sess_doc):
            pid = int(sess_doc["pid"])
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                raise ExecutorRefusal(
                    f"lock session {n} pid {pid} is UNCERTAIN "
                    "(no permission to probe) — operator "
                    "disposition, never silent takeover")
            if alive:
                raise ExecutorRefusal(
                    f"another executor holds lock session {n} "
                    f"(pid {pid} is alive; its release is "
                    "absent or does not verify)")
            if not takeover_stale:
                raise ExecutorRefusal(
                    f"lock session {n} has no VERIFIED release "
                    f"and its pid {pid} is dead — a stale or "
                    "uncertain release never frees the lock; an "
                    "explicit recorded takeover "
                    "(--takeover-stale-lock) is required")
            tk = {"schema": "agent_multi.t2_lock_takeover.v2",
                  "over_session": n, "dead_pid": pid,
                  "by_session_uuid": session_uuid,
                  "campaign_generation": T2_CAMPAIGN_GENERATION,
                  "at_wall": time.time()}
            tk["takeover_sha256"] = _self_sha(tk,
                                              "takeover_sha256")
            rr.excl_write(rr.locks_fd, f"TAKEOVER_{n:06d}.json",
                          json.dumps(tk, indent=1).encode())
            prev = None
    doc = {"schema": "agent_multi.t2_lock_session.v2",
           "session": nxt, "session_uuid": session_uuid,
           "pid": os.getpid(), "boot_id": _boot_id(),
           "campaign_generation": T2_CAMPAIGN_GENERATION,
           "started_wall": time.time()}
    doc["session_sha256"] = _self_sha(doc, "session_sha256")
    try:
        rr.excl_write(rr.locks_fd, f"SESSION_{nxt:06d}.json",
                      json.dumps(doc, indent=1).encode())
    except FileExistsError:
        raise ExecutorRefusal(
            "lost the lock-acquisition race — another executor "
            "claimed the next session")
    if prev is not None:
        # C58: post-election revalidation of BOTH release objects
        if not _release_complete(rr, prev[0], prev[1]):
            raise ExecutorRefusal(
                f"lock session {prev[0]} release no longer "
                "verifies after the election — refusing to run; "
                "this session stays unreleased for recorded "
                "recovery")
    return nxt, rr


def release_lock(rr: ResultsRoot, session_n: int,
                 session_uuid: str) -> None:
    sess_doc = _lock_doc_verify(
        rr.read_private(rr.locks_fd,
                        f"SESSION_{session_n:06d}.json",
                        f"lock session {session_n}"),
        f"lock session {session_n}", _SESSION_KEYS,
        "agent_multi.t2_lock_session.v2", "session_sha256")
    if sess_doc["session_uuid"] != session_uuid or \
            int(sess_doc["pid"]) != os.getpid():
        raise ExecutorRefusal(
            "refusing to release a lock session this process "
            "does not hold")
    intent = {"schema": "agent_multi.t2_lock_release_intent.v1",
              "session": session_n,
              "session_sha256": sess_doc["session_sha256"],
              "session_uuid": session_uuid, "pid": os.getpid(),
              "campaign_generation": T2_CAMPAIGN_GENERATION,
              "released_wall": time.time()}
    intent["intent_sha256"] = _self_sha(intent, "intent_sha256")
    iname = f"RELEASE_INTENT_{session_n:06d}.json"
    if rr.exists(rr.locks_fd, iname):
        prior = _lock_doc_verify(
            rr.read_private(rr.locks_fd, iname,
                            f"release intent {session_n}"),
            f"release intent {session_n}", _INTENT_KEYS,
            "agent_multi.t2_lock_release_intent.v1",
            "intent_sha256")
        if prior["session_sha256"] != \
                sess_doc["session_sha256"] or \
                prior["session_uuid"] != session_uuid or \
                int(prior["pid"]) != os.getpid():
            raise ExecutorRefusal(
                "a foreign release intent already exists for "
                "this session — refusing")
        intent = prior              # immutable intent, reused
    else:
        rr.excl_write(rr.locks_fd, iname,
                      json.dumps(intent, indent=1).encode())
    if rr.exists(rr.locks_fd,
                 f"RELEASE_DONE_{session_n:06d}.json"):
        raise ExecutorRefusal(
            "a completion witness already exists for this "
            "session (possibly torn from an uncertain fsync) — "
            "its state is decided by verification, never by "
            "overwrite")
    done = {"schema": "agent_multi.t2_lock_release_done.v1",
            "session": session_n,
            "intent_sha256": intent["intent_sha256"]}
    done["done_sha256"] = _self_sha(done, "done_sha256")
    # C58: two-phase completion — the DONE witness is INVALID
    # JSON until its final byte, and that byte lands only after
    # the body's fsync returned. Both physical outcomes of the
    # final fsync are modeled: (i) it persisted -> the release is
    # complete; (ii) it was lost in a crash -> the witness is
    # torn/absent, the release stays uncertain and never frees.
    payload = json.dumps(done, indent=1).encode()
    name = f"RELEASE_DONE_{session_n:06d}.json"
    rr.revalidate()
    fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                 | os.O_NOFOLLOW, 0o600, dir_fd=rr.locks_fd)
    try:
        os.write(fd, payload[:-1])
        os.fsync(fd)
        os.write(fd, payload[-1:])
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(rr.locks_fd)
    rr._post_identity("RELEASE")


# ---------------- unit reconstruction ----------------

def load_bank_unit(design: dict, uid: str,
                   manifest: dict, raw_root: Path) -> dict:
    b = design["task_population"]["unit_map"][uid]
    lid = b["dataset"]
    contract = census_mod.CONTRACTS[lid]
    admissible = conf.validate_public_manifest(
        manifest, raw_root=raw_root)
    d = admissible[lid]
    fd = conf._open_nofollow_under(Path(os.path.abspath(raw_root)),
                                  Path(d["local_relpath"]))
    try:
        chunks = []
        while True:
            x = os.read(fd, 1 << 20)
            if not x:
                break
            chunks.append(x)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != d["sha256"]:
        raise ExecutorRefusal(f"{lid}: bytes drifted")
    import zipfile
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        tsf = [n for n in zf.namelist() if n.endswith(".tsf")]
        if len(tsf) != 1:
            raise ExecutorRefusal(f"{lid}: not exactly one .tsf")
        panel = bank.parse_tsf_bytes(zf.read(tsf[0]), lid)
    built = bank.build_series_units(
        panel, d["family"], contract["period"],
        "monash_record_frequency", contract["max_gap"],
        contract["min_length"])
    if uid not in built["units"]:
        raise ExecutorRefusal(f"{uid}: not rebuildable from bytes")
    u = built["units"][uid]
    if u["series_numeric_sha256"] != b["series_numeric_sha256"]:
        raise ExecutorRefusal(f"{uid}: numeric digest differs "
                              "from the sealed binding")
    if int(u["n"]) != int(b["n_obs"]) or \
            int(u["seasonal_period"]) != int(b["seasonal_period"]):
        raise ExecutorRefusal(f"{uid}: length/period differ from "
                              "the sealed binding")
    want_w = bank.origin_windows_for(
        u["n"], int(design["role_geometry"]["rolling_origins"]),
        float(design["role_geometry"]["origin_base_frac"]),
        seasonal_period=u["seasonal_period"],
        horizon=int(design["role_geometry"]["horizon"]))
    if want_w != b["origin_windows"]:
        raise ExecutorRefusal(f"{uid}: origin windows do not "
                              "re-derive")
    return {"unit_id": uid, "y": u["y"],
            "bytes_sha256": d["sha256"],
            "dataset": lid, "family": b["family"],
            "frequency": u["frequency_declared"],
            "license_note": d["license_id"],
            "missingness": u["missingness"],
            "time_index": u["time_index"],
            "time_provenance": u["time_provenance"],
            "seasonal_period": u["seasonal_period"],
            "seasonal_period_provenance":
                u["seasonal_period_provenance"]}


def _safe_name(uid: str) -> str:
    return uid.replace("::", "__").replace("/", "_")


def _binding_for(design: dict, unit: dict, mode: str) -> dict:
    uid = unit["unit_id"]
    if mode == "confirmatory":
        return design["task_population"]["unit_map"][uid]
    y = np.asarray(unit["y"], dtype=np.float64)
    wins = bank.origin_windows_for(
        len(y), int(design["role_geometry"]["rolling_origins"]),
        float(design["role_geometry"]["origin_base_frac"]),
        seasonal_period=int(unit["seasonal_period"]),
        horizon=int(design["role_geometry"]["horizon"]))
    return {"dataset": unit["dataset"], "family": unit["family"],
            "seasonal_period": int(unit["seasonal_period"]),
            "horizon": int(design["role_geometry"]["horizon"]),
            "n_obs": int(len(y)),
            "series_numeric_sha256":
                bank.series_numeric_digest(y),
            "origin_windows": wins}


def physical_authority(design: dict, mode: str,
                       repo_root: Path = None,
                       design_file_sha: str = None) -> dict:
    """C83: live paths (rehearsal, dispositions) pass the digest
    of the ONE frozen stream they consumed, so the authority names
    the exact consumed bytes; only DEFERRED re-verification, which
    deliberately checks the current pathname's bytes, omits it."""
    if mode not in MODES:
        raise ExecutorRefusal(f"unknown execution mode {mode!r}")
    sealed_file_sha = (design_file_sha if design_file_sha
                       else _sha_file(active_design_path()))
    self_sha = _self_sha(design, "design_sha256")
    if self_sha != design.get("design_sha256"):
        raise ExecutorRefusal(
            "sealed design self identity does not re-derive")
    manifest_sha = _sha_file(MANIFEST_PATH)
    census_sha = _sha_file(CENSUS_PATH)
    review = conf.verify_design_review_record(
        design, manifest_sha, census_sha)
    if mode == "confirmatory":
        exec_rec = conf.verify_execution_record(
            design, sealed_file_sha, review["_record_sha256"],
            manifest_sha, census_sha,
            repo_root=repo_root or conf.REPO)
        exec_sha = exec_rec["_record_sha256"]
    else:
        exec_sha = REHEARSAL_EXECUTION_SENTINEL
        manifest_sha = REHEARSAL_DIGEST_SENTINEL
        census_sha = REHEARSAL_DIGEST_SENTINEL
    return {"sealed_design_file_sha256": sealed_file_sha,
            "sealed_design_self_sha256": self_sha,
            "design_review_record_sha256":
                review["_record_sha256"],
            "execution_record_sha256": exec_sha,
            "manifest_sha256": manifest_sha,
            "census_sha256": census_sha}


# -------- C59: authority-bound claims and terminals --------

def _authority_block(authority: dict, binding: dict,
                     pins: tuple) -> dict:
    return {**{k: authority[k] for k in _AUTHORITY_KEYS},
            "code_identity": code_identity(),
            "pinned_commit": pins[0], "pinned_tree": pins[1],
            "campaign_generation": T2_CAMPAIGN_GENERATION,
            "unit_binding": binding}


def _verify_authority_block(doc: dict, design: dict,
                            authority: dict, mode: str,
                            uid: str, what: str,
                            repo_root: Path = None) -> None:
    """C59: re-derive every authority field of a claim/terminal
    against the CURRENT physical objects."""
    for k in _AUTHORITY_KEYS:
        if doc[k] != authority[k]:
            raise ExecutorRefusal(
                f"{what} {k} does not match the current physical "
                "authority — stale or transplanted")
    if doc["campaign_generation"] != T2_CAMPAIGN_GENERATION:
        raise ExecutorRefusal(
            f"{what} names a foreign campaign generation")
    if doc["code_identity"] != conf.executor_code_identity(
            repo_root or REPO):
        raise ExecutorRefusal(
            f"{what} code_identity does not match the reviewed "
            "checkout")
    head, tree = _git_head_tree()
    if doc["pinned_commit"] != head or doc["pinned_tree"] != tree:
        raise ExecutorRefusal(
            f"{what} pinned commit/tree is not the executing "
            "checkout")
    binding = doc["unit_binding"]
    if mode == "confirmatory":
        tp = design["task_population"]
        if uid not in set(tp["series_ids"]):
            raise ExecutorRefusal(
                f"{what} unit {uid!r} is not in the sealed "
                "population")
        if binding != tp["unit_map"][uid]:
            raise ExecutorRefusal(
                f"{what} binding does not equal the sealed "
                "unit_map entry")
    else:
        if uid not in DEV_UNITS:
            raise ExecutorRefusal(
                f"{what} unit {uid!r} is not a development unit")
        if type(binding) is not dict or \
                set(binding) != _REHEARSAL_BINDING_KEYS:
            raise ExecutorRefusal(
                f"{what} rehearsal binding keys are not the "
                "exact schema")


def verify_unit_claim(claim_path, design: dict,
                      authority: dict = None,
                      mode_expected: str = None,
                      uid_expected: str = None,
                      repo_root: Path = None) -> dict:
    claim = _strict_parse(
        _read_evidence(claim_path, "unit claim"),
        "unit claim")
    if set(claim) != _CLAIM_KEYS:
        raise ExecutorRefusal(
            "unit claim keys are not the exact v3 schema (diff: "
            f"{sorted(set(claim) ^ _CLAIM_KEYS)[:6]})")
    if claim["schema"] != "agent_multi.t2_unit_claim.v3":
        raise ExecutorRefusal("unit claim carries a foreign "
                              "schema")
    if _self_sha(claim, "claim_sha256") != claim["claim_sha256"]:
        raise ExecutorRefusal(
            "unit claim self-digest does not re-derive")
    import re as _re
    if not _re.fullmatch(r"attempt_[0-9a-f]{16}",
                         claim["attempt_id"]):
        raise ExecutorRefusal("unit claim attempt_id is not "
                              "canonical")
    mode = claim["mode"]
    if mode not in MODES:
        raise ExecutorRefusal(f"unit claim mode {mode!r} unknown")
    if mode_expected is not None and mode != mode_expected:
        raise ExecutorRefusal(
            f"unit claim mode {mode!r} does not match the "
            f"expected {mode_expected!r}")
    uid = claim["unit_id"]
    if uid_expected is not None and uid != uid_expected:
        raise ExecutorRefusal(
            "unit claim does not name the expected unit")
    if _src_name(claim_path) != \
            f"CLAIM_{_safe_name(uid)}.json":
        raise ExecutorRefusal(
            "unit claim filename does not derive from its "
            "unit_id")
    # C71: exact domains
    if not (type(claim["pid"]) is int
            and not isinstance(claim["pid"], bool)
            and claim["pid"] > 0):
        raise ExecutorRefusal(
            "unit claim pid is not a canonical positive integer")
    import math as _m
    cw = claim["claimed_wall"]
    if type(cw) not in (int, float) or isinstance(cw, bool) or \
            not _m.isfinite(cw) or cw <= 0:
        raise ExecutorRefusal(
            "unit claim claimed_wall is not a finite positive "
            "number")
    expected = authority if authority is not None else \
        physical_authority(design, mode, repo_root=repo_root)
    _verify_authority_block(claim, design, expected, mode, uid,
                            "unit claim", repo_root)
    return claim


def verify_unit_terminal(term_path, design: dict,
                         authority: dict = None,
                         mode_expected: str = None,
                         repo_root: Path = None) -> dict:
    """C59: deep verification of a FAILED terminal under CURRENT
    physical authority, including the claim it closes. Any
    failure here must adjudicate typed UNCERTAIN — never
    TERMINAL_FAILED."""
    term = _strict_parse(
        _read_evidence(term_path, "unit terminal"),
        "unit terminal")
    if set(term) != _TERMINAL_KEYS:
        raise ExecutorRefusal(
            "unit terminal keys are not the exact v3 schema "
            f"(diff: {sorted(set(term) ^ _TERMINAL_KEYS)[:6]})")
    if term["schema"] != "agent_multi.t2_unit_terminal.v3":
        raise ExecutorRefusal(
            "unit terminal carries a foreign schema")
    if _self_sha(term, "terminal_sha256") != \
            term["terminal_sha256"]:
        raise ExecutorRefusal(
            "unit terminal self-digest does not re-derive")
    if term["terminal"] != "FAILED":
        raise ExecutorRefusal(
            "unit terminal is not a FAILED terminal")
    if type(term["operator_disposition"]) is not bool:
        raise ExecutorRefusal(
            "unit terminal operator_disposition must be a "
            "boolean")
    uid = term["unit_id"]
    mode = term["mode"]
    if mode not in MODES:
        raise ExecutorRefusal(
            f"unit terminal mode {mode!r} unknown")
    if mode_expected is not None and mode != mode_expected:
        raise ExecutorRefusal(
            f"unit terminal mode {mode!r} does not match the "
            f"expected {mode_expected!r}")
    if _src_name(term_path) != \
            f"TERMINAL_{_safe_name(uid)}.json":
        raise ExecutorRefusal(
            "unit terminal filename does not derive from its "
            "unit_id")
    # C71: exact domains
    import math as _m
    ws = term["wall_seconds"]
    if type(ws) not in (int, float) or isinstance(ws, bool) or \
            not _m.isfinite(ws) or ws < 0:
        raise ExecutorRefusal(
            "unit terminal wall_seconds is not a finite "
            "nonnegative number")
    for k in ("failure_class", "reason"):
        if type(term[k]) is not str or not term[k]:
            raise ExecutorRefusal(
                f"unit terminal {k} must be a nonempty string")
    expected = authority if authority is not None else \
        physical_authority(design, mode, repo_root=repo_root)
    _verify_authority_block(term, design, expected, mode, uid,
                            "unit terminal", repo_root)
    claim_p = _src_sibling(term_path,
                           f"CLAIM_{_safe_name(uid)}.json")
    claim = verify_unit_claim(claim_p, design,
                              authority=expected,
                              mode_expected=mode,
                              uid_expected=uid,
                              repo_root=repo_root)
    if claim["attempt_id"] != term["attempt_id"]:
        raise ExecutorRefusal(
            "unit terminal does not close this unit's claimed "
            "attempt")
    if term["closed_claim_sha256"] != claim["claim_sha256"]:
        raise ExecutorRefusal(
            "unit terminal closed_claim_sha256 does not pin the "
            "physical claim")
    if term["operator_disposition"]:
        if term["disposition_record_sha256"] == \
                NO_DISPOSITION_SENTINEL:
            raise ExecutorRefusal(
                "an operator-disposition terminal must pin its "
                "external disposition record")
    else:
        if term["disposition_record_sha256"] != \
                NO_DISPOSITION_SENTINEL:
            raise ExecutorRefusal(
                "a non-disposition terminal must carry the "
                "no-disposition sentinel")
    return term


def _write_terminal(rr: ResultsRoot, design: dict,
                    authority: dict, pins: tuple, unit: dict,
                    mode: str, attempt: str, claim_sha: str,
                    failure_class: str, reason: str,
                    wall_seconds: float,
                    operator_disposition: bool = False,
                    disposition_sha: str =
                    NO_DISPOSITION_SENTINEL) -> None:
    uid = unit["unit_id"]
    term = {"schema": "agent_multi.t2_unit_terminal.v3",
            "unit_id": uid, "attempt_id": attempt, "mode": mode,
            **_authority_block(authority,
                               _binding_for(design, unit, mode),
                               pins),
            "closed_claim_sha256": claim_sha,
            "terminal": "FAILED",
            "failure_class": failure_class,
            "reason": reason[:300],
            "operator_disposition": bool(operator_disposition),
            "disposition_record_sha256": disposition_sha,
            "wall_seconds": round(wall_seconds, 2)}
    term["terminal_sha256"] = _self_sha(term, "terminal_sha256")
    rr.excl_write(rr.units_fd, f"TERMINAL_{_safe_name(uid)}.json",
                  json.dumps(term, indent=1).encode())


# ---------------- production + verification ----------------

def run_unit(hz, co, unit: dict, design: dict, authority: dict,
             rr: ResultsRoot, mode: str, pins: tuple,
             guard=None, fit_supervisor=None) -> dict:
    """C63 failure discipline: an ASSAY failure writes its
    integral v3 terminal and raises T2AssayFailed (countable);
    a budget stop or interrupt leaves the claim UNCERTAIN; ANY
    other failure — persistence, verification, even a failed
    terminal write — raises typed uncertainty that HALTS the
    campaign."""
    if mode not in MODES:
        raise ExecutorRefusal(f"unknown execution mode {mode!r}")
    uid = unit["unit_id"]
    safe = _safe_name(uid)
    attempt = ("attempt_"
               + hashlib.sha256(os.urandom(16)).hexdigest()[:16])
    binding = _binding_for(design, unit, mode)
    claim = {"schema": "agent_multi.t2_unit_claim.v3",
             "unit_id": uid, "attempt_id": attempt, "mode": mode,
             **_authority_block(authority, binding, pins),
             "pid": os.getpid(), "claimed_wall": time.time()}
    claim["claim_sha256"] = _self_sha(claim, "claim_sha256")
    try:
        rr.excl_write(rr.units_fd, f"CLAIM_{safe}.json",
                      json.dumps(claim, indent=1).encode())
    except FileExistsError:
        raise ExecutorRefusal(
            f"{uid}: unit already claimed — attempts are never "
            "reused or overwritten; adjudication decides")
    sink = {}
    t0 = time.time()
    try:
        rec = hz.assay_unit(
            co, unit, h=int(design["role_geometry"]["horizon"]),
            sink=sink, guard=guard, fit_supervisor=fit_supervisor)
    except (T2BudgetStop, KeyboardInterrupt):
        raise                   # claim stays; UNCERTAIN by design
    except BaseException as exc:
        try:
            _write_terminal(rr, design, authority, pins, unit,
                            mode, attempt, claim["claim_sha256"],
                            type(exc).__name__, str(exc),
                            time.time() - t0)
        except BaseException as werr:
            raise ExecutorRefusal(
                f"{uid}: assay failed AND its terminal could not "
                f"be durably written ({type(werr).__name__}) — "
                "attempt UNCERTAIN; campaign halts") from exc
        raise T2AssayFailed(uid, f"{type(exc).__name__}")
    try:
        npz_name = f"ARRAYS_{safe}.npz"
        arrays = {"y": np.asarray(unit["y"], dtype=np.float64)}
        for (okey, arm, model), captured in sink.items():
            pred, obs, fit_rows = captured
            arrays[f"pred__{okey}__{arm}__{model}"] = pred
            arrays[f"obs__{okey}__{arm}__{model}"] = obs
            if fit_rows is not None:
                arrays[f"fit__{okey}__{arm}__{model}"] = fit_rows
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        rr.excl_write(rr.units_fd, npz_name, buf.getvalue())
        wrapper = {
            "schema": "agent_multi.t2_confirmatory_unit_record"
                      ".v2",
            "mode": mode, "unit_id": uid, "attempt_id": attempt,
            **{k: authority[k] for k in _AUTHORITY_KEYS},
            "unit_binding": binding,
            "code_identity": code_identity(),
            "assay_record": rec,
            "arrays_npz_sha256": hashlib.sha256(
                buf.getvalue()).hexdigest(),
            "wall_seconds": round(time.time() - t0, 2)}
        wrapper["record_sha256"] = _self_sha(wrapper,
                                             "record_sha256")
        rr.excl_write(rr.units_fd, f"RECORD_{safe}.json",
                      json.dumps(wrapper, indent=1).encode())
        verify_unit_record(
            (rr, f"RECORD_{safe}.json"), (rr, npz_name), design,
            authority=authority,
            unit_y=np.asarray(unit["y"], dtype=np.float64),
            mode_expected=mode)
    except (T2BudgetStop, KeyboardInterrupt):
        raise
    except SystemExit as exc:
        raise ExecutorRefusal(
            f"{uid}: post-assay persistence/verification failed "
            f"({str(exc)[:140]}) — attempt UNCERTAIN; campaign "
            "halts")
    except BaseException as exc:
        raise ExecutorRefusal(
            f"{uid}: post-assay persistence/verification failed "
            f"({type(exc).__name__}: {str(exc)[:120]}) — attempt "
            "UNCERTAIN; campaign halts")
    return {"record": f"RECORD_{safe}.json",
            "wall": wrapper["wall_seconds"]}


def _expect_close(got, want, path: str, tol=1e-9) -> None:
    if want is None or got is None:
        if want is None and got is None:
            return
        raise ExecutorRefusal(
            f"{path}: recomputed {got!r} vs recorded {want!r} — "
            "does not recompute from persisted arrays")
    if abs(float(got) - float(want)) > tol * max(1.0,
                                                abs(float(want))):
        raise ExecutorRefusal(
            f"{path}: recomputed {got!r} vs recorded {want!r} — "
            "does not recompute from persisted arrays")


def verify_unit_record(rec_path, npz_path,
                       design: dict, authority: dict = None,
                       unit_y=None, mode_expected: str = None,
                       repo_root: Path = None) -> dict:
    """C49/C50/C51 (+C59): RE-DERIVE, never believe — including
    the full v3 claim the record's attempt binds to."""
    import t2_assay_harness as hz
    repo_root = repo_root or conf.REPO
    wrapper = _strict_parse(
        _read_evidence(rec_path, "unit record"),
        "unit record")
    if set(wrapper) != _WRAPPER_KEYS:
        raise ExecutorRefusal(
            "unit record keys are not the exact v2 schema (diff: "
            f"{sorted(set(wrapper) ^ _WRAPPER_KEYS)})")
    for k in ("schema", "mode", "unit_id", "attempt_id",
              *_AUTHORITY_KEYS, "arrays_npz_sha256",
              "record_sha256"):
        if type(wrapper[k]) is not str or not wrapper[k]:
            raise ExecutorRefusal(
                f"unit record field {k!r} must be a nonempty "
                "string")
    for k in ("unit_binding", "code_identity", "assay_record"):
        if type(wrapper[k]) is not dict or not wrapper[k]:
            raise ExecutorRefusal(
                f"unit record field {k!r} must be a nonempty "
                "object")
    if type(wrapper["wall_seconds"]) not in (int, float) or \
            isinstance(wrapper["wall_seconds"], bool) or \
            wrapper["wall_seconds"] < 0:
        raise ExecutorRefusal(
            "unit record wall_seconds must be a nonnegative "
            "number")
    if wrapper["schema"] != \
            "agent_multi.t2_confirmatory_unit_record.v2":
        raise ExecutorRefusal(
            "unit record carries a foreign schema")
    if _self_sha(wrapper, "record_sha256") != \
            wrapper["record_sha256"]:
        raise ExecutorRefusal(
            "unit record self-digest does not re-derive")
    mode = wrapper["mode"]
    if mode not in MODES:
        raise ExecutorRefusal(
            f"unit record mode {mode!r} is not a known execution "
            "mode")
    if mode_expected is not None and mode != mode_expected:
        raise ExecutorRefusal(
            f"unit record mode {mode!r} does not match the "
            f"expected {mode_expected!r} — rehearsal records "
            "never verify as confirmatory")
    uid = wrapper["unit_id"]
    import re as _re
    if not _re.fullmatch(r"attempt_[0-9a-f]{16}",
                         wrapper["attempt_id"]):
        raise ExecutorRefusal(
            "unit record attempt_id is not a canonical attempt "
            "identifier")
    binding = wrapper["unit_binding"]
    if mode == "confirmatory":
        tp = design["task_population"]
        if uid not in set(tp["series_ids"]):
            raise ExecutorRefusal(
                f"unit {uid!r} is NOT one of the sealed "
                "population's series — a foreign unit never "
                "verifies")
        if binding != tp["unit_map"][uid]:
            raise ExecutorRefusal(
                f"unit {uid!r} binding does not equal the sealed "
                "unit_map entry — transplanted bindings refuse")
    else:
        if uid not in DEV_UNITS:
            raise ExecutorRefusal(
                f"unit {uid!r} is not a development unit — the "
                "rehearsal population is closed")
        if set(binding) != _REHEARSAL_BINDING_KEYS:
            raise ExecutorRefusal(
                "rehearsal unit binding keys are not the exact "
                "schema")
        if uid in set(design["task_population"]["series_ids"]):
            raise ExecutorRefusal(
                "a rehearsal record can never name a sealed-"
                "population series")
    safe = _safe_name(uid)
    if _src_name(rec_path) != f"RECORD_{safe}.json":
        raise ExecutorRefusal(
            f"unit record filename {_src_name(rec_path)!r} does "
            f"not derive from its unit_id ({uid!r})")
    if _src_name(npz_path) != f"ARRAYS_{safe}.npz":
        raise ExecutorRefusal(
            f"arrays filename {_src_name(npz_path)!r} does not "
            f"derive from the unit_id ({uid!r})")
    expected = authority if authority is not None else \
        physical_authority(design, mode, repo_root=repo_root)
    for k in _AUTHORITY_KEYS:
        if wrapper[k] != expected[k]:
            raise ExecutorRefusal(
                f"unit record {k} does not match the physical "
                "authority object — a repaired self-digest never "
                "converts false authority into fact")
    physical_ci = conf.executor_code_identity(repo_root)
    if wrapper["code_identity"] != physical_ci:
        raise ExecutorRefusal(
            "unit record code_identity does not match the "
            "reviewed checkout surface — declared identities "
            "grant nothing")
    # C59: the claim this record's attempt binds to must verify
    claim = verify_unit_claim(
        _src_sibling(rec_path, f"CLAIM_{safe}.json"), design,
        authority=expected, mode_expected=mode,
        uid_expected=uid, repo_root=repo_root)
    if claim["attempt_id"] != wrapper["attempt_id"]:
        raise ExecutorRefusal(
            "unit record does not bind this unit's claimed "
            "attempt")
    raw_npz = _read_evidence(npz_path, "persisted arrays")
    if hashlib.sha256(raw_npz).hexdigest() != \
            wrapper["arrays_npz_sha256"]:
        raise ExecutorRefusal(
            "persisted arrays differ from the record's digest — "
            "swapped or edited evidence")
    try:
        with np.load(io.BytesIO(raw_npz),
                     allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
    except ExecutorRefusal:
        raise
    except Exception as exc:
        raise ExecutorRefusal(
            "persisted arrays are not a loadable pickle-free NPZ "
            f"({type(exc).__name__}) — arbitrary bytes are never "
            "evidence")
    rec = wrapper["assay_record"]
    if rec.get("unit_id") != uid:
        raise ExecutorRefusal(
            "assay record unit_id does not match the wrapper")
    period = int(binding["seasonal_period"])
    horizon = int(design["role_geometry"]["horizon"])
    if int(rec.get("seasonal_period", -1)) != period or \
            int(rec.get("horizon", -1)) != horizon or \
            int(binding["horizon"]) != horizon:
        raise ExecutorRefusal(
            "seasonal period / horizon do not agree across "
            "binding, design and assay record")
    lags = int(design["role_geometry"].get("lags", hz.RIDGE_LAGS))
    if lags != hz.RIDGE_LAGS:
        raise ExecutorRefusal(
            "design lag order does not match the harness")
    arms = sorted(design["arms"])
    models = ["ridge"] + [f"mlp_seed{s}"
                          for s in design["seed_tape"]]
    wins = binding["origin_windows"]
    okeys = sorted(wins)
    if okeys != [f"origin{i}" for i in range(len(okeys))] or \
            len(okeys) != int(
                design["role_geometry"]["rolling_origins"]):
        raise ExecutorRefusal(
            "binding origin windows are not the sealed geometry")
    if sorted(rec.get("rolling_origins", {})) != okeys:
        raise ExecutorRefusal(
            "assay record origins do not match the binding")
    expected_arrays = {"y"}
    for okey in okeys:
        expected_arrays.add(f"pred__{okey}__seasonal_naive__"
                            "baseline")
        expected_arrays.add(f"obs__{okey}__seasonal_naive__"
                            "baseline")
        for arm in arms:
            for m in models:
                for kind in ("pred", "obs", "fit"):
                    expected_arrays.add(
                        f"{kind}__{okey}__{arm}__{m}")
    got_arrays = set(data)
    if got_arrays != expected_arrays:
        raise ExecutorRefusal(
            "persisted array inventory is not exact (diff: "
            f"{sorted(got_arrays ^ expected_arrays)[:6]})")
    y = data["y"]
    if y.ndim != 1 or y.dtype != np.float64 or \
            not np.all(np.isfinite(y)):
        raise ExecutorRefusal(
            "persisted series y is not 1-D finite float64")
    if int(binding["n_obs"]) != len(y):
        raise ExecutorRefusal(
            "persisted series length differs from the binding")
    if bank.series_numeric_digest(y) != \
            binding["series_numeric_sha256"]:
        raise ExecutorRefusal(
            "persisted series numeric digest differs from the "
            "binding — the arrays are not this unit's series")
    if rec.get("series_numeric_sha256") != \
            binding["series_numeric_sha256"]:
        raise ExecutorRefusal(
            "assay record series digest differs from the binding")
    if unit_y is not None and not np.array_equal(
            y, np.asarray(unit_y, dtype=np.float64)):
        raise ExecutorRefusal(
            "persisted series is not byte-identical to the "
            "physically rebuilt unit")
    want_w = bank.origin_windows_for(
        len(y), int(design["role_geometry"]["rolling_origins"]),
        float(design["role_geometry"]["origin_base_frac"]),
        seasonal_period=period, horizon=horizon)
    if want_w != wins:
        raise ExecutorRefusal(
            "binding origin windows do not re-derive from the "
            "physical series through the sealed geometry")
    for okey in okeys:
        o = rec["rolling_origins"][okey]
        o_lo, o_hi = wins[okey]["score"]
        lo_t, hi_t = 0, o_lo
        if list(o.get("train", ())) != [lo_t, hi_t] or \
                list(o.get("score", ())) != [o_lo, o_hi]:
            raise ExecutorRefusal(
                f"{okey}: assay windows do not equal the binding "
                "windows")
        targets_idx = [t + horizon for t in
                       range(o_lo + hz.RIDGE_LAGS, o_hi - horizon)]
        train_idx = [t + horizon for t in
                     range(lo_t + hz.RIDGE_LAGS, hi_t - horizon)]
        if not targets_idx or not train_idx:
            raise ExecutorRefusal(f"{okey}: degenerate geometry")
        obs_want = y[targets_idx]
        yt = y[train_idx]
        den_want = hz._mase_denominator(y, lo_t, hi_t, period)
        _expect_close(den_want,
                      o.get("mase_denominator_train_snaive"),
                      f"rolling_origins.{okey}."
                      "mase_denominator_train_snaive")
        ex_mask, ex_thresh = hz.train_innovation_extremes(
            y, lo_t, hi_t, targets_idx, period)
        _expect_close(
            ex_thresh,
            o.get("extreme_innovation_threshold_train"),
            f"rolling_origins.{okey}."
            "extreme_innovation_threshold_train")
        snv_idx = [t - period for t in targets_idx]
        if min(snv_idx) < 0:
            raise ExecutorRefusal(
                f"{okey}: seasonal naive would read before the "
                "series start")
        entries = {("seasonal_naive", "baseline"):
                   o["results"]["seasonal_naive"]["metrics"]}
        for arm in arms:
            entries[(arm, "ridge")] = o["results"][arm]["ridge"]
            for s in design["seed_tape"]:
                entries[(arm, f"mlp_seed{s}")] = \
                    o["results"][arm]["mlp_small"][f"seed{s}"]
        for (arm, m), metrics in entries.items():
            base = f"{okey}__{arm}__{m}"
            pred = data[f"pred__{base}"]
            obs = data[f"obs__{base}"]
            for name, arr in (("pred", pred), ("obs", obs)):
                if arr.ndim != 1 or arr.dtype != np.float64 or \
                        not np.all(np.isfinite(arr)):
                    raise ExecutorRefusal(
                        f"{base}: persisted {name} is not 1-D "
                        "finite float64")
                if len(arr) != len(targets_idx):
                    raise ExecutorRefusal(
                        f"{base}: persisted {name} length "
                        f"{len(arr)} differs from the re-derived "
                        f"target count {len(targets_idx)}")
            if not np.array_equal(obs, obs_want):
                raise ExecutorRefusal(
                    f"{base}: persisted obs is not the exact "
                    "slice of the physical series at the "
                    "re-derived target indices — joint obs/pred "
                    "alteration never fabricates evidence")
            if m == "baseline":
                if not np.array_equal(pred, y[snv_idx]):
                    raise ExecutorRefusal(
                        f"{base}: the baseline prediction is not "
                        "the seasonal-naive slice of the series")
                q = np.quantile(y[[t - period for t in train_idx]]
                                - yt, [0.05, 0.95])
            else:
                fit_rows = data[f"fit__{base}"]
                if fit_rows.ndim != 1 or \
                        fit_rows.dtype != np.float64 or \
                        not np.all(np.isfinite(fit_rows)) or \
                        len(fit_rows) != len(train_idx):
                    raise ExecutorRefusal(
                        f"{base}: persisted fit rows are not 1-D "
                        "finite float64 of the train-target "
                        "length")
                q = np.quantile(fit_rows - yt, [0.05, 0.95])
            err = pred - obs_want
            mpath = (f"rolling_origins.{okey}.results.{arm}."
                     + ("metrics" if m == "baseline" else
                        ("ridge" if m == "ridge" else
                         f"mlp_small.{m.replace('mlp_', '')}")))
            mae = float(np.mean(np.abs(err)))
            _expect_close(mae,
                          metrics.get("mae_per_series_diagnostic"),
                          f"{mpath}.mae_per_series_diagnostic")
            _expect_close(float(np.sqrt(np.mean(err ** 2))),
                          metrics.get(
                              "rmse_per_series_diagnostic"),
                          f"{mpath}.rmse_per_series_diagnostic")
            mase = (mae / den_want) if den_want > 0 else None
            _expect_close(mase, metrics.get("mase_primary"),
                          f"{mpath}.mase_primary")
            lo_q, hi_q = float(q[0]), float(q[1])
            _expect_close(float(np.mean((err >= lo_q)
                                        & (err <= hi_q))),
                          metrics.get(
                              "interval_coverage_train_q90"),
                          f"{mpath}.interval_coverage_train_q90")
            _expect_close(hi_q - lo_q,
                          metrics.get("interval_width_train_q90"),
                          f"{mpath}.interval_width_train_q90")
            if ex_mask is not None and ex_mask.any():
                em = (float(np.mean(np.abs(err[ex_mask])))
                      / den_want) if den_want > 0 else None
                supp = int(ex_mask.sum())
            else:
                em, supp = None, 0
            _expect_close(em, metrics.get(
                "mase_on_extreme_innovations"),
                f"{mpath}.mase_on_extreme_innovations")
            if int(metrics.get("extreme_support", -1)) != supp:
                raise ExecutorRefusal(
                    f"{mpath}.extreme_support: recomputed {supp} "
                    "vs recorded "
                    f"{metrics.get('extreme_support')!r} — does "
                    "not recompute from persisted arrays")
    return {"verified_units": 1, "unit_id": uid, "mode": mode}


# ---------------- C63: adjudication ----------------

def _unit_srcs(root, uid):
    safe = _safe_name(uid)
    names = {k: f"{k}_{safe}.{'npz' if k == 'ARRAYS' else 'json'}"
             for k in ("RECORD", "ARRAYS", "TERMINAL", "CLAIM")}
    if isinstance(root, ResultsRoot):
        return {k: (root, n) for k, n in names.items()}
    return {k: Path(root) / n for k, n in names.items()}


def adjudicate_unit_shallow(root, uid: str) -> tuple:
    """Physical-presence scan ONLY (C69: descriptor-relative when
    given a ResultsRoot): PENDING / COMPLETED / TERMINAL_PRESENT
    / UNCERTAIN. TERMINAL_FAILED exists ONLY as a deep
    adjudication outcome (C59). A unit carrying BOTH a record and
    a terminal is UNCERTAIN (C71)."""
    srcs = _unit_srcs(root, uid)
    has = {k: _src_exists(v) for k, v in srcs.items()}
    if has["RECORD"] and has["TERMINAL"]:
        return ("UNCERTAIN",
                f"{uid}: BOTH a record and a terminal exist — "
                "contradictory evidence")
    if has["RECORD"]:
        if not has["ARRAYS"]:
            return ("UNCERTAIN", f"{uid}: record without arrays")
        return ("COMPLETED", "")
    if has["TERMINAL"]:
        return ("TERMINAL_PRESENT", "")
    if has["ARRAYS"]:
        return ("UNCERTAIN",
                f"{uid}: arrays without a record — crash between "
                "NPZ and record")
    if has["CLAIM"]:
        return ("UNCERTAIN",
                f"{uid}: claim without terminal or record — a "
                "crashed or budget-stopped attempt; explicit "
                "EXTERNAL operator disposition required")
    return ("PENDING", "")


def adjudicate_unit_deep(root, uid: str, design: dict,
                         authority: dict, mode: str,
                         unit_y=None) -> tuple:
    st, why = adjudicate_unit_shallow(root, uid)
    srcs = _unit_srcs(root, uid)
    if st == "COMPLETED":
        try:
            verify_unit_record(srcs["RECORD"], srcs["ARRAYS"],
                               design, authority=authority,
                               unit_y=unit_y, mode_expected=mode)
            return ("COMPLETED_VERIFIED", "")
        except SystemExit as exc:
            return ("UNCERTAIN", f"{uid}: record does not "
                    f"verify deeply — {str(exc)[:140]}")
    if st == "TERMINAL_PRESENT":
        try:
            verify_unit_terminal(
                srcs["TERMINAL"], design,
                authority=authority, mode_expected=mode)
            return ("TERMINAL_FAILED", "")
        except SystemExit as exc:
            return ("UNCERTAIN", f"{uid}: terminal does not "
                    f"verify under current authority — "
                    f"{str(exc)[:140]}")
    return (st, why)


def final_adjudication(rr: ResultsRoot, uids, design: dict,
                       authority: dict, mode: str,
                       rebuild) -> dict:
    """C63/C69/C71: before the lock is released with success —
    the declared root identity is revalidated, the control
    directory carries EXACTLY the expected objects (foreign,
    duplicate or unrecognized objects refuse), EVERY unit
    re-adjudicates deeply under current authority through the
    held descriptors, zero UNCERTAIN, counts equal the sealed
    population exactly."""
    rr.revalidate()
    # C78: the final chain is FRESH — a new component-by-component
    # walk whose device/inode pins must equal the held ones, so a
    # replace-then-restore with another inode at the same
    # pathname refuses even between operations.
    rr_fresh = ResultsRoot(rr.path)
    if rr_fresh._pins != rr._pins:
        rr_fresh.close()
        raise ExecutorRefusal(
            "final adjudication found a REPLACED-AND-RESTORED "
            "results root (same pathname, different identity) — "
            "refusing success")
    rr = rr_fresh
    expected_names = set()
    for uid in uids:
        safe = _safe_name(uid)
        expected_names.update({f"CLAIM_{safe}.json",
                               f"RECORD_{safe}.json",
                               f"ARRAYS_{safe}.npz",
                               f"TERMINAL_{safe}.json"})
    actual = set(rr.listdir(rr.units_fd))
    foreign = actual - expected_names
    if foreign:
        raise ExecutorRefusal(
            "final adjudication found foreign or unrecognized "
            f"objects in the control directory: "
            f"{sorted(foreign)[:5]} — refusing success")
    counts = {"COMPLETED_VERIFIED": 0, "TERMINAL_FAILED": 0}
    for uid in uids:
        st, why = adjudicate_unit_shallow(rr, uid)
        y = rebuild(uid) if st == "COMPLETED" else None
        st, why = adjudicate_unit_deep(rr, uid, design,
                                       authority, mode, unit_y=y)
        if st == "UNCERTAIN":
            raise ExecutorRefusal(
                "final adjudication found typed uncertainty — "
                f"{why}; the campaign cannot exit successfully")
        if st == "PENDING":
            raise ExecutorRefusal(
                f"final adjudication found {uid} still PENDING — "
                "counts do not close the sealed population")
        counts[st] += 1
    if counts["COMPLETED_VERIFIED"] + counts["TERMINAL_FAILED"] \
            != len(list(uids)):
        raise ExecutorRefusal(
            "final adjudication counts do not equal the sealed "
            "population exactly")
    return counts


def declare_attempt_failed(out_root: Path, uid: str,
                           mode: str = "confirmatory") -> None:
    """C60: operator disposition is EXTERNAL authority. Gates
    first; the uncertain claim fully verified; then the separate
    Musashi disposition record at the private reviewer root is
    consumed — it pins claim digest, unit, attempt, current
    execution record, decision and reason. Candidate-authored
    text grants nothing."""
    if mode not in MODES:
        raise ExecutorRefusal(f"unknown execution mode {mode!r}")
    rr = ResultsRoot(out_root, create=False)
    # C83: the active design is resolved ONCE; every consumer
    # below uses the frozen snapshot — no post-gate reread.
    if mode == "confirmatory":
        facts = conf.verify_confirmatory_gates(
            MANIFEST_PATH, active_design_path(),
            census_path=CENSUS_PATH)
        design = facts["design"].doc
        authority = {
            "sealed_design_file_sha256":
                facts["design_file_sha256"],
            "sealed_design_self_sha256":
                facts["design_self_sha256"],
            "design_review_record_sha256":
                facts["review_record_sha256"],
            "execution_record_sha256":
                facts["execution_record_sha256"],
            "manifest_sha256": facts["manifest_sha256"],
            "census_sha256": facts["census_sha256"]}
    else:
        design_fz = conf.freeze_evidence_file(
            active_design_path(), "active design",
            self_sha_key="design_sha256")
        design = design_fz.doc
        authority = physical_authority(
            design, mode, design_file_sha=design_fz.bytes_sha256)
    st, why = adjudicate_unit_shallow(rr, uid)
    if not (st == "UNCERTAIN" and "claim without" in why):
        raise ExecutorRefusal(
            f"{uid}: operator disposition applies only to a "
            f"claim without terminal or record (state: {st})")
    safe = _safe_name(uid)
    claim = verify_unit_claim(
        (rr, f"CLAIM_{safe}.json"), design, authority=authority,
        mode_expected=mode, uid_expected=uid)
    disp_path = conf.AUTHORITY_ROOT / \
        f"MUSASHI_T2_DISPOSITION_{safe}.json"
    fd = conf._open_private_authority_file(
        disp_path,
        missing_msg=(
            "T2_DISPOSITION_RECORD_REQUIRED: an uncertain "
            "attempt is closed only by the EXTERNAL Musashi "
            "disposition record at the private reviewer root — "
            "candidate-authored text grants nothing"))
    try:
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    disp_sha = hashlib.sha256(raw).hexdigest()
    disp = _strict_parse(raw, "disposition record")
    _DISP_KEYS = {"schema", "reviewed_at_date", "reviewer",
                  "decision", "unit_id", "attempt_id",
                  "claim_sha256", "execution_record_sha256",
                  "reason"}
    if set(disp) != _DISP_KEYS:
        raise ExecutorRefusal(
            "disposition record keys are not the exact schema")
    for k in _DISP_KEYS:
        if type(disp[k]) is not str or not disp[k]:
            raise ExecutorRefusal(
                f"disposition record field {k!r} must be a "
                "nonempty string")
    if disp["schema"] != \
            "agent_multi.musashi_t2_disposition_record.v1":
        raise ExecutorRefusal(
            "disposition record carries a foreign schema")
    import datetime as _dt
    try:
        d_ = _dt.date.fromisoformat(disp["reviewed_at_date"])
    except ValueError:
        raise ExecutorRefusal(
            "disposition reviewed_at_date is not a canonical ISO "
            "date")
    if d_.isoformat() != disp["reviewed_at_date"]:
        raise ExecutorRefusal(
            "disposition reviewed_at_date is not canonical")
    if disp["reviewer"] != "General Musashi":
        raise ExecutorRefusal(
            "disposition record author field is not the external "
            "reviewer role")
    if disp["decision"] != "DECLARE_ATTEMPT_FAILED":
        raise ExecutorRefusal(
            "disposition record decision does not close this "
            "attempt")
    if disp["unit_id"] != uid or \
            disp["attempt_id"] != claim["attempt_id"]:
        raise ExecutorRefusal(
            "disposition record does not name this unit and "
            "attempt")
    if disp["claim_sha256"] != claim["claim_sha256"]:
        raise ExecutorRefusal(
            "disposition record does not pin the physical claim")
    if disp["execution_record_sha256"] != \
            authority["execution_record_sha256"]:
        raise ExecutorRefusal(
            "disposition record does not pin the CURRENT "
            "execution authority")
    term = {"schema": "agent_multi.t2_unit_terminal.v3",
            "unit_id": uid, "attempt_id": claim["attempt_id"],
            "mode": mode,
            **{k: claim[k] for k in
               (*_AUTHORITY_KEYS, "code_identity",
                "pinned_commit", "pinned_tree",
                "campaign_generation", "unit_binding")},
            "closed_claim_sha256": claim["claim_sha256"],
            "terminal": "FAILED",
            "failure_class": "OPERATOR_DISPOSITION",
            "reason": disp["reason"][:300],
            "operator_disposition": True,
            "disposition_record_sha256": disp_sha,
            "wall_seconds": 0.0}
    term["terminal_sha256"] = _self_sha(term, "terminal_sha256")
    rr.excl_write(rr.units_fd, f"TERMINAL_{safe}.json",
                  json.dumps(term, indent=1).encode())


def census_of_work(design: dict) -> dict:
    """C70: the census names the PHYSICAL work of the corrected
    harness, distinguishing selected model instances, selection
    sequences, candidate fits, linear solves, prediction sets and
    baseline evaluations — never a single conflated 'fits'
    number."""
    tp = design["task_population"]
    u = len(tp["series_ids"])
    o = int(design["role_geometry"]["rolling_origins"])
    seeds = len(design["seed_tape"])
    a = len(design["arms"])
    return {"units": u, "origins_per_unit": o,
            "arms": a, "mlp_seeds": seeds,
            "selected_model_instances": u * o * a * (1 + seeds),
            "linear_solves": u * o * a,
            "mlp_selection_sequences": u * o * a * seeds,
            "mlp_candidate_fits": u * o * a * seeds * 4,
            "prediction_sets": u * o * a * (1 + seeds) * 2,
            "baseline_evals": u * o}


def _heartbeat(rr: ResultsRoot, payload: dict) -> None:
    """C61/C69: random per-write temp name + descriptor-relative
    replace, after root-identity revalidation — no fixed shared
    .tmp path, no write to a replaced root."""
    rr.revalidate()
    tmp = f".hb_{os.urandom(8).hex()}"
    data = json.dumps({**payload, "pid": os.getpid(),
                       "monotonic": time.monotonic()},
                      indent=1).encode()
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                 | os.O_NOFOLLOW, 0o600, dir_fd=rr.root_fd)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.rename(tmp, "EXECUTOR_HEARTBEAT.json",
              src_dir_fd=rr.root_fd, dst_dir_fd=rr.root_fd)
    rr._post_identity("HEARTBEAT")


def _adjudication_counts(units_dir: Path, uids) -> dict:
    counts = {"PENDING": 0, "COMPLETED": 0, "TERMINAL_PRESENT": 0,
              "UNCERTAIN": 0}
    causes = []
    for uid in uids:
        st, why = adjudicate_unit_shallow(units_dir, uid)
        counts[st] += 1
        if st == "UNCERTAIN":
            causes.append(why)
    return {"counts": counts, "uncertain_causes": causes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--mechanical-rehearsal", action="store_true")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--takeover-stale-lock", action="store_true")
    ap.add_argument("--declare-attempt-failed", type=str,
                    default=None, metavar="UNIT_ID")
    ap.add_argument("--mode", type=str, default="confirmatory")
    args = ap.parse_args(argv)
    os.nice(15)
    if args.declare_attempt_failed:
        if args.out_root is None:
            raise ExecutorRefusal(
                "operator disposition requires --out-root")
        declare_attempt_failed(args.out_root,
                               args.declare_attempt_failed,
                               mode=args.mode)
        print(json.dumps({"operator_disposition": "RECORDED",
                          "unit": args.declare_attempt_failed}))
        return 0
    if args.mechanical_rehearsal:
        return rehearse(args.out_root)
    if not (args.execute or args.plan):
        raise ExecutorRefusal(
            "choose --execute/--plan (sealed bank; gated by the "
            "external v2 execution record) or "
            "--mechanical-rehearsal (dev units only)")
    out_root = args.out_root or (
        STATE / "t2_confirmatory_results_v6")
    # C83/C84: the pathnames are resolved ONCE, here; the gate
    # reads each evidence file from a single O_NOFOLLOW stream and
    # returns the frozen snapshot. From this point on, NOTHING in
    # the campaign calls active_design_path() or reparses the
    # design/manifest/census — planning, census of work, the task
    # list, unit reconstruction, limits, claims and the final
    # adjudication all consume the same frozen object.
    facts = conf.verify_confirmatory_gates(
        MANIFEST_PATH, active_design_path(),
        census_path=CENSUS_PATH)
    design = facts["design"].doc
    manifest = facts["manifest"].doc
    # C86: the executable v6-to-successor diff re-runs at the
    # FINAL point of use, on the frozen bytes the campaign will
    # consume — not only at the gate's earlier read.
    if design.get("schema") == conf.T2_SUCCESSOR_SCHEMA:
        conf.verify_resource_successor(design)
    authority = {
        "sealed_design_file_sha256": facts["design_file_sha256"],
        "sealed_design_self_sha256": facts["design_self_sha256"],
        "design_review_record_sha256":
            facts["review_record_sha256"],
        "execution_record_sha256":
            facts["execution_record_sha256"],
        "manifest_sha256": facts["manifest_sha256"],
        "census_sha256": facts["census_sha256"]}
    pins = _git_head_tree()
    work = census_of_work(design)
    uids = design["task_population"]["series_ids"]
    if args.plan:
        adj = (_adjudication_counts(out_root / "units", uids)
               if (out_root / "units").is_dir()
               else {"counts": {"PENDING": len(uids),
                                "COMPLETED": 0,
                                "TERMINAL_PRESENT": 0,
                                "UNCERTAIN": 0},
                     "uncertain_causes": []})
        print(json.dumps({"plan": work,
                          "adjudication": adj,
                          "authority": {k: v[:16] for k, v in
                                        authority.items()}},
                         indent=1))
        return 0
    # ---- durable effects begin here, gates already open ----
    session_uuid = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
    session_n, rr = acquire_lock(
        out_root, session_uuid,
        takeover_stale=args.takeover_stale_lock)
    conf.open_attempt_ledger(rr.path / "T2_ATTEMPT_LEDGER.json")
    limits = {"max_wall_seconds":
              design["resource_contract"]["max_wall_seconds"],
              "max_rss_bytes":
              design["resource_contract"]["max_rss_bytes"]}
    wall = WallAuthority(rr, limits, resolve_stop_file(design),
                         session_uuid)
    supervisor = make_fit_supervisor(limits, wall)
    adj = _adjudication_counts(rr, uids)
    if adj["counts"]["UNCERTAIN"]:
        release_lock(rr, session_n, session_uuid)
        wall.close()
        raise ExecutorRefusal(
            "UNCERTAIN units block the run until explicit "
            "EXTERNAL operator disposition: "
            + "; ".join(adj["uncertain_causes"][:4]))
    import t2_assay_harness as hz
    co = hz.load_co()
    raw_root = STATE / "t2_public_raw"

    def _rebuild(uid):
        return np.asarray(
            load_bank_unit(design, uid, manifest, raw_root)["y"],
            dtype=np.float64)

    done = failed = verified_resumed = 0
    current_uid = None
    try:
        for uid in uids:
            current_uid = uid
            wall.check(f"between_units:{uid}")
            st, why = adjudicate_unit_shallow(rr, uid)
            if st == "COMPLETED":
                st2, why2 = adjudicate_unit_deep(
                    rr, uid, design, authority,
                    "confirmatory", unit_y=_rebuild(uid))
                if st2 != "COMPLETED_VERIFIED":
                    raise ExecutorRefusal(
                        f"resume re-verification failed — {why2}")
                verified_resumed += 1
                continue
            if st == "TERMINAL_PRESENT":
                st2, why2 = adjudicate_unit_deep(
                    rr, uid, design, authority,
                    "confirmatory")
                if st2 != "TERMINAL_FAILED":
                    raise ExecutorRefusal(
                        f"terminal re-verification failed — "
                        f"{why2}")
                failed += 1     # preserved as missing, never rerun
                continue
            _heartbeat(rr, {
                "current_unit": uid, "done": done,
                "failed": failed,
                "resumed_verified": verified_resumed,
                "wall_consumed_s": round(wall.consumed(), 1)})
            unit = load_bank_unit(design, uid, manifest, raw_root)
            try:
                run_unit(hz, co, unit, design, authority, rr,
                         "confirmatory", pins,
                         guard=wall.check,
                         fit_supervisor=supervisor)
                done += 1
            except T2AssayFailed:
                failed += 1     # integral typed terminal written
    except T2BudgetStop as stop:
        in_flight = {}
        if current_uid is not None:
            st_f, why_f = adjudicate_unit_shallow(rr,
                                                  current_uid)
            in_flight = {"unit_id": current_uid, "state": st_f,
                         "why": why_f}
        stop_report = {
            "schema": "agent_multi.t2_stop_report.v2",
            "session_uuid": session_uuid,
            "checkpoint": stop.checkpoint,
            "reason": str(stop),
            "wall_consumed_s": round(wall.consumed(), 1),
            "done": done, "failed_preserved": failed,
            "resumed_verified": verified_resumed,
            "in_flight_adjudication": in_flight}
        stop_report["report_sha256"] = _self_sha(stop_report,
                                                 "report_sha256")
        rr.excl_write(rr.root_fd,
                      f"T2_STOP_REPORT_{session_uuid}.json",
                      json.dumps(stop_report, indent=1).encode())
        release_lock(rr, session_n, session_uuid)
        wall.close()
        raise
    # C63: physical adjudication controls process success
    counts = final_adjudication(rr, uids, design, authority,
                                "confirmatory", _rebuild)
    rr.revalidate()
    release_lock(rr, session_n, session_uuid)
    wall.close()
    print(json.dumps({"done": done, "failed_preserved": failed,
                      "resumed_verified": verified_resumed,
                      "final_adjudication": counts}, indent=1))
    return 0


def rehearse(out_root: Path = None) -> int:
    out_root = Path(out_root or
                    (Path.home() / ".cache/t2_rehearsal_v6"))
    import shutil
    if out_root.exists():
        if Path.home() / ".cache" not in out_root.parents and \
                "t2" not in out_root.name:
            raise ExecutorRefusal(
                "refusing to clear a rehearsal root outside the "
                "cache area")
        shutil.rmtree(out_root)
    # C83: one frozen read; the rehearsal authority names the
    # digest of the exact stream it consumed.
    design_fz = conf.freeze_evidence_file(
        active_design_path(), "active design",
        self_sha_key="design_sha256")
    design = design_fz.doc
    sealed_ids = set(design["task_population"]["series_ids"])
    authority = physical_authority(
        design, "mechanical_rehearsal",
        design_file_sha=design_fz.bytes_sha256)
    pins = _git_head_tree()
    limits = {"max_wall_seconds":
              design["resource_contract"]["max_wall_seconds"],
              "max_rss_bytes":
              design["resource_contract"]["max_rss_bytes"]}
    session_uuid = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
    session_n, rr = acquire_lock(out_root, session_uuid)
    wall = WallAuthority(rr, limits, resolve_stop_file(design),
                         session_uuid)
    supervisor = make_fit_supervisor(limits, wall)
    import t2_assay_harness as hz
    co = hz.load_co()
    import t2_public_data_census as dev_census_mod
    census = dev_census_mod.build_census()
    timings = {}
    units_y = {}
    for uid in DEV_UNITS:
        assert uid not in sealed_ids
        wall.check(f"between_units:{uid}")
        unit = hz.load_task_unit(census, uid)
        units_y[uid] = np.asarray(unit["y"], dtype=np.float64)
        t0 = time.time()
        run_unit(hz, co, unit, design, authority, rr,
                 "mechanical_rehearsal", pins, guard=wall.check,
                 fit_supervisor=supervisor)
        timings[uid] = round(time.time() - t0, 2)
    counts = final_adjudication(
        rr, DEV_UNITS, design, authority, "mechanical_rehearsal",
        lambda uid: units_y[uid])
    rr.revalidate()
    release_lock(rr, session_n, session_uuid)
    wall.close()
    print(json.dumps({
        "rehearsal": "COMPLETE_MECHANICS_ONLY",
        "units": list(DEV_UNITS),
        "per_unit_wall_seconds": timings,
        "sealed_bank_series_touched": 0,
        "records_verified_from_persisted_arrays": len(DEV_UNITS),
        "final_adjudication": counts,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
