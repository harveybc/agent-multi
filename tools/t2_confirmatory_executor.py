#!/usr/bin/env python3
"""T2 confirmatory EXECUTOR (orders C42-C47 + C48-C56) —
implemented, and STRUCTURALLY CLOSED for science: `--execute`
consumes the SEALED v6 design only after ALL pure gates open,
including the external Musashi v2 EXECUTION record (which pins the
sealed design, the review record, manifest, census, the physical
executor code identity and the FULL executing checkout). That
record does not exist; no confirmatory score is computed. The
mechanical rehearsal (`--mechanical-rehearsal`) runs the SAME real
executor over the three DEVELOPMENT units only.

C49/C50/C51: per unit, ONE immutable (O_EXCL, 0600, fsynced)
self-digested v2 record whose verification RE-DERIVES everything —
unit membership in the sealed population, the exact unit_binding,
every authority digest against the physical objects, the code
identity against the checkout, the exact NPZ inventory
(allow_pickle=False, descriptor-first custody), each `obs` as the
exact slice of the physical series at re-derived target indices,
the baseline prediction from the series itself, and EVERY consumed
metric (MASE + denominator, MAE, RMSE, coverage, width, extreme
threshold/mask/support, MASE-on-extremes) from persisted arrays
and fit rows. Changing a self-digest never converts forgery into
fact.

C52: an EXECUTING budget guard is checked between units AND inside
each unit (origins, arms, ridge, MLP seeds, epoch candidates);
non-interruptible sklearn fits run under a supervised fork worker
with wall/RSS bounds and typed harvest; accumulated wall persists
durably across resumes (restarting never renews the 4 h); the
stop-file lives at the design-declared <state_root>.

C53: `--plan` is PURE read-only (zero directories, claims, locks,
ledgers); every durable effect happens only in `--execute`, after
all gates.

C54: the executor lock is a durable MONOTONIC session protocol
(locks are never unlinked; release is a durable record; recovery
follows physical facts — a provably dead pid still requires an
explicit recorded takeover); every unit adjudicates exactly one of
PENDING / COMPLETED_VERIFIED / TERMINAL_FAILED / UNCERTAIN, and an
UNCERTAIN attempt blocks with its typed cause until an explicit
recorded operator disposition. A budget stop mid-unit leaves the
claim WITHOUT a terminal on purpose: the attempt adjudicates
UNCERTAIN and is never silently reused."""
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
MANIFEST_PATH = STATE / "t2_public_data_manifest_20260906.json"
CENSUS_PATH = STATE / "t2_bank_census_20260906.json"
DEV_UNITS = ("sm_co2", "sm_sunspots", "sm_nile")
MODES = ("confirmatory", "mechanical_rehearsal")
REHEARSAL_EXECUTION_SENTINEL = (
    "REHEARSAL_NO_EXECUTION_RECORD_MECHANICS_ONLY")
REHEARSAL_DIGEST_SENTINEL = "REHEARSAL_DEV_CENSUS_ONLY"
PER_FIT_WALL_SECONDS = 120.0
WALL_PERSIST_CADENCE_S = 5.0

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


class ExecutorRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


class T2BudgetStop(SystemExit):
    """C52.5: a typed budget stop naming its exact checkpoint. It
    deliberately does NOT write a unit terminal: an interrupted
    attempt adjudicates UNCERTAIN and awaits explicit recorded
    operator disposition — it is never silently reused."""
    def __init__(self, msg, checkpoint):
        self.checkpoint = checkpoint
        super().__init__(f"T2_BUDGET_STOP at {checkpoint}: {msg}")


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


def _fsync_dir(d: Path) -> None:
    dfd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _excl_write(path: Path, payload: bytes) -> None:
    """C50.1: exclusive creation, 0600, file AND directory fsync —
    a preexisting object is never truncated or reused."""
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _excl_write_npz(path: Path, arrays: dict) -> None:
    """C50.1 for arrays: the NPZ is created O_EXCL 0600 and fsynced
    (file and directory) — open(.., 'wb') truncation is gone."""
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as f:
            np.savez_compressed(f, **arrays)
            f.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _read_private_bytes(path: Path, what: str) -> bytes:
    """C50.2: ONE descriptor per evidence object — O_NOFOLLOW,
    regular file, executing uid, exact mode 0600; hash and parse
    consume the same byte stream."""
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


def resolve_stop_file(design: dict) -> Path:
    """C52.4: the stop-file location comes from the SEALED design's
    resource contract — `<state_root>/T2_STOP` — never from a
    reinterpretation of the results root."""
    decl = design["resource_contract"]["stop_file"]
    if decl != "<state_root>/T2_STOP":
        raise ExecutorRefusal(
            f"sealed stop_file declaration {decl!r} is not the "
            "known contract — refusing to guess a location")
    return STATE / "T2_STOP"


# ---------------- C52: executing budget guard ----------------

class BudgetGuard:
    """Wall (durably ACCUMULATED across resumes), RSS and the
    design-declared stop-file, checkable at ANY labelled point.
    Wall progress persists to an append-only JSONL ledger every few
    seconds — restarting the process never renews the 4 h."""
    def __init__(self, limits: dict, stop_file: Path,
                 wall_ledger: Path, session_id: str):
        self.limits = limits
        self.stop_file = Path(stop_file)
        self.session_id = session_id
        self.wall_ledger = Path(wall_ledger)
        self.prior_wall = self._prior_from_ledger()
        self._t0 = time.monotonic()
        self._last_persist = 0.0
        self._fd = os.open(
            str(self.wall_ledger),
            os.O_CREAT | os.O_WRONLY | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self._persist(force=True)

    def _prior_from_ledger(self) -> float:
        if not self.wall_ledger.exists():
            return 0.0
        best = {}
        for line in self.wall_ledger.read_text().splitlines():
            try:
                d = json.loads(line)
                sid, el = d["session"], float(d["elapsed_seconds"])
            except (ValueError, KeyError, TypeError):
                continue        # a torn trailing line loses <=5 s
            if sid != self.session_id:
                best[sid] = max(best.get(sid, 0.0), el)
        return float(sum(best.values()))

    def session_elapsed(self) -> float:
        return time.monotonic() - self._t0

    def total_elapsed(self) -> float:
        return self.prior_wall + self.session_elapsed()

    def _persist(self, force=False) -> None:
        now = time.monotonic()
        if not force and now - self._last_persist < \
                WALL_PERSIST_CADENCE_S:
            return
        self._last_persist = now
        line = json.dumps(
            {"session": self.session_id,
             "elapsed_seconds": round(self.session_elapsed(), 3)}
        ) + "\n"
        os.write(self._fd, line.encode())
        os.fsync(self._fd)

    def check(self, label: str) -> None:
        self._persist()
        total = self.total_elapsed()
        if total > float(self.limits["max_wall_seconds"]):
            self._persist(force=True)
            raise T2BudgetStop(
                f"accumulated wall {total:.1f}s exceeds the sealed "
                f"{self.limits['max_wall_seconds']}s (prior "
                f"sessions {self.prior_wall:.1f}s — resume never "
                "renews the budget)", label)
        rss = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss * 1024
        if rss > int(self.limits["max_rss_bytes"]):
            self._persist(force=True)
            raise T2BudgetStop(
                f"RSS {rss} bytes exceeds the sealed bound", label)
        if self.stop_file.exists():
            self._persist(force=True)
            raise T2BudgetStop(
                f"external stop request present at the "
                f"design-declared state root "
                f"({self.stop_file.name})", label)

    def close(self) -> None:
        self._persist(force=True)
        os.close(self._fd)


def make_fit_supervisor(limits: dict):
    """C52.2: run a non-interruptible fit under a supervised fork
    worker with its own wall bound, an address-space ceiling and a
    post-fit RSS check; harvest is TYPED (OK / WALL_KILLED /
    RSS_EXCEEDED / CRASH) — a single fit can never make the global
    limits invisible. The child executes the identical closure, so
    no number changes."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    rss_cap = int(limits["max_rss_bytes"])

    def supervise(fn, label):
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
                send.send(("RSS_EXCEEDED", "address-space ceiling"))
            except BaseException as exc:
                send.send(("CRASH",
                           f"{type(exc).__name__}: "
                           f"{str(exc)[:120]}"))

        p = ctx.Process(target=_child, daemon=True)
        p.start()
        send.close()
        if recv.poll(PER_FIT_WALL_SECONDS):
            kind, payload = recv.recv()
            p.join(5)
            if kind == "OK":
                return payload
            raise SupervisedFitFailure(
                f"{label} harvested {kind}: {payload}")
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join(5)
        raise SupervisedFitFailure(
            f"{label} exceeded the per-fit wall bound "
            f"({PER_FIT_WALL_SECONDS}s) — WALL_KILLED")
    return supervise


# ---------------- C54: durable monotonic lock ----------------

def _locks_dir(out_root: Path) -> Path:
    d = Path(out_root) / "locks"
    d.mkdir(mode=0o700, exist_ok=True)
    return d


def _scan_sessions(locks: Path) -> list:
    out = []
    for p in sorted(locks.glob("SESSION_*.json")):
        try:
            n = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            raise ExecutorRefusal(
                f"foreign object {p.name!r} in the lock protocol "
                "directory")
        out.append((n, p))
    return out


def acquire_lock(out_root: Path, session_uuid: str,
                 takeover_stale: bool = False) -> int:
    """C54.1: locks are NEVER unlinked. Acquisition appends the
    next monotonic SESSION record; a session is free only when its
    durable RELEASE record exists. Without one, recovery follows
    physical facts: a live pid refuses; a provably dead pid still
    requires an EXPLICIT recorded takeover; an inconclusive pid
    check refuses as UNCERTAIN. An fsync exception can only leave
    the lock HELD — never absent."""
    locks = _locks_dir(out_root)
    sessions = _scan_sessions(locks)
    nxt = 1
    if sessions:
        n, p = sessions[-1]
        nxt = n + 1
        rel = locks / f"RELEASE_{n:06d}.json"
        if not rel.exists():
            doc = _strict_parse(
                _read_private_bytes(p, f"lock SESSION_{n:06d}"),
                f"lock SESSION_{n:06d}")
            pid = int(doc["pid"])
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
                    f"(pid {pid} is alive)")
            if not takeover_stale:
                raise ExecutorRefusal(
                    f"lock session {n} has no release record and "
                    f"its pid {pid} is dead — a STALE lock is a "
                    "physical fact requiring an explicit recorded "
                    "takeover (--takeover-stale-lock), never "
                    "silent absence")
            tk = {"schema": "agent_multi.t2_lock_takeover.v1",
                  "over_session": n, "dead_pid": pid,
                  "by_session_uuid": session_uuid,
                  "at_wall": time.time()}
            tk["takeover_sha256"] = _self_sha(tk,
                                              "takeover_sha256")
            _excl_write(locks / f"TAKEOVER_{n:06d}.json",
                        json.dumps(tk, indent=1).encode())
    doc = {"schema": "agent_multi.t2_lock_session.v1",
           "session": nxt, "session_uuid": session_uuid,
           "pid": os.getpid(), "started_wall": time.time()}
    doc["session_sha256"] = _self_sha(doc, "session_sha256")
    try:
        _excl_write(locks / f"SESSION_{nxt:06d}.json",
                    json.dumps(doc, indent=1).encode())
    except FileExistsError:
        raise ExecutorRefusal(
            "lost the lock-acquisition race — another executor "
            "claimed the next session")
    return nxt


def release_lock(out_root: Path, session_n: int,
                 session_uuid: str) -> None:
    doc = {"schema": "agent_multi.t2_lock_release.v1",
           "session": session_n, "session_uuid": session_uuid,
           "released_wall": time.time()}
    doc["release_sha256"] = _self_sha(doc, "release_sha256")
    _excl_write(_locks_dir(out_root)
                / f"RELEASE_{session_n:06d}.json",
                json.dumps(doc, indent=1).encode())


# ---------------- unit reconstruction ----------------

def load_bank_unit(design: dict, uid: str,
                   manifest: dict, raw_root: Path) -> dict:
    """Reconstruct ONE sealed-population unit from physical bytes
    via the census loaders and verify every unit_map binding."""
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
    """confirmatory: the EXACT sealed unit_map entry. rehearsal: a
    binding derived from the physical dev unit through the SAME
    geometry authority."""
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


# ---------------- C49-C51: production + total verification ----

def run_unit(hz, co, unit: dict, design: dict, authority: dict,
             out_root: Path, mode: str, guard=None,
             fit_supervisor=None) -> dict:
    """One unit through the REAL harness with array capture, then
    the immutable custody record + NPZ + full independent
    re-verification. A budget stop mid-unit re-raises WITHOUT a
    terminal (the claim adjudicates UNCERTAIN — C54.3); any other
    in-unit failure writes an integral typed FAILED terminal and
    the unit is preserved as missing per the sealed rule."""
    if mode not in MODES:
        raise ExecutorRefusal(f"unknown execution mode {mode!r}")
    uid = unit["unit_id"]
    udir = Path(out_root) / "units"
    udir.mkdir(mode=0o700, exist_ok=True)
    safe = _safe_name(uid)
    claim_p = udir / f"CLAIM_{safe}.json"
    attempt = ("attempt_"
               + hashlib.sha256(os.urandom(16)).hexdigest()[:16])
    binding = _binding_for(design, unit, mode)
    claim = {"schema": "agent_multi.t2_unit_claim.v2",
             "unit_id": uid, "attempt_id": attempt, "mode": mode,
             "pid": os.getpid(), "claimed_wall": time.time()}
    claim["claim_sha256"] = _self_sha(claim, "claim_sha256")
    try:
        _excl_write(claim_p, json.dumps(claim, indent=1).encode())
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
        term = {"schema": "agent_multi.t2_unit_terminal.v2",
                "unit_id": uid, "attempt_id": attempt,
                "mode": mode, "terminal": "FAILED",
                "failure_class": type(exc).__name__,
                "reason": str(exc)[:300],
                "operator_disposition": False,
                "wall_seconds": round(time.time() - t0, 2)}
        term["terminal_sha256"] = _self_sha(term,
                                            "terminal_sha256")
        _excl_write(udir / f"TERMINAL_{safe}.json",
                    json.dumps(term, indent=1).encode())
        raise
    npz_p = udir / f"ARRAYS_{safe}.npz"
    arrays = {"y": np.asarray(unit["y"], dtype=np.float64)}
    for (okey, arm, model), captured in sink.items():
        pred, obs, fit_rows = captured
        arrays[f"pred__{okey}__{arm}__{model}"] = pred
        arrays[f"obs__{okey}__{arm}__{model}"] = obs
        if fit_rows is not None:
            arrays[f"fit__{okey}__{arm}__{model}"] = fit_rows
    _excl_write_npz(npz_p, arrays)
    wrapper = {
        "schema": "agent_multi.t2_confirmatory_unit_record.v2",
        "mode": mode, "unit_id": uid, "attempt_id": attempt,
        **authority,
        "unit_binding": binding,
        "code_identity": code_identity(),
        "assay_record": rec,
        "arrays_npz_sha256": hashlib.sha256(
            npz_p.read_bytes()).hexdigest(),
        "wall_seconds": round(time.time() - t0, 2)}
    wrapper["record_sha256"] = _self_sha(wrapper, "record_sha256")
    rec_p = udir / f"RECORD_{safe}.json"
    _excl_write(rec_p, json.dumps(wrapper, indent=1).encode())
    verify_unit_record(rec_p, npz_p, design, authority=authority,
                       unit_y=np.asarray(unit["y"],
                                         dtype=np.float64),
                       mode_expected=mode)
    return {"record": str(rec_p.name),
            "wall": wrapper["wall_seconds"]}


def physical_authority(design: dict, mode: str,
                       repo_root: Path = None) -> dict:
    """C49.3: the expected authority digests, RE-DERIVED from the
    physical objects at verification time — never accepted from a
    producer's declaration. Confirmatory mode runs the full
    external-record verifications (review AND execution)."""
    if mode not in MODES:
        raise ExecutorRefusal(f"unknown execution mode {mode!r}")
    sealed_file_sha = _sha_file(SEALED_PATH)
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


def verify_unit_record(rec_path: Path, npz_path: Path,
                       design: dict, authority: dict = None,
                       unit_y=None, mode_expected: str = None,
                       repo_root: Path = None) -> dict:
    """C49/C50/C51: RE-DERIVE, never believe.

    Schema/keys/types exact; unit membership in the sealed
    population (or the dev set, by mode) with the EXACT binding;
    every authority digest against the PHYSICAL objects (recomputed
    here when not supplied by the verified gate step); code
    identity against the checkout; claim and filenames; NPZ custody
    (descriptor-first, exact inventory, allow_pickle=False, dtype/
    finiteness/lengths); every `obs` as the exact slice of the
    series at re-derived target indices; the baseline prediction
    from the series itself; and EVERY consumed metric recomputed
    from persisted arrays and fit rows, refusing with the exact
    path of the first forged entry."""
    import t2_assay_harness as hz
    repo_root = repo_root or conf.REPO
    rec_path, npz_path = Path(rec_path), Path(npz_path)
    wrapper = _strict_parse(
        _read_private_bytes(rec_path, "unit record"),
        "unit record")
    if set(wrapper) != _WRAPPER_KEYS:
        raise ExecutorRefusal(
            "unit record keys are not the exact v2 schema (diff: "
            f"{sorted(set(wrapper) ^ _WRAPPER_KEYS)})")
    for k in ("schema", "mode", "unit_id", "attempt_id",
              "sealed_design_file_sha256",
              "sealed_design_self_sha256",
              "design_review_record_sha256",
              "execution_record_sha256", "manifest_sha256",
              "census_sha256", "arrays_npz_sha256",
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
            f"expected {mode_expected!r} — rehearsal records never "
            "verify as confirmatory")
    uid = wrapper["unit_id"]
    import re as _re
    if not _re.fullmatch(r"attempt_[0-9a-f]{16}",
                         wrapper["attempt_id"]):
        raise ExecutorRefusal(
            "unit record attempt_id is not a canonical attempt "
            "identifier")
    # C49.2: membership + EXACT binding
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
    if rec_path.name != f"RECORD_{safe}.json":
        raise ExecutorRefusal(
            f"unit record filename {rec_path.name!r} does not "
            f"derive from its unit_id ({uid!r})")
    if npz_path.name != f"ARRAYS_{safe}.npz":
        raise ExecutorRefusal(
            f"arrays filename {npz_path.name!r} does not derive "
            f"from the unit_id ({uid!r})")
    claim_p = rec_path.parent / f"CLAIM_{safe}.json"
    claim = _strict_parse(
        _read_private_bytes(claim_p, "unit claim"), "unit claim")
    if _self_sha(claim, "claim_sha256") != \
            claim.get("claim_sha256"):
        raise ExecutorRefusal(
            "unit claim self-digest does not re-derive")
    if claim.get("unit_id") != uid or \
            claim.get("attempt_id") != wrapper["attempt_id"] or \
            claim.get("mode") != mode:
        raise ExecutorRefusal(
            "unit claim does not bind this record's unit, attempt "
            "and mode")
    # C49.3: authority against PHYSICAL objects
    expected = authority if authority is not None else \
        physical_authority(design, mode, repo_root=repo_root)
    for k in ("sealed_design_file_sha256",
              "sealed_design_self_sha256",
              "design_review_record_sha256",
              "execution_record_sha256", "manifest_sha256",
              "census_sha256"):
        if wrapper[k] != expected[k]:
            raise ExecutorRefusal(
                f"unit record {k} does not match the physical "
                "authority object — a repaired self-digest never "
                "converts false authority into fact")
    # C49.4: code identity against the CHECKOUT
    physical_ci = conf.executor_code_identity(repo_root)
    if wrapper["code_identity"] != physical_ci:
        raise ExecutorRefusal(
            "unit record code_identity does not match the "
            "reviewed checkout surface — declared identities "
            "grant nothing")
    # C50: NPZ custody — one descriptor, exact inventory, no
    # pickles, dtype/finiteness/shape
    raw_npz = _read_private_bytes(npz_path, "persisted arrays")
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
    # C50.5: windows re-derive through the ONE geometry authority
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


# ---------------- C54: adjudication ----------------

def adjudicate_unit_shallow(units_dir: Path, uid: str) -> tuple:
    """PENDING / COMPLETED (record present, verification pending) /
    TERMINAL_FAILED / UNCERTAIN — from physical presence only; the
    deep verification of COMPLETED happens in the resume loop under
    the CURRENT authority."""
    safe = _safe_name(uid)
    rec_p = units_dir / f"RECORD_{safe}.json"
    npz_p = units_dir / f"ARRAYS_{safe}.npz"
    term_p = units_dir / f"TERMINAL_{safe}.json"
    claim_p = units_dir / f"CLAIM_{safe}.json"
    if rec_p.exists():
        if not npz_p.exists():
            return ("UNCERTAIN", f"{uid}: record without arrays")
        return ("COMPLETED", "")
    if term_p.exists():
        try:
            term = _strict_parse(
                _read_private_bytes(term_p, "unit terminal"),
                "unit terminal")
            if _self_sha(term, "terminal_sha256") != \
                    term.get("terminal_sha256") or \
                    term.get("unit_id") != uid or \
                    term.get("terminal") != "FAILED":
                return ("UNCERTAIN",
                        f"{uid}: terminal does not re-derive")
        except SystemExit as exc:
            return ("UNCERTAIN", f"{uid}: {exc}")
        return ("TERMINAL_FAILED", "")
    if npz_p.exists():
        return ("UNCERTAIN",
                f"{uid}: arrays without a record — crash between "
                "NPZ and record")
    if claim_p.exists():
        return ("UNCERTAIN",
                f"{uid}: claim without terminal or record — a "
                "crashed or budget-stopped attempt; explicit "
                "operator disposition required "
                "(--declare-attempt-failed)")
    return ("PENDING", "")


def declare_attempt_failed(out_root: Path, uid: str,
                           reason: str) -> None:
    """C54: the EXPLICIT recorded operator disposition for an
    UNCERTAIN claim — writes an integral typed FAILED terminal
    naming the operator decision; never deletes the claim; refuses
    when the unit is not actually in the claim-without-outcome
    state."""
    units_dir = Path(out_root) / "units"
    state, why = adjudicate_unit_shallow(units_dir, uid)
    if not (state == "UNCERTAIN" and "claim without" in why):
        raise ExecutorRefusal(
            f"{uid}: operator disposition applies only to a claim "
            f"without terminal or record (state: {state})")
    safe = _safe_name(uid)
    claim = _strict_parse(
        _read_private_bytes(units_dir / f"CLAIM_{safe}.json",
                            "unit claim"), "unit claim")
    if not reason or type(reason) is not str:
        raise ExecutorRefusal(
            "operator disposition requires a written --reason")
    term = {"schema": "agent_multi.t2_unit_terminal.v2",
            "unit_id": uid, "attempt_id": claim["attempt_id"],
            "mode": claim["mode"], "terminal": "FAILED",
            "failure_class": "OPERATOR_DISPOSITION",
            "reason": reason[:300],
            "operator_disposition": True,
            "wall_seconds": 0.0}
    term["terminal_sha256"] = _self_sha(term, "terminal_sha256")
    _excl_write(units_dir / f"TERMINAL_{safe}.json",
                json.dumps(term, indent=1).encode())


def census_of_work(design: dict) -> dict:
    tp = design["task_population"]
    n_units = len(tp["series_ids"])
    ro = int(design["role_geometry"]["rolling_origins"])
    seeds = len(design["seed_tape"])
    arms = len(design["arms"])
    return {"units": n_units, "origins_per_unit": ro,
            "arms": arms, "mlp_seeds": seeds,
            "model_fits": n_units * ro * arms * (1 + seeds),
            "baseline_evals": n_units * ro}


def _heartbeat(out_root: Path, payload: dict) -> None:
    p = Path(out_root) / "EXECUTOR_HEARTBEAT.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(
        {**payload, "pid": os.getpid(),
         "monotonic": time.monotonic()}, indent=1))
    os.replace(tmp, p)


def _adjudication_counts(units_dir: Path, uids) -> dict:
    counts = {"PENDING": 0, "COMPLETED": 0, "TERMINAL_FAILED": 0,
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
    ap.add_argument("--plan", action="store_true",
                    help="PURE read-only: gates + work census + "
                         "adjudication summary; zero writes")
    ap.add_argument("--mechanical-rehearsal", action="store_true")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--takeover-stale-lock", action="store_true")
    ap.add_argument("--declare-attempt-failed", type=str,
                    default=None, metavar="UNIT_ID")
    ap.add_argument("--reason", type=str, default=None)
    args = ap.parse_args(argv)
    os.nice(15)
    if args.declare_attempt_failed:
        if args.out_root is None:
            raise ExecutorRefusal(
                "operator disposition requires --out-root")
        declare_attempt_failed(args.out_root,
                               args.declare_attempt_failed,
                               args.reason or "")
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
    # C53.2: EVERY gate — design, fresh population, both external
    # records and the full checkout — verifies BEFORE any effect.
    facts = conf.verify_confirmatory_gates(
        MANIFEST_PATH, SEALED_PATH, census_path=CENSUS_PATH)
    design = conf.strict_json_load(SEALED_PATH, "sealed design")
    manifest = conf.strict_json_load(MANIFEST_PATH, "manifest")
    authority = {
        "sealed_design_file_sha256": facts["design_file_sha256"],
        "sealed_design_self_sha256": design["design_sha256"],
        "design_review_record_sha256":
            facts["review_record_sha256"],
        "execution_record_sha256":
            facts["execution_record_sha256"],
        "manifest_sha256": facts["manifest_sha256"],
        "census_sha256": facts["census_sha256"]}
    work = census_of_work(design)
    uids = design["task_population"]["series_ids"]
    if args.plan:
        # C53.1: read-only — no directory creation, no lock, no
        # ledger, no claim; adjudication is a presence scan.
        adj = (_adjudication_counts(out_root / "units", uids)
               if (out_root / "units").is_dir()
               else {"counts": {"PENDING": len(uids),
                                "COMPLETED": 0,
                                "TERMINAL_FAILED": 0,
                                "UNCERTAIN": 0},
                     "uncertain_causes": []})
        print(json.dumps({"plan": work,
                          "adjudication": adj,
                          "authority": {k: v[:16] for k, v in
                                        authority.items()}},
                         indent=1))
        return 0
    # ---- C53.3: durable effects begin here, gates already open --
    out_root.mkdir(mode=0o700, exist_ok=True)
    session_uuid = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
    session_n = acquire_lock(out_root, session_uuid,
                             takeover_stale=args.takeover_stale_lock)
    conf.open_attempt_ledger(out_root / "T2_ATTEMPT_LEDGER.json")
    limits = {"max_wall_seconds":
              design["resource_contract"]["max_wall_seconds"],
              "max_rss_bytes":
              design["resource_contract"]["max_rss_bytes"]}
    guard = BudgetGuard(limits, resolve_stop_file(design),
                        out_root / "T2_WALL_LEDGER.jsonl",
                        session_uuid)
    supervisor = make_fit_supervisor(limits)
    units_dir = out_root / "units"
    units_dir.mkdir(mode=0o700, exist_ok=True)
    # C54: any UNCERTAIN unit blocks BEFORE new work
    adj = _adjudication_counts(units_dir, uids)
    if adj["counts"]["UNCERTAIN"]:
        release_lock(out_root, session_n, session_uuid)
        guard.close()
        raise ExecutorRefusal(
            "UNCERTAIN units block the run until explicit "
            "operator disposition: "
            + "; ".join(adj["uncertain_causes"][:4]))
    import t2_assay_harness as hz
    co = hz.load_co()
    raw_root = STATE / "t2_public_raw"
    done = failed = verified_resumed = 0
    stop_report = None
    try:
        for uid in uids:
            guard.check(f"between_units:{uid}")
            safe = _safe_name(uid)
            st, why = adjudicate_unit_shallow(units_dir, uid)
            if st == "COMPLETED":
                # C54.4: resume skips ONLY records fully
                # re-verified under the CURRENT authority,
                # including the physical series rebuild.
                unit = load_bank_unit(design, uid, manifest,
                                      raw_root)
                verify_unit_record(
                    units_dir / f"RECORD_{safe}.json",
                    units_dir / f"ARRAYS_{safe}.npz",
                    design, authority=authority,
                    unit_y=np.asarray(unit["y"],
                                      dtype=np.float64),
                    mode_expected="confirmatory")
                verified_resumed += 1
                continue
            if st == "TERMINAL_FAILED":
                failed += 1     # preserved as missing, never rerun
                continue
            _heartbeat(out_root, {
                "current_unit": uid, "done": done,
                "failed": failed,
                "resumed_verified": verified_resumed,
                "elapsed_total_s": round(guard.total_elapsed(),
                                         1)})
            unit = load_bank_unit(design, uid, manifest, raw_root)
            try:
                run_unit(hz, co, unit, design, authority,
                         out_root, "confirmatory",
                         guard=guard.check,
                         fit_supervisor=supervisor)
                done += 1
            except (T2BudgetStop, ExecutorRefusal,
                    KeyboardInterrupt):
                raise
            except BaseException:
                failed += 1     # typed terminal already written
    except T2BudgetStop as stop:
        # C52.5: publish the exact stop point; keep every terminal
        stop_report = {
            "schema": "agent_multi.t2_stop_report.v1",
            "session_uuid": session_uuid,
            "checkpoint": stop.checkpoint,
            "reason": str(stop),
            "elapsed_total_s": round(guard.total_elapsed(), 1),
            "done": done, "failed_preserved": failed,
            "resumed_verified": verified_resumed}
        stop_report["report_sha256"] = _self_sha(stop_report,
                                                 "report_sha256")
        _excl_write(out_root
                    / f"T2_STOP_REPORT_{session_uuid}.json",
                    json.dumps(stop_report, indent=1).encode())
        release_lock(out_root, session_n, session_uuid)
        guard.close()
        raise
    release_lock(out_root, session_n, session_uuid)
    guard.close()
    print(json.dumps({"done": done, "failed_preserved": failed,
                      "resumed_verified": verified_resumed},
                     indent=1))
    return 0


def rehearse(out_root: Path = None) -> int:
    """C46/C56: the mechanical rehearsal — the SAME real executor
    (guard, supervisor, custody, verification) over the three
    DEVELOPMENT units only; refuses any sealed-bank series;
    verifiable records; zero confirmatory scores."""
    out_root = out_root or (
        Path.home() / ".cache/t2_rehearsal_v6")
    out_root = Path(out_root)
    import shutil
    if out_root.exists():
        if Path.home() / ".cache" not in out_root.parents and \
                "t2" not in out_root.name:
            raise ExecutorRefusal(
                "refusing to clear a rehearsal root outside the "
                "cache area")
        shutil.rmtree(out_root)
    os.makedirs(out_root, mode=0o700)
    design = conf.strict_json_load(SEALED_PATH, "sealed design")
    sealed_ids = set(design["task_population"]["series_ids"])
    authority = physical_authority(design, "mechanical_rehearsal")
    limits = {"max_wall_seconds":
              design["resource_contract"]["max_wall_seconds"],
              "max_rss_bytes":
              design["resource_contract"]["max_rss_bytes"]}
    session_uuid = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
    session_n = acquire_lock(out_root, session_uuid)
    guard = BudgetGuard(limits, resolve_stop_file(design),
                        out_root / "T2_WALL_LEDGER.jsonl",
                        session_uuid)
    supervisor = make_fit_supervisor(limits)
    import t2_assay_harness as hz
    co = hz.load_co()
    import t2_public_data_census as dev_census_mod
    census = dev_census_mod.build_census()
    timings = {}
    for uid in DEV_UNITS:
        assert uid not in sealed_ids
        guard.check(f"between_units:{uid}")
        unit = hz.load_task_unit(census, uid)
        t0 = time.time()
        run_unit(hz, co, unit, design, authority, out_root,
                 "mechanical_rehearsal", guard=guard.check,
                 fit_supervisor=supervisor)
        timings[uid] = round(time.time() - t0, 2)
    release_lock(out_root, session_n, session_uuid)
    guard.close()
    print(json.dumps({
        "rehearsal": "COMPLETE_MECHANICS_ONLY",
        "units": list(DEV_UNITS),
        "per_unit_wall_seconds": timings,
        "sealed_bank_series_touched": 0,
        "records_verified_from_persisted_arrays": len(DEV_UNITS),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
