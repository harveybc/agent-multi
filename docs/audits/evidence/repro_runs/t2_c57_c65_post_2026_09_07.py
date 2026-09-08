"""POST for order T2 C57-C65: the seven audited counterexamples
are DEAD through the same productive APIs the PRE used.

C57 charge-in-advance hash-chained descriptor-bound wall: the
     PRE's eight crash sessions now yield ONE accepted charge and
     seven typed refusals.
C58 an empty/bare release filename never frees a LIVE lock; a
     real release is intent + two-phase completion witness.
C59 a fabricated authority-free terminal deep-adjudicates typed
     UNCERTAIN; TERMINAL_FAILED exists only after full
     re-derivation under current authority.
C60 a fabricated two-field claim refuses at claim verification;
     an uncertain attempt without the EXTERNAL Musashi
     disposition record refuses typed.
C61 a symlinked results root refuses BEFORE any write lands
     under its target (real trusted parents).
C62 the supervisor kills at min(remaining_global, per_fit),
     refuses dispatch with no budget, and leaves no orphan.
C63 a post-assay persistence crash HALTS typed with the claim
     preserved UNCERTAIN; final physical adjudication refuses
     success over any uncertainty; the v3 rehearsal closes with
     an exact final adjudication.

Zero sealed-bank series, zero scores, zero B4 objects. CPU only."""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault(
    "B4_T1_PREPROCESSOR_ROOT",
    str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))

import numpy as np  # noqa: E402

import t2_confirmatory as conf  # noqa: E402
import t2_confirmatory_executor as ex  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="t2_c57_post_"))
ex._TRUSTED_ROOT_PARENTS = tuple(ex._TRUSTED_ROOT_PARENTS) + (TMP,)
STATE = Path.home() / ".local/share/agent-multi"
design = conf.strict_json_load(
    STATE / "t2_screen_design_SEALED_V6.json", "sealed design")
stop = TMP / "T2_STOP"


def _die(fn, needle, what):
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), (needle, str(exc)[:160])
        print(f"  {what} -> {str(exc)[:100]}")
        return
    raise AssertionError(f"{what}: did NOT refuse")


print("== C57 DEAD: a restart cannot renew the wall ==")
budget = {"max_wall_seconds": 0.08, "max_rss_bytes": 8 << 30}
rr = ex.ResultsRoot(TMP / "renew", create=True)
accepted = refused = 0
for k in range(8):
    try:
        g = ex.WallAuthority(rr, budget, stop, f"crash{k}")
        accepted += 1
        os.close(g._fd)              # crash: the reservation stays
    except SystemExit as exc:
        assert "never renews" in str(exc)
        refused += 1
print(f"crash sessions: {accepted} accepted, {refused} refused "
      f"typed (PRE accepted 8/8)")
assert accepted == 1 and refused == 7
src = (REPO / "tools/t2_confirmatory_executor.py").read_text()
assert "O_RDWR | os.O_APPEND" in src        # ONE descriptor
assert "prev_sha" in src and "record_sha" in src
assert "charged += sum(" in src              # open reserves charge
print("=> hash-chained, descriptor-bound, charge-in-advance "
      "(source facts)")

print("\n== C58 DEAD: a bare release never frees a live lock ==")
n1, rrl = ex.acquire_lock(TMP / "lock", "live-holder")
for name in ("RELEASE_000001.json", "RELEASE_INTENT_000001.json",
             "RELEASE_DONE_000001.json"):
    fd = os.open(rrl.path / "locks" / name,
                 os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)                              # zero bytes each
_die(lambda: ex.acquire_lock(rrl, "thief"), "alive",
     "empty release filenames against a LIVE holder (the PRE)")
for name in ("RELEASE_INTENT_000001.json",
             "RELEASE_DONE_000001.json"):
    (rrl.path / "locks" / name).unlink()
ex.release_lock(rrl, 1, "live-holder")
n2, _ = ex.acquire_lock(rrl, "next")
print(f"  a REAL intent+completion release frees: session {n2}")
assert n2 == 2

print("\n== C59 DEAD: terminals require current authority ==")
authority = ex.physical_authority(design, "mechanical_rehearsal")
u59 = TMP / "units59"
u59.mkdir()
term = {"schema": "agent_multi.t2_unit_terminal.v2",
        "unit_id": "ghost", "attempt_id": "attempt_" + "a" * 16,
        "mode": "confirmatory", "terminal": "FAILED",
        "failure_class": "Fabricated", "reason": "no authority",
        "operator_disposition": False, "wall_seconds": 0.0}
term["terminal_sha256"] = ex._self_sha(term, "terminal_sha256")
tp = u59 / "TERMINAL_ghost.json"
tp.write_text(json.dumps(term))
os.chmod(tp, 0o600)
st, why = ex.adjudicate_unit_deep(u59, "ghost", design,
                                  authority,
                                  "mechanical_rehearsal")
print(f"  the PRE's fabricated terminal -> {st}: {why[:84]}")
assert st == "UNCERTAIN"

print("\n== C60 DEAD: disposition is external authority ==")
root60 = TMP / "root60"
rr60 = ex.ResultsRoot(root60, create=True)
fake = {"attempt_id": "attempt_" + "b" * 16,
        "mode": "mechanical_rehearsal"}
cp = rr60.path / "units" / "CLAIM_invented__series.json"
cp.write_text(json.dumps(fake))
os.chmod(cp, 0o600)
_die(lambda: ex.declare_attempt_failed(
        root60, "invented::series", mode="mechanical_rehearsal"),
     "exact v3 schema", "the PRE's two-field fabricated claim")
assert not (rr60.path / "units"
            / "TERMINAL_invented__series.json").exists()
import t2_assay_harness as hz  # noqa: E402
import t2_public_data_census as dc  # noqa: E402
co = hz.load_co()
unit = hz.load_task_unit(dc.build_census(), "sm_nile")
pins = ex._git_head_tree()
rr60b = ex.ResultsRoot(TMP / "real60", create=True)


def _trip(label):
    if "mlp_seed12" in label:
        raise ex.T2BudgetStop("stop", label)


try:
    ex.run_unit(hz, co, unit, design, authority, rr60b,
                "mechanical_rehearsal", pins, guard=_trip)
except SystemExit:
    pass
_die(lambda: ex.declare_attempt_failed(
        TMP / "real60", "sm_nile", mode="mechanical_rehearsal"),
     "T2_DISPOSITION_RECORD_REQUIRED",
     "a REAL uncertain attempt without the external record")

print("\n== C61 DEAD: the results root cannot be a symlink ==")
cache = Path.home() / ".cache"
real61 = cache / f"t2_post_c61_target_{os.getpid()}"
link61 = cache / f"t2_post_c61_link_{os.getpid()}"
real61.mkdir(mode=0o700, exist_ok=True)
if link61.is_symlink():
    link61.unlink()
link61.symlink_to(real61)
before = sorted(p.name for p in real61.iterdir())
_die(lambda: ex.ResultsRoot(link61, create=True), "O_NOFOLLOW",
     "the PRE's symlinked results root")
after = sorted(p.name for p in real61.iterdir())
print(f"  writes under the target: {len(after) - len(before)}")
assert before == after
link61.unlink()
shutil.rmtree(real61)

print("\n== C62 DEAD: one effective wall for supervised fits ==")
rr62 = ex.ResultsRoot(TMP / "w62", create=True)
w62 = ex.WallAuthority(rr62, {"max_wall_seconds": 0.6,
                              "max_rss_bytes": 8 << 30},
                       stop, "s62")
time.sleep(0.35)
sup = ex.make_fit_supervisor({"max_rss_bytes": 8 << 30}, w62)


class _Slow:
    def __call__(self):
        time.sleep(30)


t0 = time.monotonic()
_die(lambda: sup(_Slow(), "slow-fit"), "WALL_KILLED",
     "the PRE's wall-blind fit (remaining ~0.25s of 0.6s)")
dt = time.monotonic() - t0
print(f"  killed after {dt:.2f}s = min(remaining, per-fit); "
      f"PRE ran to completion")
assert dt < 1.5
time.sleep(0.3)


class _Ok:
    def __call__(self):
        return 1


_die(lambda: sup(_Ok(), "no-budget"), "T2_BUDGET_STOP",
     "dispatch with zero remaining budget")
import multiprocessing as _mp  # noqa: E402
print("  orphan workers:", len(_mp.active_children()))
assert not _mp.active_children()
w62.close()

print("\n== C63 DEAD: adjudication controls success ==")
rr63 = ex.ResultsRoot(TMP / "halt63", create=True)
real_savez = np.savez_compressed
np.savez_compressed = lambda *a, **k: (_ for _ in ()).throw(
    OSError("simulated persistence crash"))
try:
    _die(lambda: ex.run_unit(hz, co, unit, design, authority,
                             rr63, "mechanical_rehearsal", pins),
         "campaign halts",
         "the PRE's post-assay persistence crash")
finally:
    np.savez_compressed = real_savez
st, why = ex.adjudicate_unit_shallow(rr63.path / "units",
                                     "sm_nile")
print(f"  claim preserved: {st}: {why[:74]}")
assert st == "UNCERTAIN"
rr63b = ex.ResultsRoot(TMP / "final63", create=True)


def _defect(label):
    if "ridge_done" in label:
        raise ValueError("defect")


try:
    ex.run_unit(hz, co, unit, design, authority, rr63b,
                "mechanical_rehearsal", pins, guard=_defect)
except ex.T2AssayFailed:
    pass
counts = ex.final_adjudication(rr63b, ("sm_nile",), design,
                               authority, "mechanical_rehearsal",
                               lambda uid: None)
print("  assay failure with terminal ->", counts)
assert counts == {"COMPLETED_VERIFIED": 0, "TERMINAL_FAILED": 1}
(rr63b.path / "units" / "TERMINAL_sm_nile.json").unlink()
_die(lambda: ex.final_adjudication(
        rr63b, ("sm_nile",), design, authority,
        "mechanical_rehearsal", lambda uid: None),
     "typed uncertainty",
     "final adjudication over a missing terminal")

print("\n== v3 mechanical rehearsal at the final tip ==")
reh = TMP / "t2reh"
rc = ex.rehearse(reh)
assert rc == 0

shutil.rmtree(TMP)
print("\nPOST CONFIRMED: C57-C63 dead through the productive "
      "APIs; the executor charges walls in advance, frees locks "
      "only through verified two-phase releases, adjudicates "
      "terminals under current authority, defers disposition to "
      "the external record, walks the root with O_NOFOLLOW, "
      "bounds fits by the remaining global wall, and refuses "
      "success over any physical uncertainty; scoring remains "
      "closed by the ABSENT external execution record")
