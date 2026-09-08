"""PRE freeze for order T2 C57-C65: Musashi's seven runtime
counterexamples reproduce from 4e5f19dc through the productive
APIs.

C57 restart renews the wall (8 sub-cadence sessions accepted
     against a smaller sealed budget; lossy 5 s cadence; ledger
     read via exists()+read_text(), not one descriptor).
C58 a bare/empty RELEASE_<n>.json filename frees a LIVE lock.
C59 a self-consistent terminal bound to NO authority adjudicates
     TERMINAL_FAILED shallowly and is never deeply re-verified.
C60 --declare-attempt-failed converts a two-field fabricated
     claim into an accepted TERMINAL_FAILED, before any gate.
C61 a symlinked results root is followed; the session lock lands
     under the target.
C62 the fit supervisor ignores the remaining global wall (a
     0.12 s fit returns OK against a 0.01 s wall) and reads only
     the RSS limit.
C63 a post-assay persistence failure escapes run_unit() without
     a terminal; main()'s BaseException arm counts it as
     failed_preserved and exits zero while physical adjudication
     says UNCERTAIN.

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

import t2_confirmatory_executor as ex  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="t2_c57_pre_"))

print("== C57: repeated sub-cadence crashes renew the wall ==")
led = TMP / "wall.jsonl"
stop = TMP / "T2_STOP"
budget = {"max_wall_seconds": 0.08, "max_rss_bytes": 8 << 30}
accepted = 0
for k in range(8):
    g = ex.BudgetGuard(budget, stop, led, f"crash{k}")
    time.sleep(0.03)
    try:
        g.check(f"probe{k}")          # inside the 5 s cadence:
        accepted += 1                  # nothing persisted yet
    finally:
        os.close(g._fd)                # simulate crash: no close()
print(f"sessions accepted: {accepted}/8 against a sealed "
      f"{budget['max_wall_seconds']}s wall; real elapsed ~"
      f"{8 * 0.03:.2f}s")
assert accepted == 8
src = (REPO / "tools/t2_confirmatory_executor.py").read_text()
assert "WALL_PERSIST_CADENCE_S = 5.0" in src
assert "if not self.wall_ledger.exists():" in src   # check-then-
assert "self.wall_ledger.read_text()" in src        # reopen path
assert "continue        # a torn trailing line loses" in src
print("=> lossy cadence + exists()/read_text() ledger confirmed "
      "in source")

print("\n== C58: an EMPTY release filename frees a LIVE lock ==")
lockroot = TMP / "lockroot"
lockroot.mkdir(mode=0o700)
n1 = ex.acquire_lock(lockroot, "live-holder")     # pid = ALIVE us
rel = lockroot / "locks" / "RELEASE_000001.json"
fd = os.open(rel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
os.close(fd)                                       # zero bytes
n2 = ex.acquire_lock(lockroot, "thief")
print(f"holder pid {os.getpid()} ALIVE; empty RELEASE_000001 -> "
      f"second acquire got session {n2}")
assert n2 == 2

print("\n== C59: an authority-free terminal adjudicates "
      "TERMINAL_FAILED ==")
u = TMP / "units59"
u.mkdir(mode=0o700)
term = {"schema": "agent_multi.t2_unit_terminal.v2",
        "unit_id": "ghost_unit", "attempt_id": "attempt_" + "a" * 16,
        "mode": "confirmatory", "terminal": "FAILED",
        "failure_class": "Fabricated", "reason": "no authority",
        "operator_disposition": False, "wall_seconds": 0.0}
term["terminal_sha256"] = ex._self_sha(term, "terminal_sha256")
tp = u / "TERMINAL_ghost_unit.json"
tp.write_text(json.dumps(term))
os.chmod(tp, 0o600)
st, why = ex.adjudicate_unit_shallow(u, "ghost_unit")
print("adjudication of a terminal bound to NO design/execution "
      "authority:", st)
assert st == "TERMINAL_FAILED"
assert "verify_unit_terminal" not in src        # no deep verifier
tschema = ('"schema": "agent_multi.t2_unit_terminal.v2",\n'
           '                "unit_id": uid, "attempt_id": attempt,')
assert "sealed_design_file_sha256" not in src.split(
    "def run_unit")[1].split("def physical_authority")[0].split(
    "term = {")[1].split("}")[0]
print("=> terminal schema carries no sealed/execution/code/"
      "binding/claim authority (source)")

print("\n== C60: disposition legitimizes a fabricated claim ==")
root60 = TMP / "root60"
(root60 / "units").mkdir(mode=0o700, parents=True)
fake_claim = {"attempt_id": "attempt_" + "b" * 16,
              "mode": "confirmatory"}
cp = root60 / "units" / "CLAIM_invented__series.json"
cp.write_text(json.dumps(fake_claim))
os.chmod(cp, 0o600)
ex.declare_attempt_failed(root60, "invented::series",
                          "pre evidence: fabricated claim")
st, _ = ex.adjudicate_unit_shallow(root60 / "units",
                                   "invented::series")
print("two-field fabricated claim ->", st,
      "(no schema, no self-digest, no membership, no gate)")
assert st == "TERMINAL_FAILED"

print("\n== C61: a symlinked results root is followed ==")
real61 = TMP / "real_target"
real61.mkdir(mode=0o700)
link61 = TMP / "results_link"
link61.symlink_to(real61)
ex.acquire_lock(link61, "sym-session")
written = (real61 / "locks" / "SESSION_000001.json").exists()
print("lock written UNDER the symlink target:", written)
assert written

print("\n== C62: supervised fit ignores the remaining global "
      "wall ==")
sup = ex.make_fit_supervisor({"max_rss_bytes": 8 << 30})


class _Slow:
    def __call__(self):
        time.sleep(0.12)
        return "finished"


t0 = time.monotonic()
out = sup(_Slow(), "slow-fit")
dt = time.monotonic() - t0
print(f"global wall remaining: 0.01s (never consulted); 0.12s "
      f"fit returned {out!r} after {dt:.2f}s")
assert out == "finished"
sup_src = src[src.index("def make_fit_supervisor"):
              src.index("def _locks_dir")]
assert "max_wall" not in sup_src and "guard" not in sup_src
assert 'rss_cap = int(limits["max_rss_bytes"])' in sup_src
print("=> the supervisor reads ONLY the RSS limit; poll timeout "
      "is the fixed per-fit constant (source)")

print("\n== C63: post-assay persistence failure -> rc 0 with a "
      "new UNCERTAIN ==")
import t2_confirmatory as conf  # noqa: E402
import t2_assay_harness as hz  # noqa: E402
import t2_public_data_census as dc  # noqa: E402
STATE = Path.home() / ".local/share/agent-multi"
design = conf.strict_json_load(
    STATE / "t2_screen_design_SEALED_V6.json", "sealed design")
co = hz.load_co()
unit = hz.load_task_unit(dc.build_census(), "sm_nile")
authority = ex.physical_authority(design, "mechanical_rehearsal")
root63 = TMP / "root63"
root63.mkdir(mode=0o700)
orig = ex._excl_write_npz
ex._excl_write_npz = lambda *a, **k: (_ for _ in ()).throw(
    OSError("simulated persistence crash"))
failed_preserved = 0
try:
    try:
        ex.run_unit(hz, co, unit, design, authority, root63,
                    "mechanical_rehearsal")
    except (ex.T2BudgetStop, ex.ExecutorRefusal, KeyboardInterrupt):
        raise
    except BaseException:                 # main()'s exact arm
        failed_preserved += 1             # counted, run continues
    main_rc = 0                           # loop ends, lock freed
finally:
    ex._excl_write_npz = orig
st, why = ex.adjudicate_unit_shallow(root63 / "units", "sm_nile")
print(f"main_rc={main_rc}, failed_preserved={failed_preserved}; "
      f"physical adjudication AFTER exit: {st}: {why[:66]}")
assert main_rc == 0 and failed_preserved == 1
assert st == "UNCERTAIN" and "claim without" in why
assert not (root63 / "units" / "TERMINAL_sm_nile.json").exists()
main_seg = src[src.index("def main"):src.index("def rehearse")]
assert "except BaseException:" in main_seg
assert "failed += 1" in main_seg
assert "adjudicate" not in main_seg.split("finally:")[0].split(
    "release_lock")[-1]
print("=> no final deep adjudication before release/exit "
      "(source)")

shutil.rmtree(TMP)
print("\nPRE CONFIRMED at 4e5f19dc: C57 crash-renewal, C58 bare "
      "release, C59 authority-free terminal, C60 fabricated-claim "
      "disposition, C61 symlink root, C62 wall-blind supervisor, "
      "C63 zero-exit with fresh uncertainty — all frozen")
