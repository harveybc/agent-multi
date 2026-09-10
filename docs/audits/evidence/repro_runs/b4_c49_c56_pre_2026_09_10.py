"""PRE freeze for order B4 C49-C56 at 282c5771: the quarantine
facts reproduce read-only from the preserved v7 root, and the
callback-starvation failure mode reproduces deterministically
across a REAL process boundary.

C49: the audit input is consumed by exact bytes; the third cell
adjudicates AMBIGUOUS_CLAIM through the PRODUCTIVE adjudicator
(claim + lease, no terminal, no seal); both completed cells'
terminal/seal digest pairs equal the audit's; the stale facts
(epoch 111, 2,231,000 steps, last-write times, 43,200 s cell
ceiling, operator-stop time) are frozen; CAMPAIGN_STOP and the
cell STOP are present; the run writes NOTHING to the real root
(byte-level inventory before/after).

C50: a controlled child process enters the same parent-to-cell
execution seam shape as B4 (in-process guards that live ONLY
inside the pipeline's executing callback), makes observable
progress, then stops invoking the callback while ignoring the
stop file and exceeding a short test wall — and remains ALIVE
until the external supervisor (this test) terminates it. Source
facts: the executor carries no parent-side monotonic deadline;
every wall/stop guard rides the F9 callback.

CPU only. Zero writes to the v7 root. No service, no GPU."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

AUDIT = (REPO / "docs/audits/"
         "MUSASHI_B4_V7_RUNTIME_STALL_AND_QUARANTINE_2026_09_09"
         ".md")
ROOT = (Path.home() / ".local/share/agent-multi/"
        "b4_campaign_results_v7_20260907")

facts = {}

# ---- C49.1: audit input by exact bytes ----
audit_bytes = AUDIT.read_bytes()
facts["audit_sha256"] = hashlib.sha256(audit_bytes).hexdigest()
atext = audit_bytes.decode()

# ---- root inventory BEFORE (proves zero writes at the end) ----
def _inventory(root):
    inv = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            inv[str(p.relative_to(root))] = (st.st_size,
                                             st.st_mtime_ns)
    return inv


inv_before = _inventory(ROOT)

# ---- C49.2: productive adjudication of all 12 cells ----
import importlib.util as ilu
spec = ilu.spec_from_file_location(
    "b4orch", REPO / "tools/b4_campaign_orchestrator.py")
orch = ilu.module_from_spec(spec)
spec.loader.exec_module(orch)
led = json.loads((ROOT / "CAMPAIGN_LEDGER.json").read_text())
cells = sorted(c for c in led["cells"]) if "cells" in led else \
    sorted(p.name for p in ROOT.iterdir() if p.is_dir()
           and p.name.startswith("o20"))
states = {}
import contextlib
for cid in ["o2022_seed101", "o2022_seed202", "o2022_seed303"]:
    states[cid] = orch.adjudicate_cell_state(ROOT, cid)
pending = 0
for p in (ROOT,):
    pass
all_cells = [f"o{y}_seed{s}" for y in (2022, 2023, 2024)
             for s in (101, 202, 303, 404)]
for cid in all_cells:
    st = orch.adjudicate_cell_state(ROOT, cid)
    states[cid] = st
    if st == "PENDING":
        pending += 1
facts["adjudication"] = {
    "o2022_seed101": states["o2022_seed101"],
    "o2022_seed202": states["o2022_seed202"],
    "o2022_seed303": states["o2022_seed303"],
    "pending_count": pending}
assert states["o2022_seed101"] == "COMPLETED_VERIFIED"
assert states["o2022_seed202"] == "COMPLETED_VERIFIED"
assert states["o2022_seed303"] == "AMBIGUOUS_CLAIM"
assert pending == 9
c3 = ROOT / "o2022_seed303"
facts["third_cell_claim_lease_no_terminal_no_seal"] = (
    (c3 / "CLAIM_b4_campaign_generation_v7_20260907.json"
     ).is_file()
    and (c3 / "LEASE_attempt_b459e72dc47a4d72.json").is_file()
    and not (c3 / "B4_CELL_TERMINAL.json").exists()
    and not any(c3.glob("SEAL_COMPLETE_*")))
assert facts["third_cell_claim_lease_no_terminal_no_seal"]

# ---- C49.3: completed digest pairs equal the audit ----
pairs = {
    "o2022_seed101/B4_CELL_TERMINAL.json":
        "3a836df218702adfa8eb8d436b81064e3f517e59f16ab9a750"
        "b58304a8c921d5",
    "o2022_seed101/SEAL_COMPLETE_attempt_f16bff0d0a624d63.json":
        "aea56c68cfbe12d1ed10e5c9dbb6751ec6ffddb4d6c1d677aae"
        "69c9b65b6624e",
    "o2022_seed202/B4_CELL_TERMINAL.json":
        "cd3e61e7e33772663e808745a1135a1d4cebf6ddd7c4b67f54c"
        "134f0c3c5fd7c",
    "o2022_seed202/SEAL_COMPLETE_attempt_1015641321af481b.json":
        "ec763fd82b9958cf1e4789e2c9d178b8fc9f174c669e25172027"
        "f742c7118526",
}
for rel, want in pairs.items():
    got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    assert got == want, (rel, got)
    assert want in atext.replace("\n  ", "")\
        .replace("\n", "") or want[:32] in atext
facts["completed_digest_pairs_match_audit"] = True

# ---- C49.4: stale-status facts ----
status = json.loads((c3 / "cell_runtime/status.json"
                     ).read_text())
st_stat = (c3 / "cell_runtime/status.json").stat()
import datetime as dt
facts["stale"] = {
    "epoch": status.get("epoch_completed"),
    "status_mtime": dt.datetime.fromtimestamp(
        st_stat.st_mtime).astimezone().isoformat(),
    "claim_time_in_audit": "2026-09-08T03:34:53-05:00" in atext,
    "progress_steps_2231000_in_audit": "2,231,000" in atext,
    "cell_ceiling_43200_in_audit": "43,200" in atext,
    "operator_stop_in_audit":
        "2026-09-09T14:19:28-05:00" in atext,
}
assert facts["stale"]["epoch"] == 111
assert "2026-09-08T08:56:11" in facts["stale"]["status_mtime"]
assert all(v is True for k, v in facts["stale"].items()
           if k.endswith("_in_audit"))

# ---- C49.5: stop signals preserved ----
facts["campaign_stop_present"] = (ROOT / "CAMPAIGN_STOP"
                                  ).is_file()
facts["cell_stop_present"] = (c3 / "STOP").is_file()
assert facts["campaign_stop_present"]
assert facts["cell_stop_present"]

# ---- C50: callback starvation across a REAL process boundary --
esrc = (REPO / "tools/b4_campaign_executor.py").read_text()
osrc = (REPO / "tools/b4_campaign_orchestrator.py").read_text()
facts["c50_source"] = {
    "guards_ride_f9_callback":
        "resource guards ride the pipeline's own F9 executing "
        "callback" in esrc,
    "executor_has_no_monotonic_parent_deadline":
        "time.monotonic" not in esrc
        and "deadline" not in esrc,
    "orchestrator_runs_cell_in_process":
        "executor.execute_cell(" in osrc
        and "Popen" not in osrc,
}
assert all(facts["c50_source"].values())

# the controlled child: same seam SHAPE — budget/stop guards are
# closures that fire ONLY when the executing callback is invoked;
# after observable progress the body stops invoking them.
import tempfile
TMP = Path(tempfile.mkdtemp(prefix="b4_c50_pre_"))
stop_file = TMP / "STOP"
progress = TMP / "progress.json"
child_code = f'''
import json, os, time
stop_file = {str(stop_file)!r}
progress = {str(progress)!r}
budget_max_wall_seconds = 2.0
t0 = time.time()
def f9_callback():
    # the production guard shape: consulted ONLY when invoked
    if os.path.exists(stop_file):
        raise SystemExit("EXTERNALLY_STOPPED")
    if time.time() - t0 > budget_max_wall_seconds:
        raise SystemExit("TIMED_OUT")
# observable progress WITH the callback (healthy phase)
for i in range(3):
    f9_callback()
    with open(progress, "w") as fh:
        json.dump({{"step": i}}, fh)
# model.learn-shaped stall: alive, CPU-consuming, NEVER invokes
# the callback again — exactly the audited state
x = 1.0
while True:
    x = x * 1.0000000001 + 1e-12
'''
t_start = time.monotonic()
child = subprocess.Popen([sys.executable, "-c", child_code],
                         start_new_session=True)
for _ in range(100):
    if progress.exists():
        break
    time.sleep(0.05)
assert progress.exists()
stop_file.write_text("stop")            # external stop request
time.sleep(3.0)                          # wall (2s) long expired
alive_after_wall_and_stop = child.poll() is None
facts["c50_child"] = {
    "observable_progress": True,
    "stop_written_and_wall_expired": True,
    "child_STILL_ALIVE": alive_after_wall_and_stop,
}
assert alive_after_wall_and_stop, \
    "PRE expectation: in-process guards cannot bound this state"
# only the external supervisor ends it
os.killpg(os.getpgid(child.pid), signal.SIGKILL)
child.wait(timeout=10)
facts["c50_child"]["terminated_only_by_external_supervisor"] = \
    True
facts["c50_child"]["lifetime_s"] = round(
    time.monotonic() - t_start, 2)
import shutil
shutil.rmtree(TMP)

# ---- zero writes to the real root ----
inv_after = _inventory(ROOT)
facts["v7_root_untouched"] = inv_after == inv_before
assert facts["v7_root_untouched"]

print(json.dumps(facts, indent=1))
print("\nPRE CONFIRMED at 282c5771: the third cell adjudicates "
      "AMBIGUOUS_CLAIM with claim+lease and no terminal/seal, "
      "both completed digest pairs equal the audit bytes, the "
      "stale facts and stop signals are frozen, the v7 root "
      "received zero writes — and a child that stops invoking "
      "the F9 callback after observable progress survives an "
      "expired wall AND a written stop file until an EXTERNAL "
      "supervisor kills it; the executor has no parent-side "
      "monotonic deadline and the orchestrator runs cells "
      "in-process")
