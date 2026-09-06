"""POST for order C23-C25: the three PRE findings no longer
reproduce. Same probes, inverted expectations."""
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import stat
import sys
import unittest.mock as um
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "b4_c23_post"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
os.chmod(SCRATCH, 0o700)


def _load(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


import b4_authority as a  # noqa: E402
orch = _load("b4orch", "tools/b4_campaign_orchestrator.py")

print("== C23: uncertain release admits NO second holder ==")
root = SCRATCH / "lock"
root.mkdir(mode=0o700)
lk = orch.GlobalLock(root)
lk.__enter__()
real = orch._excl_write


def completion_lost(path, payload, mode=0o600):
    if "LOCK_RELEASE_COMPLETE" in Path(path).name:
        raise OSError("injected completion failure")
    return real(path, payload, mode)


try:
    with um.patch.object(orch, "_excl_write", completion_lost):
        lk.__exit__()
    raise AssertionError("release saw success — POST FAILS")
except SystemExit as exc:
    print("FIRST_RELEASE_SAW", str(exc)[:60])
st = orch.current_lock_epoch(root)
print("EPOCH_STATE", st["state"], "LOCK_STILL_PRESENT",
      (root / "LOCK_EPOCH_1.json").exists())
assert st["state"] == "RELEASING"


def _second(q):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "b4orch_child",
        str(REPO / "tools/b4_campaign_orchestrator.py"))
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    try:
        om.GlobalLock(root).__enter__()
        q.put(("entered", None))
    except SystemExit as exc:
        q.put(("refused", str(exc)))


ctx = mp.get_context("fork")
q = ctx.Queue()
p2 = ctx.Process(target=_second, args=(q,))
p2.start()
p2.join(30)
kind, msg = q.get(timeout=5)
print("SECOND_HOLDER_ENTERED", kind == "entered", "|",
      (msg or "")[:70])
assert kind == "refused" and "UNCERTAIN" in msg
print("-> nothing was ever unlinked; the epoch stays for the "
      "operator")

print("\n== C24: private descriptor-bound control plane ==")
root2 = SCRATCH / "modes"
root2.mkdir(mode=0o700)
lk2 = orch.GlobalLock(root2)
lk2.__enter__()
claim = orch.claim_attempt(root2, "o2024_seed101")
(root2 / "B4_MATERIALIZATION.json").write_text("{}")
lease_p = orch.issue_lease(root2, "o2024_seed101", claim,
                           "a" * 64, root2)
for path in (root2 / "LOCK_EPOCH_1.json",
             root2 / "o2024_seed101" /
             f"CLAIM_{a.CAMPAIGN_GENERATION}.json",
             lease_p):
    mode = stat.S_IMODE(os.stat(path).st_mode)
    print(f"{path.name} {oct(mode)}")
    assert mode == 0o600
print("cell dir",
      oct(stat.S_IMODE(os.stat(root2 / 'o2024_seed101').st_mode)))
src = (REPO / "tools/b4_campaign_orchestrator.py").read_text()
lc = src[src.index("def load_claim"):src.index("def verify_lease")]
print("load_claim consumes ONE descriptor (_secure_json):",
      "_secure_json" in lc and "read_bytes()" not in lc)
assert "_secure_json" in lc and "read_bytes()" not in lc
# the PRE swap window is gone: is_symlink hook never fires a
# pathname reopen — a swapped PUBLIC file dies on mode, a swapped
# private forged file dies on the self-integral digest
cp = (root2 / "o2024_seed101" /
      f"CLAIM_{a.CAMPAIGN_GENERATION}.json")
forged = json.loads(cp.read_text())
forged["attempt_id"] = "attempt_SWAPPED"
cp.unlink()
fd = os.open(str(cp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
os.write(fd, json.dumps(forged).encode())
os.close(fd)
try:
    orch.load_claim(root2, "o2024_seed101")
    raise AssertionError("swapped bytes consumed — POST FAILS")
except SystemExit as exc:
    print("swapped claim refused:", str(exc)[:70])

print("\n== C25: a missing checkpoint refuses ==")
sys.path.insert(0, str(REPO / "tests"))
import importlib.util as ilu
tspec = ilu.spec_from_file_location(
    "t_b4", REPO / "tests/test_b4_materializer_authority.py")
tmod = ilu.module_from_spec(tspec)
tspec.loader.exec_module(tmod)
(SCRATCH / "fx").mkdir(parents=True)
led, results = tmod._ledger_fixture(SCRATCH / "fx")
removed = results / "o2023_seed202" / "checkpoint_o2023_seed202.zip"
removed.unlink()
tmod.ledger_mod.verify_ledger = lambda lp, mr: led
tmod.ledger_mod.b4a.resolve_source_ref = (
    lambda ref: SCRATCH / "fx" / f"source_{ref.split(':')[1]}.csv")
tmod.ledger_mod._derive_comparator_dir = (
    lambda mr: SCRATCH / "fx" / "comp_default")
try:
    tmod.ledger_mod.verify_campaign_results(
        SCRATCH / "fx" / "ledger.json", SCRATCH / "fx", results)
    raise AssertionError("absent checkpoint accepted — POST FAILS")
except SystemExit as exc:
    print("refused:", str(exc)[:90])

shutil.rmtree(SCRATCH)
print("\nPOST CONFIRMED: C23-C25 findings no longer reproduce")
