"""PRE freeze for order C23-C25: the three residual findings
reproduce against agent-multi@7d2571c9 (tip d97c3f62 carries the
identical code)."""
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
SCRATCH = Path.home() / ".cache" / "b4_c23_pre"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)


def _load(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


import b4_authority as a  # noqa: E402
orch = _load("b4orch", "tools/b4_campaign_orchestrator.py")
ledger_mod = _load("b4led", "tools/b4_campaign_ledger.py")

print("== C23: uncertain release admits a second holder ==")
root = SCRATCH / "lock"
root.mkdir()
lk = orch.GlobalLock(root)
lk.__enter__()
real_fsync = os.fsync


def dir_fsync_fails(fd):
    st = os.fstat(fd)
    if stat.S_ISDIR(st.st_mode) and \
            not (root / "CAMPAIGN_LOCK").exists():
        # only the FINAL directory fsync, after the unlink
        raise OSError("injected final dir fsync failure")
    return real_fsync(fd)


try:
    with um.patch.object(os, "fsync", dir_fsync_fails):
        lk.__exit__()
    print("release saw success (unexpected)")
except OSError as exc:
    print("FIRST_RELEASE_SAW OSError", "LOCK_EXISTS",
          (root / "CAMPAIGN_LOCK").exists(), "WITNESSES",
          len(list(root.glob("LOCK_RELEASE_*.json"))))


def _second(q):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "b4orch_child",
        str(REPO / "tools/b4_campaign_orchestrator.py"))
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    try:
        om.GlobalLock(root).__enter__()
        q.put(("entered", True))
    except SystemExit as exc:
        q.put(("refused", str(exc)))


ctx = mp.get_context("fork")
q = ctx.Queue()
p2 = ctx.Process(target=_second, args=(q,))
p2.start()
p2.join(30)
kind, val = q.get(timeout=5)
print("SECOND_HOLDER_ENTERED", kind == "entered")
assert kind == "entered", "PRE expects the bypass to reproduce"

print("\n== C24: public-mode, path-raced control objects ==")
root2 = SCRATCH / "modes"
root2.mkdir()
lk2 = orch.GlobalLock(root2)
lk2.__enter__()
claim = orch.claim_attempt(root2, "o2024_seed101")
(root2 / "B4_MATERIALIZATION.json").write_text("{}")
lease_p = orch.issue_lease(root2, "o2024_seed101", claim,
                           "a" * 64, root2)
for path in (root2 / "CAMPAIGN_LOCK",
             root2 / "o2024_seed101" /
             f"CLAIM_{a.CAMPAIGN_GENERATION}.json",
             lease_p):
    mode = stat.S_IMODE(os.stat(path).st_mode)
    print(f"{path.name} {oct(mode)}")
    assert mode == 0o644, "PRE expects public 0644"
src = (REPO / "tools/b4_campaign_orchestrator.py").read_text()
lc = src[src.index("def load_claim"):src.index("def verify_lease")]
print("load_claim checks by path then reopens by path:",
      "is_symlink()" in lc and "read_bytes()" in lc)
assert "is_symlink()" in lc and "read_bytes()" in lc
# deterministic demonstration of the race window: swap the file
# AFTER the symlink check and BEFORE the pathname read
claim_path = (root2 / "o2024_seed101" /
              f"CLAIM_{a.CAMPAIGN_GENERATION}.json")
honest = claim_path.read_bytes()
forged = json.loads(honest)
forged["attempt_id"] = "attempt_SWAPPED"
real_is_symlink = Path.is_symlink


def swap_after_check(self):
    out = real_is_symlink(self)
    if self.name.startswith("CLAIM_"):
        self.write_text(json.dumps(forged))
    return out


with um.patch.object(Path, "is_symlink", swap_after_check):
    rec = orch.load_claim(root2, "o2024_seed101")
print("CONSUMED_SWAPPED_BYTES", rec["attempt_id"])
assert rec["attempt_id"] == "attempt_SWAPPED"
claim_path.write_bytes(honest)

print("\n== C25: a missing checkpoint is accepted ==")
sys.path.insert(0, str(REPO / "tests"))
import importlib.util as ilu
tspec = ilu.spec_from_file_location(
    "t_b4", REPO / "tests/test_b4_materializer_authority.py")
tmod = ilu.module_from_spec(tspec)
tspec.loader.exec_module(tmod)


class _MP:
    def setattr(self, obj, name, val):
        setattr(obj, name, val)


(SCRATCH / "fx").mkdir(parents=True, exist_ok=True)
led, results = tmod._ledger_fixture(SCRATCH / "fx")
removed = results / "o2023_seed202" / "checkpoint_o2023_seed202.zip"
removed.unlink()
mpatch = _MP()
tmod.ledger_mod.verify_ledger = lambda lp, mr: led
tmod.ledger_mod.b4a.resolve_source_ref = (
    lambda ref: SCRATCH / "fx" / f"source_{ref.split(':')[1]}.csv")
tmod.ledger_mod._derive_comparator_dir = (
    lambda mr: SCRATCH / "fx" / "comp_default")
facts = tmod.ledger_mod.verify_campaign_results(
    SCRATCH / "fx" / "ledger.json", SCRATCH / "fx", results)
print("ACCEPTED_ABSENT_CHECKPOINT", facts["n"],
      removed.exists())
assert facts["n"] == 12 and not removed.exists()

shutil.rmtree(SCRATCH)
print("\nPRE CONFIRMED: C23-C25 findings all reproduce")
