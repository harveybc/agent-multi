"""PRE freeze for order C17-C22: the six residual findings
reproduced against tip 5a11f858 code."""
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import unittest.mock as um
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "b4_c17_pre"
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
executor = _load("b4exec", "tools/b4_campaign_executor.py")
ledger_mod = _load("b4led", "tools/b4_campaign_ledger.py")
MAT = Path.home() / ".local/share/agent-multi/b4_materialization_v5_20260906"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


print("== C17: a foreign execution lease is accepted ==")
root = SCRATCH / "lease"
(root / "CAMPAIGN_LOCK").parent.mkdir(parents=True)
(root / "CAMPAIGN_LOCK").write_text(json.dumps(
    {"pid": 12345, "generation": a.CAMPAIGN_GENERATION}))
claim = orch.claim_attempt(root, "o2024_seed101")
lease_p = orch.issue_lease(root, "o2024_seed101", claim,
                           "a" * 64, MAT)
doc = json.loads(lease_p.read_text())
doc["schema"] = "attacker.anything.v9"
doc["authorization_sha256"] = "b" * 64
doc["holder_pid"] = 999999
lease_p.chmod(0o644)
lease_p.write_text(json.dumps(doc))
try:
    lease = orch.verify_lease(lease_p, root, "o2024_seed101", MAT)
    print("ACCEPTED_FOREIGN_LEASE",
          lease["authorization_sha256"][:4],
          lease["holder_pid"], lease["schema"])
    accepted = True
except SystemExit as exc:
    print("refused:", exc)
    accepted = False
assert accepted

print("\n== C18: failed sealing recognized as successful ==")
root2 = SCRATCH / "seal"
c2 = orch.claim_attempt(root2, "o2022_seed101")
executor.write_terminal(root2, "o2022_seed101", "COMPLETED", {
    "attempt_id": c2["attempt_id"], "cell_config_sha256": "c" * 64,
    "per_bar_csv": "x.csv", "per_bar_sha256": "d" * 64,
    "scored_index_sha256": "e" * 64, "checkpoint_sha256": "f" * 64,
    "sealed_2025_used": False})
real_open = os.open
calls = {"n": 0}


def failing_open(path, flags, *args):
    fd = real_open(path, flags, *args)
    return fd


real_fsync = os.fsync


def failing_fsync(fd):
    st = os.fstat(fd)
    import stat as _st
    if _st.S_ISDIR(st.st_mode):
        calls["n"] += 1
        if calls["n"] >= 1:
            raise OSError("injected dir fsync failure")
    return real_fsync(fd)


try:
    with um.patch.object(os, "fsync", failing_fsync):
        orch.seal_attempt(root2, "o2022_seed101", c2["attempt_id"])
    print("caller saw success (unexpected)")
except OSError as exc:
    print("CALLER_SAW OSError", exc)
state = orch.adjudicate_cell_state(root2, "o2022_seed101")
print("FRESH_STATE", state)
assert state == "COMPLETED_VERIFIED"
print("-> the caller saw failure; a fresh process accepts the "
      "seal — partial recognition")

print("\n== C19: unacknowledged unlink release ==")
root3 = SCRATCH / "lock"
root3.mkdir()
lk = orch.GlobalLock(root3)
lk.__enter__()
# ANY process can unlink another holder's lock: no ownership check,
# no durable release witness
(root3 / "CAMPAIGN_LOCK").unlink()
lk2 = orch.GlobalLock(root3)
lk2.__enter__()          # second holder while the first believes
print("second holder acquired while first still believes it holds:",
      (root3 / "CAMPAIGN_LOCK").exists())
lk2.__exit__()
src = (REPO / "tools/b4_campaign_orchestrator.py").read_text()
print("release fsyncs directory or writes witness:",
      "RELEASE" in src)
assert "RELEASE" not in src

print("\n== C20: completion without the strong verifier ==")
body = src[src.index("def run_campaign"):]
print("run_campaign calls verify_campaign_results:",
      "verify_campaign_results" in body)
assert "verify_campaign_results" not in body
lsrc = (REPO / "tools/b4_campaign_ledger.py").read_text()
print("CLI verify passes comparator_dir:",
      "comparator_dir" in lsrc[lsrc.index("def main"):])
assert "comparator_dir" not in lsrc[lsrc.index("def main"):]

print("\n== C21: present-but-not-factual fields ==")
vc = lsrc[lsrc.index("def verify_campaign_results"):
          lsrc.index("def verify_single_cell_result")]
for fact, tok in (
        ("seed column vs cell seed", 'df["seed"]'),
        ("scored_index sequence", 'df["scored_index"]'),
        ("source_row recompute", "source_row_sha256 ="),
        ("checkpoint bytes verify", "checkpoint_sha256 =="),
        ("net_return recompute", "net_return recompute")):
    print(f"verifier checks {fact}:", tok in vc)
assert 'df["seed"]' not in vc and 'df["scored_index"]' not in vc

print("\n== C22: caller-provided global remainder enlarges ==")
esrc = (REPO / "tools/b4_campaign_executor.py").read_text()
eb = esrc[esrc.index("def execute_cell"):]
print("executor derives remainder ONLY when caller passes None:",
      "if global_wall_remaining_seconds is None" in eb)
assert "if global_wall_remaining_seconds is None" in eb
print("-> execute_cell(..., global_wall_remaining_seconds=1e12) "
      "is accepted verbatim; the hard limit is delegated to the "
      "caller")

shutil.rmtree(SCRATCH)
print("\nPRE CONFIRMED: C17-C22 findings all reproduce")
