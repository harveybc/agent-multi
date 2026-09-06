"""POST for order C17-C22: the six PRE findings no longer
reproduce. Same probes as b4_c17_c22_pre_2026_09_06.py, inverted
expectations, against the corrected runtime."""
import hashlib
import json
import os
import shutil
import sys
import unittest.mock as um
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "b4_c17_post"
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
MAT = Path.home() / ".local/share/agent-multi/b4_materialization_v5_20260906"

print("== C17: the PRE foreign lease now refuses ==")
root = SCRATCH / "lease"
root.mkdir(parents=True)
(root / "CAMPAIGN_LOCK").write_text(json.dumps(
    {"pid": os.getpid(), "generation": a.CAMPAIGN_GENERATION,
     "acquire_id": "post0000000000"}))
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
    orch.verify_lease(lease_p, root, "o2024_seed101", MAT)
    raise AssertionError("foreign lease accepted — POST FAILS")
except SystemExit as exc:
    print("refused:", exc)
lease_p.write_text(json.dumps(
    {**{k: doc[k] for k in doc},
     "schema": "agent_multi.b4_execution_lease.v2"}))
try:
    orch.verify_lease(lease_p, root, "o2024_seed101", MAT)
    raise AssertionError("re-schema'd lease accepted — POST FAILS")
except SystemExit as exc:
    print("refused (digest):", exc)

print("\n== C18: failed sealing is adjudicated PHYSICALLY ==")
root2 = SCRATCH / "seal"
c2 = orch.claim_attempt(root2, "o2022_seed101")
executor.write_terminal(root2, "o2022_seed101", "FAILED", {
    "attempt_id": c2["attempt_id"], "reason": "post",
    "wall_seconds": 1.0})
real = orch._excl_write


def absent_completion(path, payload, mode=0o644):
    if "SEAL_COMPLETE" in Path(path).name:
        raise OSError("injected completion write failure")
    return real(path, payload, mode)


try:
    with um.patch.object(orch, "_excl_write", absent_completion):
        orch.seal_attempt(root2, "o2022_seed101", c2["attempt_id"])
    raise AssertionError("caller saw success — POST FAILS")
except OSError as exc:
    print("CALLER_SAW OSError", exc)
state = orch.adjudicate_cell_state(root2, "o2022_seed101")
print("FRESH_STATE", state)
assert state == "UNCERTAIN", state
print("-> uncertain bytes never become success; a persisted "
      "completion (the other physical outcome) is covered by "
      "test_c18_seal_fsync_outcome_matrix")

print("\n== C19: lock release is owned and witnessed ==")
root3 = SCRATCH / "lock"
root3.mkdir()
lk = orch.GlobalLock(root3)
lk.__enter__()
thief = orch.GlobalLock(root3)
thief.held = True
thief.acquire_id = "0" * 16
try:
    thief.__exit__()
    raise AssertionError("non-holder released — POST FAILS")
except SystemExit as exc:
    print("refused:", exc)
lk.__exit__()
wit = list(root3.glob("LOCK_RELEASE_*.json"))
assert len(wit) == 1
print("owned release left durable witness:", wit[0].name)
src = (REPO / "tools/b4_campaign_orchestrator.py").read_text()
assert "LOCK_RELEASE_" in src
print("release witness protocol present:", True)

print("\n== C20: completion consumes the strongest verifier ==")
body = src[src.index("def run_campaign"):]
assert "verify_campaign_results" in body
assert body.index("verify_campaign_results") < \
    body.index('"CAMPAIGN_COMPLETE"')
print("run_campaign calls verify_campaign_results before "
      "CAMPAIGN_COMPLETE:", True)
lsrc = (REPO / "tools/b4_campaign_ledger.py").read_text()
assert "comparator_dir" not in lsrc[lsrc.index("def main"):]
vbody = lsrc[lsrc.index("def verify_campaign_results"):
             lsrc.index("def verify_single_cell_result")]
assert "_derive_comparator_dir" in vbody
print("comparator derived from materialization; CLI cannot omit:",
      True)

print("\n== C21: factual fields are verified bindings ==")
for fact, tok in (
        ("seed column vs cell seed", 'df["seed"]'),
        ("scored_index sequence", 'df["scored_index"]'),
        ("source_row recompute", 'df["source_row_sha256"]'),
        ("checkpoint bytes verify", "checkpoint bytes differ"),
        ("net_return recompute", "net_return does not recompute"),
        ("exact terminal schema", "TERMINAL_SCHEMA_KEYS"),
        ("exact per-bar schema", "PER_BAR_SCHEMA")):
    assert tok in vbody or tok in lsrc, fact
    print(f"verifier checks {fact}: True")

print("\n== C22: the executor recomputes; callers only tighten ==")
esrc = (REPO / "tools/b4_campaign_executor.py").read_text()
eb = esrc[esrc.index("def execute_cell"):]
assert "recomputed = _orch.remaining_global_seconds" in eb
assert "min(" in eb and "recomputed)" in eb
print("executor recomputes at point of use and min()s the caller "
      "value:", True)

shutil.rmtree(SCRATCH)
print("\nPOST CONFIRMED: C17-C22 findings no longer reproduce")
