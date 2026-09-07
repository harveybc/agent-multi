"""POST for order B4 C35-C38: the acta is a verified object, the
witness gates every public path, and the recovered authority lives
in custody. Zero GPU, zero scientific cell, zero sealed-2025; the
real Musashi acta remains ABSENT and production stays closed."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import b4_authority as b4a  # noqa: E402


def _load(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


orch = _load("b4orch_post35", "tools/b4_campaign_orchestrator.py")
executor = _load("b4exec_post35", "tools/b4_campaign_executor.py")
ledger_mod = _load("b4led_post35", "tools/b4_campaign_ledger.py")

STATE = Path.home() / ".local/share/agent-multi"
MAT = STATE / "b4_materialization_v5_20260906"
CELL = "o2022_seed101"


def dies(label, fn, needle):
    try:
        fn()
    except SystemExit as e:
        ok = needle in str(e)
        print(f"{label}: DIES [{str(e)[:84]}] match={ok}")
        assert ok, (label, str(e))
        return
    raise AssertionError(f"{label}: DID NOT DIE")


print("== 0. PRODUCTION: the real gate is CLOSED (no acta) ==")
dies("REAL_GATE", b4a.require_v6_launch_open,
     "READY_FOR_FINAL_MUSASHI_AUDIT")

print("\n== 1. C35: the four PRE forgeries and the malformed "
      "date are DEAD ==")
scratch = Path.home() / ".cache/b4_c35_post"
if scratch.exists():
    shutil.rmtree(scratch)
os.makedirs(scratch, mode=0o700)
head = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                       "HEAD"], capture_output=True,
                      text=True).stdout.strip()
a13_sha = hashlib.sha256(
    b4a.AMENDMENT_13_PATH.read_bytes()).hexdigest()


def _acta(**over):
    rec = {"schema":
           "agent_multi.musashi_b4_v6_recovery_audit.v2",
           "reviewed_at_date": "2026-09-07",
           "reviewer": "General Musashi",
           "decision": "OPEN_B4_V6_LAUNCH",
           "latest_amendment_sha256": a13_sha,
           "pinned_commit": head,
           "preflight_reviewed": True,
           "ledger_v6_reviewed": True,
           "v5_v6_scientific_equality_reviewed": True}
    rec.update(over)
    p = scratch / "acta.json"
    if p.exists():
        p.unlink()
    p.write_text(json.dumps(rec))
    os.chmod(p, 0o600)
    b4a.RECOVERY_AUDIT_RECORD_PATH = p
    return p


real_surface = b4a.RECOVERY_SURFACE_FILES
b4a.RECOVERY_SURFACE_FILES = (
    "docs/audits/MUSASHI_B4_DISPATCH_ENVIRONMENT_INCIDENT_"
    "2026_09_06.md",)
# each PRE forgery dies at its own typed layer (the malformed
# date is its own earlier layer, so the pin forms are probed
# with a valid date to expose their own refusals)
for pin, needle in ((None, "nonempty string"),
                    (False, "nonempty string"),
                    ("../../foreign", "40 lowercase"),
                    ("0" * 40, "existing git commit")):
    _acta(pinned_commit=pin)
    dies(f"FORGED_PIN {str(pin)[:14]}",
         b4a.require_v6_launch_open, needle)
for date in ("not-a-date", "2026-13-40", "2026-9-7"):
    _acta(reviewed_at_date=date)
    dies(f"MALFORMED_DATE {date}", b4a.require_v6_launch_open,
         "canonical")
# surface mismatch: a commit whose reviewed bytes differ
pre = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                      "9cad8df4"], capture_output=True,
                     text=True).stdout.strip()
_acta(pinned_commit=pre)
b4a.RECOVERY_SURFACE_FILES = ("tools/b4_authority.py",)
dies("SURFACE_MISMATCH", b4a.require_v6_launch_open,
     "differs from the reviewed surface")
b4a.RECOVERY_SURFACE_FILES = (
    "docs/audits/MUSASHI_B4_DISPATCH_ENVIRONMENT_INCIDENT_"
    "2026_09_06.md",)
# a12-only link grants nothing
_acta(latest_amendment_sha256=hashlib.sha256(
    b4a.AMENDMENT_12_PATH.read_bytes()).hexdigest())
dies("A12_ONLY_LINK", b4a.require_v6_launch_open,
     "LATEST recovery amendment")

print("\n== 2. C36: every public path refuses with the gate "
      "closed ==")
b4a.RECOVERY_AUDIT_RECORD_PATH = scratch / "absent.json"
TR = scratch / "root"
os.makedirs(TR, mode=0o700)
dies("claim_attempt",
     lambda: orch.claim_attempt(TR, CELL),
     "READY_FOR_FINAL_MUSASHI_AUDIT")
print("no claim object created:",
      not list(TR.rglob("CLAIM_*.json")))
dies("execute_cell(direct, no lease path even)",
     lambda: executor.execute_cell(CELL, MAT, TR, "cpu",
                                   lease_path=scratch /
                                   "none.json"),
     "READY_FOR_FINAL_MUSASHI_AUDIT")
dies("standalone CLI --action execute",
     lambda: executor.main([
         "--cell-id", CELL,
         "--materialization-root", str(MAT),
         "--output-root", str(TR), "--device", "cpu",
         "--action", "execute",
         "--lease", str(scratch / "none.json")]),
     "READY_FOR_FINAL_MUSASHI_AUDIT")

print("\n== 3. C36: with a VALID fixture witness the flow works "
      "and custody carries all four bindings ==")
_acta()
wit = b4a.require_v6_launch_open()
with orch.GlobalLock(TR):
    claim = orch.claim_attempt(TR, CELL)
    lease_p = orch.issue_lease(TR, CELL, claim,
                               executor.CAMPAIGN_AUTH_SHA, MAT)
    lease = json.loads(lease_p.read_text())
assert claim["recovery_acta_sha256"] == wit["acta_sha256"]
assert lease["recovery_acta_sha256"] == wit["acta_sha256"]
assert lease["pinned_execution_commit"] == wit["pinned_commit"]
print("claim+lease bind acta", wit["acta_sha256"][:10],
      "and pin", wit["pinned_commit"][:8])
src = (REPO / "tools/b4_campaign_executor.py").read_text()
comp = src[src.index('if terminal == "COMPLETED"'):]
comp = comp[:comp.index("b4a.verify_language")]
for k in ("campaign_generation", "recovery_acta_sha256",
          "pinned_execution_commit", "latest_amendment_sha256"):
    assert k in comp, k
print("COMPLETED terminal REQUIRES the four recovery bindings")
osrc = (REPO / "tools/b4_campaign_orchestrator.py").read_text()
for fn in ("def claim_attempt", "def issue_lease",
           "def verify_lease"):
    seg = osrc[osrc.index(fn):]
    assert "require_v6_launch_open" in seg[:2600], fn
esrc = src[src.index("def execute_cell"):]
assert "require_v6_launch_open" in esrc[:2400]
print("structural domination: claim/lease/verify/execute all "
      "re-derive the witness")
rep = osrc[osrc.index('status = "CAMPAIGN_COMPLETE"'):]
rep = rep[:rep.index("def main")]
for k in ("recovery_acta_sha256", "pinned_execution_commit",
          "latest_amendment_sha256", "campaign_generation"):
    assert k in rep, k
print("campaign report binds the recovered authority")
shutil.rmtree(scratch)

print("\nPOST CONFIRMED: acta verified-object, witness on every "
      "path, recovered custody; real acta ABSENT, launch CLOSED")
