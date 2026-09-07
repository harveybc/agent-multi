"""POST for order B4 C39-C42: the PRE bypasses are dead against
the productive authority. Zero GPU; the real external acta remains
ABSENT and the launch CLOSED."""
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


def dies(label, fn, needle):
    try:
        fn()
    except SystemExit as e:
        ok = needle in str(e)
        print(f"{label}: DIES [{str(e)[:80]}] match={ok}")
        assert ok, (label, str(e))
        return
    raise AssertionError(f"{label}: DID NOT DIE")


print("== 0. PRODUCTION: gate closed; acta path OUTSIDE git ==")
p = b4a.RECOVERY_AUDIT_RECORD_PATH
try:
    p.relative_to(REPO)
    raise AssertionError("acta path still inside the repo")
except ValueError:
    pass
print("productive acta path: ~/.config/agent-multi/"
      "reviewer_authority/ (outside the candidate repo)")
dies("REAL_GATE", b4a.require_v6_launch_open,
     "READY_FOR_FINAL_MUSASHI_AUDIT")

print("\n== 1. C39: the PRE's modified-SAC checkout now refuses "
      "==")
TIP = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                      "HEAD"], capture_output=True,
                     text=True).stdout.strip()
SC = Path.home() / ".cache/b4_c39_post_checkout"
subprocess.run(["git", "-C", str(REPO), "worktree", "remove",
                "--force", str(SC)], capture_output=True)
r = subprocess.run(["git", "-C", str(REPO), "worktree", "add",
                    "--detach", str(SC), TIP],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-200:]
real_repo = b4a.REPO
b4a.REPO = SC
try:
    ok = b4a.verify_checkout_identity(TIP)
    print("clean exact-commit checkout: head",
          ok["head"][:10], "tree", ok["tree"][:10])
    sac = SC / "agent_plugins/sac_agent.py"
    orig = sac.read_bytes()
    sac.write_bytes(orig + b"\n# harmless appended line\n")
    dies("MODIFIED_SAC", lambda: b4a.verify_checkout_identity(TIP),
         "not clean against the pinned commit")
    sac.write_bytes(orig)
    shadow = SC / "pipeline_plugins/zz_shadow.py"
    shadow.write_text("# shadow\n")
    dies("UNTRACKED_SHADOW",
         lambda: b4a.verify_checkout_identity(TIP),
         "shadow repository imports")
    shadow.unlink()
    parent = subprocess.run(
        ["git", "-C", str(SC), "rev-parse", "HEAD~1"],
        capture_output=True, text=True).stdout.strip()
    dies("DIFFERENT_HEAD",
         lambda: b4a.verify_checkout_identity(parent),
         "differs from the acta's pinned commit")
finally:
    b4a.REPO = real_repo
    subprocess.run(["git", "-C", str(REPO), "worktree",
                    "remove", "--force", str(SC)],
                   capture_output=True)

print("\n== 2. C40: a repo lookalike grants nothing; custody "
      "walk enforced ==")
look = REPO / ("docs/audits/evidence/"
               "MUSASHI_B4_V6_RECOVERY_AUDIT_RECORD.json")
assert not look.exists()
look.write_text(json.dumps({"schema":
                            "agent_multi.musashi_b4_v6_recovery_"
                            "audit.v2"}))
try:
    dies("REPO_LOOKALIKE", b4a.require_v6_launch_open,
         "READY_FOR_FINAL_MUSASHI_AUDIT")
finally:
    look.unlink()
gate = Path.home() / ".cache/b4_c40_post_gate"
if gate.exists():
    shutil.rmtree(gate)
am = gate / "agent-multi"
ra = am / "reviewer_authority"
for d in (gate, am, ra):
    d.mkdir(mode=0o700)
a14_sha = hashlib.sha256(
    b4a.AMENDMENT_14_PATH.read_bytes()).hexdigest()
acta = {"schema": "agent_multi.musashi_b4_v6_recovery_audit.v2",
        "reviewed_at_date": "2026-09-07",
        "reviewer": "General Musashi",
        "decision": "OPEN_B4_V6_LAUNCH",
        "latest_amendment_sha256": a14_sha,
        "pinned_commit": TIP,
        "preflight_reviewed": True,
        "ledger_v6_reviewed": True,
        "v5_v6_scientific_equality_reviewed": True}
ap_ = ra / "MUSASHI_B4_V6_RECOVERY_AUDIT_RECORD.json"
ap_.write_text(json.dumps(acta))
os.chmod(ap_, 0o600)
b4a.RECOVERY_AUDIT_RECORD_PATH = ap_
os.chmod(ra, 0o755)
dies("PERMISSIVE_CHAIN", b4a.require_v6_launch_open,
     "not the private 0700")
os.chmod(ra, 0o700)
os.chmod(ap_, 0o644)
dies("PERMISSIVE_FILE", b4a.require_v6_launch_open, "0600")
os.chmod(ap_, 0o600)
wit = b4a.require_v6_launch_open()
print("valid private fixture witness v2:",
      wit["acta_sha256"][:10], "| tree",
      wit["checkout_tree_sha"][:10])
assert wit["schema"] == "agent_multi.b4_v6_recovery_witness.v2"
shutil.rmtree(gate)
src = (REPO / "tools/b4_authority.py").read_text()
assert "cryptographically identify an author" in src
print("honest custody-not-authorship prose present")

print("\nPOST CONFIRMED: full-checkout authority + external "
      "private custody; real acta ABSENT, launch CLOSED")
