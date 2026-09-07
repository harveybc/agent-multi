"""PRE freeze for order B4 C39-C42 (+ the B4 half of the custody
findings): the two remaining authority defects reproduce at
agent-multi@367aa53e through productive APIs.

1. C39: in an ISOLATED checkout at the reviewed tip, changing ONLY
   agent_plugins/sac_agent.py (the module that builds/trains SAC)
   is ACCEPTED by both the amendment chain and the launch gate,
   because the authority binds nine hand-listed files — not the
   executable checkout. The loaded-plugin digest is recorded by
   the preflight but never compared to a reviewed value.
2. C40: the productive acta path lives INSIDE the candidate
   repository (docs/audits/evidence/); a lookalike JSON there is
   consumed while no private external record exists — content
   checks (reviewer string) prove declared content, not custody
   or authorship.

Zero GPU, zero model construction, zero writes under any real
campaign root; the isolated checkout is a scratch worktree,
removed at the end."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TIP = "367aa53ee298782116aa5ba5077998bac8d465ff"
SCRATCH = Path.home() / ".cache/b4_c39_pre_checkout"

print("== C39: isolated checkout at the reviewed tip ==")
if SCRATCH.exists():
    subprocess.run(["git", "-C", str(REPO), "worktree", "remove",
                    "--force", str(SCRATCH)],
                   capture_output=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)
r = subprocess.run(["git", "-C", str(REPO), "worktree", "add",
                    "--detach", str(SCRATCH), TIP],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-300:]
head = subprocess.run(["git", "-C", str(SCRATCH), "rev-parse",
                       "HEAD"], capture_output=True,
                      text=True).stdout.strip()
print("scratch HEAD:", head[:12], "== reviewed tip:",
      head == TIP)

sac = SCRATCH / "agent_plugins/sac_agent.py"
before = hashlib.sha256(sac.read_bytes()).hexdigest()
sac.write_bytes(sac.read_bytes() +
                b"\n# harmless appended line (PRE probe)\n")
after = hashlib.sha256(sac.read_bytes()).hexdigest()
print("sac_agent.py modified:", before[:10], "->", after[:10])

sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(SCRATCH / "tools"))
import b4_authority as b4a  # noqa: E402  (the TIP's authority)

a13_sha = hashlib.sha256(
    b4a.AMENDMENT_13_PATH.read_bytes()).hexdigest()
gate_dir = Path.home() / ".cache/b4_c39_pre_gate"
if gate_dir.exists():
    shutil.rmtree(gate_dir)
os.makedirs(gate_dir, mode=0o700)
acta = {"schema": "agent_multi.musashi_b4_v6_recovery_audit.v2",
        "reviewed_at_date": "2026-09-07",
        "reviewer": "General Musashi",
        "decision": "OPEN_B4_V6_LAUNCH",
        "latest_amendment_sha256": a13_sha,
        "pinned_commit": TIP,
        "preflight_reviewed": True,
        "ledger_v6_reviewed": True,
        "v5_v6_scientific_equality_reviewed": True}
ap_ = gate_dir / "acta.json"
ap_.write_text(json.dumps(acta))
os.chmod(ap_, 0o600)
b4a.RECOVERY_AUDIT_RECORD_PATH = ap_

surf = b4a.RECOVERY_SURFACE_FILES
print("sac_in_review_surface",
      "agent_plugins/sac_agent.py" in surf)
chain = b4a.verify_amendment_chain()
pins = chain["final_code_pins"]
print("sac_in_final_pins",
      "agent_plugins/sac_agent.py" in pins)
print("chain ACCEPTED", len(chain["amendment_shas"]))
wit = b4a.require_v6_launch_open()
print("launch_gate ACCEPTED (witness acta",
      wit["acta_sha256"][:10] + ")")
assert "agent_plugins/sac_agent.py" not in surf
assert "agent_plugins/sac_agent.py" not in pins
assert len(chain["amendment_shas"]) == 13
print("=> the MODIFIED SAC module is admitted: the authority "
      "binds nine files, not the executable checkout")
esrc = (SCRATCH / "tools/b4_campaign_executor.py").read_text()
seg = esrc[esrc.index("def preflight_environment"):]
seg = seg[:seg.index("\ndef ", 10)]
print("preflight records plugin digest:",
      "module_sha256" in seg,
      "| compares it to a reviewed value:",
      "reviewed" in seg.split("module_sha256")[1][:400])

print("\n== C40: the acta path is candidate-writable, inside "
      "the repo ==")
import importlib
del sys.modules["b4_authority"]
sys.path.remove(str(SCRATCH))
sys.path.remove(str(SCRATCH / "tools"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import b4_authority as live_a  # noqa: E402
rel = live_a.RECOVERY_AUDIT_RECORD_PATH
try:
    inside = rel.relative_to(REPO)
    print("productive acta path (repo-relative):", inside)
    assert str(inside).startswith("docs/audits/evidence/")
except ValueError:
    raise AssertionError("path not inside the repo?")
ext_root = Path.home() / ".config/agent-multi/reviewer_authority"
print("private external authority root exists:",
      ext_root.exists())
# a lookalike at the repo path IS consumed (content checks only)
lookalike = dict(acta)
rel.write_text(json.dumps(lookalike))
os.chmod(rel, 0o600)
try:
    wit2 = live_a.require_v6_launch_open()
    print("repo lookalike CONSUMED: witness",
          wit2["acta_sha256"][:10],
          "— declared authorship, not external custody")
    consumed = True
finally:
    rel.unlink()
assert consumed

subprocess.run(["git", "-C", str(REPO), "worktree", "remove",
                "--force", str(SCRATCH)], capture_output=True)
shutil.rmtree(gate_dir, ignore_errors=True)
print("\nPRE CONFIRMED: partial-surface authority and in-repo "
      "custody both reproduce at the reviewed tip")
