"""PRE freeze for order B4 C35-C38: the three authority bypasses
reproduce against agent-multi@9d4aec7f through the PUBLIC APIs.

1. C35: require_v6_launch_open() accepts pinned_commit None /
   False / a traversal-like string / forty zeroes, together with a
   malformed reviewed_at_date — a reviewer label plus booleans and
   the real amendment-12 digest open the gate.
2. C36: with the recovery gate raising UNCONDITIONALLY, the public
   sequence GlobalLock -> claim_attempt -> issue_lease ->
   execute_cell still reaches a post-claim typed PLUGIN terminal —
   the gate protects only run_campaign.
3. C37: a v6 COMPLETED terminal and the campaign report carry ONLY
   the historical authorization + amendment_11_sha256 — amendment
   12, the recovery-acta digest, the pinned execution commit and
   the v6 generation are absent from the productive schemas.

Zero GPU, zero score, zero sealed-2025; v5 and v6 state untouched
(throwaway roots only)."""
import hashlib
import json
import os
import shutil
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


orch = _load("b4orch_pre35", "tools/b4_campaign_orchestrator.py")
executor = _load("b4exec_pre35", "tools/b4_campaign_executor.py")

STATE = Path.home() / ".local/share/agent-multi"
MAT = STATE / "b4_materialization_v5_20260906"
CELL = "o2022_seed101"

print("== 1. C35: the gate accepts forged pins and dates ==")
a12_sha = hashlib.sha256(
    b4a.AMENDMENT_12_PATH.read_bytes()).hexdigest()
scratch = Path.home() / ".cache/b4_c35_pre_20260907"
if scratch.exists():
    shutil.rmtree(scratch)
os.makedirs(scratch, mode=0o700)
real_path = b4a.RECOVERY_AUDIT_RECORD_PATH
accepted = []
for pin in (None, False, "../../foreign", "0" * 40):
    rec = {"schema": "agent_multi.musashi_b4_v6_recovery_audit.v1",
           "reviewed_at_date": "not-a-date",
           "reviewer": "General Musashi",
           "decision": "OPEN_B4_V6_LAUNCH",
           "amendment_12_sha256": a12_sha,
           "pinned_commit": pin,
           "preflight_reviewed": True,
           "ledger_v6_reviewed": True,
           "v5_v6_scientific_equality_reviewed": True}
    p = scratch / "acta.json"
    p.write_text(json.dumps(rec))
    b4a.RECOVERY_AUDIT_RECORD_PATH = p
    try:
        out = b4a.require_v6_launch_open()
        accepted.append((repr(pin), "ACCEPTED"))
        print(f"ACCEPTED {pin!r} 'not-a-date'")
    except SystemExit as exc:
        accepted.append((repr(pin), f"refused: {exc}"))
        print(f"refused {pin!r}: {str(exc)[:60]}")
b4a.RECOVERY_AUDIT_RECORD_PATH = real_path
assert all(v == "ACCEPTED" for _, v in accepted), accepted
print("all four forged actas OPEN the gate; no descriptor "
      "binding, no independent digest, no date/commit validation")

print("\n== 2. C36: the public claim/lease/executor sequence "
      "bypasses the closed gate ==")


def _closed():
    raise SystemExit("REFUSED: GATE_CLOSED_FOR_THIS_PROBE")


b4a.require_v6_launch_open = _closed
import importlib.metadata as md  # noqa: E402
import app.plugin_loader as apl  # noqa: E402
_real_md, _real_apl = md.entry_points, apl.entry_points


def _view():
    class _V:
        def select(self, group):
            eps = _real_md().select(group=group)
            if group == "agent.plugins":
                return [e for e in eps if e.name != "sac_agent"]
            return eps
    return _V()


md.entry_points = _view
apl.entry_points = _view
TR = scratch / "root"
os.makedirs(TR, mode=0o700)
try:
    with orch.GlobalLock(TR):
        claim = orch.claim_attempt(TR, CELL)
        lease = orch.issue_lease(TR, CELL, claim,
                                 executor.CAMPAIGN_AUTH_SHA, MAT)
        gate_hit = False
        try:
            executor.execute_cell(CELL, MAT, TR, "cpu",
                                  lease_path=lease)
        except SystemExit as exc:
            gate_hit = "GATE_CLOSED_FOR_THIS_PROBE" in str(exc)
            print("SystemExit:", str(exc)[:70])
        except ImportError as exc:
            print("ImportError reached (post-claim):",
                  str(exc)[:60])
finally:
    md.entry_points = _real_md
    apl.entry_points = _real_apl
term = TR / CELL / "B4_CELL_TERMINAL.json"
print("claim created:", (TR / CELL /
      f"CLAIM_{b4a.CAMPAIGN_GENERATION}.json").exists(),
      "| typed terminal written:", term.exists(),
      "| gate ever consulted:", gate_hit)
assert term.exists() and not gate_hit
t = json.loads(term.read_text())
print("terminal:", t["terminal"], "— the gate protects ONLY "
      "run_campaign; claim+lease+constructors ran while CLOSED")

print("\n== 3. C37: recovered authority absent from custody ==")
src = (REPO / "tools/b4_campaign_executor.py").read_text()
comp_block = src[src.index('if terminal == "COMPLETED"'):]
comp_block = comp_block[:comp_block.index("b4a.verify_language")]
checks = {
    "terminal_requires_amendment_11":
        "amendment_11_sha256" in comp_block,
    "terminal_requires_amendment_12":
        "amendment_12_sha256" in comp_block,
    "terminal_requires_recovery_acta":
        "recovery_acta_sha256" in comp_block,
    "terminal_requires_pinned_commit":
        "pinned_execution_commit" in comp_block,
    "terminal_requires_generation":
        "campaign_generation" in comp_block,
}
for k, v in checks.items():
    print(f"{k}: {v}")
assert checks["terminal_requires_amendment_11"]
assert not any(v for k, v in checks.items()
               if k != "terminal_requires_amendment_11")
osrc = (REPO / "tools/b4_campaign_orchestrator.py").read_text()
rep = osrc[osrc.index('status = "CAMPAIGN_COMPLETE"'):]
rep = rep[:rep.index("def main")]
print("report carries amendment_11 only:",
      "amendment_11_sha256" in rep and
      "amendment_12" not in rep and
      "recovery" not in rep)
assert "amendment_11_sha256" in rep and \
    "amendment_12" not in rep and "recovery" not in rep
bind_block = src[src.index("b4_cell_auth_binding"):]
bind_block = bind_block[:bind_block.index("_orch.verify_lease")]
print("per-attempt binding fields: authorization + amendment_11 "
      "only:", "amendment_11_sha256" in bind_block and
      "recovery" not in bind_block)
shutil.rmtree(scratch)

print("\nPRE CONFIRMED: C35 forged-acta acceptance, C36 public "
      "bypass, C37 stale custody all reproduce at 9d4aec7f")
