"""PRE freeze for order @0b4d2748 (C23-C25): the formerly omitted
forged self-hash call, the permissive nested design field, the
unconsumed owner act, and the coerced build boundary."""
import copy
import hashlib
import importlib.util
import json
import sys
import unittest.mock as um
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location(
    "n4a", REPO / "tools" / "n4_target_audit.py")
n4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n4a)

EV = REPO / "docs/audits/evidence"
RESULT_V1 = EV / "N4_SCREEN_RESULT_2026_09_04.json"
DESIGN_V3 = EV / "N4_TARGET_AUDIT_DESIGN_V3_2026_09_04.json"
RUN_ROOT = (Path.home() / ".local/share/agent-multi/"
            "target_horizon_census_n2_20260903")

print("== C23: the omitted second call — forged v1 + its OWN "
      "correct hash ==")
v1 = json.loads(RESULT_V1.read_text())
for ck in ("mfemae_h6", "mfemae_h12"):
    for wk, rec in v1["per_window_records"][ck].items():
        rec["losses"]["target_history"] = [
            round(v * 0.01, 8) for v in rec["losses"]["prior"]]
forged = Path.home() / ".cache" / "c23_forged_v1.json"
forged.write_text(json.dumps(v1, default=float))
fsha = hashlib.sha256(forged.read_bytes()).hexdigest()
print("forged sha:", fsha)
try:
    out = n4a.rebind(
        forged, fsha,
        "ae05f1878305cc3aee9003849d4f147f2685a159ed3afbdc3870ec"
        "7e8c58f4ef",
        hashlib.sha256(DESIGN_V3.read_bytes()).hexdigest(),
        RUN_ROOT, Path.home() / ".cache" / "c23_out.json")
    print("accepted:", True)
    print("verdict:", out["verdict"])
    print("passers:", out["passers"])
    accepted = True
except n4a.N4Refusal as r:
    print("refused:", r)
    accepted = False
finally:
    forged.unlink(missing_ok=True)
    (Path.home() / ".cache" / "c23_out.json").unlink(
        missing_ok=True)
assert accepted, "expected the substitution to still succeed"

print("\n== C24-A: nested design field remains permissive ==")
d = json.loads(DESIGN_V3.read_text())
d["classification"] = {"attacker_field": True}
p = Path.home() / ".cache" / "c24_design.json"
p.write_text(json.dumps(d))
sha = hashlib.sha256(p.read_bytes()).hexdigest()
try:
    with um.patch.object(n4a, "DESIGN_V3", str(p)):
        n4a.validate_design(sha)
    print("ACCEPTED_UNKNOWN_NESTED_FIELD")
    nested_ok = True
except n4a.N4Refusal as r:
    print("refused:", r)
    nested_ok = False
finally:
    p.unlink()
assert nested_ok

print("\n== C24-B: the owner act is named but not consumed ==")
act = EV / ("OWNER_RATIFICATION_OBSERVATION_V2_AND_MT5_BUILD_6140"
            "_2026_09_04.json")
print("act present on this branch:", act.exists())
contract = json.loads(
    (REPO / "examples/config/phase_3_eth_sac_dynamics/systems/"
     "ethusdt_4h_l1_system_v2.json").read_text())
print("$doc still says pending:",
      "pending" in contract.get("$doc", "").lower()
      or "AWAITING" in contract.get("$doc", ""))
assert not act.exists()

print("\n== C25: LTS build coercion (documented from the order's "
      "independent run) ==")
lts = Path.home() / "Documents/GitHub/lts"
src = (lts / "tools/collector_activation_preflight.py").read_text()
print("judge still coerces via int():",
      "int(expected_terminal_build)" in src)
assert "int(expected_terminal_build)" in src

print("\nPRE CONFIRMED: all findings reproduce")
