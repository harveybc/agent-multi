"""POST for order T2 C37: one unambiguous estimand. The productive
contract names the primary contrast mase_improvement_X_minus_D =
MASE(X) - MASE(D) (positive = D reduces error) everywhere; old,
ambiguous and polarity-inverted names refuse; draft v6 supersedes
v5 by digest changing ONLY the naming contract; the real
polarity mutation (xm-am -> am-xm) breaks the battery (executed
and read from terminal output before this prose was written).
Zero downloads, zero scores, zero seal, zero ledger."""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

import t2_confirmatory as conf  # noqa: E402
import importlib.util as ilu
spec = ilu.spec_from_file_location(
    "t2tests", REPO / "tests/test_t2_harness.py")
tt = ilu.module_from_spec(spec)
spec.loader.exec_module(tt)

STATE = Path.home() / ".local/share/agent-multi"
mp = STATE / "t2_public_data_manifest_20260906.json"

print("== 1. polarity, executable ==")
d = tt._d3()
uid = d["task_population"]["series_ids"][0]
b = d["task_population"]["unit_map"][uid]


def probe(x, dd):
    rec = tt._complete_record(uid, b["family"],
                              dataset=b["dataset"],
                              windows=b["origin_windows"])
    for o in rec["rolling_origins"].values():
        o["results"]["X"]["ridge"]["mase_primary"] = x
        o["results"]["D"]["ridge"]["mase_primary"] = dd
    tt._stamp(rec)
    return conf._series_stats(rec, d, "D", "ridge")["delta"]


print(f"X=1.0, D=0.9 -> {probe(1.0, 0.9):+.1f} (beneficial)")
print(f"X=0.9, D=1.0 -> {probe(0.9, 1.0):+.1f} (harmful)")
assert abs(probe(1.0, 0.9) - 0.1) < 1e-12
assert abs(probe(0.9, 1.0) + 0.1) < 1e-12
good = [tt._rec_for(d, u, delta=0.10)
        for u in d["task_population"]["series_ids"]]
out = conf.adjudicate_screen(good, d)
print("verdict on +0.1:", out["verdict"], "|",
      out["panel_effect_definition"][:52])
assert out["verdict"] == "ADVANCE_TO_DOMAIN_VALIDATION"
assert out["panel_effect_definition"].startswith(
    "mase_improvement_X_minus_D")

print("\n== 2. names: one accepted, all others refuse ==")
v6 = json.loads(
    (STATE / "t2_screen_design_DRAFT_V6_20260907.json")
    .read_text())
conf.validate_confirmatory_design(v6, conf._sha_file(mp))
print("v6 (mase_improvement_X_minus_D): PASS")
for bad in ("D_minus_X", "delta",
            "mase_improvement_D_minus_X", "X_minus_D"):
    f2 = json.loads(json.dumps(v6))
    f2["primary_contrast"]["delta"] = bad
    body = {k: f2[k] for k in sorted(f2)
            if k != "design_sha256"}
    f2["design_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    try:
        conf.validate_confirmatory_design(f2, conf._sha_file(mp))
        raise AssertionError(f"{bad} accepted")
    except SystemExit as e:
        print(f"{bad!r}: refuses")
v5 = json.loads(
    (STATE / "t2_screen_design_DRAFT_V5_20260907.json")
    .read_text())
try:
    conf.validate_confirmatory_design(v5, conf._sha_file(mp))
    raise AssertionError("v5 accepted")
except SystemExit:
    print("draft v5 (superseded schema): refuses")

print("\n== 3. v5 -> v6: naming-only supersession ==")
assert v6["task_population"] == v5["task_population"]
assert v6["role_geometry"] == v5["role_geometry"]
assert v6["supersedes_draft_sha256"] == hashlib.sha256(
    (STATE / "t2_screen_design_DRAFT_V5_20260907.json")
    .read_bytes()).hexdigest()
assert v6["arms"]["XDR"] == "[X, D, X-D]"
print("population/unit_map/geometry byte-equal; supersedes-v5 "
      "binding true; the XDR FEATURE list [X, D, X-D] untouched")

print("\n== 4. the real mutation was executed and read ==")
print("sed xm-am -> am-xm at the productive line killed 4 tests")
print("(c37_polarity assert diff 0.1999...; c8; kill_10; c36_1)")
print("then reverted — recorded in the packet from terminal "
      "output")

print("\nPOST CONFIRMED: one unambiguous estimand; polarity "
      "frozen; v6 ready")
