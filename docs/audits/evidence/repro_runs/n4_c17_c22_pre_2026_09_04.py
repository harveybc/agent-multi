"""PRE freeze for order @af1ca667 (C17-C22): executable
reproducers through public functions, before any edit."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location(
    "n4a", REPO / "tools" / "n4_target_audit.py")
n4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n4a)

DESIGN_V2 = (REPO / "docs/audits/evidence/"
             "N4_TARGET_AUDIT_DESIGN_V2_2026_09_04.json")
RESULT_V2 = (REPO / "docs/audits/evidence/"
             "N4_SCREEN_RESULT_V2_2026_09_04.json")
RESULT_V1 = (REPO / "docs/audits/evidence/"
             "N4_SCREEN_RESULT_2026_09_04.json")

print("== C17: validate_design is not strict ==")
design = json.loads(DESIGN_V2.read_text())
import tempfile, os
cases = {}
d1 = json.loads(json.dumps(design))
d1["unknown_top_level_field"] = {"x": 1}
cases["unknown_top_field"] = json.dumps(d1)
cases["duplicate_key"] = DESIGN_V2.read_text().replace(
    '"schema": "agent_multi.n4_target_audit_design.v2"',
    '"schema": "agent_multi.n4_target_audit_design.v2", '
    '"schema": "agent_multi.n4_target_audit_design.v2"', 1)
d3 = json.loads(json.dumps(design))
d3["executable_binding"]["support_min"] = 30.0
cases["float_support_min"] = json.dumps(d3)
accepted = {}
for tag, text in cases.items():
    p = Path.home() / ".cache" / f"c17_{tag}.json"
    p.write_text(text)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    import unittest.mock as um
    with um.patch.object(n4a, "DESIGN_V2", str(p)):
        try:
            n4a.validate_design(sha)
            accepted[tag] = True
        except n4a.N4Refusal as r:
            accepted[tag] = f"refused: {str(r)[:60]}"
    p.unlink()
print(json.dumps(accepted, indent=1))
assert accepted["unknown_top_field"] is True
assert accepted["duplicate_key"] is True
assert accepted["float_support_min"] is True

print("\n== C18: fabricated supports license and pass tm_h6 ==")
records = json.loads(RESULT_V2.read_text())["per_window_records"]
recs = copy.deepcopy(records)
for wk, rec in recs["tm_h6"].items():
    rec["class_support_score"] = {"0": 80, "1": 80, "2": 60}
    rec["losses"]["volatility_history"] = [
        round(v * 0.5, 8) for v in rec["losses"]["prior"]]
out = n4a.adjudicate(recs)
print("tm_h6 outcome:", out["per_candidate"]["tm_h6"]["outcome"],
      "| passers:", out["passers"])
assert "tm_h6:volatility_history" in out["passers"]

recs2 = copy.deepcopy(records)
for wk, rec in recs2["tm_h6"].items():
    rec.pop("window"); rec.pop("n_score")
    rec["class_support_score"] = {"0": 30, "1": 30, "2": 30}
try:
    out2 = n4a.adjudicate(recs2)
    print("missing window/n_score + 30/30/30 supports:",
          "ACCEPTED —", out2["per_candidate"]["tm_h6"]["outcome"],
          "(counts never checked against loss cardinality)")
    missing_ok = True
except n4a.N4Refusal as r:
    print("refused:", r)
    missing_ok = False
assert missing_ok

print("\n== C19: a forged v1 mints a positive successor ==")
v1 = json.loads(RESULT_V1.read_text())
for ck in ("mfemae_h6", "mfemae_h12"):
    for wk, rec in v1["per_window_records"][ck].items():
        rec["losses"]["target_history"] = [
            round(v * 0.01, 8) for v in rec["losses"]["prior"]]
        rec["losses"]["causal_linear"] = [
            round(v * 0.01, 8) for v in rec["losses"]["prior"]]
forged = Path.home() / ".cache" / "c19_forged_v1.json"
forged.write_text(json.dumps(v1, default=float))
out3 = n4a.readjudicate(forged, Path.home() / ".cache"
                        / "c19_out.json")
print("forged-v1 successor verdict:", out3["verdict"],
      "| passers:", out3["passers"])
assert out3["verdict"] == \
    "TARGET_FORMULATION_CANDIDATE_FOR_FUTURE_CONFIRMATION"
forged.unlink()
(Path.home() / ".cache" / "c19_out.json").unlink()

print("\n== C20: numeric string survives on the unlicensed path ==")
recs3 = copy.deepcopy(records)
recs3["tm_h6"]["w1"]["losses"]["prior"][0] = str(
    recs3["tm_h6"]["w1"]["losses"]["prior"][0])
try:
    out4 = n4a.adjudicate(recs3)
    print("numeric-string loss in unlicensed candidate: ACCEPTED "
          f"(verdict {out4['verdict']})")
    str_ok = True
except n4a.N4Refusal as r:
    print("refused:", r)
    str_ok = False
assert str_ok

print("\n== C21: v2 claims a predeclaration it cannot have ==")
print("design v2 sealed_before_any_new_score:",
      design.get("sealed_before_any_new_score"),
      "(v2 was added in the correction commit AFTER v1 scores "
      "were known)")
assert design.get("sealed_before_any_new_score") is True

print("\n== C22: N5 ledger misorders the roadmap ==")
n5 = json.loads((REPO / "docs/audits/evidence/"
                 "N5_TRANSITION_LEDGER_2026_09_04.json")
                .read_text())
paths = list(n5["paths"])
print("ledger paths:", paths)
print("Screen B/B4 named as next node:",
      any("screen_b" in p or "b4" in p for p in paths))
joined = json.dumps(n5)
print("stale claims present: C38-awaiting:",
      "awaiting review" in joined,
      "| non-flat MT5:", "non-flat" in joined
      or "open MT5 position" in joined)
assert not any("screen_b" in p or "b4" in p for p in paths)

print("\nPRE CONFIRMED: all six findings reproduce")
