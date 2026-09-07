"""PRE freeze for order T2 C37: the primary estimand's declared
sign contradicts the implemented quantity at agent-multi@0e20e497.

Frozen facts:
1. _series_stats computes MASE(X) - MASE(D): with MASE(X)=1.0 and
   MASE(D)=0.9 the implemented delta is +0.1;
2. draft v5 and validate_confirmatory_design() name the primary
   contrast `D_minus_X`, whose mathematical value for the same
   inputs is -0.1 — the published name inverts the meaning;
3. the adjudicator treats the implemented POSITIVE value as
   improvement (decision polarity correct) — the defect is the
   declared estimand name, not the arithmetic.

Zero downloads, zero scores, zero ledger."""
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

print("== 1. the implemented quantity: MASE(X)=1.0, MASE(D)=0.9 "
      "==")
d = tt._d3()
uid = d["task_population"]["series_ids"][0]
b = d["task_population"]["unit_map"][uid]
rec = tt._complete_record(uid, b["family"], dataset=b["dataset"],
                          windows=b["origin_windows"], delta=0.0)
for o in rec["rolling_origins"].values():
    o["results"]["X"]["ridge"]["mase_primary"] = 1.0
    o["results"]["D"]["ridge"]["mase_primary"] = 0.9
tt._stamp(rec)
s = conf._series_stats(rec, d, "D", "ridge")
print(f"implemented delta        = {s['delta']:+.1f}")
assert abs(s["delta"] - 0.1) < 1e-12
print(f"declared D_minus_X value = {0.9 - 1.0:+.1f}  "
      "(MASE(D) - MASE(X))")
print("the name inverts the implemented meaning")

print("\n== 2. the contract NAMES it D_minus_X ==")
v5 = json.loads(
    (STATE / "t2_screen_design_DRAFT_V5_20260907.json")
    .read_text())
print("draft v5 primary_contrast.delta:",
      v5["primary_contrast"]["delta"])
assert v5["primary_contrast"]["delta"] == "D_minus_X"
vsrc = (REPO / "tools/t2_confirmatory.py").read_text()
line = next(ln for ln in vsrc.splitlines()
            if 'get("delta")' in ln)
print("validator check:", line.strip())
assert '"D_minus_X"' in line
print("prose calls the panel effect a 'paired D-X mean':",
      "paired D-X mean" in json.dumps(v5["estimand"]))

print("\n== 3. the adjudicator treats POSITIVE as improvement ==")
good = [tt._rec_for(d, u, delta=0.10)
        for u in d["task_population"]["series_ids"]]
out = conf.adjudicate_screen(good, d)
print("uniform +0.1 implemented deltas ->", out["verdict"])
assert out["verdict"] == "ADVANCE_TO_DOMAIN_VALIDATION"
bad = [tt._rec_for(d, u, delta=-0.10)
       for u in d["task_population"]["series_ids"]]
out2 = conf.adjudicate_screen(bad, d)
print("uniform -0.1 implemented deltas ->", out2["verdict"])
assert out2["verdict"] == "DOES_NOT_ADVANCE"
print("decision polarity agrees with the ARITHMETIC — the "
      "defect is the published estimand name only")

print("\nPRE CONFIRMED: sign contradiction frozen at the "
      "executable contract level")
