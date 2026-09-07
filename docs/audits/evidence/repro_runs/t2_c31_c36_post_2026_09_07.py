"""POST for order T2 C31-C36: the final screen contract stands —
per-panel extreme support, single geometry authority with the
common two-origin rule, hospital a full member (2x17), every
unit_map field independently forgeable-and-dying, exact costs,
symlink root dead, fresh verifier executing. Zero downloads, zero
scores, zero scientific ledger."""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

import t2_bank as bank  # noqa: E402
import t2_confirmatory as conf  # noqa: E402
import t2_fresh_verifier as fv  # noqa: E402
import importlib.util as ilu
spec = ilu.spec_from_file_location(
    "t2tests", REPO / "tests/test_t2_harness.py")
tt = ilu.module_from_spec(spec)
spec.loader.exec_module(tt)

STATE = Path.home() / ".local/share/agent-multi"

print("== C31: one evaluable extreme series never licenses ==")
d = tt._d3()
recs = []
for uid in d["task_population"]["series_ids"]:
    only = uid.endswith("::s0")
    recs.append(tt._rec_for(d, uid,
                            ex_support=(4 if only else 0)))
out = conf.adjudicate_screen(recs, d)
print("SUPPORT_1_OF_N:", out["verdict"], "|",
      out["reason"][:70])
assert out["verdict"] == "INCONCLUSIVE"
assert "evaluable extreme evidence" in out["reason"]

print("\n== C35: hospital 2x17 under the ONE authority ==")
w = bank.origin_windows_for(84, 2, 0.6, seasonal_period=12)
print("windows:", w)
assert all(wb["score"][1] - wb["score"][0] == 17
           for wb in w.values())
v5 = conf.strict_json_load(
    STATE / "t2_screen_design_DRAFT_V5_20260907.json", "d")
gf = v5["task_population"]["geometry_feasibility"]["hospital"]
print("v5 hospital:", json.dumps(gf))
assert gf["n_selected"] == 40 and gf["model_minimums_kept"]
assert len(v5["task_population"]["series_ids"]) == 242
print("population 242; hospital FULL member; v4 superseded:",
      v5["supersedes_draft_sha256"][:12])

print("\n== C32/C34: the generator's windows are never the "
      "authority — live full re-derivation ==")
manifest = conf.strict_json_load(
    STATE / "t2_public_data_manifest_20260906.json", "m")
census = conf.strict_json_load(
    STATE / "t2_bank_census_20260906.json", "c")
out2 = fv.fresh_verify(manifest, census, v5,
                       manifest_sha=conf._sha_file(
                           STATE /
                           "t2_public_data_manifest_20260906"
                           ".json"))
print(json.dumps(out2))
assert out2["units_rederived"] == 4650
assert out2["design_series"] == 242
forged = json.loads(json.dumps(v5))
uid = v5["task_population"]["series_ids"][0]
forged["task_population"]["unit_map"][uid]["origin_windows"][
    "origin0"]["train"] = [0, 51]
try:
    fv.fresh_verify(manifest, census, forged, manifest_sha=None)
    raise AssertionError("shifted design window accepted")
except SystemExit as exc:
    print("one-row design-window forgery:", str(exc)[:80])

print("\n== battery-frozen kills (each individually in the "
      "focal battery) ==")
print("- every unit_map field forged dies live "
      "(test_c36_2, 8 fields)")
print("- record window shifted one row refuses "
      "(test_c30_kill_4)")
print("- each cost phase omitted refuses via the productive "
      "adjudicator (test_c36_4, 5 phases + extra keys)")
print("- symlink root refuses (test_c30_kill_7)")
print("- fresh verifier wired before the ledger "
      "(test_c30_kill_8, live)")
print("- damaged panel -> DOES_NOT_ADVANCE (test_c30_kill_10)")
print("- dominant panel fails LOPO (test_c30_kill_9)")

print("\n== sims re-executed under the new geometry: "
      "byte-identical (panel-effect level) ==")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


assert sha(STATE / "t2_coverage_sim_20260906.json").startswith(
    "e7fae781")
assert sha(STATE / "t2_screen_sim_20260906.json").startswith(
    "76581fbb")
print("coverage e7fae781... | screen 76581fbb...")

print("\nPOST CONFIRMED: C31-C35 executable; hospital resolved "
      "without exception; single geometry authority live")
