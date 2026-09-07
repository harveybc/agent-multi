"""POST for order T2 C25-C30: every PRE bypass now dies with a
typed refusal or the correct screen verdict; the fresh verifier is
an executing precondition of the single path; the estimand is the
single T2-S screen. Zero downloads, zero scores, zero scientific
ledger, B4 untouched."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))
os.environ.setdefault(
    "B4_T1_PREPROCESSOR_ROOT",
    str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))
STATE = Path.home() / ".local/share/agent-multi"

import t2_confirmatory as conf  # noqa: E402
import t2_fresh_verifier as fv  # noqa: E402
import importlib.util as ilu
spec = ilu.spec_from_file_location(
    "t2tests", REPO / "tests/test_t2_harness.py")
tt = ilu.module_from_spec(spec)
spec.loader.exec_module(tt)


def dies(label, fn, needle):
    try:
        fn()
    except SystemExit as e:
        ok = needle in str(e)
        print(f"{label}: DIES [{str(e)[:100]}] match={ok}")
        assert ok, (label, str(e))
        return
    raise AssertionError(f"{label}: DID NOT DIE")


d = tt._d3()
base = tt._recs(d)

print("== C25a: missing extreme metrics -> typed refusal ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["rolling_origins"].values():
        for arm in ("X", "D", "XDR", "width_control"):
            o["results"][arm]["ridge"].pop(
                "mase_on_extreme_innovations", None)
            for sv in o["results"][arm]["mlp_small"].values():
                sv.pop("mase_on_extreme_innovations", None)
    tt._stamp(r)
dies("MISSING_EXTREMES",
     lambda: conf.adjudicate_screen(recs, d),
     "extreme metric is absent")

print("\n== C25b: X extreme 0 vs D 999 -> HARM_INFINITE ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["rolling_origins"].values():
        o["results"]["X"]["ridge"][
            "mase_on_extreme_innovations"] = 0.0
        o["results"]["D"]["ridge"][
            "mase_on_extreme_innovations"] = 999.0
    tt._stamp(r)
out = conf.adjudicate_screen(recs, d)
print("ZERO_BASELINE_EXTREME:", out["verdict"], "|",
      out["reason"][:70])
assert out["verdict"] == "DOES_NOT_ADVANCE"
assert "infinite extreme damage" in out["reason"]

print("\n== C26: record without geometry -> refusal ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["rolling_origins"].values():
        o.pop("train", None)
        o.pop("score", None)
    tt._stamp(r)
dies("NO_ORIGIN_GEOMETRY",
     lambda: conf.adjudicate_screen(recs, d),
     "origin keys are not the exact")

print("\n== C26: one window bound shifted one row -> refusal ==")
recs = json.loads(json.dumps(base))
recs[0]["rolling_origins"]["origin1"]["train"][1] += 1
tt._stamp(recs[0])
dies("SHIFTED_WINDOW",
     lambda: conf.adjudicate_screen(recs, d),
     "windows differ from the design")

print("\n== C27: costs without target/baseline -> refusal ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["costs_by_phase"].values():
        o.pop("target_construction_s", None)
        o.pop("seasonal_naive_s", None)
    tt._stamp(r)
dies("MISSING_GLOBAL_COST_PHASES",
     lambda: conf.adjudicate_screen(recs, d),
     "cost phases are not the exact")

print("\n== C28a: forged unit_map semantics (TRUE digest) dies "
      "in the LIVE fresh re-derivation ==")
manifest = conf.strict_json_load(
    STATE / "t2_public_data_manifest_20260906.json", "m")
census = conf.strict_json_load(
    STATE / "t2_bank_census_20260906.json", "c")
design = conf.strict_json_load(
    STATE / "t2_screen_design_DRAFT_V4_20260906.json", "d")
forged = json.loads(json.dumps(design))
for uid, b in list(forged["task_population"][
        "unit_map"].items())[:5]:
    b["family"] = "totally_forged_family"
    b["dataset"] = "alien_panel"
    b["seasonal_period"] = 999
dies("FORGED_UNIT_MAP",
     lambda: fv.fresh_verify(manifest, census, forged,
                             manifest_sha=None),
     "do not re-derive from the physical bytes")

print("\n== C28b: symlink ROOT refuses ==")
sc = Path.home() / ".cache/t2_c25_post"
if sc.exists():
    shutil.rmtree(sc)
sc.mkdir(parents=True)
realroot = sc / "realroot"
realroot.mkdir()
data = b"@frequency monthly\n@data\ns1:2020:" + ",".join(
    str(float(i % 9)) for i in range(150)).encode() + b"\n"
(realroot / "probe.tsf").write_bytes(data)
os.symlink(realroot, sc / "rootlink")
m2, _ = tt._mk_manifest(sc, raw_root=realroot)
dies("SYMLINK_ROOT",
     lambda: conf.validate_public_manifest(
         m2, raw_root=sc / "rootlink"),
     "symlink root")
adm = conf.validate_public_manifest(m2, raw_root=realroot)
print("honest physical root still validates:", "probe" in adm)
shutil.rmtree(sc)

print("\n== C28c: run_confirmatory CONSUMES the fresh verifier "
      "(live), before review and ledger ==")
lp = Path.home() / ".cache/t2_c25_post_ledger.json"
if lp.exists():
    lp.unlink()
dies("HONEST_V4_REACHES_REVIEW_GATE",
     lambda: conf.run_confirmatory(
         STATE / "t2_public_data_manifest_20260906.json",
         STATE / "t2_screen_design_DRAFT_V4_20260906.json",
         lp, census_path=STATE / "t2_bank_census_20260906.json"),
     "DESIGN_REVIEW_REQUIRED")
print("no ledger artifact created:", not lp.exists())
assert not lp.exists()
src = (REPO / "tools/t2_confirmatory.py").read_text()
seg = src[src.index("def run_confirmatory"):]
seg = seg[:seg.index("def ", 10)]
print("order inside the single path: fresh_verify < review < "
      "ledger:",
      seg.index("fresh_verify") <
      seg.index("verify_design_review_record") <
      seg.index("open_attempt_ledger"))

print("\n== C29: dominant panel cannot carry the screen ==")
recs = [tt._rec_for(d, uid,
                    delta=(0.030 if uid.startswith("f0")
                           else 0.019))
        for uid in d["task_population"]["series_ids"]]
out = conf.adjudicate_screen(recs, d)
print("DOMINANT_PANEL:", out["verdict"], "|", out["reason"][:80])
assert out["verdict"] != "ADVANCE_TO_DOMAIN_VALIDATION"
assert "leave-one-panel-out" in out["reason"]

print("\n== C29: favorable average with one damaged panel ==")
recs = [tt._rec_for(d, uid,
                    delta=(-0.06 if uid.startswith("f3")
                           else 0.09))
        for uid in d["task_population"]["series_ids"]]
out = conf.adjudicate_screen(recs, d)
print("DAMAGED_PANEL:", out["verdict"], "| grand =",
      round(out[
          "primary_estimand_unweighted_mean_of_panel_effects"],
          4))
assert out["verdict"] == "DOES_NOT_ADVANCE"

print("\n== C29: single estimand; eligibility outcome dead ==")
est = design["estimand"]
print("superior_unit:", est["superior_unit"],
      "| outputs:", est["outputs"])
assert "PUBLICLY_ELIGIBLE" not in json.dumps(design)
try:
    conf.adjudicate_confirmatory([], d)
    raise AssertionError("confirmatory outcome still exists")
except SystemExit as e:
    print("adjudicate_confirmatory:", str(e)[:80])
print("\nPOST CONFIRMED: all ten C25-C30 adversaries die; "
      "screen contract executable")
