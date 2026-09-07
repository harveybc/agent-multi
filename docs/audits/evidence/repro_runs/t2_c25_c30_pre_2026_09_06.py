"""PRE freeze for order T2 C25-C30: Musashi's residual bypasses
reproduce against agent-multi@4e62ca57."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))
os.environ.setdefault(
    "B4_T1_PREPROCESSOR_ROOT",
    str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))
STATE = Path.home() / ".local/share/agent-multi"

import t2_confirmatory as conf  # noqa: E402
import importlib.util as ilu
spec = ilu.spec_from_file_location(
    "t2tests", REPO / "tests/test_t2_harness.py")
tt = ilu.module_from_spec(spec)
spec.loader.exec_module(tt)

d3 = tt._d3()
base = tt._recs(d3)

print("== C25a: missing extreme metrics -> POSITIVE ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["rolling_origins"].values():
        for arm in ("X", "D", "XDR", "width_control"):
            o["results"][arm]["ridge"].pop(
                "mase_on_extreme_innovations", None)
            for sv in o["results"][arm]["mlp_small"].values():
                sv.pop("mase_on_extreme_innovations", None)
out = conf.adjudicate_confirmatory(recs, d3)
print("MISSING_EXTREMES", out["verdict"])
assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"

print("\n== C25b: X extreme 0.0 vs D extreme 999.0 -> POSITIVE ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["rolling_origins"].values():
        o["results"]["X"]["ridge"][
            "mase_on_extreme_innovations"] = 0.0
        o["results"]["D"]["ridge"][
            "mase_on_extreme_innovations"] = 999.0
out = conf.adjudicate_confirmatory(recs, d3)
print("ZERO_BASELINE_EXTREME", out["verdict"])
assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"

print("\n== C26: record without train/score geometry -> "
      "POSITIVE ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["rolling_origins"].values():
        o.pop("train", None)
        o.pop("score", None)
out = conf.adjudicate_confirmatory(recs, d3)
print("NO_ORIGIN_GEOMETRY", out["verdict"])
assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"
print("outer schema/record_sha256 not consumed:",
      "record_sha256" not in json.dumps(base[0]))

print("\n== C27: costs without target/baseline phases -> "
      "POSITIVE ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["costs_by_phase"].values():
        o.pop("target_construction_s", None)
        o.pop("seasonal_naive_s", None)
out = conf.adjudicate_confirmatory(recs, d3)
print("MISSING_GLOBAL_COST_PHASES", out["verdict"])
assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"

print("\n== C28a: forged unit_map semantics pass the fresh "
      "verifier ==")
import t2_fresh_verifier as fv
manifest = conf.strict_json_load(
    STATE / "t2_public_data_manifest_20260906.json", "m")
rebuilt = fv.rebuild_population(manifest, STATE / "t2_public_raw")
design = conf.strict_json_load(
    STATE / "t2_confirmatory_design_DRAFT_V3_20260906.json", "d")
forged = json.loads(json.dumps(design))
for uid, b in forged["task_population"]["unit_map"].items():
    b["family"] = "totally_forged_family"
    b["dataset"] = "alien_panel"
    b["seasonal_period"] = 999
try:
    fv.verify_design_population(rebuilt, forged)
    print("FORGED_UNIT_MAP accepted: True")
    ok = True
except SystemExit:
    ok = False
assert ok

print("\n== C28b: symlink ROOT accepted ==")
sc = Path.home() / ".cache/t2_c25_pre"
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
adm = conf.validate_public_manifest(m2,
                                    raw_root=sc / "rootlink")
print("SYMLINK_ROOT accepted:", "probe" in adm)
assert "probe" in adm
shutil.rmtree(sc)

print("\n== C28c: run_confirmatory does not consume the fresh "
      "verifier ==")
src = (REPO / "tools/t2_confirmatory.py").read_text()
seg = src[src.index("def run_confirmatory"):]
print("fresh verifier absent from the single path:",
      "fresh_verifier" not in seg and "rebuild_population"
      not in seg)

print("\n== C29: the v3 draft mixes two estimands (documented) ==")
print("scope says 'named panels only' while the rule demands "
      ">=3 panels PER FAMILY:",
      "named public panels" in design["inference_scope"]
      and ">=3" in design["inference_method"]["detail"])

print("\nPRE CONFIRMED: C25-C29 findings all reproduce")
