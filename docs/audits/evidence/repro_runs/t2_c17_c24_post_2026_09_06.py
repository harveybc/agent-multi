"""POST for order T2 C17-C24: the ten bypasses die against the
corrected semantic-consumption stack."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))
STATE = Path.home() / ".local/share/agent-multi"

import t2_bank as bank  # noqa: E402
import t2_confirmatory as conf  # noqa: E402

import importlib.util as ilu
spec = ilu.spec_from_file_location(
    "t2tests", REPO / "tests/test_t2_harness.py")
tt = ilu.module_from_spec(spec)
os.environ.setdefault(
    "B4_T1_PREPROCESSOR_ROOT",
    str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))
spec.loader.exec_module(tt)

d3 = tt._d3()
recs = tt._recs(d3)

print("== 1: NaN refuses with the exact path ==")
r = json.loads(json.dumps(recs))
r[0]["rolling_origins"]["origin0"]["results"]["D"]["ridge"][
    "mase_primary"] = float("nan")
try:
    conf.adjudicate_confirmatory(r, d3)
    raise AssertionError("POST FAILS")
except SystemExit as exc:
    print("refused:", str(exc)[:80])

print("\n== 2: family relabel refuses ==")
r = json.loads(json.dumps(recs))
for x in r:
    i = int(x["family"][1])
    x["family"] = f"f{(i + 1) % 6}"
try:
    conf.adjudicate_confirmatory(r, d3)
    raise AssertionError("POST FAILS")
except SystemExit as exc:
    print("refused:", str(exc)[:80])

print("\n== 3/4: null costs and forged MLP refuse ==")
r = json.loads(json.dumps(recs))
for x in r:
    for o in x["costs_by_phase"]:
        x["costs_by_phase"][o] = {"arm_X": None}
try:
    conf.adjudicate_confirmatory(r, d3)
    raise AssertionError("POST FAILS")
except SystemExit as exc:
    print("costs refused:", str(exc)[:60])
r = json.loads(json.dumps(recs))
for x in r:
    for o in x["rolling_origins"].values():
        for arm in ("X", "D", "XDR", "width_control"):
            o["results"][arm]["mlp_small"] = {
                f"seed{s}": "forged" for s in (11, 12, 13)}
try:
    conf.adjudicate_confirmatory(r, d3)
    raise AssertionError("POST FAILS")
except SystemExit as exc:
    print("forged MLP refused:", str(exc)[:60])

print("\n== 5/6: manifest schema/identity/license + symlink ==")
m = json.loads((STATE / "t2_public_data_manifest_20260906.json"
                ).read_text())
m["smuggled"] = True
try:
    conf.validate_public_manifest(m)
    raise AssertionError("POST FAILS")
except SystemExit as exc:
    print("extra top key refused:", str(exc)[:60])
sc = Path.home() / ".cache/t2_c17_post_raw"
if sc.exists():
    shutil.rmtree(sc)
sc.mkdir(parents=True)
data = b"@frequency monthly\n@data\ns1:2020:" + ",".join(
    str(float(i % 9)) for i in range(150)).encode() + b"\n"
(sc / "real.tsf").write_bytes(data)
os.symlink("real.tsf", sc / "link.tsf")
row = {"logical_id": "probe", "family": "f1",
       "final_url": "https://example.org/x",
       "archival_record": "doi:10/x",
       "record_metadata_sha256": hashlib.sha256(
           b"doi:10/x").hexdigest(),
       "retrieved_at_utc": "2026-09-06T00:00:00Z",
       "byte_size": len(data),
       "sha256": hashlib.sha256(data).hexdigest(),
       "upstream_checksum": "UNAVAILABLE",
       "license_id": "cc-by-4.0",
       "license_id_sha256": hashlib.sha256(
           b"cc-by-4.0").hexdigest(),
       "license_text_sha256": "UNAVAILABLE",
       "citation": "x", "local_relpath": "link.tsf",
       "admission": "ADMISSIBLE"}
m2 = {"schema": "agent_multi.t2_public_data_manifest.v2",
      "acquired_at_utc": "x", "remanifested_at_utc": "x",
      "byte_cap": 2 << 30, "bytes_downloaded_total": len(data),
      "raw_root_note": "x", "etth1_disposition": "x",
      "datasets": {"probe": row}}
try:
    conf.validate_public_manifest(m2, raw_root=sc)
    raise AssertionError("POST FAILS")
except SystemExit as exc:
    print("internal symlink refused:", str(exc)[:60])
shutil.rmtree(sc)

print("\n== 7: fresh verifier kills a non-derivable census ==")
import t2_fresh_verifier as fv
manifest = conf.strict_json_load(
    STATE / "t2_public_data_manifest_20260906.json", "m")
rebuilt = fv.rebuild_population(manifest,
                                STATE / "t2_public_raw")
census = conf.strict_json_load(
    STATE / "t2_bank_census_20260906.json", "c")
fv.verify_census_semantic(rebuilt, census)
print("honest census re-derives from bytes: True")
forged = json.loads(json.dumps(census))
forged["population"]["hospital"]["unit_numeric_digests"] = {
    k: "f" * 64 for k in
    forged["population"]["hospital"]["unit_numeric_digests"]}
try:
    fv.verify_census_semantic(rebuilt, forged)
    raise AssertionError("POST FAILS")
except SystemExit as exc:
    print("forged census refused:", str(exc)[:60])

print("\n== 8: family cap is global and exact ==")
sel = bank.family_top_k(
    {"pa": [f"pa::s{i}" for i in range(60)],
     "pb": [f"pb::s{i}" for i in range(60)]}, 40, "t2_design_v3")
print("two same-family panels select:", len(sel), "(cap 40)")
assert len(sel) == 40

print("\n== 9: duplicates and bool die in the design validator ==")
try:
    conf._unique_list([11, 11, 12], "seed_tape", elem_type=int)
    raise AssertionError("POST FAILS")
except SystemExit as exc:
    print("refused:", str(exc)[:50])
try:
    conf._unique_list([11, True, 12], "seed_tape", elem_type=int)
    raise AssertionError("POST FAILS")
except SystemExit as exc:
    print("refused:", str(exc)[:50])

print("\n== 10: no fabricated precision ==")
d1 = tt._d3(panels=1)
out = conf.adjudicate_confirmatory(tt._recs(d1), d1)
print("single panel per family ->", out["verdict"])
assert out["verdict"] == "INCONCLUSIVE"
sim = json.loads((STATE / "t2_coverage_sim_20260906.json"
                  ).read_text())
row = [r for r in sim["rows"] if r["icc_true"] == 0.5][0]
print("ICC=0.5: naive", row["naive_single_panel_coverage"],
      "| within-panel ICC", row["within_panel_icc_coverage"],
      "| panel-level (t, K=4)", row["panel_level_coverage"])
assert row["panel_level_coverage"] >= 0.93

print("\nPOST CONFIRMED: C17-C24 findings no longer reproduce")
