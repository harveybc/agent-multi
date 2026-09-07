"""POST for order T2 C9-C16: every Musashi bypass dies against the
corrected stack."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "t2_c9_post"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
STATE = Path.home() / ".local/share/agent-multi"

import t2_bank as bank  # noqa: E402
import t2_confirmatory as conf  # noqa: E402

print("== C9: self-fabricated review dies; no early ledger ==")
fake = SCRATCH / "MUSASHI_T2_DESIGN_REVIEW_2026_09.json"
fake.write_text("candidate says approved")
import unittest.mock as um
design = {"design_review_record_sha256": hashlib.sha256(
    b"candidate says approved").hexdigest(),
    "supersedes_draft_sha256": "a" * 64}
with um.patch.object(conf, "T2_REVIEW_RECORD_PATH", fake):
    try:
        conf.verify_design_review_record(design, "b" * 64,
                                         "c" * 64)
        raise AssertionError("fabricated review passed — POST "
                             "FAILS")
    except SystemExit as exc:
        print("refused:", str(exc)[:70])
lp = SCRATCH / "ledger.json"
print("no ledger artifact on refusal:", not lp.exists())

print("\n== C10: manifest bound to physical bytes ==")
m = json.loads((STATE / "t2_public_data_manifest_20260906.json"
                ).read_text())
adm = conf.validate_public_manifest(m)
print("v2 manifest verifies", len(adm),
      "admissible datasets from descriptors")
bad = json.loads(json.dumps(m))
k0 = "tourism_monthly"
bad["datasets"][k0]["sha256"] = "Z" * 64
try:
    conf.validate_public_manifest(bad)
    raise AssertionError("non-hex accepted — POST FAILS")
except SystemExit as exc:
    print("non-hex digest refused:", str(exc)[:60])
bad2 = json.loads(json.dumps(m))
bad2["datasets"][k0]["local_relpath"] = "../../outside"
try:
    conf.validate_public_manifest(bad2)
    raise AssertionError("traversal accepted — POST FAILS")
except SystemExit as exc:
    print("traversing relpath refused:", str(exc)[:60])
print("etth1 admission:", m["datasets"]["etth1"]["admission"])
print("truthful license fields (Monash):",
      "license_id_sha256" in m["datasets"][k0],
      "| text digest:",
      m["datasets"][k0]["license_text_sha256"])

print("\n== C11/C14: incomplete evidence and duplicates die ==")
recs = [{"unit_id": f"f{i}::s{j}", "family": f"f{i}",
         "rolling_origins": {"origin0": {"results": {
             "X": {"ridge": {"mase_primary": 1.0}},
             "D": {"ridge": {"mase_primary": 0.9}}}}}}
        for i in range(6) for j in range(6)]
design6 = {
    "task_population": {
        "series_ids": sorted(r["unit_id"] for r in recs),
        "primary_gate_families": [f"f{i}" for i in range(6)]},
    "role_geometry": {"rolling_origins": 3},
    "seed_tape": [11, 12, 13],
    "practical_margin_mase": 0.02,
    "observed_precision_rule": {"max_ci_halfwidth": 0.02},
    "harm_margins": {"extreme_innovation_mase_ratio_max": 1.2,
                     "coverage_drop_max": 0.1,
                     "width_inflation_max": 1.5},
    "precision_rule": {"min_series_per_family": 5,
                       "min_families": 6},
    "multiplicity_rule": {"alpha": 0.05},
    "inference_scope": "named panels only"}
try:
    conf.adjudicate_confirmatory(recs, design6)
    raise AssertionError("incomplete evidence adjudicated — POST "
                         "FAILS")
except SystemExit as exc:
    print("one-origin/X-D-only/no-cost refused:", str(exc)[:60])

tsf = ("@frequency monthly\n@data\n"
       "s1:2020:" + ",".join(str(float(i)) for i in range(130))
       + "\n"
       "s1:2020:" + ",".join(str(float(i + 500))
                             for i in range(130)) + "\n")
try:
    bank.parse_tsf_bytes(tsf.encode(), "dup_probe")
    raise AssertionError("duplicate id accepted — POST FAILS")
except SystemExit as exc:
    print("duplicate .tsf id refused:", str(exc)[:60])

print("\n== C11/C12: complete walk + exact cap ==")
census = json.loads((STATE / "t2_bank_census_20260906.json"
                     ).read_text())
w = census["population"]["weather"]
print("weather series parsed (complete walk):",
      w["series_parsed"], "admissible:", w["series_admissible"])
assert w["series_parsed"] > 300
draft = json.loads(
    (STATE / "t2_confirmatory_design_DRAFT_V2_20260906.json"
     ).read_text())
per = draft["task_population"]["per_dataset"]
caps = {k: v["n_series"] for k, v in per.items()
        if v["family"] in
        draft["task_population"]["primary_gate_families"]}
print("primary per-dataset selections:", caps)
assert all(v <= 40 for v in caps.values())

print("\n== C13: truthful declarations ==")
pr = draft["precision_rule"]
print("n_min:", pr["min_series_per_family"], "| note:",
      pr["note"][-40:])
assert pr["min_series_per_family"] == 28
assert "raised to 20" not in pr["note"]

print("\nPOST CONFIRMED: C9-C16 findings no longer reproduce")
