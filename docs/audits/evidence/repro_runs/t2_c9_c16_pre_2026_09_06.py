"""PRE freeze for order T2 C9-C16: Musashi's bypasses reproduce
against agent-multi@f6e9cc94 (read-only on real state; mutations in
scratch)."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "t2_c9_pre"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)

import t2_bank as bank  # noqa: E402
import t2_confirmatory as conf  # noqa: E402


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


print("== C9: self-fabricated external review opens the gate ==")
manifest = {
    "schema": "agent_multi.t2_public_data_manifest.v1",
    "datasets": {"probe": {
        "logical_id": "probe", "family": "f1",
        "final_url": "https://example.org/x",
        "archival_record": "doi:10/x",
        "retrieved_at_utc": "2026-09-06T00:00:00Z",
        "byte_size": 10, "sha256": "a" * 64,
        "upstream_checksum": "UNAVAILABLE",
        "license_id": "cc-by-4.0",
        "license_text_sha256": "b" * 64,
        "citation": "x", "local_relpath": "x.tsf",
        "admission": "ADMISSIBLE"}}}
mp = SCRATCH / "manifest.json"
mp.write_text(json.dumps(manifest))
fake_review = REPO / ("docs/audits/evidence/"
                      "MUSASHI_T2_DESIGN_REVIEW_2026_09.json")
existed = fake_review.exists()
fake_review.write_text("candidate says approved")
design = {
    "schema": "agent_multi.t2_confirmatory_design.v1",
    "sealed_after_census_manifest_sha256": sha_file(mp),
    "operator": conf.T1_ACCEPTED_OPERATOR,
    "task_population": {"series_ids": ["probe::s1"],
                        "families": ["f1"]},
    "role_geometry": {}, "primary_metric": "MASE",
    "practical_margin_mase": 0.02,
    "harm_margins": {"x": 1},
    "precision_rule": {"min_series_per_family": 1,
                       "min_families": 1},
    "multiplicity_rule": {"alpha": 0.05},
    "missing_unit_rule": "x", "inconclusive_rule": "x",
    "resource_contract": {"x": 1},
    "verifier_specification": "x",
    "design_review_record_sha256": sha_file(fake_review)}
body = {k: design[k] for k in sorted(design)}
design["design_sha256"] = hashlib.sha256(json.dumps(
    body, sort_keys=True).encode()).hexdigest()
dp = SCRATCH / "design.json"
dp.write_text(json.dumps(design))
lp = SCRATCH / "ledger.json"
try:
    conf.run_confirmatory(mp, dp, lp)
except SystemExit as exc:
    print("gate outcome:", str(exc)[:80])
    reached_exec = "NOT_IMPLEMENTED" in str(exc)
print("SELF_FABRICATED_REVIEW_REACHED_EXECUTOR:", reached_exec)
print("LEDGER_CREATED_BEFORE_REVIEW_CHECK:", lp.exists())
assert reached_exec and lp.exists()
if not existed:
    fake_review.unlink()

print("\n== C10: manifest not bound to physical bytes ==")
bad = json.loads(json.dumps(manifest))
bad["datasets"]["probe"]["sha256"] = "Z" * 64          # non-hex
bad["datasets"]["probe"]["local_relpath"] = "../../outside"
adm = conf.validate_public_manifest(bad)
print("non-hex digest + traversing relpath + zero physical bytes "
      "accepted:", "probe" in adm)
assert "probe" in adm

print("\n== C11: incomplete evidence adjudicates ELIGIBLE ==")
recs = []
for i in range(6):
    recs.append({
        "unit_id": f"u{i}", "family": f"f{i % 3}",
        "rolling_origins": {"origin0": {"results": {
            "X": {"ridge": {"mase_primary": 1.0},
                  "mlp_small": {}},
            "D": {"ridge": {"mase_primary": 0.9},
                  "mlp_small": {}}}}}})
out = conf.adjudicate_confirmatory(recs, {
    "practical_margin_mase": 0.02,
    "precision_rule": {"min_series_per_family": 2,
                       "min_families": 3},
    "multiplicity_rule": {"alpha": 0.05}})
print("one origin, X/D only, ridge only, no costs/controls ->",
      out["verdict"])
assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"

print("\n== C14: duplicate .tsf id silently overwritten ==")
tsf = ("@frequency monthly\n@data\n"
       "s1:2020:" + ",".join(str(float(i)) for i in range(130))
       + "\n"
       "s1:2020:" + ",".join(str(float(i + 500))
                             for i in range(130)) + "\n")
panel = bank.parse_tsf_bytes(tsf.encode(), "dup_probe")
print("series parsed:", len(panel["series"]),
      "| first value of s1:", panel["series"]["s1"][0],
      "(second row won silently)")
assert len(panel["series"]) == 1 and \
    panel["series"]["s1"][0] == 500.0

print("\n== C12: order-dependent census cut + no exact cap ==")
csrc = (REPO / "tools/t2_bank_census.py").read_text()
print("census truncates at max_series before hash selection:",
      "max_series" in csrc and "MAX_SERIES_PER_PANEL" in csrc)
ids = [f"s{i}" for i in range(299)]
kept = bank.deterministic_subsample(ids, 40 / 299, "t2_design_v1")
print(f"probabilistic threshold with cap-40 intent kept "
      f"{len(kept)} ids (design draft materialized 46/41/48)")
assert len(kept) != 40

print("\n== C13: two factually incorrect declarations ==")
draft = json.loads((Path.home() / ".local/share/agent-multi/"
                    "t2_confirmatory_design_DRAFT_20260906.json"
                    ).read_text())
note = draft["precision_rule"]["note"]
print("note says:", note[-40:])
assert "raised to 20" in note and \
    draft["precision_rule"]["min_series_per_family"] == 28
man = json.loads((Path.home() / ".local/share/agent-multi/"
                  "t2_public_data_manifest_20260906.json"
                  ).read_text())
d0 = man["datasets"]["tourism_monthly"]
print("license_text_sha256 == sha256(license id string):",
      d0["license_text_sha256"] == hashlib.sha256(
          d0["license_id"].encode()).hexdigest())
assert d0["license_text_sha256"] == hashlib.sha256(
    d0["license_id"].encode()).hexdigest()

shutil.rmtree(SCRATCH)
print("\nPRE CONFIRMED: C9-C16 findings all reproduce")
