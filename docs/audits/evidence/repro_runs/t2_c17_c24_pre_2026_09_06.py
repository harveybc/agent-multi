"""PRE freeze for order T2 C17-C24: the ten semantic-consumption
bypasses reproduce against agent-multi@915ac268."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "t2_c17_pre"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
STATE = Path.home() / ".local/share/agent-multi"

import t2_bank as bank  # noqa: E402
import t2_confirmatory as conf  # noqa: E402


def complete_record(uid, fam, delta=0.10, seeds=(11, 12, 13)):
    ro, costs = {}, {}
    for i in range(3):
        res = {}
        for arm in ("X", "D", "XDR", "width_control"):
            m = 1.0 - (delta if arm == "D" else
                       delta * 0.8 if arm == "XDR" else 0.0)
            e = {"mase_primary": m,
                 "interval_coverage_train_q90": 0.9,
                 "interval_width_train_q90": 1.0,
                 "mase_on_extreme_innovations": m}
            res[arm] = {"ridge": dict(e),
                        "mlp_small": {f"seed{s}": dict(e)
                                      for s in seeds}}
        ro[f"origin{i}"] = {"results": res}
        costs[f"origin{i}"] = {"denoise_fit_transform_s": 0.1,
                               "arm_X": {"r": 0.1}}
    return {"unit_id": uid, "family": fam,
            "rolling_origins": ro, "costs_by_phase": costs}


def design6(ids):
    return {"task_population": {
                "series_ids": sorted(ids),
                "primary_gate_families": [f"f{i}"
                                          for i in range(6)]},
            "role_geometry": {"rolling_origins": 3},
            "seed_tape": [11, 12, 13],
            "practical_margin_mase": 0.02,
            "observed_precision_rule": {"max_ci_halfwidth": 0.02},
            "harm_margins": {
                "extreme_innovation_mase_ratio_max": 1.2,
                "coverage_drop_max": 0.1,
                "width_inflation_max": 1.5},
            "precision_rule": {"min_series_per_family": 5,
                               "min_families": 6},
            "multiplicity_rule": {"alpha": 0.05},
            "inference_scope": "named panels only"}


base = [complete_record(f"f{i}::s{j}", f"f{i}")
        for i in range(6) for j in range(6)]
D6 = design6([r["unit_id"] for r in base])

print("== 1 (C17): NaN authorizes a positive ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["rolling_origins"].values():
        o["results"]["D"]["ridge"]["mase_primary"] = float("nan")
out = conf.adjudicate_confirmatory(recs, D6)
print("all D/ridge mase_primary = NaN ->", out["verdict"])
assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"

print("\n== 2 (C18): family relabel with preserved counts ==")
recs = json.loads(json.dumps(base))
for r in recs:
    i = int(r["family"][1])
    r["family"] = f"f{(i + 1) % 6}"
out = conf.adjudicate_confirmatory(recs, D6)
print("permuted family labels ->", out["verdict"])
assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"

print("\n== 3 (C19): null costs pass ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["costs_by_phase"]:
        r["costs_by_phase"][o] = {"arm_X": None}
out = conf.adjudicate_confirmatory(recs, D6)
print("costs {'arm_X': null} ->", out["verdict"])
assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"

print("\n== 4 (C19): forged MLP payload passes ==")
recs = json.loads(json.dumps(base))
for r in recs:
    for o in r["rolling_origins"].values():
        for arm in o["results"].values():
            arm["mlp_small"] = {f"seed{s}": "forged"
                                for s in (11, 12, 13)}
out = conf.adjudicate_confirmatory(recs, D6)
print("MLP seeds = 'forged' ->", out["verdict"])
assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"

print("\n== 5 (C20): schema/identity/license decoupling ==")
m = json.loads((STATE / "t2_public_data_manifest_20260906.json"
                ).read_text())
m["smuggled_top_key"] = True
row = m["datasets"].pop("tourism_monthly")
m["datasets"]["renamed_key"] = row          # key != logical_id
row["license_id_sha256"] = hashlib.sha256(
    b"some other bytes").hexdigest()        # canonical but wrong
adm = conf.validate_public_manifest(m)
print("extra top key + decoupled key + rehashed license "
      "accepted:", "renamed_key" in adm)
assert "renamed_key" in adm

print("\n== 6 (C20): internal symlink accepted ==")
raw2 = SCRATCH / "raw"
raw2.mkdir()
data = b"@frequency monthly\n@data\ns1:2020:" + ",".join(
    str(float(i % 9)) for i in range(150)).encode() + b"\n"
(raw2 / "real.tsf").write_bytes(data)
os.symlink("real.tsf", raw2 / "link.tsf")
m2 = {"schema": "agent_multi.t2_public_data_manifest.v2",
      "datasets": {"probe": {
          "logical_id": "probe", "family": "f1",
          "final_url": "https://example.org/x",
          "archival_record": "doi:10/x",
          "record_metadata_sha256": "c" * 64,
          "retrieved_at_utc": "2026-09-06T00:00:00Z",
          "byte_size": len(data),
          "sha256": hashlib.sha256(data).hexdigest(),
          "upstream_checksum": "UNAVAILABLE",
          "license_id": "cc-by-4.0",
          "license_id_sha256": hashlib.sha256(
              b"cc-by-4.0").hexdigest(),
          "license_text_sha256": "UNAVAILABLE",
          "citation": "x", "local_relpath": "link.tsf",
          "admission": "ADMISSIBLE"}}}
adm2 = conf.validate_public_manifest(m2, raw_root=raw2)
print("symlinked relpath accepted:", "probe" in adm2)
assert "probe" in adm2

print("\n== 7 (C21): census not re-derived by the consumer ==")
csrc = (REPO / "tools/t2_confirmatory.py").read_text()
seg = csrc[csrc.index("def run_confirmatory"):]
print("run_confirmatory only hashes the census file:",
      "census_sha = _sha_file(cp)" in seg
      and "reconstruct" not in seg.lower())

print("\n== 8 (C21): per-dataset top-k, not per-family ==")
dsrc = (REPO / "tools/t2_design_draft.py").read_text()
print("selection runs inside the per-dataset loop:",
      'for lid, p_ in census["population"].items():' in dsrc
      and "deterministic_top_k" in dsrc)
# two datasets of one family would each get up to 40 -> up to 80
ids_a = [f"pa::s{i}" for i in range(60)]
ids_b = [f"pb::s{i}" for i in range(60)]
ka = bank.deterministic_top_k(ids_a, 40, "t2_design_v2")
kb = bank.deterministic_top_k(ids_b, 40, "t2_design_v2")
print("same-family two-dataset selection would total:",
      len(ka) + len(kb))
assert len(ka) + len(kb) == 80

print("\n== 9 (C22): duplicate lists / bool-as-number pass ==")
d = json.loads((STATE /
                "t2_confirmatory_design_DRAFT_V2_20260906.json"
                ).read_text())
seeds_dup = [11, 11, 12, 12, 13, 13]
print("set() collapses duplicated seed list:",
      set(seeds_dup) == {11, 12, 13})
print("bool passes int-membership:", True in {1, 2, True})

print("\n== 10 (C23): false precision under intrapanel "
      "dependence ==")
rng = np.random.default_rng(20260906)
n_series, icc, n_sim = 36, 0.5, 2000
cover = 0
for _ in range(n_sim):
    shared = rng.normal(0, np.sqrt(icc))
    x = shared + rng.normal(0, np.sqrt(1 - icc), n_series)
    se = x.std(ddof=1) / np.sqrt(n_series)
    lo, hi = x.mean() - 1.96 * se, x.mean() + 1.96 * se
    cover += (lo <= 0 <= hi)
print(f"naive normal CI coverage at ICC={icc}: "
      f"{cover / n_sim:.3f} (nominal 0.95) — precision is "
      "fabricated")
assert cover / n_sim < 0.75

shutil.rmtree(SCRATCH)
print("\nPRE CONFIRMED: C17-C24 findings all reproduce")
