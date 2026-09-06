"""PRE freeze for order T0-T1 C13-C20: the eight residual bypasses
reproduce against T0 preprocessor@4b1d2d4e and T1 agent-multi@
8c3fb4af (v2 artifacts). Read-only on the sealed v2 state; every
mutation happens on copies in scratch."""
import copy
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
PREP = Path(os.environ.get(
    "B4_T1_PREPROCESSOR_ROOT",
    Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))
sys.path.insert(0, str(PREP))
SCRATCH = Path.home() / ".cache" / "t0t1_c13_pre"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
STATE = Path.home() / ".local/share/agent-multi"

from app import causal_operators as co  # noqa: E402

SPEC = {"schema": co.SCHEMA_VERSION, "operator_id": "pre_ewma",
        "kind": "ewma", "version": "1", "params": {"alpha": 0.5},
        "columns": ["x"], "fit_role": "train", "lookback": 0,
        "availability_rule": "bar_close"}
train = np.linspace(1.0, 2.0, 64).reshape(-1, 1)
art = co.fit(co.validate_spec(dict(SPEC)), train, ["x"],
             "train")

print("== C13: unbound state accepted and steers output ==")
honest = co.init_state(art)
tc = co.make_bar_close_contract(1)
out_h, _ = co.transform_incremental(art, honest,
                                    np.array([[2.0]]), ["x"],
                                    time_contract=tc)
forged = co.init_state(art)
forged["version"] = "foreign"
forged["rows_seen"] = 1000
forged["payload"]["ewma"] = [-1986.0]
p = co.save_state(forged, art, SCRATCH, "forged")
loaded = co.load_state(p, art)
out_f, _ = co.transform_incremental(art, loaded,
                                    np.array([[2.0]]), ["x"],
                                    time_contract=tc)
print("ACCEPTED_UNBOUND_STATE", loaded["rows_seen"],
      loaded["version"], "OUTPUT", float(out_f[0, 0]),
      "vs honest", float(out_h[0, 0]))
assert loaded["version"] == "foreign" and out_f[0, 0] < -900

print("\n== C15: understated causal semantics accepted ==")
bad_alpha = dict(SPEC, operator_id="pre_a2",
                 params={"alpha": 2.0})
co.validate_spec(bad_alpha)
print("ewma alpha=2.0 accepted:", True)
bad_look = dict(SPEC, operator_id="pre_lk", kind="trailing_mean",
                params={"window": 5}, lookback=0)
co.validate_spec(bad_look)
print("trailing window=5 with declared lookback=0 accepted:", True)
print("fit() binds no train interval/prefix digest:",
      "train_interval" not in art
      and "source_prefix_sha256" not in art)

print("\n== C14: state persistence is a pathname overwrite ==")
s1 = co.init_state(art)
p1 = co.save_state(s1, art, SCRATCH, "dur")
first = p1.read_bytes()
s1["rows_seen"] = 0
s1["payload"]["ewma"] = [1.5]
p2 = co.save_state(s1, art, SCRATCH, "dur")
print("same path silently overwritten:", p1 == p2,
      "bytes changed:", p1.read_bytes() != first)
assert p1 == p2 and p1.read_bytes() != first
src = (PREP / "app/causal_operators.py").read_text()
body = src[src.index("def save_state"):src.index("def load_state")]
print("save_state uses O_EXCL/fsync/content address:",
      any(t in body for t in ("O_EXCL", "fsync")))
assert not any(t in body for t in ("O_EXCL", "fsync"))

print("\n== C16: three-record forgery -> verifier REPRODUCED ==")
meas = json.loads((STATE /
                   "t1_measurements_v2_20260906.json").read_text())
forged_n = 0
for r in meas["records"]:
    if r["operator"] == "ewma" and \
            r["unit_id"].startswith("am__white__snr5__"):
        for pv in r["per_variable"]:
            for h in pv["assays_score_fit_train"].values():
                h["D"] = h["X"] + 0.05        # utility now favorable
                h["XDR"] = h["width_control_X_nuisance"] + 0.05
                h["residual_incremental_r2"] = 0.05
            sc = pv["by_role"]["score"]
            sc["extreme_retention"] = 0.99
            sc["tail_ratio"] = 0.5
        forged_n += 1
mut_meas = SCRATCH / "measurements_forged.json"
mut_meas.write_text(json.dumps(meas))
import t1_adjudicator as adj  # noqa: E402
design = adj._strict_json(
    REPO / "docs/audits/evidence/T1_LAB_DESIGN_V2_2026_09_06.json")
inventory = adj._strict_json(
    STATE / "t1_bank_v2_20260906/BANK_INVENTORY.json")
cand = adj.adjudicate(design, inventory, json.loads(
    mut_meas.read_text()),
    bank_dir=STATE / "t1_bank_v2_20260906",
    npz_dir=STATE / "t1_npz_v2_20260906", rederive_sample=0)
verdict = cand["verdicts"]["ewma::am|white|snr5"]["verdict"]
real = json.loads((STATE / "t1_adjudication_v2_20260906.json"
                   ).read_text())["verdicts"][
    "ewma::am|white|snr5"]["verdict"]
mut_pub = SCRATCH / "published_forged.json"
mut_pub.write_text(json.dumps(cand))
print("FORGED_RECORDS", forged_n, "FORGED_VERDICT", verdict,
      "(real:", real + ")")
design_sha = hashlib.sha256(
    (REPO / "docs/audits/evidence/T1_LAB_DESIGN_V2_2026_09_06.json"
     ).read_bytes()).hexdigest()
rc = subprocess.run(
    [sys.executable, str(REPO / "tools/t1_independent_verifier.py"),
     "--design",
     str(REPO / "docs/audits/evidence/T1_LAB_DESIGN_V2_2026_09_06"
         ".json"),
     "--design-sha", design_sha,
     "--bank-dir", str(STATE / "t1_bank_v2_20260906"),
     "--npz-dir", str(STATE / "t1_npz_v2_20260906"),
     "--measurements", str(mut_meas),
     "--published", str(mut_pub)],
    capture_output=True, text=True)
print(rc.stdout.strip()[:200])
print("VERIFIER_RC", rc.returncode)
assert real == "LAB_REJECTED" and verdict == "LAB_CALIBRATED"
assert rc.returncode == 0

print("\n== C17: inventory binds names, not unit evidence ==")
inv_raw = (STATE / "t1_bank_v2_20260906/BANK_INVENTORY.json"
           ).read_text()
inv = json.loads(inv_raw)
unit_entry = json.dumps(inv)[:2000]
has_digests = ("clean_signal_sha256" in unit_entry or
               "unit_sha256" in unit_entry or
               "arrays_sha256" in unit_entry)
print("inventory carries per-unit array digests:", has_digests)
assert not has_digests
bank_copy = SCRATCH / "bank_replaced"
shutil.copytree(STATE / "t1_bank_v2_20260906", bank_copy)
u = bank_copy / "am__white__snr5__hom__seed11"
clean = np.load(u / "clean_signal.npy")
np.save(u / "clean_signal.npy", clean * 0.5)   # physical replacement
same_inventory = (bank_copy / "BANK_INVENTORY.json"
                  ).read_text() == inv_raw
print("arrays replaced while sealed inventory unchanged:",
      same_inventory)
assert same_inventory

print("\n== C18: coherent measurement+publication pair accepted ==")
print("verifier consumes caller-supplied measurement/publication "
      "paths with no externally reviewed digest:",
      "--measurements" in
      (REPO / "tools/t1_independent_verifier.py").read_text())
print("-> demonstrated live by the C16 probe: forged pair returned "
      "REPRODUCED rc 0")

print("\n== C19: circular-shift nuisance crosses role boundaries ==")
lab = (REPO / "tools/t1_lab_run.py").read_text()
print("width control uses np.roll:", "np.roll" in lab)
assert "np.roll" in lab
n = 2048
series = np.arange(n, dtype=float)      # value == time index
shift = 517
rolled = np.roll(series, shift)
train_end = 1024
leaked = rolled[:shift]
print("first", shift, "train positions hold values from indices",
      int(leaked.min()), "-", int(leaked.max()),
      "(the series END, i.e. validation/score rows)")
assert leaked.min() >= n - shift

print("\n== C20: unknown/mistyped nested fields reach "
      "adjudication ==")
meas2 = json.loads((STATE /
                    "t1_measurements_v2_20260906.json").read_text())
for r in meas2["records"]:
    if r["status"] == "MEASURED":
        r["attacker_note"] = "smuggled"
        r["per_variable"][0]["extra_nested"] = {"x": float("nan")}
        break
mut2 = SCRATCH / "measurements_extra.json"
mut2.write_text(json.dumps(meas2).replace("NaN", "1e999"))
out2 = adj.adjudicate(design, inventory,
                      json.loads(mut2.read_text()),
                      bank_dir=STATE / "t1_bank_v2_20260906",
                      npz_dir=STATE / "t1_npz_v2_20260906",
                      rederive_sample=0)
print("adjudication completed over smuggled/mistyped nested "
      "fields:", out2["verdict_counts"])

shutil.rmtree(SCRATCH)
print("\nPRE CONFIRMED: C13-C20 findings all reproduce")
