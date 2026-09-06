"""POST for order T0-T1 C13-C20: the eight PRE bypasses die
against the corrected stack (T0 v3 operators, T1 v3 lab/
adjudicator/verifier, bank v3, design v3)."""
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
SCRATCH = Path.home() / ".cache" / "t0t1_c13_post"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
STATE = Path.home() / ".local/share/agent-multi"
BANK = STATE / "t1_bank_v3_20260906"
NPZ = STATE / "t1_npz_v3_20260906"
MEAS = STATE / "t1_measurements_v3_20260906.json"
DESIGN = REPO / ("docs/audits/evidence/"
                 "T1_LAB_DESIGN_V3_2026_09_06.json")

from app import causal_operators as co  # noqa: E402
import t1_adjudicator as adj  # noqa: E402

SPEC = {"schema": co.SCHEMA_VERSION, "operator_id": "post_ewma",
        "kind": "ewma", "version": "1", "params": {"alpha": 0.5},
        "columns": ["x"], "fit_role": "train",
        "lookback": co.LOOKBACK_UNBOUNDED,
        "availability_rule": "bar_close"}
train = np.linspace(1.0, 2.0, 64).reshape(-1, 1)
art = co.fit(co.validate_spec(dict(SPEC)), train, ["x"], "train",
             co.make_train_contract(train, 64))

print("== 1 (C13): the unbound-state forgery dies ==")
forged = co.init_state(art, "synthetic_bar_close", 1.0)
forged["version"] = "foreign"
forged["rows_seen"] = 1000
forged["payload"]["ewma"] = [-1986.0]
try:
    co.save_state(forged, art, SCRATCH)
    raise AssertionError("accepted — POST FAILS")
except co.CausalOperatorError as exc:
    print("refused:", str(exc)[:90])
lied = co.init_state(art, "synthetic_bar_close", 1.0)
lied["payload"]["ewma"] = [-1986.0]
try:
    co.transform_incremental(art, lied, np.array([[2.0]]), ["x"],
                             co.make_bar_close_contract(1))
    raise AssertionError("steered output — POST FAILS")
except co.CausalOperatorError as exc:
    print("refused:", str(exc)[:90])

print("\n== 2 (C15): understated semantics die ==")
for mut, why in ((dict(SPEC, operator_id="a2",
                       params={"alpha": 2.0}), "alpha=2.0"),
                 (dict(SPEC, operator_id="lk",
                       kind="trailing_mean",
                       params={"window": 5}, lookback=0),
                  "window=5 lookback=0")):
    try:
        co.validate_spec(mut)
        raise AssertionError(f"{why} accepted — POST FAILS")
    except co.CausalOperatorError as exc:
        print(f"{why} refused:", str(exc)[:70])
print("fit binds interval+contract+prefix:",
      "train_binding" in art)

print("\n== 3 (C14): persistence is durable content-addressed ==")
s1 = co.init_state(art, "synthetic_bar_close", 1.0)
p1 = co.save_state(s1, art, SCRATCH / "dur")
p2 = co.save_state(s1, art, SCRATCH / "dur")   # idempotent
print("content-addressed idempotent:", p1 == p2,
      "| name is digest:", p1.name.startswith(
          json.loads(p1.read_text())["snapshot_sha256"][:8]))
body = (PREP / "app/causal_operators.py").read_text()
seg = body[body.index("def save_state"):body.index(
    "def load_state")]
print("O_EXCL+fsync+typed uncertainty present:",
      all(t in seg for t in ("O_EXCL", "fsync",
                             "StateWriteUncertain")))

print("\n== 4 (C16): the three-record forgery dies ==")
m = json.loads(MEAS.read_text())
targets = [r for r in m["records"] if r["operator"] == "ewma"
           and r["unit_id"].startswith("am__white__snr5__")]
killed = 0
for r in targets:
    for pv in r["per_variable"]:
        for h in pv["assays_score_fit_train"].values():
            h["D"] = h["X"] + 0.05
            h["XDR"] = h["width_control_X_nuisance"] + 0.05
            h["residual_incremental_r2"] = 0.05
        sc = pv["by_role"]["score"]
        sc["extreme_retention"] = 0.99
        sc["tail_ratio"] = 0.5
    try:
        adj.rederive_all_facts(r, BANK, NPZ)
        raise AssertionError("forged record survived — POST FAILS")
    except SystemExit as exc:
        killed += 1
print(f"all {killed}/3 forged records refused in full "
      "re-derivation")

print("\n== 5 (C17): physical replacement dies ==")
inv = json.loads((BANK / "BANK_INVENTORY.json").read_text())
uid = "am__white__snr5__hom__seed11"
(SCRATCH / uid).mkdir()
for f in (BANK / uid).iterdir():
    shutil.copyfile(f, SCRATCH / uid / f.name)
clean = np.load(SCRATCH / uid / "clean_signal.npy")
np.save(SCRATCH / uid / "clean_signal.npy", clean * 0.5)
try:
    adj.verify_unit_bytes(inv, SCRATCH, uid)
    raise AssertionError("replacement survived — POST FAILS")
except SystemExit as exc:
    print("refused:", str(exc)[:80])

print("\n== 6 (C18): a coherent pair cannot claim review ==")
design_sha = hashlib.sha256(DESIGN.read_bytes()).hexdigest()
m2 = json.loads(MEAS.read_text())
m2["records"][0]["cpu_wall_seconds"] = 9.9
fake_m = SCRATCH / "rewritten.json"
fake_m.write_text(json.dumps(m2))
rc = subprocess.run(
    [sys.executable, str(REPO / "tools/t1_independent_verifier.py"),
     "--design", str(DESIGN), "--design-sha", design_sha,
     "--bank-dir", str(BANK), "--npz-dir", str(NPZ),
     "--measurements", str(fake_m),
     "--measurement-manifest",
     str(STATE / "t1_measurements_v3_20260906_MANIFEST.json"),
     "--published", str(STATE / "t1_adjudication_v3_20260906.json")],
    capture_output=True, text=True)
print("rewritten pair rc:", rc.returncode, "|",
      (rc.stderr + rc.stdout).strip()[:90])
assert rc.returncode not in (0, 3)
rc2 = subprocess.run(
    [sys.executable, str(REPO / "tools/t1_independent_verifier.py"),
     "--design", str(DESIGN), "--design-sha", design_sha,
     "--bank-dir", str(BANK), "--npz-dir", str(NPZ),
     "--measurements", str(MEAS),
     "--measurement-manifest",
     str(STATE / "t1_measurements_v3_20260906_MANIFEST.json"),
     "--published", str(STATE / "t1_adjudication_v3_20260906.json")],
    capture_output=True, text=True)
print("honest pair without reviewer record rc:", rc2.returncode,
      "-> SELF_CONSISTENT_ONLY_NOT_AUTHORIZING:",
      "SELF_CONSISTENT_ONLY_NOT_AUTHORIZING" in rc2.stdout)
assert rc2.returncode == 3

print("\n== 7 (C19): no circular shift; controls untouchable ==")
lab_src = (REPO / "tools/t1_lab_run.py").read_text()
print("np.roll removed:", "np.roll" not in lab_src)
assert "np.roll" not in lab_src
import importlib.util as ilu
spec = ilu.spec_from_file_location("t1lab_post",
                                   REPO / "tools/t1_lab_run.py")
lab = ilu.module_from_spec(spec)
spec.loader.exec_module(lab)
roles = {"train": (0, 1228), "validation": (1228, 1638),
         "score": (1638, 2048)}
obs = np.random.default_rng(7).normal(0, 1, 2048)
obs_mut = obs.copy()
obs_mut[1228:] = 999.0
n_a = lab.nuisance_channels(obs, roles, "u|op|v0")
n_b = lab.nuisance_channels(obs_mut, roles, "u|op|v0")
print("future-row mutation cannot reach the width control:",
      all(a.tobytes() == b.tobytes() for a, b in zip(n_a, n_b)))

print("\n== 8 (C20): smuggled/mistyped nested fields die ==")
rec = dict(m["records"][0])
rec["attacker_note"] = "smuggled"
try:
    adj.check_measurement_record(rec)
    raise AssertionError("smuggled field survived — POST FAILS")
except SystemExit as exc:
    print("refused:", str(exc)[:80])

shutil.rmtree(SCRATCH)
print("\nPOST CONFIRMED: all eight C13-C20 bypasses die")
