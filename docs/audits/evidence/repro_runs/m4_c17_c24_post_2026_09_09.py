"""POST for order M4 C17-C24: the foundation exists, every
adversary refuses against the corrected verifier, and each guard
surface BITES under mutation.

Phase 1 (corrected, in-process, reduced sealed-rule world):
  - forged screen metric           -> does not replay
  - relabeled learnability outcome -> does not re-derive
  - forged censoring/stopping fact -> facts do not equal replay
  - CONFIRMATION-role unit in the pre-result ledger
                                   -> population equality refuses
  - random-label licensing         -> margin rises, never passes
Phase 2 (subprocess, one mutant runner each):
  A. screen replay-equality OFF    -> forged metric verifies
  B. table re-derivation OFF       -> relabel verifies
  C. arm facts-equality OFF        -> censoring forgery verifies
  D. ledger row-equality OFF       -> foreign role verifies
  E. generator bank train-only scaling broken (train_slice ->
     full series) -> the battery's future-leak assertion FAILS,
     proving that guard bites.

CPU only, tmp roots, DEVELOPMENT only; the committed
m4_development_run_20260909 evidence is untouched."""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CHILD = os.environ.get("M4_C17_POST_CHILD")


def small_design(dz, m4):
    d = copy.deepcopy(dz.load_design_v4())
    cp = d["candidate_population"]
    cp["structured_boolean_families"] = ["identity", "parity4"]
    cp["temporal_families"] = ["sine"]
    cp["noise_regimes_temporal"] = ["clean", "white"]
    cp["hidden_widths"] = [16]
    d["population_census"][
        "development_generators_per_cell"] = 1
    d["four_unit_rule"]["units"] = [
        {"family": "identity", "noise": "clean", "width": 16,
         "generator_index": 0, "model_seed": 0}]
    del d["design_sha256"]
    d["design_sha256"] = m4._self_sha(d, "design_sha256")
    return d


def adversary(rn, m4, out, name):
    def repair_report():
        p = out / "RUN_REPORT.json"
        rep = json.loads(p.read_text())
        rep["artifacts_sha256"] = rn._inventory(out)
        rep.pop("record_sha256")
        rep["record_sha256"] = m4._self_sha(rep, "record_sha256")
        p.write_text(json.dumps(rep, indent=1, sort_keys=True))
    if name == "metric_forge":
        p = sorted((out / "screen").glob("*identity*"))[0]
        doc = json.loads(p.read_text())
        doc["metric_heldout"] = 0.123456
        doc.pop("record_sha256")
        doc["record_sha256"] = m4._self_sha(doc, "record_sha256")
        p.write_text(json.dumps(doc, indent=1, sort_keys=True))
    elif name == "relabel":
        p = out / "LEARNABILITY_TABLE.json"
        tab = json.loads(p.read_text())
        k = next(iter(tab["cells"]))
        cur = tab["cells"][k]["outcome"]
        tab["cells"][k]["outcome"] = (
            "OPTIMIZATION_LIMITED"
            if cur == "LEARNABLE_UNDER_FROZEN_BUDGET"
            else "LEARNABLE_UNDER_FROZEN_BUDGET")
        tab.pop("record_sha256")
        tab["record_sha256"] = m4._self_sha(tab, "record_sha256")
        p.write_text(json.dumps(tab, indent=1, sort_keys=True))
    elif name == "censor_forge":
        p = next((out / "intervention").glob("*_summary.json"))
        doc = json.loads(p.read_text())
        doc["arms"]["control"]["stopping_cause"] = "MAX_BATCHES"
        doc["arms"]["control"]["censored"] = True
        doc.pop("record_sha256")
        doc["record_sha256"] = m4._self_sha(doc, "record_sha256")
        p.write_text(json.dumps(doc, indent=1, sort_keys=True))
    elif name == "foreign_role":
        p = out / "RUN_LEDGER.json"
        led = json.loads(p.read_text())
        led["units"][0]["generator_id"] = \
            led["units"][0]["generator_id"].replace(
                "DEVELOPMENT", "CONFIRMATION")
        led.pop("record_sha256")
        led["record_sha256"] = m4._self_sha(led, "record_sha256")
        p.write_text(json.dumps(led, indent=1, sort_keys=True))
    else:
        raise AssertionError(name)
    repair_report()


def run_one(rn, dz, m4, name):
    tmp = Path(tempfile.mkdtemp(prefix="m4c17_post_"))
    try:
        d = small_design(dz, m4)
        out = tmp / "run"
        rn.execute(d, out)
        adversary(rn, m4, out, name)
        try:
            v = rn.verify_run(d, out)
            return {"adversary": name, "result": "VERIFIED",
                    "verified": v["verified"]}
        except SystemExit as exc:
            return {"adversary": name, "result": "REFUSED",
                    "reason": str(exc)[:130]}
    finally:
        shutil.rmtree(tmp)


if CHILD:
    tools_dir = Path(os.environ["M4_C17_TOOLS_DIR"])
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(tools_dir))
    # the mutated module must be imported FIRST — the design and
    # runner modules prepend the real tools path when they load
    if CHILD == "bank_leak":
        import numpy as np
        import m4_generator_bank as gb
        assert Path(gb.__file__).parent == tools_dir
        rng = np.random.default_rng(7)
        latent = np.concatenate([np.ones(200) * 0.5,
                                 np.ones(64) * 50.0])
        tr = slice(0, 200)
        dd = gb._disturbance("white", rng, latent, tr)
        expected = 0.1 * float(np.std(latent[tr]) + 1e-12)
        observed = float(np.std(dd[tr]))
        leaked = not (observed <
                      0.1 * float(np.std(latent)) / 10)
        print(json.dumps({"adversary": "bank_leak",
                          "result": "LEAKED" if leaked
                          else "TRAIN_ONLY",
                          "observed": round(observed, 6),
                          "train_only_expected":
                              round(expected, 6)}))
        sys.exit(0)
    else:
        import m4_intervention_runner as rn
        import m4_residual_capacity as m4
        import m4_intervention_design as dz
        assert Path(rn.__file__).parent == tools_dir
        print(json.dumps(run_one(rn, dz, m4, CHILD)))
        sys.exit(0)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import m4_residual_capacity as m4  # noqa: E402
import m4_intervention_design as dz  # noqa: E402
import m4_intervention_runner as rn  # noqa: E402
import m4_generator_bank as gb  # noqa: E402

# ---- Phase 1: every adversary refuses on the corrected code ----
for adv, needle in (
        ("metric_forge", "does not replay"),
        ("relabel", "does not re-derive"),
        ("censor_forge", "do not equal the replayed|does not "
                         "reconstruct"),
        ("foreign_role", "does not enumerate the sealed "
                         "population")):
    r = run_one(rn, dz, m4, adv)
    print("corrected:", json.dumps(r))
    import re
    assert r["result"] == "REFUSED" and \
        re.search(needle, r["reason"]), r

# random-label licensing: margin monotonicity (pure function)
d_small = small_design(dz, m4)
recs = [{"family": "random_label", "noise_coord": "clean",
         "unit_id": f"screen::random_label::clean::w16::g{i}",
         "improvement": 0.4, "numerically_invalid": False}
        for i in range(2)] + \
    [{"family": "identity", "noise_coord": "clean",
      "unit_id": "screen::identity::clean::w16::g0",
      "improvement": 0.3, "numerically_invalid": False}]
tab = rn.derive_learnability_table(d_small, recs)
print("corrected: random_label licensing ->",
      tab["cells"]["screen::identity::clean::w16::g0"]["outcome"],
      "margin", tab["margin"])
assert tab["margin"] >= 0.4
assert tab["cells"]["screen::identity::clean::w16::g0"][
    "outcome"] == "OPTIMIZATION_LIMITED"

# ---- Phase 2: one mutant per guard ----
SRC = (REPO / "tools/m4_intervention_runner.py").read_text()
GSRC = (REPO / "tools/m4_generator_bank.py").read_text()
MUTANTS = {
    "A_screen_replay_off": (
        "m4_intervention_runner.py", SRC,
        [("        if cmp_f != cmp_r:",
          "        if False and cmp_f != cmp_r:")],
        "metric_forge"),
    "B_table_rederive_off": (
        "m4_intervention_runner.py", SRC,
        [("    if fresh_tab != tab:",
          "    if False and fresh_tab != tab:")],
        "relabel"),
    "C_arm_facts_off": (
        "m4_intervention_runner.py", SRC,
        [(('            if a["endpoint_last_passing"] != '
           'endpoint or \\\n'
           '                    a["censored"] is not censored '
           'or \\\n'
           '                    a["stopping_cause"] != cause '
           'or \\\n'
           '                    a["updates_done"] != '
           'st["updates_done"]:'),
          "            if False:")],
        "censor_forge"),
    "D_ledger_rows_off": (
        "m4_intervention_runner.py", SRC,
        [("    if led[\"units\"] != want_rows:",
          "    if False and led[\"units\"] != want_rows:")],
        "foreign_role"),
    "E_bank_future_leak": (
        "m4_generator_bank.py", GSRC,
        [("    base = 0.1 * float(np.std(latent[train_slice])"
          " + 1e-12)",
          "    base = 0.1 * float(np.std(latent) + 1e-12)")],
        "bank_leak"),
}
TMP = Path(tempfile.mkdtemp(prefix="m4c17_mut_"))
try:
    for mname, (fname, base_src, subs, adv) in MUTANTS.items():
        mut = base_src
        for old, new in subs:
            assert old in mut, (mname, old)
            mut = mut.replace(old, new)
        mdir = TMP / mname
        mdir.mkdir()
        (mdir / fname).write_text(mut)
        env = {**os.environ, "M4_C17_POST_CHILD": adv,
               "M4_C17_TOOLS_DIR": str(mdir)}
        rc = subprocess.run([sys.executable, __file__],
                            capture_output=True, text=True,
                            env=env)
        assert rc.returncode == 0, rc.stderr[-400:]
        r = json.loads(rc.stdout.strip().splitlines()[-1])
        print(f"mutant {mname}:", json.dumps(r))
        if adv == "bank_leak":
            assert r["result"] == "LEAKED", \
                "mutant E must leak the future scale"
        else:
            assert r["result"] == "VERIFIED", \
                f"mutant {mname} must accept its adversary"
finally:
    shutil.rmtree(TMP)

print("\nPOST CONFIRMED: population/ledger, screen replay, "
      "table re-derivation, censoring facts and train-only "
      "scaling all refuse their adversaries on the corrected "
      "code, and each guard alone is load-bearing under "
      "mutation")
