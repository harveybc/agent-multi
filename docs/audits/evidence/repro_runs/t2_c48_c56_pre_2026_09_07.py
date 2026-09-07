"""PRE freeze for order T2 C48-C56: Musashi's three accepted
bypasses reproduce from b49aa1f6 through the REAL productive APIs.

P1. verify_execution_record() requires only a NONEMPTY STRING for
    candidate_commit — no hex form, no existence, no HEAD, no clean
    tree, no executor code identity. An attacker-controlled string
    is ACCEPTED: the record binds the design but not the program.
P2. verify_unit_record() believes the self-signed wrapper: with
    unit_id, execution_record_sha256, code_identity and ALL 34
    mase_on_extreme_innovations forged and ONLY the producer
    self-digest recomputed (NPZ intact), the public productive API
    returns {'verified_units': 1}.
P3. The sealed bounds do not govern the interior of a unit
    (_budget only before each unit; assay_unit takes no guard);
    wall restarts on every resume; the stop-file is read from
    <out_root> while the sealed design declares <state_root>;
    --plan traverses run_confirmatory() and creates the ledger;
    main() creates out_root BEFORE the external record verifies;
    the lock is released by unlink(); the NPZ is written with
    open(.., "wb").

Zero sealed-bank series, zero scientific ledger (fixture ledgers in
throwaway roots only), zero scores, CPU only."""
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault(
    "B4_T1_PREPROCESSOR_ROOT",
    str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))

import numpy as np  # noqa: E402

import t2_confirmatory as conf  # noqa: E402
import t2_confirmatory_executor as ex  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"
SEALED = STATE / "t2_screen_design_SEALED_V6.json"
design = conf.strict_json_load(SEALED, "sealed design")
TMP = Path(tempfile.mkdtemp(prefix="t2_c48_pre_"))


def _chain(base):
    ra = base / "agent-multi" / "reviewer_authority"
    for d in (base, base / "agent-multi", ra):
        d.mkdir(mode=0o700, exist_ok=True)
        os.chmod(d, 0o700)
    return ra


def _install(path, doc):
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(doc))
    os.chmod(path, 0o600)


print("== P1: attacker-controlled candidate_commit is ACCEPTED ==")
ra = _chain(TMP / "auth")
er = ra / "MUSASHI_T2_V6_EXECUTION_RECORD.json"
conf.T2_EXECUTION_RECORD_PATH = er
attacker = "attacker-controlled-nonempty-string"
good_v1 = {"schema": "agent_multi.musashi_t2_execution_record.v1",
           "reviewed_at_date": "2026-09-07",
           "reviewer": "General Musashi",
           "decision": "OPEN_T2_CONFIRMATORY_EXECUTION",
           "sealed_design_file_sha256": conf._sha_file(SEALED),
           "sealed_design_self_sha256": design["design_sha256"],
           "candidate_commit": attacker}
_install(er, good_v1)
out = conf.verify_execution_record(design, conf._sha_file(SEALED))
print("verify_execution_record ->",
      "ACCEPTED_UNPINNED_EXECUTOR_IDENTITY",
      out["candidate_commit"])
assert out["candidate_commit"] == attacker
src = (REPO / "tools/t2_confirmatory.py").read_text()
seg = src[src.index("def verify_execution_record"):]
seg = seg[:seg.index("\nMETRIC_DOMAINS")]
for probe in ("git", "pinned_commit", "pinned_tree", "[0-9a-f]{40}",
              "code_identity"):
    assert probe not in seg, probe
print("=> no form / existence / HEAD / tree / code-identity check "
      "exists in the gate")

print("\n== P2: forged wrapper with repaired self-digest is "
      "ACCEPTED ==")
run_root = TMP / "unit_run"
os.makedirs(run_root, mode=0o700)
import t2_assay_harness as hz  # noqa: E402
import t2_public_data_census as dc  # noqa: E402
co = hz.load_co()
census = dc.build_census()
unit = hz.load_task_unit(census, "sm_nile")
sealed_ids = set(design["task_population"]["series_ids"])
assert "sm_nile" not in sealed_ids
authority = {"sealed_design_file_sha256": conf._sha_file(SEALED),
             "sealed_design_self_sha256": design["design_sha256"],
             "design_review_record_sha256":
                 design["design_review_record_sha256"],
             "execution_record_sha256":
                 "REHEARSAL_NO_EXECUTION_RECORD_MECHANICS_ONLY",
             "manifest_sha256": "REHEARSAL",
             "census_sha256": "REHEARSAL"}
ex.run_unit(hz, co, unit, design, authority, run_root)
rp = run_root / "units" / "RECORD_sm_nile.json"
npz = run_root / "units" / "ARRAYS_sm_nile.npz"
wrapper = json.loads(rp.read_text())
wrapper["unit_id"] = "attacker::not_in_sealed_population"
wrapper["execution_record_sha256"] = "0" * 64
wrapper["code_identity"] = {"attacker.py": "f" * 64}
n_forged = 0
for o in wrapper["assay_record"]["rolling_origins"].values():
    for arm, entry in o["results"].items():
        pools = ([entry["metrics"]] if arm == "seasonal_naive"
                 else [entry["ridge"],
                       *entry["mlp_small"].values()])
        for m in pools:
            m["mase_on_extreme_innovations"] = 999.0
            n_forged += 1
print("forged fields: unit_id, execution_record_sha256 (64 zeros),"
      " code_identity, and", n_forged,
      "mase_on_extreme_innovations -> 999.0")
assert n_forged == 34
wrapper["record_sha256"] = ex._self_sha(wrapper, "record_sha256")
fp = TMP / "forged_record.json"
fp.write_text(json.dumps(wrapper))
res = ex.verify_unit_record(fp, npz, design)
print("verify_unit_record ->", res,
      "ACCEPTED_FORGED_AUTHORITY_UNIT_AND_EXTREMES")
assert res == {"verified_units": 1}

print("\n== P3: bounds do not govern the interior; effects precede "
      "gates ==")
esrc = (REPO / "tools/t2_confirmatory_executor.py").read_text()
hsrc = (REPO / "tools/t2_assay_harness.py").read_text()
main_seg = esrc[esrc.index("def main"):esrc.index("def rehearse")]
assert main_seg.count("_budget(") == 1        # once, before a unit
assert "def assay_unit(co, unit: dict, h: int = 1, " \
       "sink: dict = None) -> dict" in hsrc   # no guard parameter
assert "guard" not in esrc
print("a) _budget() runs once per unit; assay_unit() takes no "
      "guard — wall/RSS/stop cannot fire inside a unit")
assert "started = time.time()" in main_seg
assert "T2_WALL" not in esrc
print("b) started = time.time() at every launch — resume restarts "
      "the 4 h wall; no durable accumulation exists")
stop_decl = design["resource_contract"]["stop_file"]
assert stop_decl == "<state_root>/T2_STOP"
assert 'Path(out_root) / "T2_STOP"' in esrc
print(f"c) sealed design declares {stop_decl!r}; the executor "
      "polls <out_root>/T2_STOP — a stop at the declared root is "
      "invisible")
assert "lock.unlink(missing_ok=True)" in esrc
assert 'with open(npz_p, "wb") as f:' in esrc
print("d) lock released by unlink(); NPZ written via open('wb') — "
      "no exclusive creation, no durable release protocol")

# behavioral: out_root is created BEFORE the execution gate refuses
conf.T2_EXECUTION_RECORD_PATH = ra / "ABSENT.json"
miss_root = TMP / "plan_norecord"
assert not miss_root.exists()
rc = None
try:
    ex.main(["--plan", "--out-root", str(miss_root)])
except SystemExit as exc:
    rc = str(exc)
assert "T2_EXECUTION_RECORD_REQUIRED" in rc
assert miss_root.exists()
print("e) --plan with the record ABSENT: typed refusal, but "
      "out_root WAS created — an effect before the gate")

# behavioral: --plan with a (v1) record CREATES the ledger
conf.T2_EXECUTION_RECORD_PATH = er
plan_root = TMP / "plan_ledger"
rc = ex.main(["--plan", "--out-root", str(plan_root)])
assert rc == 0
led = plan_root / "T2_ATTEMPT_LEDGER.json"
print("f) --plan exit 0; ledger created by the plan:",
      led.exists(), "->", json.loads(led.read_text())["schema"])
assert led.exists()

shutil.rmtree(TMP)
print("\nPRE CONFIRMED at b49aa1f6: P1 unpinned executor identity, "
      "P2 forged wrapper accepted, P3 ungoverned interior + "
      "effectful plan — all frozen")
