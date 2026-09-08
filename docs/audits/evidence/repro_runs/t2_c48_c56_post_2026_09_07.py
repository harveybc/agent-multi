"""POST for order T2 C48-C56: the three accepted bypasses are DEAD
through the SAME productive APIs the PRE used, the hooks change no
scientific number, and the v2 mechanical rehearsal completes with
fully re-derived custody.

P1 -> C48: the execution record is v2 and pins the executor — an
     attacker-controlled pinned_commit dies on FORM; a v1-shaped
     record is a foreign schema; a nonexistent commit dies typed;
     the absent record still closes the chain with zero ledger.
P2 -> C49/C50/C51: the EXACT quadruple forgery (foreign unit_id,
     64-zero execution record, attacker code_identity, all 34
     mase_on_extreme_innovations -> 999.0, self-digest repaired,
     NPZ intact) refuses; the 34-extremes-only forgery refuses
     NAMING ITS PATH; joint obs+pred alteration refuses on the
     physical-series anchor; the intact record verifies.
P3 -> C52/C53/C54: the guard executes INSIDE units (origins, arms,
     ridge, seeds, epoch candidates) and a mid-unit stop leaves a
     typed UNCERTAIN attempt for explicit operator disposition;
     accumulated wall persists (resume never renews); the
     stop-file resolves at the design-declared <state_root>;
     --plan and failed gates leave ZERO writes (no out_root); the
     lock is monotonic (never unlinked); the NPZ is O_EXCL.

Zero sealed-bank series, zero scientific ledger, zero scores."""
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
TMP = Path(tempfile.mkdtemp(prefix="t2_c48_post_"))


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


def _die(fn, needle, what):
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), (needle, str(exc)[:160])
        print(f"  {what} -> {str(exc)[:104]}")
        return
    raise AssertionError(f"{what}: did NOT refuse")


print("== P1 DEAD (C48): the record pins the executor ==")
ra = _chain(TMP / "auth")
er = ra / "MUSASHI_T2_V6_EXECUTION_RECORD.json"
conf.T2_EXECUTION_RECORD_PATH = er
args = (design, conf._sha_file(SEALED),
        conf._sha_file(conf.T2_REVIEW_RECORD_PATH),
        conf._sha_file(STATE / "t2_public_data_manifest_20260906"
                               ".json"),
        conf._sha_file(STATE / "t2_bank_census_20260906.json"))
import subprocess  # noqa: E402
head = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                       "HEAD"], capture_output=True,
                      text=True).stdout.strip()
tree = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                       "HEAD^{tree}"], capture_output=True,
                      text=True).stdout.strip()
good_v2 = {"schema": "agent_multi.musashi_t2_execution_record.v2",
           "reviewed_at_date": "2026-09-07",
           "reviewer": "General Musashi",
           "decision": "OPEN_T2_CONFIRMATORY_EXECUTION",
           "sealed_design_file_sha256": conf._sha_file(SEALED),
           "sealed_design_self_sha256": design["design_sha256"],
           "design_review_record_sha256":
               conf._sha_file(conf.T2_REVIEW_RECORD_PATH),
           "manifest_sha256": args[3], "census_sha256": args[4],
           "executor_code_identity":
               conf.executor_code_identity(),
           "pinned_commit": head, "pinned_tree": tree}
attacker = "attacker-controlled-nonempty-string"
_install(er, {**good_v2, "pinned_commit": attacker})
_die(lambda: conf.verify_execution_record(*args),
     "40 lowercase hex", "PRE's exact attacker commit")
_install(er, {**good_v2, "pinned_commit": "f" * 40})
_die(lambda: conf.verify_execution_record(*args),
     "existing commit", "nonexistent 40-hex commit")
v1 = {"schema": "agent_multi.musashi_t2_execution_record.v1",
      "reviewed_at_date": "2026-09-07",
      "reviewer": "General Musashi",
      "decision": "OPEN_T2_CONFIRMATORY_EXECUTION",
      "sealed_design_file_sha256": conf._sha_file(SEALED),
      "sealed_design_self_sha256": design["design_sha256"],
      "candidate_commit": attacker}
_install(er, v1)
_die(lambda: conf.verify_execution_record(*args),
     "exact v2 schema", "the PRE's whole v1 record shape")
_install(er, {**good_v2,
              "executor_code_identity": {"attacker.py": "f" * 64}})
_die(lambda: conf.verify_execution_record(*args),
     "physical checkout surface", "forged code identity")
er.unlink()
lp = TMP / "ledger.json"
_die(lambda: conf.run_confirmatory(
        STATE / "t2_public_data_manifest_20260906.json", SEALED,
        lp, census_path=STATE / "t2_bank_census_20260906.json"),
     "T2_EXECUTION_RECORD_REQUIRED", "absent record, single path")
assert not lp.exists()
print("  => structurally closed; zero ledger")

print("\n== scientific identity: hooks change NO number ==")
import t2_assay_harness as hz  # noqa: E402
import t2_public_data_census as dc  # noqa: E402
co = hz.load_co()
census = dc.build_census()
unit = hz.load_task_unit(census, "sm_nile")
plain = hz.assay_unit(co, unit)
sink, labels = {}, []
sup = ex.make_fit_supervisor({"max_rss_bytes": 8 << 30})
hooked = hz.assay_unit(co, unit, sink=sink, guard=labels.append,
                       fit_supervisor=sup)
same = json.dumps(plain["rolling_origins"], sort_keys=True) == \
    json.dumps(hooked["rolling_origins"], sort_keys=True)
print("rolling_origins identical with/without hooks:", same,
      "| guard checkpoints:", len(labels),
      "| epoch candidates guarded:",
      any("epoch_candidate" in l for l in labels),
      "| sink entries:", len(sink), "| fit rows:",
      sum(1 for v in sink.values() if v[2] is not None))
assert same and len(sink) == 34

print("\n== v2 mechanical rehearsal at the final tip ==")
reh = TMP / "t2reh"
rc = ex.rehearse(reh)
assert rc == 0

print("\n== P2 DEAD (C49/C50/C51) on a REAL rehearsal record ==")
rp = reh / "units" / "RECORD_sm_nile.json"
npz = reh / "units" / "ARRAYS_sm_nile.npz"
base = json.loads(rp.read_text())


def _forged(mut):
    w = json.loads(json.dumps(base))
    mut(w)
    w["record_sha256"] = ex._self_sha(w, "record_sha256")
    fp = reh / "units" / "RECORD_sm_nile.json"
    fp.unlink()
    fp.write_text(json.dumps(w))
    os.chmod(fp, 0o600)
    return fp


def _quad(w):
    w["unit_id"] = "attacker::not_in_sealed_population"
    w["execution_record_sha256"] = "0" * 64
    w["code_identity"] = {"attacker.py": "f" * 64}
    n = 0
    for o in w["assay_record"]["rolling_origins"].values():
        for arm, entry in o["results"].items():
            pools = ([entry["metrics"]]
                     if arm == "seasonal_naive" else
                     [entry["ridge"], *entry["mlp_small"].values()])
            for m in pools:
                m["mase_on_extreme_innovations"] = 999.0
                n += 1
    assert n == 34


fp = _forged(_quad)
_die(lambda: ex.verify_unit_record(fp, npz, design),
     "not a development unit",
     "the PRE's EXACT quadruple forgery (34 extremes -> 999.0)")


def _extremes_only(w):
    for o in w["assay_record"]["rolling_origins"].values():
        for arm, entry in o["results"].items():
            pools = ([entry["metrics"]]
                     if arm == "seasonal_naive" else
                     [entry["ridge"], *entry["mlp_small"].values()])
            for m in pools:
                m["mase_on_extreme_innovations"] = 999.0


fp = _forged(_extremes_only)
_die(lambda: ex.verify_unit_record(fp, npz, design),
     "mase_on_extreme_innovations", "34 extremes alone (path "
     "named)")
fp = _forged(lambda w: w.update(
    {"execution_record_sha256": "0" * 64}))
_die(lambda: ex.verify_unit_record(fp, npz, design),
     "physical authority", "64-zero execution digest alone")
# joint obs+pred alteration (metrics still recompute!) dies on
# the physical-series anchor
with np.load(npz, allow_pickle=False) as z:
    data = {k: z[k] for k in z.files}
key = next(k for k in data if k.startswith("pred__")
           and "ridge" in k)
data[key] = data[key] + 3.7
data[key.replace("pred__", "obs__")] = \
    data[key.replace("pred__", "obs__")] + 3.7
npz.unlink()
with open(npz, "wb") as f:
    np.savez_compressed(f, **data)
os.chmod(npz, 0o600)
fp = _forged(lambda w: w.update(
    {"arrays_npz_sha256": ex._sha_file(npz)}))
_die(lambda: ex.verify_unit_record(fp, npz, design),
     "slice of the physical series",
     "obs+pred altered JOINTLY")
# restore the intact pair: the true record still verifies
shutil.copy(reh / "units" / "RECORD_sm_co2.json",
            TMP / "co2_probe.json")
os.chmod(TMP / "co2_probe.json", 0o600)
out = ex.verify_unit_record(
    reh / "units" / "RECORD_sm_co2.json",
    reh / "units" / "ARRAYS_sm_co2.npz", design)
print("  intact record ->", out)
assert out["verified_units"] == 1

print("\n== P3 DEAD (C52/C53/C54) ==")
calls = []


def tripping(label):
    calls.append(label)
    if len(calls) == 7:
        raise ex.T2BudgetStop("post probe bound", label)


root = TMP / "stoproot"
root.mkdir(mode=0o700)
authority = ex.physical_authority(design, "mechanical_rehearsal")
try:
    ex.run_unit(hz, co, unit, design, authority, root,
                "mechanical_rehearsal", guard=tripping)
    raise AssertionError("no stop")
except SystemExit as exc:
    print("a) mid-unit stop:", str(exc)[:88])
    assert "T2_BUDGET_STOP" in str(exc)
st, why = ex.adjudicate_unit_shallow(root / "units", "sm_nile")
print("   adjudication:", st, "|", why[:76])
assert st == "UNCERTAIN"
ex.declare_attempt_failed(root, "sm_nile", "post evidence: "
                          "operator disposition of the stopped "
                          "attempt")
st, _ = ex.adjudicate_unit_shallow(root / "units", "sm_nile")
print("   after recorded operator disposition:", st)
assert st == "TERMINAL_FAILED"
led = TMP / "wall.jsonl"
led.write_text(json.dumps({"session": "prior",
                           "elapsed_seconds": 100.0}) + "\n")
g = ex.BudgetGuard({"max_wall_seconds": 50,
                    "max_rss_bytes": 8 << 30},
                   TMP / "T2_STOP", led, "fresh")
_die(lambda: g.check("probe"), "never renews",
     "b) resume with 100s prior against a 50s wall")
g.close()
print("c) stop-file resolves at the design-declared state root:",
      ex.resolve_stop_file(design) == ex.STATE / "T2_STOP",
      "| declared:", design["resource_contract"]["stop_file"])
conf.T2_EXECUTION_RECORD_PATH = ra / "ABSENT.json"
miss_root = TMP / "plan_zero"
try:
    ex.main(["--plan", "--out-root", str(miss_root)])
except SystemExit as exc:
    assert "T2_EXECUTION_RECORD_REQUIRED" in str(exc)
print("d) --plan with the record ABSENT: typed refusal; out_root "
      "created:", miss_root.exists())
assert not miss_root.exists()
try:
    ex.main(["--execute", "--out-root", str(miss_root)])
except SystemExit:
    pass
assert not miss_root.exists()
print("   --execute refused likewise: ZERO writes before gates")
lockroot = TMP / "lockroot"
lockroot.mkdir(mode=0o700)
n1 = ex.acquire_lock(lockroot, "post-a")
_die(lambda: ex.acquire_lock(lockroot, "post-b"), "alive",
     "e) second executor against a live lock")
ex.release_lock(lockroot, n1, "post-a")
n2 = ex.acquire_lock(lockroot, "post-b")
print("   monotonic sessions:", n1, "->", n2,
      "| RELEASE_000001 exists:",
      (lockroot / "locks/RELEASE_000001.json").exists())
esrc = (REPO / "tools/t2_confirmatory_executor.py").read_text()
assert "lock.unlink" not in esrc
assert 'with open(npz_p, "wb")' not in esrc
assert "_excl_write_npz(npz_p, arrays)" in esrc
print("   lock never unlinked; NPZ exclusive-create (source)")

shutil.rmtree(TMP)
print("\nPOST CONFIRMED: P1/P2/P3 dead through the productive "
      "APIs; hooks change no number; the v2 rehearsal verifies "
      "with fully re-derived custody; scoring remains closed by "
      "the ABSENT external v2 record")
