"""POST for the M4 CONFIRMATION execution BODY (2026-09-26).

What was missing: execute_confirmation() verified its gates, wrote
the pre-result ledger and RETURNED. Its closing note promised
"unit execution proceeds only beyond this point" and nothing in
the module's execution path went beyond it — the only call to
_run_intervention_unit_v5 was inside the DEVELOPMENT mechanics
probe. With every approval installed, the screen would have
produced a PENDING ledger and zero fitted units.

Phase 1 — the gate is untouched and still first:
  - run_confirmation() refuses at the two-record gate with the
    records ABSENT and creates nothing;
  - with the body stubbed to raise, the refusal still comes
    first: the body is never entered;
  - plan still reports 3024 units with execution_open false;
  - the bank's C33 kill-17 guard still refuses a CONFIRMATION
    unit for a caller that does not pass the new flag;
  - the sealed DEVELOPMENT mechanics probe is behaviour
    identical (2 records, 0 CONFIRMATION artifacts).

Phase 2 — the body, over DEVELOPMENT units only, through the real
process boundary: a fresh census, an idempotent re-run, a batched
session, a SIGKILL inside a unit followed by a restart that
completes the census, and the set-aside step proven load-bearing
(removed, the sealed "partial log without its durable predecessor
state — UNCERTAIN" refusal bites).

Phase 3 — the sealed evidence: the five M4 batteries and the
frozen C37/POST battery of 2026-09-10, re-run unchanged.

Phase 4 — nothing was authorized and nothing was created: both
external records are still absent and no CONFIRMATION array,
score or ledger exists anywhere.

CPU only. No CONFIRMATION screen was run.
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))

import m4_confirmation_protocol as cp  # noqa: E402
import m4_confirmation_runner as cr  # noqa: E402
import m4_generator_bank as gb  # noqa: E402
import m4_v5_runner as rn  # noqa: E402

RUNNER = str(REPO / "tools" / "m4_confirmation_runner.py")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
TMP = Path(tempfile.mkdtemp(prefix="m4body_post_"))


def head(msg):
    print("\n== " + msg)


try:
    # ------------------------- Phase 1 -------------------------
    head("Phase 1: the gate chain, unchanged and still first")
    assert not cp.MUSASHI_REVIEW_RECORD_PATH.exists() and \
        not cp.OWNER_EXECUTION_RECORD_PATH.exists(), (
            "an external record exists on this host — this POST "
            "never installs one")
    out = TMP / "gate"
    try:
        cr.run_confirmation(REPO, out)
        raise AssertionError("run_confirmation did NOT refuse")
    except SystemExit as e:
        assert "ABSENT" in str(e), str(e)
        assert not out.exists(), (
            "artifacts were created before the refusal")
        print("run_confirmation: REFUSED at the two-record gate "
              "before any artifact —", str(e)[:70])

    body_calls = []
    real_body = cr.execute_confirmation_units

    def _never(*a, **k):
        body_calls.append(1)
        raise AssertionError("the body ran without records")

    cr.execute_confirmation_units = _never
    try:
        cr.run_confirmation(REPO, TMP / "gate2")
        raise AssertionError("run_confirmation did NOT refuse")
    except SystemExit as e:
        assert "ABSENT" in str(e)
    finally:
        cr.execute_confirmation_units = real_body
    assert body_calls == [], body_calls
    assert not (TMP / "gate2").exists()
    print("with the body stubbed to raise: still REFUSED, body "
          "entered", len(body_calls), "times")

    plan = cr.plan_confirmation(REPO)
    assert plan["units_total"] == 3024
    assert plan["execution_open"] is False
    print("plan:", plan["units_total"], "units,",
          plan["eligible_slots"], "slots, execution_open",
          plan["execution_open"], "| records present:",
          plan["musashi_review_record_present"],
          plan["owner_execution_record_present"])

    design = cp.bind_calibration_evidence(REPO)["design"]
    u_conf = rn._iv_unit("CONFIRMATION", "sine", "white", 16,
                         0, 0)
    o = TMP / "kill17"
    (o / "intervention").mkdir(parents=True)
    try:
        rn._run_intervention_unit_v5(design, u_conf, o,
                                     cr.new_accounting())
        raise AssertionError("kill 17 did not refuse")
    except SystemExit as e:
        assert "RESERVED" in str(e), str(e)
        print("kill 17 (default caller, CONFIRMATION unit): "
              "REFUSED —", str(e)[:60])
    assert list((o / "intervention").iterdir()) == []

    sealed = cr.development_mechanics_probe(REPO, TMP / "sealed")
    assert sealed["records"] == 2
    assert sealed["confirmation_artifacts"] == 0
    print("sealed DEVELOPMENT mechanics probe:",
          sealed["records"], "records,",
          sealed["confirmation_artifacts"],
          "CONFIRMATION artifacts")

    # ------------------------- Phase 2 -------------------------
    head("Phase 2: the body over DEVELOPMENT units, real "
         "process boundary")

    def cli(out_root, *extra):
        return subprocess.run(
            [sys.executable, RUNNER,
             "development-execution-probe", "--out",
             str(out_root), *extra],
            capture_output=True, text=True, env=ENV, timeout=900)

    full = TMP / "body_full"
    t0 = time.monotonic()
    r = cli(full)
    assert r.returncode == 0, r.stderr[-600:]
    b = json.loads(r.stdout)
    wall = round(time.monotonic() - t0, 2)
    assert b["census_complete"] is True
    assert b["units_complete"] == b["units_total"] == 4
    assert b["confirmation_artifacts"] == 0
    per_unit = b["wall_seconds"] / b["units_total"]
    print(json.dumps({k: b[k] for k in (
        "units_total", "units_complete",
        "units_new_this_session",
        "generators_disjointness_verified",
        "prior_role_digests", "session_status", "accounting",
        "wall_seconds")}))
    print(f"measured basis {per_unit:.3f} s/unit "
          f"(process wall {wall} s) -> 3024 CONFIRMATION units "
          f"project to {per_unit * 3024 / 60:.1f} min of one "
          f"CPU inside the sealed "
          f"{design['resources']['max_wall_seconds']} s wall")

    r2 = cli(full)
    assert r2.returncode == 0, r2.stderr[-600:]
    b2 = json.loads(r2.stdout)
    assert b2["units_new_this_session"] == 0
    assert b2["units_complete"] == 4
    print("idempotent re-run: 0 new units,",
          b2["units_complete"], "verified complete")

    batched = TMP / "body_batched"
    bb = json.loads(cli(batched, "--batch-units", "2").stdout)
    assert bb["units_new_this_session"] == 2
    assert bb["units_pending"] == 2
    assert bb["session_status"] == "BATCH_FILLED_RESUMABLE"
    kept = sorted((batched / "intervention"
                   ).glob("*_summary.json"))
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in kept}
    bb2 = json.loads(cli(batched).stdout)
    assert bb2["census_complete"] is True
    assert bb2["units_new_this_session"] == 2
    for p in kept:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == \
            digests[p.name], f"{p.name} was rewritten on resume"
    print("batched session 2+2: complete, and the first two "
          "records are byte-identical after the resume")

    # SIGKILL inside a unit, then restart
    interrupted = TMP / "body_interrupted"
    iv = interrupted / "intervention"
    p = subprocess.Popen(
        [sys.executable, RUNNER, "development-execution-probe",
         "--out", str(interrupted)], env=ENV,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    trigger = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < 120:
        if iv.is_dir():
            ns = len(list(iv.glob("*_summary.json")))
            nl = len(list(iv.glob("*.jsonl")))
            if ns >= 1 and nl > 4 * ns:
                p.send_signal(signal.SIGKILL)
                trigger = (ns, nl)
                break
        if p.poll() is not None:
            break
        time.sleep(0.002)
    p.wait()
    assert trigger is not None, "never caught an in-flight unit"
    print("SIGKILL delivered inside a unit: rc", p.returncode,
          "| records", trigger[0], "| arm logs", trigger[1],
          "| session reports",
          len(list(interrupted.glob("SESSION_*"))))

    # the set-aside step is load-bearing: remove it and the
    # SEALED partial-log refusal bites on the same state
    src = (REPO / "tools" / "m4_confirmation_runner.py"
           ).read_text()
    anchor = "        aborted += abort_partial_unit(out, u)"
    assert anchor in src, "set-aside anchor missing"
    mdir = TMP / "mutant_set_aside_off"
    mdir.mkdir()
    (mdir / "m4_confirmation_runner.py").write_text(
        src.replace(anchor, "        pass  # set-aside REMOVED"))
    drv = mdir / "driver.py"
    drv.write_text(
        "import importlib.util, json, sys\n"
        f"sys.path.insert(0, {str(REPO / 'tools')!r})\n"
        "spec = importlib.util.spec_from_file_location("
        f"'mut', {str(mdir / 'm4_confirmation_runner.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['mut'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        f"assert mod.__file__.startswith({str(mdir)!r}), "
        "mod.__file__\n"
        "try:\n"
        "    mod.development_execution_probe("
        f"{str(REPO)!r}, {str(interrupted)!r})\n"
        "    print(json.dumps({'result': 'COMPLETED'}))\n"
        "except SystemExit as e:\n"
        "    print(json.dumps({'result': 'REFUSED', "
        "'reason': str(e)[:200]}))\n")
    mr = subprocess.run([sys.executable, str(drv)], env=ENV,
                        capture_output=True, text=True,
                        timeout=900)
    assert mr.returncode == 0, mr.stderr[-500:]
    mut = json.loads(mr.stdout.strip().splitlines()[-1])
    assert mut["result"] == "REFUSED", mut
    assert "partial intervention log without" in mut["reason"], \
        mut
    print("mutant set_aside_off:", json.dumps(mut)[:220])

    rr = cli(interrupted)
    assert rr.returncode == 0, rr.stderr[-600:]
    br = json.loads(rr.stdout)
    assert br["census_complete"] is True
    assert br["units_complete"] == 4
    assert br["partial_attempts_set_aside"] >= 1
    assert br["confirmation_artifacts"] == 0
    preserved = sorted(
        str(q.relative_to(interrupted))
        for q in (interrupted / "ABORTED_PARTIALS").rglob("*")
        if q.is_file())
    print("restart after the kill:",
          json.dumps({k: br[k] for k in (
              "units_complete", "units_new_this_session",
              "partial_attempts_set_aside", "session_status")}))
    print("partial bytes PRESERVED, not deleted:",
          len(preserved))
    for q in preserved:
        print("   ", q)

    # ------------------------- Phase 3 -------------------------
    head("Phase 3: the sealed batteries, re-run unchanged")
    m4_tests = ["tests/test_m4_confirmation_protocol.py",
                "tests/test_m4_v5_protocol.py",
                "tests/test_m4_intervention.py",
                "tests/test_m4_numeric_incident.py",
                "tests/test_m4_residual_capacity.py",
                "tests/test_m4_confirmation_execution_body.py"]
    t = subprocess.run([sys.executable, "-m", "pytest",
                        *m4_tests, "-q"], cwd=REPO,
                       capture_output=True, text=True, env=ENV,
                       timeout=3600)
    tail = [ln for ln in t.stdout.strip().splitlines() if ln][-1]
    print("pytest (5 sealed M4 batteries + the new body "
          "battery):", tail)
    assert t.returncode == 0, t.stdout[-1500:]

    frozen = subprocess.run(
        [sys.executable, str(REPO / "docs/audits/evidence"
                             / "repro_runs"
                             / "m4_c32_c38_post_2026_09_10.py")],
        capture_output=True, text=True, env=ENV, timeout=3600)
    assert frozen.returncode == 0, frozen.stderr[-800:]
    for ln in frozen.stdout.strip().splitlines():
        if ln.strip():
            print("   [frozen C37/POST]", ln)

    # ------------------------- Phase 4 -------------------------
    head("Phase 4: nothing authorized, nothing created")
    assert not cp.MUSASHI_REVIEW_RECORD_PATH.exists()
    assert not cp.OWNER_EXECUTION_RECORD_PATH.exists()
    state = Path.home() / ".local/share/agent-multi"
    assert list(state.glob("*m4*confirmation*")) == []
    leaked = [str(q) for q in TMP.rglob("*")
              if "CONFIRMATION" in q.name]
    assert leaked == [], leaked
    print("both external records: ABSENT | state root: zero "
          "CONFIRMATION artifacts | this POST's roots: zero "
          "CONFIRMATION-named artifacts")
    conf_gen = gb.generate("CONFIRMATION", "sine", "white", 0,
                           allow_confirmation=True)
    assert cr.generator_array_digests(conf_gen).isdisjoint(
        cr.role_array_digests(design, [("sine", "white")]))
    print("role namespaces remain byte-disjoint in the array "
          "domain (checked, not assumed)")

    print("\nPOST CONFIRMED: the CONFIRMATION screen now has an "
          "execution body — the 3024 units are driven through "
          "the sealed v5 limit machinery, resumably, with "
          "per-array role disjointness and atomic per-unit "
          "records — and the two-record gate still refuses "
          "FIRST with the records absent, creating nothing and "
          "never entering the body. No CONFIRMATION array, "
          "score or ledger was created.")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
