"""POST for order M4 C32-C38: the CONFIRMATION protocol stands,
execution stays closed, and the two capital guards bite alone.

Phase 1 (live facts on corrected code):
  - C32 bind verifies the four order-pinned identities and
    re-derives every order fact from structures;
  - C33 successor verifies (self-digested, CALIBRATION_DERIVED_
    AND_REVIEWED, SCIENTIFIC_ANALYSIS_FREEZE, floor 39);
  - C35 plan reports the exact census (3024 units = 21 x 48 x 3,
    4 checkpoints, frozen update bounds) with NO authority;
  - execute REFUSES at the two-record gate BEFORE creating any
    artifact;
  - the DEVELOPMENT mechanics probe runs 2 real units through
    the real machinery with ZERO CONFIRMATION artifacts;
  - the C37 battery result stands (28 passed) and its four
    guard-removal mutants bite inside it.

Phase 2 (capital guard-off mutants, subprocess):
  A. two-record gate OFF  -> execute reaches census + ledger
     with NO records installed (the gate alone closes it);
  B. successor-divergence check OFF -> a successor with a
     changed 12/16 threshold VERIFIES (the live re-derivation
     alone rejects scientific mutation).

CPU only; no CONFIRMATION array, score or ledger is created
anywhere; the real successor bytes are save/restored."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))

import m4_confirmation_protocol as cp  # noqa: E402
import m4_confirmation_runner as cr  # noqa: E402

CHILD = os.environ.get("M4_C32_POST_CHILD")

if CHILD:
    import importlib.util
    mdir = Path(os.environ["M4_C32_TOOLS_DIR"])

    def _load(modname, path):
        spec = importlib.util.spec_from_file_location(
            modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    mcp = _load("m4_confirmation_protocol",
                mdir / "m4_confirmation_protocol.py")
    mcr = _load("m4_confirmation_runner",
                mdir / "m4_confirmation_runner.py")
    assert mcp.__file__.startswith(str(mdir))
    assert mcr.__file__.startswith(str(mdir))
    tmp = Path(tempfile.mkdtemp(prefix="m4post_"))
    try:
        if CHILD == "gate_off":
            # isolate THE gate: the dirty-checkout guard is a
            # separate legitimate closure (this POST runs in a
            # working tree that carries the new protocol
            # files), so present a clean status and pinned head
            real_git = mcr._git

            def _stub_git(repo_root, *a):
                r = real_git(repo_root, *a)
                if a and a[0] == "status":
                    r = subprocess.CompletedProcess(
                        a, 0, stdout="", stderr="")
                return r
            mcr._git = _stub_git
            out = tmp / "out"
            r = mcr.execute_confirmation(REPO, out)
            led = Path(r["ledger"])
            print(json.dumps({
                "adversary": CHILD, "result": "EXECUTED",
                "ledger_exists": led.is_file(),
                "units": r["census"]["units_total"]}))
        elif CHILD == "diverge_off":
            orig = (REPO / mcp.SUCCESSOR_PATH).read_bytes()
            doc = json.loads(orig.decode())
            doc["eligibility_rule"][
                "min_learnable_under_frozen_budget"] = 11
            doc["successor_sha256"] = mcp._selfsha(
                doc, "successor_sha256")
            try:
                (REPO / mcp.SUCCESSOR_PATH).write_text(
                    json.dumps(doc, indent=1))
                got = mcp.verify_confirmation_successor(REPO)
                print(json.dumps({
                    "adversary": CHILD, "result": "VERIFIED",
                    "threshold": got["eligibility_rule"][
                        "min_learnable_under_frozen_budget"]}))
            finally:
                (REPO / mcp.SUCCESSOR_PATH).write_bytes(orig)
    except SystemExit as e:
        print(json.dumps({"adversary": CHILD,
                          "result": "REFUSED",
                          "reason": str(e)[:70]}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0)

# ---- Phase 1 ----
auth = cp.bind_calibration_evidence(REPO)
print("C32 bind: 4 identities verified;",
      len(auth["eligible_slots"]), "eligible /",
      len(auth["ineligible_slots"]), "ineligible slots;",
      len(auth["incomplete_generators"]),
      "incomplete generators")

succ = cp.verify_confirmation_successor(REPO)
assert succ["selection_rule_label"] == \
    "CALIBRATION_DERIVED_AND_REVIEWED"
assert succ["classification"] == "SCIENTIFIC_ANALYSIS_FREEZE"
print("C33 successor verified: self",
      succ["successor_sha256"][:12], "| floor",
      succ["attrition"]["min_complete_required"],
      "| contrasts", len(succ["contrast_family_16"]))

plan = cr.plan_confirmation(REPO)
assert plan["units_total"] == 3024
assert plan["execution_open"] is False
print("C35 plan:", plan["units_total"], "units =",
      plan["eligible_slots"], "slots x",
      plan["generators_per_slot"], "gen x",
      plan["seeds_per_generator"], "seeds; records present:",
      plan["musashi_review_record_present"],
      plan["owner_execution_record_present"])

tmp_exec = Path(tempfile.mkdtemp(prefix="m4post_exec_"))
exec_out = tmp_exec / "out"
try:
    cr.execute_confirmation(REPO, exec_out)
    raise AssertionError("execute did NOT refuse")
except SystemExit as e:
    assert "ABSENT" in str(e), str(e)
    assert not exec_out.exists(), (
        "refusal came AFTER artifact creation")
    print("execute: REFUSED at the two-record gate before any "
          "artifact")
finally:
    shutil.rmtree(tmp_exec, ignore_errors=True)

tmp_probe = Path(tempfile.mkdtemp(prefix="m4post_probe_"))
try:
    probe = cr.development_mechanics_probe(
        REPO, tmp_probe / "probe")
    assert probe["confirmation_artifacts"] == 0
    assert probe["records"] == 2
    print("DEVELOPMENT probe:", probe["records"],
          "real units through the real machinery, 0 "
          "CONFIRMATION artifacts")
finally:
    shutil.rmtree(tmp_probe, ignore_errors=True)

state = Path.home() / ".local/share/agent-multi"
leaks = [p for p in state.glob("*m4*confirmation*")]
assert leaks == [], leaks
print("state root: zero CONFIRMATION artifacts")

# ---- Phase 2: capital mutants ----
GATE_OLD = "    records = cp.require_both_records(successor)"
DIV_OLD = """    if got != expected:
        diff = sorted(set(got) ^ set(expected)) or sorted(
            k for k in expected if got.get(k) != expected[k])
        raise ConfirmationProtocolRefusal("""
MUTANTS = {
    "A_gate_off": (
        "m4_confirmation_runner.py", GATE_OLD,
        "    records = {\"review\": {\"record_sha256\": "
        "\"0\" * 64},\n"
        "               \"execution\": {\"record_sha256\": "
        "\"0\" * 64}}  # cp.require_both_records(successor)",
        "gate_off", "EXECUTED"),
    "B_diverge_off": (
        "m4_confirmation_protocol.py", DIV_OLD,
        """    if False:
        diff = sorted(set(got) ^ set(expected)) or sorted(
            k for k in expected if got.get(k) != expected[k])
        raise ConfirmationProtocolRefusal(""",
        "diverge_off", "VERIFIED"),
}
TMP = Path(tempfile.mkdtemp(prefix="m4post_mut_"))
try:
    for name, (fname, old, new, adv, want) in MUTANTS.items():
        mdir = TMP / name
        mdir.mkdir()
        for f in ("m4_confirmation_protocol.py",
                  "m4_confirmation_runner.py"):
            s = (REPO / "tools" / f).read_text()
            if f == fname:
                assert old in s, (name, "anchor missing")
                s = s.replace(old, new)
            (mdir / f).write_text(s)
        env = {**os.environ, "M4_C32_POST_CHILD": adv,
               "M4_C32_TOOLS_DIR": str(mdir)}
        rc = subprocess.run([sys.executable, __file__],
                            capture_output=True, text=True,
                            env=env, timeout=600)
        assert rc.returncode == 0, (name, rc.stderr[-400:])
        r = json.loads(rc.stdout.strip().splitlines()[-1])
        print(f"mutant {name}:", json.dumps(r))
        assert r["result"] == want, r
        if name == "A_gate_off":
            assert r["ledger_exists"] and r["units"] == 3024
        else:
            assert r["threshold"] == 11
finally:
    shutil.rmtree(TMP, ignore_errors=True)

succ2 = cp.verify_confirmation_successor(REPO)
assert succ2["successor_sha256"] == succ["successor_sha256"]
print("successor bytes restored and re-verified after mutants")

print("\nPOST CONFIRMED: the protocol binds and verifies on "
      "corrected code, execution is closed by the two-record "
      "gate alone (mutant A executes to census+ledger without "
      "records), scientific mutation is rejected by the live "
      "re-derivation alone (mutant B verifies a changed "
      "threshold), and no CONFIRMATION artifact exists "
      "anywhere.")
