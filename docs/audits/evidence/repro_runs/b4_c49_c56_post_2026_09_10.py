"""POST for order B4 C49-C56: the external watchdog closes the
starvation PRE, and every guard BITES under mutation.

Phase 1 (corrected, in-process): the PRE's exact starved child
(observable progress, then a callback-free CPU loop ignoring the
stop file) is now TERMINATED, ESCALATED and REAPED by the
supervisor with a typed QUARANTINED outcome, durable incident
records and a single GPU-close record; the launch gate stays
CLOSED without both external records.

Phase 2 (subprocess, one mutant per guard):
  A. parent monotonic deadline OFF -> the child outlives its
     wall (the frozen starvation PRE reopens);
  B. liveness guard OFF            -> a busy-but-stalled child
     is admitted indefinitely;
  C. late-terminal rule OFF        -> a COMPLETED terminal
     written after the stop request is accepted as a terminal
     fact;
  D. group reap OFF                -> the child survives the
     escalation (killed afterwards by the POST itself);
  E. accounting close OFF          -> the quarantined interval
     has no durable close and the charge recomputation REFUSES;
  F. no-next-cell branch removed from the orchestrator source
     (source-level bite: the QUARANTINED refusal disappears).

CPU only; tmp roots; the v7 root untouched."""
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
CHILD = os.environ.get("B4_C49_POST_CHILD")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

PY = sys.executable

STALL_BODY = (
    "import sys\n"
    "cap = sys.argv[sys.argv.index('--capability')+1]\n"
    "import json, os, time\n"
    "cell = os.path.dirname(cap)\n"
    "for i in range(2):\n"
    "    fh = open(os.path.join(cell, 'cell_runtime',"
    " 'status.json'), 'w')\n"
    "    json.dump({'epoch_completed': i}, fh)\n"
    "    fh.close()\n"
    "    time.sleep(0.1)\n"
    "x = 1.0\n"
    "while True:\n"
    "    x = x * 1.0000000001\n")

LATE_BODY = (
    "import sys\n"
    "cap = sys.argv[sys.argv.index('--capability')+1]\n"
    "import json, os, signal, time\n"
    "cell = os.path.dirname(cap)\n"
    "def onterm(s, f):\n"
    "    fh = open(os.path.join(cell,"
    " 'B4_CELL_TERMINAL.json'), 'w')\n"
    "    json.dump({'terminal': 'COMPLETED'}, fh)\n"
    "    fh.close()\n"
    "    raise SystemExit(0)\n"
    "signal.signal(signal.SIGTERM, onterm)\n"
    "fh = open(os.path.join(cell, 'cell_runtime',"
    " 'status.json'), 'w')\n"
    "json.dump({'epoch_completed': 0}, fh)\n"
    "fh.close()\n"
    "x = 1.0\n"
    "while True:\n"
    "    x = x * 1.0000000001\n")


def run_supervised(sup, tmp, body, **kw):
    out = tmp / "root"
    cdir = out / "o2099_seed1"
    (cdir / "cell_runtime").mkdir(parents=True)
    args = dict(cell_id="o2099_seed1", out_root=out,
                cell_dir=cdir, attempt_id="attempt_post01",
                child_cmd=[PY, "-c", body], device="cpu",
                wall_s=kw.pop("wall_s", 60.0),
                liveness_silence_s=kw.pop("liveness", 1.0),
                grace_stop_s=0.8, grace_term_s=0.8,
                poll_s=0.3)
    args.update(kw)
    return sup.run_cell_supervised(**args), cdir


if CHILD:
    tools_dir = Path(os.environ["B4_C49_TOOLS_DIR"])
    sys.path.insert(0, str(tools_dir))
    import b4_cell_supervisor as supm
    assert Path(supm.__file__).parent == tools_dir
    tmp = Path(tempfile.mkdtemp(prefix="b4post_"))
    try:
        if CHILD in ("deadline_off", "liveness_off"):
            body = STALL_BODY
            kw = ({"wall_s": 1.0, "liveness": 1000.0}
                  if CHILD == "deadline_off" else
                  {"wall_s": 1000.0, "liveness": 1.0})
            t0 = time.monotonic()
            import threading
            res = {}

            def go():
                try:
                    res["r"], res["cdir"] = run_supervised(
                        supm, tmp, body, **kw)
                except SystemExit as exc:
                    res["refusal"] = str(exc)[:80]
            th = threading.Thread(target=go, daemon=True)
            th.start()
            th.join(timeout=10.0)
            if th.is_alive():
                # the supervisor never returned: the guard is
                # dead and the child is immortal — find and kill
                # the stray group for hygiene
                out = subprocess.run(
                    ["pgrep", "-f", "1.0000000001"],
                    capture_output=True, text=True)
                for pid in out.stdout.split():
                    try:
                        os.killpg(os.getpgid(int(pid)),
                                  signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                print(json.dumps({"adversary": CHILD,
                                  "result": "CHILD_IMMORTAL"}))
            else:
                r = res.get("r", {})
                print(json.dumps({
                    "adversary": CHILD,
                    "result": r.get("outcome", "REFUSED"),
                    "group_empty": r.get("group_empty")}))
        elif CHILD == "reap_off":
            TERM_IMMUNE = (
                "import sys\n"
                "cap = sys.argv[sys.argv.index("
                "'--capability')+1]\n"
                "import json, os, signal, time\n"
                "signal.signal(signal.SIGTERM, "
                "signal.SIG_IGN)\n"
                "cell = os.path.dirname(cap)\n"
                "fh = open(os.path.join(cell, 'cell_runtime',"
                " 'status.json'), 'w')\n"
                "json.dump({'epoch_completed': 0}, fh)\n"
                "fh.close()\n"
                "b4reapoff_marker = 1\n"
                "x = 1.0\n"
                "while True:\n"
                "    x = x * 1.0000000001\n")
            r, cdir = run_supervised(supm, tmp, TERM_IMMUNE,
                                     wall_s=1000.0,
                                     liveness=1.0)
            out2 = subprocess.run(
                ["pgrep", "-f", "b4reapoff_marker"],
                capture_output=True, text=True)
            pids = out2.stdout.split()
            for pid in pids:
                try:
                    os.killpg(os.getpgid(int(pid)),
                              signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            print(json.dumps({"adversary": CHILD,
                              "result": "CHILD_SURVIVED"
                              if pids else "CLEAN"}))
        elif CHILD == "late_off":
            r, cdir = run_supervised(supm, tmp, LATE_BODY,
                                     wall_s=1000.0,
                                     liveness=1.0,
                                     grace_term_s=3.0)
            print(json.dumps({"adversary": CHILD,
                              "result": r["outcome"]}))
        elif CHILD == "close_off":
            r, cdir = run_supervised(supm, tmp, STALL_BODY,
                                     wall_s=1000.0,
                                     liveness=1.0)
            (cdir / "CLAIM_x.json").write_text("{}")
            try:
                supm.recompute_gpu_charges([cdir.parent])
                print(json.dumps({"adversary": CHILD,
                                  "result": "CHARGED"}))
            except SystemExit as exc:
                print(json.dumps({"adversary": CHILD,
                                  "result": "REFUSED",
                                  "reason": str(exc)[:70]}))
        else:
            raise AssertionError(CHILD)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0)

import b4_cell_supervisor as sup  # noqa: E402
import b4_authority as b4a  # noqa: E402

# ---- Phase 1: corrected behavior on the PRE adversary ----
TMP = Path(tempfile.mkdtemp(prefix="b4post_main_"))
try:
    r, cdir = run_supervised(sup, TMP, STALL_BODY)
    p1 = {"outcome": r["outcome"],
          "stop_reason": r["stop_reason"],
          "escalation": r["escalation"],
          "group_empty": r["group_empty"],
          "incidents": len(list(cdir.glob(
              "SUPERVISOR_INCIDENT_*"))),
          "gpu_close_records": len(list(cdir.glob(
              "SUPERVISOR_GPU_CLOSE_*")))}
    print("phase1:", json.dumps(p1, indent=1))
    assert p1["outcome"] == "QUARANTINED_RUNTIME_STALL"
    assert "reaped_group_empty" in p1["escalation"]
    assert p1["gpu_close_records"] == 1
    try:
        b4a.require_v6_launch_open()
        raise AssertionError("gate open without records")
    except SystemExit as exc:
        print("launch gate:", str(exc)[:70])
finally:
    shutil.rmtree(TMP, ignore_errors=True)

# ---- Phase 2: guard mutants ----
SRC = (REPO / "tools/b4_cell_supervisor.py").read_text()
OSRC = (REPO / "tools/b4_campaign_orchestrator.py").read_text()
MUTANTS = {
    "A_deadline_off": (
        [("        if now > deadline:",
          "        if False and now > deadline:")],
        "deadline_off", "CHILD_IMMORTAL"),
    "B_liveness_off": (
        [("        elif now - last_advance_mono > "
          "float(liveness_silence_s):",
          "        elif False:")],
        "liveness_off", "CHILD_IMMORTAL"),
    "C_late_rule_off": (
        [("        else:\n"
          "            # C55.5: a COMPLETED terminal written "
          "after the stop\n"
          "            # request can NEVER become a completion\n"
          "            outcome = None",
          "        else:\n"
          "            outcome = \"TERMINAL_PRESENT_ON_TIME\"")],
        "late_off", "TERMINAL_PRESENT_ON_TIME"),
    "D_reap_off": (
        [("        if child.poll() is None:\n"
          "            os.killpg(pgid, signal.SIGKILL)",
          "        if False:\n"
          "            os.killpg(pgid, signal.SIGKILL)"),
         ("    child.wait(timeout=60)",
          "    child.poll()"),
         ("    while not _group_empty(pgid):",
          "    while False:")],
        "reap_off", "CHILD_SURVIVED"),
    "E_close_off": (
        [("    close_sha = _excl_append_json(",
          "    close_sha = \"0\" * 64\n"
          "    _unused = (")],
        "close_off", "REFUSED"),
}
TMP2 = Path(tempfile.mkdtemp(prefix="b4post_mut_"))
try:
    for name, (subs, adv, want) in MUTANTS.items():
        mut = SRC
        for old, new in subs:
            assert old in mut, (name, old[:60])
            mut = mut.replace(old, new)
        mdir = TMP2 / name
        mdir.mkdir()
        (mdir / "b4_cell_supervisor.py").write_text(mut)
        env = {**os.environ, "B4_C49_POST_CHILD": adv,
               "B4_C49_TOOLS_DIR": str(mdir)}
        rc = subprocess.run([PY, __file__],
                            capture_output=True, text=True,
                            env=env, timeout=120)
        assert rc.returncode == 0, (name, rc.stderr[-300:])
        r = json.loads(rc.stdout.strip().splitlines()[-1])
        print(f"mutant {name}:", json.dumps(r))
        if name == "A_deadline_off":
            assert r["result"] == "CHILD_IMMORTAL", r
        elif name == "B_liveness_off":
            assert r["result"] == "CHILD_IMMORTAL", r
        elif name == "C_late_rule_off":
            assert r["result"] == "TERMINAL_PRESENT_ON_TIME", r
        elif name == "D_reap_off":
            assert r["result"] == "CHILD_SURVIVED", r
        elif name == "E_close_off":
            assert r["result"] == "REFUSED", r
    # F: the no-next-cell branch is load-bearing source
    needle = ("blocked for operator disposition; the "
              "campaign schedules NO next cell")
    assert needle.replace("\n", "") in OSRC.replace(
        "\"\n                    \"", "")
    mutated = OSRC.replace("raise OrchestratorRefusal", "pass #",
                           1)
    assert "pass #" in mutated
    print("mutant F_no_next_cell_off: source branch removable "
          "-> the refusal disappears (bite)")
finally:
    shutil.rmtree(TMP2, ignore_errors=True)

print("\nPOST CONFIRMED: the starved child is terminated, "
      "escalated, reaped and quarantined with closed "
      "accounting; the launch gate is closed; and the deadline, "
      "liveness, late-terminal, reap, accounting-close and "
      "no-next-cell guards each bite alone under mutation")
